import json
import os
import re
from typing import Any, Dict, List, Tuple

import streamlit as st
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv


# =========================================================
# ENVIRONMENT / OPENAI
# =========================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error(
        "🔑 OpenAI API Key not found. "
        "Set OPENAI_API_KEY in your .env file or Streamlit secrets."
    )
    st.stop()

client = OpenAI(api_key=api_key)


# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Policy Tester",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AI Policy Tester")
st.write(
    "Extract complete policy information from a PDF, validate JSON test data, "
    "or compare two policy PDFs."
)


# =========================================================
# CONSTANTS
# =========================================================

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# IMPORTANT:
# Do NOT send the entire PDF to one AI request.
# The uploaded policy can contain 40+ pages and thousands of fields.
# We split it into smaller page-based chunks.
MAX_CHARS_PER_CHUNK = 12000

# Number of test-data fields sent to one validation request.
VALIDATION_FIELDS_PER_BATCH = 30


# =========================================================
# GENERIC HELPERS
# =========================================================

def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Parse JSON returned by the model.

    Handles occasional markdown fences even though the prompt asks
    for JSON only.
    """
    text = (text or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to recover the outermost JSON object.
        start = text.find("{")
        end = text.rfind("}")

        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])

        raise


def normalize_field_name(value: Any) -> str:
    """
    Normalizes field names for deduplication/comparison.
    """
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalize_value(value: Any) -> str:
    """
    Normalizes values for duplicate detection.
    """
    if value is None:
        return ""

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)

    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def chunk_list(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_pages(uploaded_file) -> Tuple[List[Dict[str, Any]], int]:
    """
    Extract text page-by-page.

    We keep page boundaries because the AI needs the original page number.
    """
    reader = PdfReader(uploaded_file)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        text = text.strip()

        pages.append(
            {
                "page": page_number,
                "text": text,
            }
        )

    return pages, len(reader.pages)


def build_page_chunks(
    pages: List[Dict[str, Any]],
    max_chars: int = MAX_CHARS_PER_CHUNK,
) -> List[str]:
    """
    Build chunks without losing page numbers.

    A chunk can contain multiple PDF pages, but every page keeps its
    own PAGE marker.
    """
    chunks = []
    current_parts = []
    current_length = 0

    for item in pages:
        page_number = item["page"]
        text = item["text"]

        page_block = (
            f"\n\n========== PDF PAGE {page_number} ==========\n"
            f"{text}\n"
            f"========== END PDF PAGE {page_number} ==========\n"
        )

        # If one page itself is larger than the limit, split that page.
        if len(page_block) > max_chars:
            if current_parts:
                chunks.append("".join(current_parts))
                current_parts = []
                current_length = 0

            raw_text = text

            start = 0
            while start < len(raw_text):
                end = min(start + max_chars - 500, len(raw_text))

                chunk_text = (
                    f"\n\n========== PDF PAGE {page_number} "
                    f"(PART) ==========\n"
                    f"{raw_text[start:end]}\n"
                    f"========== END PAGE PART ==========\n"
                )

                chunks.append(chunk_text)
                start = end

            continue

        if current_length + len(page_block) > max_chars and current_parts:
            chunks.append("".join(current_parts))
            current_parts = []
            current_length = 0

        current_parts.append(page_block)
        current_length += len(page_block)

    if current_parts:
        chunks.append("".join(current_parts))

    return chunks


# =========================================================
# AI EXTRACTION - ONE CHUNK
# =========================================================

def extract_fields_from_chunk(
    chunk_text: str,
    chunk_number: int,
    total_chunks: int,
) -> List[Dict[str, Any]]:
    """
    Extract fields from ONE chunk.

    This is the main fix for the 30-40 field problem.
    The old implementation sent the complete document in one request,
    which made the model return only a limited subset of fields.
    """

    prompt = f"""
You are a highly accurate insurance policy document extraction engine.

You are processing chunk {chunk_number} of {total_chunks}.

The text below is only one portion of a larger insurance policy.
Extract EVERY meaningful field, table row, benefit, condition,
limit, amount, date, person, relationship, status, option,
coverage, exclusion, eligibility rule, premium, deductible,
address component, identifier, and other structured information
that is actually present in this chunk.

VERY IMPORTANT:

1. Extract ALL meaningful information from this chunk.
2. Do NOT stop after 30, 40, or any arbitrary number of fields.
3. Do NOT summarize.
4. Do NOT invent information.
5. Do NOT infer a value that is not present.
6. Preserve the value exactly as written whenever possible.
7. Preserve the PDF page number.
8. If a table has multiple rows, extract every row.
9. If a table has multiple insured persons, extract information for
   every insured person.
10. If the same field has different values for different people,
    keep separate records.
11. If a field appears multiple times with different values,
    keep the separate occurrences when they represent different
    contexts.
12. Do not merge different people, plans, covers, premiums, limits,
    or rows into one value.
13. Extract headings and section names when they help identify the
    meaning of a value.
14. Extract "Not Opted", "Opted", "Covered", "Not Covered",
    "NA", "Yes", "No", and similar statuses when they are actual
    values.
15. Extract numeric amounts such as premiums, sum insured,
    deductibles, limits, percentages, quantities, and counts.
16. Extract dates and periods.
17. Extract policy wording conditions and eligibility criteria as
    fields when they contain testable information.
18. Repeated PDF headers such as company name should normally be
    ignored unless they are meaningful policy data.
19. Never fabricate vehicle information if this is a health policy.
20. Never fabricate health information if it is not present.
21. Page number must refer to the PDF page marker in the input.
22. Return ONLY JSON.

Recommended categories:
- Policy Details
- Customer / Policyholder
- Insured Person
- Address
- Contact
- Nominee
- Intermediary
- Coverage
- Sum Insured
- Benefits
- Add-ons / Riders
- Premium
- Discount
- Deductible
- Claims
- Dates
- Payment
- Tax / GST
- Exclusions
- Conditions
- Waiting Period
- Eligibility
- Limits
- Definitions
- Other

Return exactly:

{{
  "policy_fields": [
    {{
      "field_name": "Policy Number",
      "value": "example",
      "page": 1,
      "category": "Policy Details"
    }}
  ]
}}

If the chunk contains no meaningful fields, return:

{{
  "policy_fields": []
}}

DOCUMENT CHUNK
==============

{chunk_text}
"""

    response = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract structured insurance policy data. "
                    "Completeness is more important than brevity. "
                    "Never invent values."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    data = safe_json_loads(response.choices[0].message.content)

    fields = data.get("policy_fields", [])

    if not isinstance(fields, list):
        return []

    cleaned = []

    for field in fields:
        if not isinstance(field, dict):
            continue

        field_name = str(field.get("field_name", "")).strip()

        if not field_name:
            continue

        cleaned.append(
            {
                "field_name": field_name,
                "value": field.get("value"),
                "page": field.get("page"),
                "category": str(
                    field.get("category", "Other")
                ).strip() or "Other",
            }
        )

    return cleaned


# =========================================================
# COMPLETE POLICY EXTRACTION
# =========================================================

def deduplicate_policy_fields(
    fields: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Remove only true duplicates.

    Important:
    We DO NOT deduplicate only by field_name because a policy can contain:
        Premium -> Person A
        Premium -> Person B
        Premium -> Person C

    Those must remain separate.
    """
    result = []
    seen = set()

    for field in fields:
        key = (
            normalize_field_name(field.get("field_name")),
            normalize_value(field.get("value")),
            str(field.get("page", "")),
            normalize_field_name(field.get("category")),
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(field)

    return result


def extract_policy_fields(
    pages: List[Dict[str, Any]],
    progress_callback=None,
) -> List[Dict[str, Any]]:
    """
    Extract the COMPLETE policy by processing all chunks independently.
    """

    chunks = build_page_chunks(pages)

    all_fields = []

    for index, chunk in enumerate(chunks, start=1):

        if progress_callback:
            progress_callback(index, len(chunks))

        chunk_fields = extract_fields_from_chunk(
            chunk_text=chunk,
            chunk_number=index,
            total_chunks=len(chunks),
        )

        all_fields.extend(chunk_fields)

    all_fields = deduplicate_policy_fields(all_fields)

    # Sort by page so the UI follows the document order.
    all_fields.sort(
        key=lambda x: (
            int(x.get("page") or 999999)
            if str(x.get("page") or "").isdigit()
            else 999999
        )
    )

    return all_fields


# =========================================================
# VALIDATION HELPERS
# =========================================================

def flatten_json(data: Any, parent_path: str = "") -> List[Dict[str, Any]]:
    """
    Flatten nested JSON while preserving array indexes.

    Example:
        {"ProposalObj": [{"Proposer_FName": "Varsha"}]}

    becomes:
        ProposalObj[0].Proposer_FName -> Varsha

    Container objects/arrays are not returned as fields; only leaf values
    are validated. This is important for insurance request payloads where
    ProposalObj and InsuredObj are arrays of records.
    """
    flattened: List[Dict[str, Any]] = []

    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{parent_path}.{key}" if parent_path else str(key)
            flattened.extend(flatten_json(value, current_path))

    elif isinstance(data, list):
        for index, value in enumerate(data):
            current_path = f"{parent_path}[{index}]"
            flattened.extend(flatten_json(value, current_path))

    else:
        field_name = parent_path.split(".")[-1]
        flattened.append(
            {
                "json_path": parent_path,
                "field": field_name,
                "value": data,
            }
        )

    return flattened


def normalize_comparison_value(value: Any) -> str:
    """Normalize common insurance values for deterministic comparison."""
    if value is None:
        return ""

    if isinstance(value, bool):
        return "true" if value else "false"

    value = str(value).strip().lower()
    value = value.replace("₹", "")
    value = value.replace(",", "")
    value = re.sub(r"\\s+", " ", value)
    return value.strip()


def make_policy_context_for_validation(
    policy_fields: List[Dict[str, Any]]
) -> str:
    """Create a compact policy representation for the AI validator."""
    lines = []

    for index, field in enumerate(policy_fields, start=1):
        lines.append(
            json.dumps(
                {
                    "id": index,
                    "field_name": field.get("field_name", ""),
                    "value": field.get("value", ""),
                    "page": field.get("page", ""),
                    "category": field.get("category", ""),
                },
                ensure_ascii=False,
            )
        )

    return "\n".join(lines)


def validate_policy_batch(
    policy_fields: List[Dict[str, Any]],
    test_batch: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Validate a batch of flattened JSON leaf fields.

    The important difference from the old implementation is that the model
    receives the JSON PATH (for example InsuredObj[0].Member_DOB), rather than
    only a top-level key. This makes ProposalObj/InsuredObj payloads validate
    correctly.
    """

    policy_context = make_policy_context_for_validation(policy_fields)

    prompt = f"""
You are a highly accurate insurance policy validation engine.

Your task is to validate TEST JSON fields against the POLICY FIELDS extracted
from an insurance policy PDF.

The TEST JSON has already been flattened into leaf fields. Each item contains:
- json_path: exact location in the original JSON
- field: field name
- value: supplied test value

==================================================
IMPORTANT JSON STRUCTURE RULES
==================================================

The original JSON can contain structures such as:

ProposalObj[0].Proposer_FName
ProposalObj[0].Proposer_LName
ProposalObj[0].Floater_Suminsured
InsuredObj[0].Member_FName
InsuredObj[0].Member_LName
InsuredObj[0].Member_DOB

Do NOT treat ProposalObj or InsuredObj as a single field.
Validate every leaf field individually.

Do NOT compare one insured person's information with another person's
information.

When the policy contains multiple insured people, first identify the correct
person using the available name, DOB, gender, member ID, or other contextual
information. Only then compare that person's values.

==================================================
FIELD MAPPING
==================================================

JSON field names will often be different from PDF labels.
Use semantic mapping.

Examples:

Proposer_FName -> proposer/policyholder/insured first name
Proposer_LName -> proposer/policyholder/insured last name
Proposer_Email -> proposer/customer email
Proposer_Mobile -> proposer/customer mobile
Member_FName -> insured person first name
Member_LName -> insured person last name
Member_DOB -> insured person's date of birth
Member_Suminsured -> insured person's sum insured
Floater_Suminsured -> floater/base sum insured
Policy_StartDate -> policy start/valid-from date
Policy_Duration -> policy duration/period
Policy_No -> policy number
Product_Code -> product code, if present in policy
Plan_Code -> plan code, if present in policy

Do not require exact spelling of labels, but do not make a match merely because
words look similar.

==================================================
VALIDATION RULES
==================================================

1. Validate EVERY leaf field in TEST DATA.

2. PASS means the supplied value actually matches the corresponding policy
   value or an explicit policy statement.

3. FAIL means a corresponding policy value exists and conflicts with the
   supplied value.

4. NOT APPLICABLE means the policy does not contain enough information to
   confidently validate the supplied field.

5. NOT FOUND means no corresponding policy field can reasonably be found.

6. Do NOT mark PASS merely because a field exists.

7. Do NOT mark a field MISSING simply because it is not in TEST JSON.
   The test JSON is a subset of policy data and does not need to contain every
   policy clause.

8. Do NOT generate MISSING results for unrelated policy clauses, exclusions,
   benefits, waiting periods, or narrative text.

9. Ignore case and harmless whitespace differences.

10. Numeric values should ignore commas, currency symbols, and equivalent
    numeric formatting.

11. Date values should be compared by actual date, not string formatting.

    Example:
    09/09/2026
    09-09-2026
    2026-09-09

    are the same date.

12. Do not automatically treat 0, false, No, null, and empty string as equal.

13. Empty string/null in TEST DATA means the value is not supplied. If the
    policy contains a real value, this should normally FAIL or NOT APPLICABLE,
    not PASS.

14. Names may be compared intelligently, including first-name/last-name
    components, but only for the same person.

15. If a policy contains a family/insured table, do not use a random person's
    value to validate another person's JSON record.

16. Preserve the exact policy value in policy_value.

17. Include the policy page number whenever a matching policy field exists.

18. Do not invent policy values, fields, pages, people, or rules.

19. Technical API metadata such as Vendor_TxnId, Flag, Source_Name, or
    Intermediary_Code should only PASS if the exact value is actually present
    in the PDF. Otherwise return NOT FOUND.

20. A field that is NOT FOUND must never be silently converted to PASS.

==================================================
POLICY FIELDS
==================================================

{policy_context}

==================================================
TEST JSON BATCH
==================================================

{json.dumps(test_batch, indent=2, ensure_ascii=False)}

==================================================
OUTPUT
==================================================

Return ONLY valid JSON.

Return exactly this structure:

{{
  "results": [
    {{
      "json_path": "ProposalObj[0].Proposer_FName",
      "field": "Proposer_FName",
      "input_value": "Varsha",
      "policy_field": "Insured Name",
      "policy_value": "Kajal Dattatray Chavan",
      "policy_page": 4,
      "status": "FAIL",
      "reason": "The supplied first name does not match the corresponding insured person in the policy."
    }}
  ]
}}

Allowed statuses:
PASS
FAIL
NOT APPLICABLE
NOT FOUND
"""

    response = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict insurance policy validator. "
                    "Validate actual values and never invent policy data."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    data = safe_json_loads(response.choices[0].message.content)
    results = data.get("results", [])

    return results if isinstance(results, list) else []


def _deterministic_exact_match(
    input_value: Any,
    policy_value: Any,
) -> bool:
    """Handle safe, obvious scalar equality without relying on AI."""
    a = normalize_comparison_value(input_value)
    b = normalize_comparison_value(policy_value)

    if not a or not b:
        return False

    # Exact normalized string equality.
    if a == b:
        return True

    # Numeric equality.
    try:
        if re.fullmatch(r"[-+]?\\d+(?:\\.\\d+)?", a) and re.fullmatch(
            r"[-+]?\\d+(?:\\.\\d+)?", b
        ):
            return float(a) == float(b)
    except Exception:
        pass

    # Common date equality.
    date_patterns = [
        r"^(\\d{4})[-/](\\d{1,2})[-/](\\d{1,2})$",
        r"^(\\d{1,2})[-/](\\d{1,2})[-/](\\d{4})$",
    ]

    def parse_date(value: str):
        from datetime import datetime
        for pattern in date_patterns:
            match = re.fullmatch(pattern, value)
            if not match:
                continue
            parts = [int(x) for x in match.groups()]
            if len(str(parts[0])) == 4:
                year, month, day = parts
            else:
                day, month, year = parts
            try:
                return datetime(year, month, day).date()
            except ValueError:
                return None
        return None

    da = parse_date(a)
    db = parse_date(b)
    if da and db:
        return da == db

    return False


def _build_policy_name_index(policy_fields: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = {}

    for field in policy_fields:
        normalized = normalize_field_name(field.get("field_name", ""))
        if normalized:
            index.setdefault(normalized, []).append(field)

    return index


def _fallback_policy_match(
    test_field: Dict[str, Any],
    policy_fields: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    """
    Safe fallback for exact/obvious matches when the AI omits a result.
    It intentionally does not perform fuzzy value matching because that could
    create false PASS results.
    """
    field_name = normalize_field_name(test_field.get("field", ""))
    input_value = test_field.get("value")

    aliases = {
        "policy_no": {"policy_number", "policy_no", "policy_number_"},
        "policy_number": {"policy_no", "policy_number"},
        "member_dob": {"dob", "date_of_birth", "member_dob"},
        "proposer_fname": {"first_name", "proposer_fname", "proposer_first_name"},
        "proposer_lname": {"last_name", "proposer_lname", "proposer_last_name"},
        "member_fname": {"first_name", "member_fname", "insured_first_name"},
        "member_lname": {"last_name", "member_lname", "insured_last_name"},
        "floater_suminsured": {"sum_insured", "floater_sum_insured", "base_sum_insured"},
        "member_suminsured": {"sum_insured", "member_sum_insured", "individual_sum_insured"},
        "policy_startdate": {"start_date", "policy_start_date", "valid_from"},
    }

    candidate_names = {field_name}
    candidate_names.update(aliases.get(field_name, set()))

    candidates = []
    for policy_field in policy_fields:
        policy_name = normalize_field_name(policy_field.get("field_name", ""))
        if policy_name in candidate_names:
            candidates.append(policy_field)

    # Exact normalized value is a safe fallback.
    for candidate in candidates:
        if _deterministic_exact_match(input_value, candidate.get("value")):
            return candidate

    return candidates[0] if len(candidates) == 1 else None


def finalize_validation(
    policy_fields: List[Dict[str, Any]],
    flattened_test_data: List[Dict[str, Any]],
    ai_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Reconcile AI output with every flattened JSON field.

    We do NOT add hundreds of MISSING policy rows. Only supplied JSON fields
    are counted because this application validates an API/request payload
    against the policy.
    """

    # Index AI results by JSON path first, then by field name for compatibility
    # with models that omit json_path.
    by_path: Dict[str, Dict[str, Any]] = {}
    by_field: Dict[str, List[Dict[str, Any]]] = {}

    for item in ai_results:
        path = str(item.get("json_path", "")).strip()
        field = normalize_field_name(item.get("field", ""))

        if path:
            by_path[path] = item
        if field:
            by_field.setdefault(field, []).append(item)

    results: List[Dict[str, Any]] = []

    for test_item in flattened_test_data:
        path = test_item["json_path"]
        field = test_item["field"]
        value = test_item["value"]
        normalized_field = normalize_field_name(field)

        selected = by_path.get(path)

        if selected is None:
            candidates = by_field.get(normalized_field, [])
            if len(candidates) == 1:
                selected = candidates[0]

        if selected is None:
            fallback = _fallback_policy_match(test_item, policy_fields)
            if fallback:
                selected = {
                    "json_path": path,
                    "field": field,
                    "input_value": value,
                    "policy_field": fallback.get("field_name", ""),
                    "policy_value": fallback.get("value", ""),
                    "policy_page": fallback.get("page", ""),
                    "status": (
                        "PASS"
                        if _deterministic_exact_match(value, fallback.get("value"))
                        else "NOT APPLICABLE"
                    ),
                    "reason": (
                        "The supplied value exactly matches the policy value."
                        if _deterministic_exact_match(value, fallback.get("value"))
                        else "A possible policy field was found, but the value could not be confidently matched."
                    ),
                }

        if selected is None:
            result = {
                "json_path": path,
                "field": field,
                "input_value": value,
                "policy_field": "",
                "policy_value": "",
                "policy_page": "",
                "status": "NOT FOUND",
                "reason": "No corresponding field could be confidently found in the policy PDF.",
            }
        else:
            status = str(selected.get("status", "NOT APPLICABLE")).upper().strip()
            if status not in {"PASS", "FAIL", "NOT APPLICABLE", "NOT FOUND"}:
                status = "NOT APPLICABLE"

            result = {
                "json_path": path,
                "field": field,
                "input_value": value,
                "policy_field": selected.get("policy_field", ""),
                "policy_value": selected.get("policy_value", ""),
                "policy_page": selected.get("policy_page", ""),
                "status": status,
                "reason": selected.get(
                    "reason",
                    "The field could not be confidently validated."
                ),
            }

        results.append(result)

    passed = sum(1 for item in results if item["status"] == "PASS")
    failed = sum(1 for item in results if item["status"] == "FAIL")
    not_applicable = sum(
        1 for item in results if item["status"] == "NOT APPLICABLE"
    )
    not_found = sum(1 for item in results if item["status"] == "NOT FOUND")

    # For this application, NOT FOUND is a validation failure because the
    # requested JSON value could not be verified against the policy.
    status = "PASS" if failed == 0 and not_found == 0 and len(results) > 0 else "FAIL"

    return {
        "status": status,
        "summary": (
            "All supplied JSON fields were successfully validated against the policy."
            if status == "PASS"
            else "One or more supplied JSON fields could not be validated or did not match the policy."
        ),
        "total_fields": len(results),
        "passed": passed,
        "failed": failed,
        "not_applicable": not_applicable,
        "not_found": not_found,
        "missing_from_test_data": 0,
        "extra_test_fields": 0,
        "results": results,
    }


def validate_policy(
    policy_fields: List[Dict[str, Any]],
    test_data: Dict[str, Any],
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Validate the complete nested test JSON.

    The original JSON is flattened first, so nested ProposalObj and InsuredObj
    records are independently validated.
    """

    flattened = flatten_json(test_data)

    batches = [
        flattened[i:i + VALIDATION_FIELDS_PER_BATCH]
        for i in range(0, len(flattened), VALIDATION_FIELDS_PER_BATCH)
    ]

    all_results: List[Dict[str, Any]] = []

    if not batches:
        return {
            "status": "FAIL",
            "summary": "The supplied JSON does not contain any leaf fields to validate.",
            "total_fields": 0,
            "passed": 0,
            "failed": 0,
            "not_applicable": 0,
            "not_found": 0,
            "missing_from_test_data": 0,
            "extra_test_fields": 0,
            "results": [],
        }

    for index, batch in enumerate(batches, start=1):
        if progress_callback:
            progress_callback(index, len(batches))

        batch_results = validate_policy_batch(
            policy_fields,
            batch,
        )
        all_results.extend(batch_results)

    return finalize_validation(
        policy_fields,
        flattened,
        all_results,
    )


# =========================================================
# PDF VS PDF COMPARISON
# =========================================================

def compare_policy_batch(
    policy_1_fields: List[Dict[str, Any]],
    policy_2_fields: List[Dict[str, Any]],
    batch_1: List[Dict[str, Any]],
    batch_2: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    prompt = f"""
You are an insurance policy comparison engine.

Compare the supplied field records from two policies.

Policy 1 fields:
{json.dumps(batch_1, indent=2, ensure_ascii=False)}

Policy 2 fields:
{json.dumps(batch_2, indent=2, ensure_ascii=False)}

Rules:
1. Compare actual values, not just field existence.
2. Match fields when their meaning clearly corresponds.
3. Ignore case, extra whitespace, commas in numeric values,
   currency symbols, and obvious formatting differences.
4. Compare dates intelligently.
5. Preserve exact values from both policies.
6. Do not invent values.
7. Do not treat two different insured persons as the same person.
8. If a field is only in Policy 1, use ONLY_IN_POLICY_1.
9. If a field is only in Policy 2, use ONLY_IN_POLICY_2.
10. If values differ, use DIFFERENT.
11. If values match, use MATCH.
12. Return every meaningful record in the supplied batches.

Return ONLY JSON:

{{
  "results": [
    {{
      "field": "Policy Number",
      "policy_1_value": "123",
      "policy_2_value": "456",
      "policy_1_page": 1,
      "policy_2_page": 1,
      "status": "DIFFERENT",
      "reason": "The policy numbers are different."
    }}
  ]
}}

Allowed statuses:
MATCH
DIFFERENT
ONLY_IN_POLICY_1
ONLY_IN_POLICY_2
"""

    response = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You compare insurance policy fields accurately "
                    "without inventing information."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    data = safe_json_loads(response.choices[0].message.content)

    results = data.get("results", [])

    return results if isinstance(results, list) else []


def compare_policies(
    policy_1_fields: List[Dict[str, Any]],
    policy_2_fields: List[Dict[str, Any]],
    progress_callback=None,
) -> Dict[str, Any]:

    # Compare manageable batches.
    p1_batches = chunk_list(policy_1_fields, 60)
    p2_batches = chunk_list(policy_2_fields, 60)

    total_batches = max(len(p1_batches), len(p2_batches), 1)

    all_results = []

    for index in range(total_batches):

        batch_1 = (
            p1_batches[index]
            if index < len(p1_batches)
            else []
        )

        batch_2 = (
            p2_batches[index]
            if index < len(p2_batches)
            else []
        )

        if progress_callback:
            progress_callback(index + 1, total_batches)

        results = compare_policy_batch(
            policy_1_fields,
            policy_2_fields,
            batch_1,
            batch_2,
        )

        all_results.extend(results)

    # Deduplicate exact comparison records.
    seen = set()
    final_results = []

    for item in all_results:
        key = (
            normalize_field_name(item.get("field")),
            normalize_value(item.get("policy_1_value")),
            normalize_value(item.get("policy_2_value")),
            str(item.get("policy_1_page", "")),
            str(item.get("policy_2_page", "")),
        )

        if key in seen:
            continue

        seen.add(key)
        final_results.append(item)

    matching = sum(
        1 for x in final_results
        if x.get("status") == "MATCH"
    )

    different = sum(
        1 for x in final_results
        if x.get("status") == "DIFFERENT"
    )

    only_1 = sum(
        1 for x in final_results
        if x.get("status") == "ONLY_IN_POLICY_1"
    )

    only_2 = sum(
        1 for x in final_results
        if x.get("status") == "ONLY_IN_POLICY_2"
    )

    overall_status = (
        "MATCH"
        if different == 0 and only_1 == 0 and only_2 == 0
        else "DIFFERENT"
    )

    return {
        "overall_status": overall_status,
        "summary": (
            "The compared policy values match."
            if overall_status == "MATCH"
            else "The policies contain one or more differences."
        ),
        "total_compared_fields": len(final_results),
        "matching_fields": matching,
        "different_fields": different,
        "only_in_policy_1": only_1,
        "only_in_policy_2": only_2,
        "results": final_results,
    }


# =========================================================
# UI HELPERS
# =========================================================

def display_policy_fields(
    policy_fields: List[Dict[str, Any]]
):
    display = []

    for field in policy_fields:
        display.append(
            {
                "Field": str(field.get("field_name", "")),
                "Value": str(field.get("value", "")),
                "Page": str(field.get("page", "")),
                "Category": str(field.get("category", "")),
            }
        )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )


def display_validation_results(result: Dict[str, Any]):

    results = result.get("results", [])

    for item in results:

        field = item.get("field", "Unknown Field")
        json_path = item.get("json_path", field)
        policy_field = item.get("policy_field", "")
        input_value = item.get("input_value", "")
        policy_value = item.get("policy_value", "")
        policy_page = item.get("policy_page", "")
        status = item.get("status", "NOT APPLICABLE")
        reason = item.get("reason", "")

        st.markdown(f"**JSON Path:** `{json_path}`")
        if policy_field:
            st.caption(f"Policy Field: {policy_field}")

        if status == "PASS":
            st.success(
                f"✅ {field}: {reason}"
            )

        elif status == "FAIL":
            st.error(
                f"❌ {field}: {reason}"
            )

        elif status == "MISSING":
            st.warning(
                f"⚠️ {field}: Missing from test data. {reason}"
            )

        elif status == "EXTRA":
            st.warning(
                f"⚠️ {field}: Extra test field. {reason}"
            )

        elif status == "NOT FOUND":
            st.warning(
                f"⚠️ {field}: Could not be found in the policy. {reason}"
            )

        else:
            st.info(
                f"ℹ️ {field}: {reason}"
            )

        col1, col2, col3 = st.columns([1, 1, 0.35])

        with col1:
            st.write("**Input Value**")
            st.code(
                json.dumps(
                    input_value,
                    ensure_ascii=False,
                )
            )

        with col2:
            st.write("**Policy Value**")
            st.code(
                json.dumps(
                    policy_value,
                    ensure_ascii=False,
                )
            )

        with col3:
            st.write("**Page**")
            st.write(policy_page)

        st.divider()


def display_comparison_results(
    comparison_result: Dict[str, Any]
):

    results = comparison_result.get("results", [])

    for item in results:

        field = item.get("field", "Unknown Field")
        value_1 = item.get("policy_1_value", "")
        value_2 = item.get("policy_2_value", "")
        page_1 = item.get("policy_1_page", "")
        page_2 = item.get("policy_2_page", "")
        status = item.get("status", "")
        reason = item.get("reason", "")

        if status == "MATCH":
            st.success(
                f"✅ {field}: {reason}"
            )

        elif status == "DIFFERENT":
            st.error(
                f"❌ {field}: {reason}"
            )

        elif status == "ONLY_IN_POLICY_1":
            st.warning(
                f"⚠️ {field}: Only in Policy 1. {reason}"
            )

        elif status == "ONLY_IN_POLICY_2":
            st.warning(
                f"⚠️ {field}: Only in Policy 2. {reason}"
            )

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Policy 1 — Page {page_1}**")
            st.code(str(value_1))

        with col2:
            st.write(f"**Policy 2 — Page {page_2}**")
            st.code(str(value_2))

        st.divider()


# =========================================================
# FIRST POLICY PDF
# =========================================================

st.subheader("📄 Policy Document")

uploaded_file = st.file_uploader(
    "Upload Policy Document",
    type=["pdf"],
    key="first_policy_pdf",
)

pages = []
page_count = 0

if uploaded_file:

    try:
        pages, page_count = extract_pdf_pages(uploaded_file)

        readable_pages = sum(
            1 for p in pages if p["text"].strip()
        )

        total_chars = sum(
            len(p["text"])
            for p in pages
        )

        st.success(
            f"✅ Policy loaded: {page_count} PDF pages | "
            f"{readable_pages} readable pages | "
            f"{total_chars:,} extracted characters"
        )

        if readable_pages < page_count:
            st.warning(
                f"⚠️ {page_count - readable_pages} page(s) have no "
                "extractable text. If those pages are scanned images, "
                "OCR will be required."
            )

    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        st.stop()


# =========================================================
# JSON INPUT
# =========================================================

st.subheader("🧪 Test Data")

default_json = """{
  "ProposalObj": [
    {
      "Quote_Number": "",
      "Vendor_TxnId": "24",
      "Intermediary_Code": "IMD1001063",
      "Flag": "2",
      "Product_Code": "4216",
      "Plan_Code": "4216100004",
      "Policy_Varient": "1",
      "Policy_StartDate": "09/09/2026 02:32:36",
      "Policy_Duration": "1",
      "Floater_Suminsured": "300000",
      "Floater_Suminsured_Unit": "1",
      "Proposer_FName": "Varsha",
      "Proposer_LName": "Shende",
      "Proposer_Email": "varsha.shende@libertyinsurance.in",
      "Source_Name": "Customer Portal",
      "Proposer_Mobile": "9579697572",
      "Floater_deductible": null,
      "IsloyaltyApplicable": "0",
      "Policy_No": null,
      "PospCode": null,
      "PospName": null,
      "StateCode": "27",
      "ECS_Registration": "0",
      "EMI_Frequency": "0",
      "Business_Type": "New Business",
      "Business_Source": "Internet",
      "GSTIN": ""
    }
  ],
  "InsuredObj": [
    {
      "Member_No": "1",
      "Member_FName": "Varsha",
      "Member_LName": "Shende",
      "Member_DOB": "1997-02-22",
      "Member_Suminsured": "300000",
      "Member_Suminsured_Unit": "1",
      "Member_RelationCode": "R001",
      "Member_OccupationCode": "",
      "Wellness_program": "0",
      "Member_Deductible": "300000"
    }
  ]
}"""

json_input = st.text_area(
    "Enter Test Data (JSON)",
    value=default_json,
    height=300,
    placeholder="Leave empty to compare two policy PDFs",
)


# =========================================================
# SECOND PDF
# =========================================================

second_pdf = None

if not json_input.strip():

    st.info(
        "ℹ️ JSON data is empty. Upload a second policy PDF "
        "to compare both documents."
    )

    second_pdf = st.file_uploader(
        "Upload Second Policy PDF",
        type=["pdf"],
        key="second_policy_pdf",
    )


# =========================================================
# TEST BUTTON
# =========================================================

if st.button(
    "🚀 Test Policy",
    type="primary",
    use_container_width=True,
):

    if not uploaded_file:
        st.error("Please upload the first policy PDF.")
        st.stop()

    if not pages:
        st.error(
            "No readable PDF pages were found."
        )
        st.stop()

    # =====================================================
    # MODE 1 - JSON VALIDATION
    # =====================================================

    if json_input.strip():

        st.header("🧪 Policy vs JSON Validation")

        # -------------------------------------------------
        # Parse JSON
        # -------------------------------------------------

        try:
            test_data = json.loads(json_input)

            if not isinstance(test_data, dict):
                st.error(
                    "Test data must be a JSON object."
                )
                st.stop()

        except json.JSONDecodeError as e:
            st.error(
                f"Invalid JSON: {e}"
            )
            st.stop()

        # -------------------------------------------------
        # Extract COMPLETE policy
        # -------------------------------------------------

        st.info(
            "🔍 Extracting the complete policy page-by-page. "
            "The document is split into multiple AI requests so "
            "large policies are not truncated."
        )

        progress = st.progress(0)

        status_text = st.empty()

        def extraction_progress(current, total):
            progress.progress(
                min(current / total, 1.0)
            )
            status_text.info(
                f"Extracting chunk {current} of {total}..."
            )

        try:
            policy_fields = extract_policy_fields(
                pages,
                progress_callback=extraction_progress,
            )

        except Exception as e:
            progress.empty()
            status_text.empty()
            st.error(
                f"Policy extraction failed: {e}"
            )
            st.stop()

        progress.empty()
        status_text.empty()

        # -------------------------------------------------
        # Extraction summary
        # -------------------------------------------------

        st.success(
            f"✅ Complete extraction finished: "
            f"{len(policy_fields)} unique field records found "
            f"across {page_count} PDF pages."
        )

        if len(policy_fields) <= 40:
            st.warning(
                "⚠️ Only a small number of fields were extracted. "
                "This can happen when the PDF contains scanned/image "
                "pages. Check the raw PDF text below."
            )

        # -------------------------------------------------
        # Display extracted fields
        # -------------------------------------------------

        st.subheader(
            "📋 Extracted Policy Fields"
        )

        display_policy_fields(policy_fields)

        # -------------------------------------------------
        # Extraction by page
        # -------------------------------------------------

        page_counts = {}

        for field in policy_fields:
            page = str(field.get("page", "Unknown"))
            page_counts[page] = page_counts.get(page, 0) + 1

        with st.expander(
            "📊 Fields Extracted Per PDF Page"
        ):
            st.write(
                dict(
                    sorted(
                        page_counts.items(),
                        key=lambda x: (
                            int(x[0])
                            if x[0].isdigit()
                            else 999999
                        )
                    )
                )
            )

        # -------------------------------------------------
        # Raw extracted text
        # -------------------------------------------------

        with st.expander(
            "📄 View Raw PDF Text"
        ):
            raw_text = "\n\n".join(
                f"========== PAGE {p['page']} ==========\n"
                f"{p['text']}"
                for p in pages
            )
            st.text_area(
                "PDF Text",
                raw_text,
                height=500,
            )

        if not policy_fields:
            st.error(
                "No policy fields were extracted."
            )
            st.stop()

        # -------------------------------------------------
        # Validate JSON
        # -------------------------------------------------

        st.info(
            "🤖 Validating test data against the complete "
            "extracted policy..."
        )

        validation_progress = st.progress(0)
        validation_status = st.empty()

        def validation_progress_callback(current, total):
            validation_progress.progress(
                min(current / total, 1.0)
            )
            validation_status.info(
                f"Validating batch {current} of {total}..."
            )

        try:
            result = validate_policy(
                policy_fields,
                test_data,
                progress_callback=validation_progress_callback,
            )

        except Exception as e:
            validation_progress.empty()
            validation_status.empty()
            st.error(
                f"Policy validation failed: {e}"
            )
            st.stop()

        validation_progress.empty()
        validation_status.empty()

        # -------------------------------------------------
        # Summary
        # -------------------------------------------------

        st.subheader(
            "📊 Validation Summary"
        )

        status = result.get(
            "status",
            "FAIL",
        )

        if status == "PASS":
            st.success(
                "✅ POLICY TEST PASSED"
            )
        else:
            st.error(
                "❌ POLICY TEST FAILED"
            )

        st.write(
            result.get(
                "summary",
                "",
            )
        )

        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

        with col1:
            st.metric(
                "Total",
                result.get("total_fields", 0),
            )

        with col2:
            st.metric(
                "Passed",
                result.get("passed", 0),
            )

        with col3:
            st.metric(
                "Failed",
                result.get("failed", 0),
            )

        with col4:
            st.metric(
                "Not Applicable",
                result.get("not_applicable", 0),
            )

        with col5:
            st.metric(
                "Missing",
                result.get(
                    "missing_from_test_data",
                    0,
                ),
            )

        with col6:
            st.metric(
                "Not Found",
                result.get(
                    "not_found",
                    0,
                ),
            )

        with col7:
            st.metric(
                "Extra",
                result.get(
                    "extra_test_fields",
                    0,
                ),
            )

        # -------------------------------------------------
        # Detailed results
        # -------------------------------------------------

        st.subheader(
            "🔎 Validation Results"
        )

        display_validation_results(result)

        with st.expander(
            "View Raw Validation JSON"
        ):
            st.json(result)

    # =====================================================
    # MODE 2 - PDF VS PDF
    # =====================================================

    else:

        st.header(
            "📑 Policy PDF vs Policy PDF Comparison"
        )

        if not second_pdf:
            st.warning(
                "Please upload the second policy PDF."
            )
            st.stop()

        # -------------------------------------------------
        # Read second PDF
        # -------------------------------------------------

        try:
            pages_2, page_count_2 = extract_pdf_pages(
                second_pdf
            )

        except Exception as e:
            st.error(
                f"Error reading second PDF: {e}"
            )
            st.stop()

        readable_2 = sum(
            1 for p in pages_2
            if p["text"].strip()
        )

        chars_2 = sum(
            len(p["text"])
            for p in pages_2
        )

        st.success(
            f"✅ Second policy loaded: "
            f"{page_count_2} pages | "
            f"{readable_2} readable pages | "
            f"{chars_2:,} characters"
        )

        if not pages_2:
            st.error(
                "No readable pages found in second policy."
            )
            st.stop()

        # -------------------------------------------------
        # Extract Policy 1
        # -------------------------------------------------

        st.info(
            "🔍 Extracting complete Policy 1..."
        )

        progress_1 = st.progress(0)
        status_1 = st.empty()

        def progress_policy_1(current, total):
            progress_1.progress(
                min(current / total, 1.0)
            )
            status_1.info(
                f"Policy 1: chunk {current} of {total}"
            )

        try:
            policy_1_fields = extract_policy_fields(
                pages,
                progress_callback=progress_policy_1,
            )

        except Exception as e:
            progress_1.empty()
            status_1.empty()
            st.error(
                f"Policy 1 extraction failed: {e}"
            )
            st.stop()

        progress_1.empty()
        status_1.empty()

        st.success(
            f"Policy 1: {len(policy_1_fields)} fields extracted."
        )

        # -------------------------------------------------
        # Extract Policy 2
        # -------------------------------------------------

        st.info(
            "🔍 Extracting complete Policy 2..."
        )

        progress_2 = st.progress(0)
        status_2 = st.empty()

        def progress_policy_2(current, total):
            progress_2.progress(
                min(current / total, 1.0)
            )
            status_2.info(
                f"Policy 2: chunk {current} of {total}"
            )

        try:
            policy_2_fields = extract_policy_fields(
                pages_2,
                progress_callback=progress_policy_2,
            )

        except Exception as e:
            progress_2.empty()
            status_2.empty()
            st.error(
                f"Policy 2 extraction failed: {e}"
            )
            st.stop()

        progress_2.empty()
        status_2.empty()

        st.success(
            f"Policy 2: {len(policy_2_fields)} fields extracted."
        )

        if not policy_1_fields:
            st.error(
                "No fields extracted from Policy 1."
            )
            st.stop()

        if not policy_2_fields:
            st.error(
                "No fields extracted from Policy 2."
            )
            st.stop()

        # -------------------------------------------------
        # Display extracted fields
        # -------------------------------------------------

        with st.expander(
            f"📋 Policy 1 Extracted Fields "
            f"({len(policy_1_fields)})"
        ):
            display_policy_fields(
                policy_1_fields
            )

        with st.expander(
            f"📋 Policy 2 Extracted Fields "
            f"({len(policy_2_fields)})"
        ):
            display_policy_fields(
                policy_2_fields
            )

        # -------------------------------------------------
        # Compare
        # -------------------------------------------------

        st.info(
            "🤖 Comparing extracted policy information..."
        )

        compare_progress = st.progress(0)
        compare_status = st.empty()

        def comparison_progress(current, total):
            compare_progress.progress(
                min(current / total, 1.0)
            )
            compare_status.info(
                f"Comparing batch {current} of {total}..."
            )

        try:
            comparison_result = compare_policies(
                policy_1_fields,
                policy_2_fields,
                progress_callback=comparison_progress,
            )

        except Exception as e:
            compare_progress.empty()
            compare_status.empty()
            st.error(
                f"Policy comparison failed: {e}"
            )
            st.stop()

        compare_progress.empty()
        compare_status.empty()

        # -------------------------------------------------
        # Comparison Summary
        # -------------------------------------------------

        st.subheader(
            "📊 Comparison Summary"
        )

        overall_status = comparison_result.get(
            "overall_status",
            "DIFFERENT",
        )

        if overall_status == "MATCH":
            st.success(
                "✅ BOTH POLICY DOCUMENTS MATCH"
            )
        else:
            st.error(
                "❌ POLICY DOCUMENTS ARE DIFFERENT"
            )

        st.write(
            comparison_result.get(
                "summary",
                "",
            )
        )

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Compared",
                comparison_result.get(
                    "total_compared_fields",
                    0,
                ),
            )

        with col2:
            st.metric(
                "Matching",
                comparison_result.get(
                    "matching_fields",
                    0,
                ),
            )

        with col3:
            st.metric(
                "Different",
                comparison_result.get(
                    "different_fields",
                    0,
                ),
            )

        with col4:
            st.metric(
                "Only Policy 1",
                comparison_result.get(
                    "only_in_policy_1",
                    0,
                ),
            )

        with col5:
            st.metric(
                "Only Policy 2",
                comparison_result.get(
                    "only_in_policy_2",
                    0,
                ),
            )

        # -------------------------------------------------
        # Detailed comparison
        # -------------------------------------------------

        st.subheader(
            "🔎 Detailed Policy Comparison"
        )

        display_comparison_results(
            comparison_result
        )

        with st.expander(
            "View Raw Comparison JSON"
        ):
            st.json(comparison_result)
