import json
import os

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
        "🔑 OpenAI API Key not found! "
        "Please set OPENAI_API_KEY in your .env file or environment."
    )
    st.stop()

client = OpenAI(api_key=api_key)


# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Policy Tester",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Policy Tester")

st.write(
    "Validate an insurance policy using JSON test data "
    "or compare two policy PDFs."
)


# =========================================================
# FUNCTION: EXTRACT PDF TEXT
# =========================================================

def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    pages_text = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text:

            pages_text.append(
                f"\n--- PAGE {page_number} ---\n{text}"
            )

    policy_text = "\n".join(pages_text)

    return policy_text, len(reader.pages)


# =========================================================
# FUNCTION: EXTRACT POLICY FIELDS USING AI
# =========================================================

def extract_policy_fields(policy_text):

    extraction_prompt = f"""
You are an insurance policy document extraction engine.

Analyze the policy document carefully.

Extract ALL meaningful policy fields.

For every field return:

- field_name
- value
- page
- category

Important rules:

1. Do not invent values.

2. Preserve the exact value from the document.

3. Preserve the page number where the field appears.

4. Extract numbers as numbers where clearly appropriate.

5. Extract dates exactly as written.

6. Extract meaningful information including:

   - Policy details
   - Policy number
   - Customer details
   - Insured details
   - Vehicle details
   - Vehicle registration
   - Vehicle age
   - Vehicle damage
   - Driving license
   - Premium
   - Annual premium
   - Coverage
   - Insured declared value
   - Sum insured
   - Policy dates
   - Expiry dates
   - Deductibles
   - Add-ons
   - Exclusions
   - Limits
   - Conditions
   - Eligibility criteria

7. If the same field appears multiple times, use the
   most relevant occurrence.

8. Do not summarize the document.

9. Do not invent missing fields.

10. Return ONLY valid JSON.

POLICY DOCUMENT
================

{policy_text}

Return exactly this structure:

{{
    "policy_fields": [
        {{
            "field_name": "Policy Number",
            "value": "4216-400301-26-7001556-00-000",
            "page": 1,
            "category": "Policy Details"
        }}
    ]
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={
            "type": "json_object"
        },
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a highly accurate insurance "
                    "policy extraction engine."
                )
            },
            {
                "role": "user",
                "content": extraction_prompt
            }
        ]
    )

    content = response.choices[0].message.content

    extracted_data = json.loads(content)

    return extracted_data.get(
        "policy_fields",
        []
    )


# =========================================================
# FUNCTION: VALIDATE JSON AGAINST POLICY
# =========================================================

def validate_policy(policy_fields, test_data):

    validation_prompt = f"""
You are an insurance policy validation engine.

You have two sources:

1. POLICY FIELDS extracted from the policy PDF.
2. TEST DATA provided by the user.

Your job is to compare the test data against the policy.

POLICY FIELDS
=============

{json.dumps(policy_fields, indent=2)}


TEST DATA
=========

{json.dumps(test_data, indent=2)}


VALIDATION RULES
================

1. Check every field present in TEST DATA.

2. Determine whether the corresponding field exists
   in the policy.

3. Compare the supplied test value against the policy value.

4. PASS only when the supplied value actually matches
   the corresponding policy value or requirement.

5. Do NOT mark a field PASS merely because the field exists.

6. FAIL when the supplied value conflicts with the policy.

7. NOT APPLICABLE when the policy does not provide enough
   information to validate the field.

8. Identify policy fields that are missing from TEST DATA.

9. Identify TEST DATA fields that do not exist in the policy.

10. Do not invent policy values.

11. Do not invent policy rules.

12. Preserve the exact policy value in policy_value.

13. Numeric comparison:

    Treat numeric values as numbers.

14. Ignore formatting differences such as:

    - uppercase/lowercase
    - extra spaces
    - commas in numbers

15. Boolean values:

    true and "Yes" should NOT automatically be considered
    identical unless the policy clearly represents them
    as equivalent.

16. Dates:

    Compare dates intelligently while accounting for
    common formatting differences.

17. If a value cannot be confidently matched, mark it
    NOT APPLICABLE rather than guessing.

18. Every result must contain:

    - field
    - input_value
    - policy_value
    - status
    - reason

19. Status must be one of:

    PASS
    FAIL
    NOT APPLICABLE
    MISSING
    EXTRA

20. MISSING means the field exists in the policy but
    was not supplied in TEST DATA.

21. EXTRA means TEST DATA contains a field that does not
    exist in the policy.

22. The overall status should be FAIL if at least one
    field has status FAIL, MISSING, or EXTRA.

23. The overall status can be PASS only when all supplied
    fields match and there are no MISSING or EXTRA fields.

Return ONLY valid JSON.

Use exactly this structure:

{{
    "status": "PASS",
    "summary": "Overall validation result",
    "total_fields": 0,
    "passed": 0,
    "failed": 0,
    "not_applicable": 0,
    "missing_from_test_data": 0,
    "extra_test_fields": 0,
    "results": [
        {{
            "field": "Age",
            "input_value": 35,
            "policy_value": 35,
            "status": "PASS",
            "reason": "The supplied age matches the policy value."
        }}
    ]
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={
            "type": "json_object"
        },
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a highly accurate insurance "
                    "policy validation engine."
                )
            },
            {
                "role": "user",
                "content": validation_prompt
            }
        ]
    )

    content = response.choices[0].message.content

    return json.loads(content)


# =========================================================
# FUNCTION: COMPARE TWO POLICY PDFs
# =========================================================

def compare_policies(
    policy_1_fields,
    policy_2_fields
):

    comparison_prompt = f"""
You are an insurance policy comparison engine.

Compare two insurance policy documents using their
extracted structured fields.

POLICY 1 FIELDS
===============

{json.dumps(policy_1_fields, indent=2)}


POLICY 2 FIELDS
===============

{json.dumps(policy_2_fields, indent=2)}


Your job is to identify:

1. Fields that match.

2. Fields where values are different.

3. Fields present only in Policy 1.

4. Fields present only in Policy 2.

5. Meaningful policy differences.

Compare fields intelligently.

For example:

"Annual Premium"

and

"Annual_Premium"

may represent the same field.

Rules:

1. Do not invent information.

2. Preserve exact values from both policies.

3. Ignore differences caused only by:

   - uppercase/lowercase
   - extra spaces
   - commas in numbers
   - obvious formatting differences

4. Numeric values should be compared numerically.

5. Dates should be compared intelligently.

6. If a field exists only in Policy 1, mark it
   ONLY_IN_POLICY_1.

7. If a field exists only in Policy 2, mark it
   ONLY_IN_POLICY_2.

8. Do not assume differently named fields are identical
   unless their meaning clearly matches.

9. Compare meaningful insurance information including:

   - Policy number
   - Customer information
   - Vehicle information
   - Registration number
   - Premium
   - Coverage
   - Sum insured
   - Insured declared value
   - Policy start date
   - Policy end date
   - Vehicle damage
   - Vehicle age
   - Deductibles
   - Add-ons
   - Exclusions
   - Limits
   - Conditions

10. Do not mark two fields MATCH merely because both fields
    exist. Their values must actually match.

Return ONLY valid JSON.

Use exactly this structure:

{{
    "overall_status": "MATCH",
    "summary": "Both policies contain matching values.",
    "total_compared_fields": 0,
    "matching_fields": 0,
    "different_fields": 0,
    "only_in_policy_1": 0,
    "only_in_policy_2": 0,

    "results": [
        {{
            "field": "Annual Premium",
            "policy_1_value": 25000,
            "policy_2_value": 25000,
            "status": "MATCH",
            "reason": "Annual premium is identical in both policies."
        }}
    ]
}}

Allowed result statuses:

MATCH
DIFFERENT
ONLY_IN_POLICY_1
ONLY_IN_POLICY_2

Overall status:

MATCH
or
DIFFERENT
"""


    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={
            "type": "json_object"
        },
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a highly accurate insurance "
                    "policy comparison engine."
                )
            },
            {
                "role": "user",
                "content": comparison_prompt
            }
        ]
    )

    content = response.choices[0].message.content

    return json.loads(content)


# =========================================================
# FIRST PDF
# =========================================================

st.subheader("📄 Policy Document")

uploaded_file = st.file_uploader(
    "Upload Policy Document",
    type=["pdf"],
    key="first_policy_pdf"
)

policy_text = ""

if uploaded_file:

    try:

        policy_text, page_count = extract_pdf_text(
            uploaded_file
        )

        st.success(
            f"✅ Policy loaded successfully "
            f"({page_count} pages)"
        )

        if not policy_text.strip():

            st.warning(
                "⚠️ No readable text was extracted from this PDF. "
                "The PDF may be scanned/image-based."
            )

    except Exception as e:

        st.error(
            f"Error reading PDF: {e}"
        )

        st.stop()


# =========================================================
# JSON INPUT
# =========================================================

st.subheader("🧪 Test Data")

default_json = """{
    "Quote_Number": "20260909421670000101",
    "Proposer_Salutation": "Mr",
    "Proposer_FName": "Varsha",
    "Proposer_MName": "",

}"""

json_input = st.text_area(
    "Enter Test Data (JSON)",
    value=default_json,
    height=250,
    placeholder="Leave empty to compare two PDF policies"
)


# =========================================================
# SECOND PDF
# =========================================================

second_pdf = None

if not json_input.strip():

    st.info(
        "ℹ️ JSON data is empty. "
        "You can upload a second policy PDF "
        "to compare both documents."
    )

    second_pdf = st.file_uploader(
        "Upload Second Policy PDF",
        type=["pdf"],
        key="second_policy_pdf"
    )


# =========================================================
# TEST BUTTON
# =========================================================

if st.button(
    "🚀 Test Policy",
    type="primary"
):

    # =====================================================
    # CHECK FIRST PDF
    # =====================================================

    if not uploaded_file:

        st.error(
            "Please upload the first policy PDF."
        )

        st.stop()


    if not policy_text.strip():

        st.error(
            "No readable text was found in the first PDF."
        )

        st.stop()


    # =====================================================
    # MODE 1: JSON VALIDATION
    # =====================================================

    if json_input.strip():

        st.header("🧪 Policy vs JSON Validation")

        # -------------------------------------------------
        # Validate JSON
        # -------------------------------------------------

        try:

            test_data = json.loads(
                json_input
            )

            if not isinstance(
                test_data,
                dict
            ):

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
        # Extract policy fields
        # -------------------------------------------------

        st.info(
            "🔍 Step 1/2: AI is extracting policy fields..."
        )

        try:

            policy_fields = extract_policy_fields(
                policy_text
            )

        except Exception as e:

            st.error(
                f"Policy extraction failed: {e}"
            )

            st.stop()


        # -------------------------------------------------
        # Display extracted fields
        # -------------------------------------------------

        st.subheader(
            "📋 Extracted Policy Fields"
        )

        if policy_fields:

            display_fields = []

            for field in policy_fields:

                display_fields.append(
                    {
                        "Field": str(
                            field.get(
                                "field_name",
                                ""
                            )
                        ),

                        "Value": str(
                            field.get(
                                "value",
                                ""
                            )
                        ),

                        "Page": str(
                            field.get(
                                "page",
                                ""
                            )
                        ),

                        "Category": str(
                            field.get(
                                "category",
                                ""
                            )
                        )
                    }
                )

            st.dataframe(
                display_fields,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "No policy fields were extracted."
            )

            st.stop()


        # -------------------------------------------------
        # Validate policy
        # -------------------------------------------------

        st.info(
            "🤖 Step 2/2: AI is validating the test data..."
        )

        try:

            result = validate_policy(
                policy_fields,
                test_data
            )

        except Exception as e:

            st.error(
                f"Policy validation failed: {e}"
            )

            st.stop()


        # -------------------------------------------------
        # Validation Summary
        # -------------------------------------------------

        st.subheader(
            "📊 Validation Summary"
        )

        validation_status = result.get(
            "status",
            "FAIL"
        )

        if validation_status == "PASS":

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
                ""
            )
        )


        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:

            st.metric(
                "Total Fields",
                result.get(
                    "total_fields",
                    0
                )
            )

        with col2:

            st.metric(
                "Passed",
                result.get(
                    "passed",
                    0
                )
            )

        with col3:

            st.metric(
                "Failed",
                result.get(
                    "failed",
                    0
                )
            )

        with col4:

            st.metric(
                "Not Applicable",
                result.get(
                    "not_applicable",
                    0
                )
            )

        with col5:

            st.metric(
                "Missing",
                result.get(
                    "missing_from_test_data",
                    0
                )
            )

        with col6:

            st.metric(
                "Extra",
                result.get(
                    "extra_test_fields",
                    0
                )
            )


        # -------------------------------------------------
        # Detailed Results
        # -------------------------------------------------

        st.subheader(
            "🔎 Validation Results"
        )

        results = result.get(
            "results",
            []
        )

        for item in results:

            field = item.get(
                "field",
                "Unknown Field"
            )

            input_value = item.get(
                "input_value",
                ""
            )

            policy_value = item.get(
                "policy_value",
                ""
            )

            item_status = item.get(
                "status",
                "NOT APPLICABLE"
            )

            reason = item.get(
                "reason",
                ""
            )


            # ---------------------------------------------
            # Status
            # ---------------------------------------------

            if item_status == "PASS":

                st.success(
                    f"✅ {field}: {reason}"
                )

            elif item_status == "FAIL":

                st.error(
                    f"❌ {field}: {reason}"
                )

            elif item_status == "MISSING":

                st.warning(
                    f"⚠️ {field}: Missing from test data. "
                    f"{reason}"
                )

            elif item_status == "EXTRA":

                st.warning(
                    f"⚠️ {field}: Extra test field. "
                    f"{reason}"
                )

            else:

                st.info(
                    f"ℹ️ {field}: {reason}"
                )


            # ---------------------------------------------
            # Values
            # ---------------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.write(
                    "**Input Value**"
                )

                st.code(
                    json.dumps(
                        input_value,
                        ensure_ascii=False
                    )
                )

            with col2:

                st.write(
                    "**Policy Value**"
                )

                st.code(
                    json.dumps(
                        policy_value,
                        ensure_ascii=False
                    )
                )


        # -------------------------------------------------
        # Raw JSON
        # -------------------------------------------------

        with st.expander(
            "View Raw Validation JSON"
        ):

            st.json(result)


    # =====================================================
    # MODE 2: PDF VS PDF
    # =====================================================

    else:

        st.header(
            "📑 Policy PDF vs Policy PDF Comparison"
        )


        # -------------------------------------------------
        # Check second PDF
        # -------------------------------------------------

        if not second_pdf:

            st.warning(
                "Please upload the second policy PDF "
                "to compare both documents."
            )

            st.stop()


        # -------------------------------------------------
        # Extract second PDF
        # -------------------------------------------------

        st.info(
            "📄 Reading the second policy PDF..."
        )

        try:

            second_policy_text, second_page_count = (
                extract_pdf_text(
                    second_pdf
                )
            )

        except Exception as e:

            st.error(
                f"Error reading second PDF: {e}"
            )

            st.stop()


        if not second_policy_text.strip():

            st.error(
                "No readable text was found in the "
                "second PDF."
            )

            st.stop()


        st.success(
            f"✅ Second policy loaded successfully "
            f"({second_page_count} pages)"
        )


        # -------------------------------------------------
        # Extract Policy 1 fields
        # -------------------------------------------------

        st.info(
            "🔍 Step 1/3: Extracting fields from Policy 1..."
        )

        try:

            policy_1_fields = extract_policy_fields(
                policy_text
            )

        except Exception as e:

            st.error(
                f"Policy 1 extraction failed: {e}"
            )

            st.stop()


        # -------------------------------------------------
        # Extract Policy 2 fields
        # -------------------------------------------------

        st.info(
            "🔍 Step 2/3: Extracting fields from Policy 2..."
        )

        try:

            policy_2_fields = extract_policy_fields(
                second_policy_text
            )

        except Exception as e:

            st.error(
                f"Policy 2 extraction failed: {e}"
            )

            st.stop()


        # -------------------------------------------------
        # Validate extraction
        # -------------------------------------------------

        if not policy_1_fields:

            st.error(
                "No fields were extracted from Policy 1."
            )

            st.stop()


        if not policy_2_fields:

            st.error(
                "No fields were extracted from Policy 2."
            )

            st.stop()


        # -------------------------------------------------
        # Display extracted fields
        # -------------------------------------------------

        with st.expander(
            "📋 View Policy 1 Extracted Fields"
        ):

            policy_1_display = []

            for field in policy_1_fields:

                policy_1_display.append(
                    {
                        "Field": str(
                            field.get(
                                "field_name",
                                ""
                            )
                        ),

                        "Value": str(
                            field.get(
                                "value",
                                ""
                            )
                        ),

                        "Page": str(
                            field.get(
                                "page",
                                ""
                            )
                        ),

                        "Category": str(
                            field.get(
                                "category",
                                ""
                            )
                        )
                    }
                )

            st.dataframe(
                policy_1_display,
                use_container_width=True,
                hide_index=True
            )


        with st.expander(
            "📋 View Policy 2 Extracted Fields"
        ):

            policy_2_display = []

            for field in policy_2_fields:

                policy_2_display.append(
                    {
                        "Field": str(
                            field.get(
                                "field_name",
                                ""
                            )
                        ),

                        "Value": str(
                            field.get(
                                "value",
                                ""
                            )
                        ),

                        "Page": str(
                            field.get(
                                "page",
                                ""
                            )
                        ),

                        "Category": str(
                            field.get(
                                "category",
                                ""
                            )
                        )
                    }
                )

            st.dataframe(
                policy_2_display,
                use_container_width=True,
                hide_index=True
            )


        # -------------------------------------------------
        # Compare Policies
        # -------------------------------------------------

        st.info(
            "🤖 Step 3/3: AI is comparing both policies..."
        )

        try:

            comparison_result = compare_policies(
                policy_1_fields,
                policy_2_fields
            )

        except Exception as e:

            st.error(
                f"Policy comparison failed: {e}"
            )

            st.stop()


        # -------------------------------------------------
        # Comparison Summary
        # -------------------------------------------------

        st.subheader(
            "📊 Comparison Summary"
        )

        overall_status = comparison_result.get(
            "overall_status",
            "DIFFERENT"
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
                ""
            )
        )


        # -------------------------------------------------
        # Comparison Metrics
        # -------------------------------------------------

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:

            st.metric(
                "Compared",
                comparison_result.get(
                    "total_compared_fields",
                    0
                )
            )

        with col2:

            st.metric(
                "Matching",
                comparison_result.get(
                    "matching_fields",
                    0
                )
            )

        with col3:

            st.metric(
                "Different",
                comparison_result.get(
                    "different_fields",
                    0
                )
            )

        with col4:

            st.metric(
                "Only Policy 1",
                comparison_result.get(
                    "only_in_policy_1",
                    0
                )
            )

        with col5:

            st.metric(
                "Only Policy 2",
                comparison_result.get(
                    "only_in_policy_2",
                    0
                )
            )


        # -------------------------------------------------
        # Detailed Comparison
        # -------------------------------------------------

        st.subheader(
            "🔎 Detailed Policy Comparison"
        )

        comparison_results = comparison_result.get(
            "results",
            []
        )


        for item in comparison_results:

            field = item.get(
                "field",
                "Unknown Field"
            )

            policy_1_value = item.get(
                "policy_1_value",
                ""
            )

            policy_2_value = item.get(
                "policy_2_value",
                ""
            )

            item_status = item.get(
                "status",
                ""
            )

            reason = item.get(
                "reason",
                ""
            )


            # ---------------------------------------------
            # Result status
            # ---------------------------------------------

            if item_status == "MATCH":

                st.success(
                    f"✅ {field}: {reason}"
                )

            elif item_status == "DIFFERENT":

                st.error(
                    f"❌ {field}: {reason}"
                )

            elif item_status == "ONLY_IN_POLICY_1":

                st.warning(
                    f"⚠️ {field}: "
                    f"Only present in Policy 1. "
                    f"{reason}"
                )

            elif item_status == "ONLY_IN_POLICY_2":

                st.warning(
                    f"⚠️ {field}: "
                    f"Only present in Policy 2. "
                    f"{reason}"
                )

            else:

                st.info(
                    f"ℹ️ {field}: {reason}"
                )


            # ---------------------------------------------
            # Side-by-side values
            # ---------------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.write(
                    "**Policy 1 Value**"
                )

                st.code(
                    str(policy_1_value)
                )

            with col2:

                st.write(
                    "**Policy 2 Value**"
                )

                st.code(
                    str(policy_2_value)
                )


        # -------------------------------------------------
        # Raw Comparison JSON
        # -------------------------------------------------

        with st.expander(
            "View Raw Comparison JSON"
        ):

            st.json(
                comparison_result
            )