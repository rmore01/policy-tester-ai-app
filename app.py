import json
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

client = OpenAI()

st.set_page_config(
    page_title='AI Policy Tester',
    page_icon="📄"
)

st.title("AI Policy Tester")

#upload file
uploaded_file = st.file_uploader(
    "Uploade policy document",
    type = ["pdf"]
)

policy_text=''

if uploaded_file:
    reader = PdfReader(uploaded_file)
    
    for page in reader.pages:
        text = page.extract_text()
        
        if text:
            policy_text += text + '\n'
            
    st.success(f"Policy loaded successfully ({len(reader.pages)} pages)")
    
    
#JSON Input 

default_json = """{
    "age": 35,
    "driving_license": true,
    "vehicle_damage": "Yes",
    "annual_premium": 25000
}"""

json_input = st.text_area(
    "Enter Test Data(Json)",
    value=default_json,
    height=250
)

#Test Policy

if st.button("Test Policy"):
    
    if not uploaded_file:
        st.error("Please upload a policy PDF.")
        st.stop()
        
#validate JSON first
    try:
        test_data =json.loads(json_input)
    
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON:{e}")
        st.stop()
        
    st.info("AI is analysing the Policy...")
    
    prompt= f"""
    
    You are a policy validation engine.
    Analyse the policy document and validate the provided JSON test data against the policy.
    
    POLICY DOCUMENT:
    ---------------
    {policy_text}
    
    TEST DATA:
    ---------------
    {json.dumps(test_data, indent=2)}
    
    Return the result in this exact JSON structure:
    
    {{
    "status": "PASS or FAIL",
    "summary": "short explanation",
    "results": [
        {{
            "field": "field name",
            "input_value": "value",
            "expected": "policy requirement",
            "status": "PASS or FAIL",
            "reason": "explanation"
        }}
    ]
    }}
    
    Important:
    - Check every applicable policy rule.
    - Do not invent policy rules.
    - If a Policy does not mention a field, mark it as "Not Applicable".
    - Use only information present in the policy
    
    """
    
    try:
        response = client.chat.completions.create(
            model = 'gpt-4o-mini',
            response_format={"type": "json_object"},
            messages=[
                {
                    "role":"system",
                    "content":"You are a policy validation engine"
                },
                {
                    "role":"user",
                    "content":prompt
                }
            ]
        )
        result = json.loads(response.choices[0].message.content)
        
        #display result
        
        if result["status"] == "PASS":
            st.success("✅ POLICY TEST PASSED")
        else:
            st.error("❌ POLICY TEST FAILED")

        st.write(result["summary"])

        st.subheader("Validation Results")

        for item in result["results"]:

            status = item["status"]

            if status == "PASS":
                st.success(
                    f"✅ {item['field']}: {item['reason']}"
                )

            elif status == "FAIL":
                st.error(
                    f"❌ {item['field']}: {item['reason']}"
                )

            else:
                st.warning(
                    f"⚠️ {item['field']}: {item['reason']}"
                )

            st.write(
                f"Input: `{item['input_value']}`  \n"
                f"Expected: `{item['expected']}`"
            )
    
    except Exception as e:
        st.error(f"Error: {e}")
    
    
