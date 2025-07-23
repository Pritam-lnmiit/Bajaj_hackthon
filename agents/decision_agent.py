from typing import List, Dict
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def decide_claim(parsed_query: dict, chunks: List[Dict], web_results: List[Dict] = [], medical_decision: dict = None) -> dict:
    procedure = parsed_query.get("procedure")
    months = parsed_query.get("policy_duration_months")

    context_clauses = "\n\n".join([chunk["text"] for chunk in chunks])
    web_clauses = "\n\n".join([web["snippet"] for web in web_results])

    medical_context = json.dumps(medical_decision) if medical_decision else "None"
    medical_decision_text = medical_decision.get("reason", ["No medical policy input"])[0].get("clause_text", "") if medical_decision else ""

    prompt = f"""You are an expert health insurance claim analyst.

    Given:
    - Age: {parsed_query.get('age', 'N/A')}
    - Gender: {parsed_query.get('gender', 'N/A')}
    - Location: {parsed_query.get('location', 'N/A')}
    - Procedure: {procedure}
    - Policy Duration: {months} months

    Relevant Policy Clauses (prioritize these over web results if conflicts arise):
    {context_clauses if context_clauses else "None"}

    Web Results:
    {web_clauses if web_clauses else "None"}

    Medical Policy Decision:
    {medical_context}

    Instructions:
    - Prioritize policy clauses from 'Relevant Policy Clauses' over 'Web Results' and 'Medical Policy Decision' if they conflict.
    - For planned surgeries, check if pre-authorization is required (e.g., from Medical Policy Decision) and deny if missing.
    - Suggest an amount based on the procedure: use 15000 for hip replacement, 5000 for knee surgery, 20000 for heart bypass surgery, 3000 for appendectomy, 2000 for cataract surgery, or 1000 as default. Scale by policy duration (max 100% of base amount over 12 months).
    - Return a JSON object with:
      - decision: "approved", "partially approved", or "rejected"
      - amount: number (e.g., 0 or 150000)
      - justification: concise explanation
      - matched_clauses: list of objects with "clause_text" and "source" (e.g., "policy", "web", "medical")

    Only return valid JSON. Do not include markdown or commentary.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert insurance claims analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )

        reply = response.choices[0].message.content.strip()

        if reply.startswith("```"):
            reply = reply.strip("`").strip()
            if reply.startswith("json"):
                reply = reply[4:].strip()

        return json.loads(reply)

    except Exception as e:
        return {
            "decision": "rejected",
            "amount": 0,
            "justification": f"Error during decision making: {str(e)}",
            "matched_clauses": []
        }