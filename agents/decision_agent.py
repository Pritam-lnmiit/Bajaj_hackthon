from typing import List, Dict
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

# ✅ Load environment
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def decide_claim(parsed_query: dict, chunks: List[Dict], web_results: List[Dict] = []) -> dict:
    procedure = parsed_query.get("procedure")
    months = parsed_query.get("policy_duration_months")

    context_clauses = "\n\n".join([chunk["text"] for chunk in chunks])
    web_clauses = "\n\n".join([web["snippet"] for web in web_results])

    prompt = f"""You are an expert health insurance claim analyst.

Given:
- Age: {parsed_query.get('age')}
- Gender: {parsed_query.get('gender')}
- Location: {parsed_query.get('location')}
- Procedure: {procedure}
- Policy Duration: {months} months

Relevant Policy Clauses:
{context_clauses if context_clauses else "None"}

Web Results:
{web_clauses if web_clauses else "None"}

Based on the above, return a JSON object with:
- decision: "approved", "partially approved", or "rejected"
- amount: number (e.g., 0 or 150000)
- justification: concise explanation
- matched_clauses: list of objects with "clause_text" and "source"

Only return valid JSON. Do not include markdown or commentary.
"""

    try:
        response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:personal:insurance-parser-chat:BvL1R17y",
            messages=[
                {"role": "system", "content": "You are an expert insurance claims analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )

        reply = response.choices[0].message.content.strip()

        # 🧼 Remove markdown if it exists
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
