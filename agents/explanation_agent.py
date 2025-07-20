import os
import openai
import json
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def explain_decision(parsed_query: dict, decision: dict) -> str:
    """
    Generates a simple human-readable explanation based on the claim and decision.
    """
    user_query = f"""
    A user filed a health insurance claim with these details:

    Age: {parsed_query.get('age')}
    Gender: {parsed_query.get('gender')}
    Procedure: {parsed_query.get('procedure')}
    Location: {parsed_query.get('location')}
    Policy Duration: {parsed_query.get('policy_duration_months')} months

    The claim decision was: {decision.get('decision')}

    Justification from the evaluator:
    {decision.get('justification')}

    Please summarize this claim decision in 3-4 lines using simple, clear language suitable for the customer.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful customer support assistant that explains health insurance decisions in simple terms."},
            {"role": "user", "content": user_query}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()

# For quick testing
if __name__ == "__main__":
    parsed = {
        "age": 46,
        "gender": "male",
        "procedure": "knee surgery",
        "location": "Pune",
        "policy_duration_months": 3
    }

    decision = {
        "decision": "rejected",
        "justification": "The policyholder has only been covered for 3 months, which is typically less than the waiting period required for surgeries like knee surgery."
    }

    summary = explain_decision(parsed, decision)
    print("\n📢 Customer Explanation:\n")
    print(summary)
