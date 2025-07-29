
import os
import openai
import json
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def explain_decision(parsed_query: dict, decision: dict) -> str:
    if not parsed_query or not decision:
        logger.error(f"Invalid input - parsed_query: {parsed_query}, decision: {decision}")
        return "We couldn't process your claim due to missing information. Please contact support."

    age = parsed_query.get('age', 'N/A')
    gender = parsed_query.get('gender', 'N/A')
    procedure = parsed_query.get('procedure', 'N/A')
    location = parsed_query.get('location', 'N/A')
    policy_duration = parsed_query.get('policy_duration_months', 'N/A')
    claim_decision = decision.get('decision', 'rejected')
    amount = decision.get('amount', 0)
    justification = decision.get('justification', 'No justification provided') if isinstance(decision.get('justification'), str) else 'No justification provided'

    user_query = f"""
    A user filed a health insurance claim with these details:

    Age: {age}
    Gender: {gender}
    Procedure: {procedure}
    Location: {location}
    Policy Duration: {policy_duration} months

    The claim decision was: {claim_decision}
    Approved Amount: ₹{amount}

    Justification from the evaluator:
    {justification}

    Please summarize this claim decision in 3-4 lines using simple, clear language suitable for the customer.
    Include the approved amount in the summary if the claim is approved or partially approved.
    Ensure the explanation is grammatically correct and clear.
    """

    try:
        logger.debug(f"Generating explanation for procedure: {procedure}, decision: {claim_decision}, amount: {amount}")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful customer support assistant that explains health insurance decisions in simple terms."},
                {"role": "user", "content": user_query}
            ],
            temperature=0.3
        )
        explanation = response.choices[0].message.content.strip()
        logger.debug(f"Generated explanation: {explanation}")
        return explanation
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        return f"We couldn't process your claim explanation due to an error: {str(e)}. Please contact support."
