import openai
import os
from dotenv import load_dotenv
import json

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def parse_user_query(raw_query: str) -> dict:
    system_prompt = (
        "You are an expert at parsing insurance claim queries. "
        "Your task is to extract and return structured JSON with keys: "
        "`age` (int), `gender` (male/female), `procedure` (string), "
        "`location` (city), and `policy_duration_months` (int). "
        "Only return valid JSON. No explanations."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_query}
        ],
        temperature=0.0
    )

    reply = response.choices[0].message.content.strip()

    # ✅ Strip triple backticks and language tag if present
    if reply.startswith("```"):
        reply = reply.strip("```").strip()
        if reply.startswith("json"):
            reply = reply[4:].strip()

    try:
        return json.loads(reply)
    except Exception:
        return {"error": "Failed to parse JSON", "raw_response": reply}
