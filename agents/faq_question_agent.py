# agents/faq_question_agent.py

import os
import logging
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from upstash_vector import Index

# Load env variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UPSTASH_URL = os.getenv("UPSTASH_VECTOR_URL")
UPSTASH_TOKEN = os.getenv("UPSTASH_VECTOR_TOKEN")

# Clients
client = OpenAI(api_key=OPENAI_API_KEY)
index = Index(url=UPSTASH_URL, token=UPSTASH_TOKEN)

logger = logging.getLogger(__name__)

BAJAJ_SUPPORT_MESSAGE = (
    "We couldn’t find this answer in our policy database. Please contact Bajaj Allianz Health Insurance "
    "Customer Care at 1800-209-5858 or visit https://www.bajajallianz.com for more information."
)

DOCUMENTED_QUESTIONS = [
    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
    "What is the waiting period for pre-existing diseases (PED) to be covered?",
]

def answer_policy_questions(questions: List[str]) -> dict:
    answers = []
    for q in questions:
        try:
            embedding = client.embeddings.create(input=q, model="text-embedding-ada-002")
            query_vector = embedding.data[0].embedding

            response = index.query(vector=query_vector, top_k=3, include_metadata=True)
            matches = getattr(response, "matches", [])

            match_answers = [m.get("metadata", {}).get("answer") for m in matches if m.get("metadata", {}).get("answer")]

            if match_answers:
                answers.append(match_answers[0])
            else:
                try:
                    gpt_response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": (
                                "You are an expert on Bajaj Allianz Health Insurance policies. Answer questions using your own knowledge, internet knowledge, or public Bajaj policy FAQs. "
                                "If the question can't be answered from known info, provide a polite fallback with Bajaj contact details."
                            )},
                            {"role": "user", "content": q}
                        ],
                        temperature=0.3
                    )
                    gpt_answer = gpt_response.choices[0].message.content.strip()
                    answers.append(gpt_answer if gpt_answer else BAJAJ_SUPPORT_MESSAGE)
                except Exception as gex:
                    logger.error(f"GPT fallback failed for question '{q}': {gex}")
                    answers.append(BAJAJ_SUPPORT_MESSAGE)

        except Exception as e:
            logger.error(f"Error answering question '{q}': {e}")
            answers.append(BAJAJ_SUPPORT_MESSAGE)

    return {"answers": answers}
