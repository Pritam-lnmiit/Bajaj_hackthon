# agents/faq_question_agent.py

import os
import logging
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from upstash_vector import Index
from langchain_openai import OpenAIEmbeddings

# ✅ Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UPSTASH_URL = os.getenv("UPSTASH_VECTOR_URL")
UPSTASH_TOKEN = os.getenv("UPSTASH_VECTOR_TOKEN")

# ✅ Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
embedding_model = OpenAIEmbeddings()
index = Index(url=UPSTASH_URL, token=UPSTASH_TOKEN)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ✅ Fallback message
FALLBACK_MESSAGE = (
    "We couldn’t find this answer in our policy database. "
    "Please contact Bajaj Allianz Health Insurance Customer Care at 1800-209-5858 "
    "or visit https://www.bajajallianz.com for more information."
)

def answer_policy_questions(questions: List[str]) -> dict:
    """
    ✅ Uses retriever-style logic to query Upstash for FAQ answers.
    ✅ Falls back to GPT ONLY if no chunk is found at all.
    """
    answers = []

    for q in questions:
        try:
            # ✅ Step 1: Dense embedding for query
            query_vector = embedding_model.embed_query(q)

            # ✅ Step 2: Query Upstash Vector DB
            response = index.query(vector=query_vector, top_k=5, include_metadata=True)
            matches = getattr(response, "matches", [])

            if matches:
                logger.info(f"✅ Found {len(matches)} matches for: {q}")

                # ✅ Look for a chunk with actual text/answer
                best_match_text = None
                for m in matches:
                    meta = m.get("metadata", {})
                    # ✅ Prefer answer field, fallback to text field
                    content = meta.get("answer") or meta.get("text")
                    if content and len(content.strip()) > 20:
                        best_match_text = content.strip()
                        break

                if best_match_text:
                    # ✅ Clean formatting
                    clean_answer = " ".join(best_match_text.split())
                    answers.append(clean_answer)
                    continue  # 🛑 Skip GPT fallback for this question

            # 🚨 If Upstash had NO matches → GPT fallback
            logger.warning(f"⚠ No match found for: {q}, using GPT fallback.")
            gpt_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": (
                        "You are an expert on Bajaj Allianz Health Insurance policies. "
                        "Answer the question factually. If unsure, respond with fallback message."
                    )},
                    {"role": "user", "content": q}
                ],
                temperature=0.2
            )
            gpt_answer = gpt_response.choices[0].message.content.strip()
            answers.append(gpt_answer if gpt_answer else FALLBACK_MESSAGE)

        except Exception as e:
            logger.error(f"❌ Error answering question '{q}': {e}")
            answers.append(FALLBACK_MESSAGE)

    return {"answers": answers}


# ✅ ----------------------------
# ✅ SELF-TEST SECTION
# ✅ ----------------------------
if __name__ == "__main__":
    print("\n=== 🧪 FAQ QUESTION AGENT SELF-TEST ===\n")

    test_questions = [
        "What is the cataract surgery limit under this policy?",
        "Does the policy cover maternity expenses?",
        "Are AYUSH treatments covered?",
        "What is the ambulance coverage limit?"
    ]

    result = answer_policy_questions(test_questions)

    for i, ans in enumerate(result["answers"], 1):
        print(f"Q{i}: {test_questions[i-1]}")
        print(f"A{i}: {ans}")
        print("-" * 60)
