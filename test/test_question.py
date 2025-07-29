# test/test_question.py

import sys
import os
import json

# ✅ Add parent folder to Python path so 'agents' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.retriever_agent import retrieve_chunks
from agents.question_answer_agent import QuestionAnswerAgent

# 🔍 HackRx test questions
questions = [
    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
    "What is the waiting period for pre-existing diseases (PED) to be covered?",
    "Does this policy cover maternity expenses, and what are the conditions?",
    "What is the waiting period for cataract surgery?",
    "Are the medical expenses for an organ donor covered under this policy?",
    "What is the No Claim Discount (NCD) offered in this policy?",
    "Is there a benefit for preventive health check-ups?",
    "How does the policy define a 'Hospital'?",
    "What is the extent of coverage for AYUSH treatments?",
    "Are there any sub-limits on room rent and ICU charges for Plan A?"
]

# ✅ Instantiate agents
retriever = RetrieverAgent(index_name="hackrx-policy")  # ensure your index is named correctly
qa_agent = QuestionAnswerAgent(retriever)

# 🧪 Run QA for each question
answers = []
print("\n📘 HackRx Q&A Test\n------------------\n")

for i, question in enumerate(questions, 1):
    print(f"🔹 Q{i}: {question}")
    try:
        answer = qa_agent.answer(question)
    except Exception as e:
        answer = f"❌ Error answering question: {str(e)}"
    answers.append(answer)
    print(f"✅ A{i}: {answer}\n")

# 📦 Final JSON-style output
result = {"answers": answers}
print("📦 Final JSON Response:\n")
print(json.dumps(result, indent=4))
