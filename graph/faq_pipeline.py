# graph/faq_pipeline.py

from agents.faq_question_agent import answer_policy_questions

def run_faq_pipeline(query: str) -> dict:
    result = answer_policy_questions([query])
    return {"answers": result["answers"]}
