# test_faq.py

from agents import faq_question_agent
import os
import logging
import json

if __name__ == "__main__":
    sample_questions = [
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

    result = faq_question_agent(sample_questions)
    print("\n✅ FAQ TEST OUTPUT:\n")
    for i, ans in enumerate(result['answers']):
        print(f"Q{i+1}: {sample_questions[i]}\nA: {ans}\n")
