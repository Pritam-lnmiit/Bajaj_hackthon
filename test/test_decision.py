import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.query_parser_agent import parse_user_query
from agents.retriever_agent import retrieve_chunks
from agents.decision_agent import decide_claim

query = "46M, knee surgery in Pune, 3-month-old policy"

parsed = parse_user_query(query)
chunks = retrieve_chunks(query)
decision = decide_claim(parsed, chunks)

print("✅ Final Decision:\n")
print(decision)
