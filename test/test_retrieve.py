import sys
import os

# ✅ This is correct
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.retriever_agent import retrieve_chunks

query = "Is knee surgery covered under 3-month-old policy in Pune?"
chunks = retrieve_chunks(query)

for i, chunk in enumerate(chunks):
    print(f"\n--- Match {i+1} (score: {chunk['score']:.2f}) ---")
    print(chunk["text"][:500])
