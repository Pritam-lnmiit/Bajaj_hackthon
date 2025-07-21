import os
import json
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "insurance-claims"
index = pc.Index(INDEX_NAME)

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def generate_embedding(text: str) -> list:
    """Generate embedding for text using OpenAI."""
    try:
        response = openai_client.embeddings.create(input=text, model="text-embedding-ada-002")
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding generation error: {str(e)}")
        return []

# Fetch all vectors (for small datasets; use pagination for large ones)
def fetch_all_vectors():
    # Query to get all IDs
    query_response = index.query(vector=[0]*1536, top_k=10000, include_metadata=True)
    ids = [match["id"] for match in query_response["matches"]]
    # Fetch vectors
    fetch_response = index.fetch(ids=ids)
    return fetch_response.vectors

# Process data into statistics
def calculate_statistics(vectors):
    rejections = 0
    total_claims = len(vectors)
    policy_durations = []
    procedures = {}
    justifications = {}

    for vector_id, vector_data in vectors.items():
        metadata = vector_data["metadata"]
        if metadata.get("decision") == "rejected":
            rejections += 1
        parsed_query = json.loads(metadata.get("parsed_query", "{}"))
        if isinstance(parsed_query, dict):
            policy_duration = parsed_query.get("policy_duration_months")
            policy_durations.append(policy_duration if policy_duration is not None else 0)
            procedure = parsed_query.get("procedure", "Unknown")
            procedures[procedure] = procedures.get(procedure, 0) + 1
        just = json.loads(metadata.get("justifications", "[]"))[0] if metadata.get("justifications") else {"clause_text": "No justification", "source": "system"}
        key = just if isinstance(just, str) else just.get("clause_text", "No justification")
        justifications[key] = justifications.get(key, 0) + 1

    # Calculate stats
    rejection_rate = (rejections / total_claims * 100) if total_claims > 0 else 0
    avg_policy_duration = sum(policy_durations) / len(policy_durations) if policy_durations else 0
    top_procedures = dict(sorted(procedures.items(), key=lambda x: x[1], reverse=True)[:5])
    top_justifications = dict(sorted(justifications.items(), key=lambda x: x[1], reverse=True)[:5])

    return {
        "rejection_rate": rejection_rate,
        "avg_policy_duration": avg_policy_duration,
        "top_procedures": top_procedures,
        "top_justifications": top_justifications
    }

# Generate and display statistics
vectors = fetch_all_vectors()
stats = calculate_statistics(vectors)

print("Business Improvement Statistics:")
print(f"Total Claims Analyzed: {len(vectors)}")
print(f"Rejection Rate: {stats['rejection_rate']:.1f}%")
print(f"Average Policy Duration (months): {stats['avg_policy_duration']:.1f}")
print("Top 5 Procedures by Frequency:")
for proc, count in stats["top_procedures"].items():
    print(f"  {proc}: {count} claims")
print("Top 5 Rejection Justifications:")
for just, count in stats["top_justifications"].items():
    print(f"  {just}: {count} instances")