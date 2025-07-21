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

# Query Pinecone for similar vectors
query_text = "46M, knee surgery in Pune, 3-month-old policy"
query_vector = generate_embedding(query_text)
results = index.query(vector=query_vector, top_k=5, include_metadata=True)
print("Stored vectors matching query:")
for match in results["matches"]:
    print(f"ID: {match['id']}")
    print(f"Score: {match['score']}")
    print(f"Metadata: {json.dumps(match['metadata'], indent=2)}")
    print("-" * 50)