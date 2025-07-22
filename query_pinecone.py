from pinecone import Pinecone
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv(dotenv_path="C:/bajaj/.env")  # Explicit path
print("Current working directory:", os.getcwd())  # Debug
print("Checking .env file at C:/bajaj/.env:", os.path.exists("C:/bajaj/.env"))  # Debug

# Initialize Pinecone
api_key = os.getenv("PINECONE_API_KEY")
print("PINECONE_API_KEY:", api_key)  # Debug
if not api_key:
    raise ValueError("PINECONE_API_KEY not found in environment variables")
pc = Pinecone(api_key=api_key)
index = pc.Index("insurance-claims")

# Query for the specific query
query_response = index.query(
    vector=[0]*1536,  # Dummy vector for metadata filtering
    top_k=10,
    include_metadata=True,
    filter={"query": "45F, knee replacement surgery, Mumbai, 24-month policy"}
)

# Check results
if query_response["matches"]:
    print("Query found in Pinecone:")
    for match in query_response["matches"]:
        print(f"ID: {match['id']}, Metadata: {match['metadata']}")
else:
    print("Query not found in Pinecone")