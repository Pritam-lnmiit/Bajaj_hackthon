import streamlit as st
import json
import sys
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# Fix Python path to import pipeline
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

try:
    from graph.pipeline import run_pipeline
except ImportError as e:
    st.error(f"Failed to import pipeline: {str(e)}")
    st.stop()

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
UPSTASH_VECTOR_URL = os.getenv("UPSTASH_VECTOR_URL")
UPSTASH_VECTOR_TOKEN = os.getenv("UPSTASH_VECTOR_TOKEN")

# Initialize Pinecone and OpenAI
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "insurance-claims"
index = pc.Index(INDEX_NAME)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def generate_embedding(text: str) -> list:
    """Generate embedding for text using OpenAI."""
    try:
        response = openai_client.embeddings.create(input=text, model="text-embedding-ada-002")
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Embedding generation error: {str(e)}")
        return []

# Streamlit app configuration
st.set_page_config(page_title="Insurance Claim Processor", layout="wide")

# Title and description
st.title("Insurance Claim Processor")
st.markdown("""
Enter a claim query (e.g., "46M, knee surgery in Pune, 3-month-old policy") to process an insurance claim.
The system will parse the query, evaluate it against policy documents, and provide a decision with justifications.
""")

# Input form
with st.form("claim_form"):
    query = st.text_input("Enter your claim query:", placeholder="e.g., 46M, knee surgery in Pune, 3-month-old policy")
    submitted = st.form_submit_button("Process Claim")

# Process query and display results
if submitted and query:
    with st.spinner("Processing your claim..."):
        try:
            result = run_pipeline(query)
            st.success("Claim processed successfully!")

            # Display JSON output
            st.subheader("Claim Processing Result")
            st.json(result)

            # Human-readable summary
            st.subheader("Summary")
            st.write(f"**Query**: {result['query']}")
            st.write(f"**Decision**: {result['decision'].capitalize()}")
            st.write(f"**Amount**: {result['amount']}")
            st.write("**Justifications**:")
            justifications = result.get("justifications", [])
            if isinstance(justifications, str):
                st.write(f"- {justifications} (Source: system)")
            else:
                for just in justifications:
                    st.write(f"- {just.get('clause_text', 'No justification')} (Source: {just.get('source', 'system')})")
            st.write(f"**Explanation**: {result['explanation']}")

            # Confirm Pinecone storage
            query_embedding = generate_embedding(result["query"])
            results = index.query(vector=query_embedding, top_k=1, include_metadata=True)
            if results["matches"]:
                st.info("Claim data successfully stored in Pinecone!")

        except Exception as e:
            st.error(f"Error processing claim: {str(e)}")
            st.write("Please check your query or system configuration and try again.")

# Statistical Analysis Section
st.subheader("Business Improvement Statistics")
if st.button("Generate Statistics"):
    with st.spinner("Calculating statistics..."):
        try:
            # Fetch all vectors
            query_response = index.query(vector=[0]*1536, top_k=10000, include_metadata=True)
            ids = [match["id"] for match in query_response["matches"]]
            fetch_response = index.fetch(ids=ids)
            vectors = fetch_response.vectors

            # Calculate statistics
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
                policy_durations.append(parsed_query.get("policy_duration_months", 0))
                procedure = parsed_query.get("procedure", "Unknown")
                procedures[procedure] = procedures.get(procedure, 0) + 1
                just = json.loads(metadata.get("justifications", "[]"))[0] if metadata.get("justifications") else {"clause_text": "No justification", "source": "system"}
                key = just if isinstance(just, str) else just.get("clause_text", "No justification")
                justifications[key] = justifications.get(key, 0) + 1

            rejection_rate = (rejections / total_claims * 100) if total_claims > 0 else 0
            avg_policy_duration = sum(policy_durations) / len(policy_durations) if policy_durations else 0
            top_procedures = dict(sorted(procedures.items(), key=lambda x: x[1], reverse=True)[:5])
            top_justifications = dict(sorted(justifications.items(), key=lambda x: x[1], reverse=True)[:5])

            # Display statistics as percentages where applicable
            st.metric("Rejection Rate", f"{rejection_rate:.1f}%")
            st.write(f"**Average Policy Duration**: {avg_policy_duration:.1f} months")
            st.write("**Top 5 Procedures by Frequency**:")
            for proc, count in top_procedures.items():
                percentage = (count / total_claims * 100) if total_claims > 0 else 0
                st.write(f"- {proc}: {count} claims ({percentage:.1f}%)")
            st.write("**Top 5 Rejection Justifications**:")
            for just, count in top_justifications.items():
                percentage = (count / total_claims * 100) if total_claims > 0 else 0
                st.write(f"- {just}: {count} instances ({percentage:.1f}%)")

        except Exception as e:
            st.error(f"Error generating statistics: {str(e)}")
            st.write("Ensure data is stored in Pinecone and try again.")

# Footer
st.markdown("---")
st.markdown("Powered by ssmc_AI | Built for Bajaj Insurance Claim Processing")