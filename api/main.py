from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from graph.pipeline import run_pipeline
from pinecone import Pinecone
import logging
import time
import json
import os
from typing import Dict, List

# 📦 Define input structure
class QueryRequest(BaseModel):
    query: str

# 🚀 Initialize FastAPI
app = FastAPI(
    title="Insurance Claim Analyzer (RAG)",
    description="An API to analyze medical insurance claims using a multi-agent RAG pipeline and provide statistics.",
    version="1.0.0"
)

# 🌍 CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🗂️ Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 🌲 Initialize Pinecone lazily (on demand)
def get_pinecone_index():
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        pc = Pinecone(api_key=api_key)
        return pc.Index("insurance-claims")
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone index: {str(e)}")
        raise

# ✅ API route with think_mode option
@app.post("/api/claim")
def analyze_claim(data: QueryRequest, think_mode: bool = False):
    try:
        logger.info(f"Processing claim for query: {data.query} with think_mode: {think_mode}")
        if think_mode:
            time.sleep(2)  # Simulate think mode delay
        result = run_pipeline(data.query, think_mode=think_mode)
        if not result:
            raise ValueError("Pipeline returned empty result")
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Error processing claim: {str(e)} - Query: {data.query}")
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

# 📊 API route to get statistics
@app.get("/api/stats")
def get_statistics():
    try:
        logger.info("Generating statistics from Pinecone index")
        index = get_pinecone_index()
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

        stats = {
            "rejection_rate": f"{rejection_rate:.1f}%",
            "avg_policy_duration": f"{avg_policy_duration:.1f} months",
            "top_procedures": {k: v for k, v in top_procedures.items()},
            "top_justifications": {k: v for k, v in top_justifications.items()}
        }
        return {"status": "success", "data": stats}
    except Exception as e:
        logger.error(f"Error generating statistics: {str(e)}")
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

# 🧪 Optional root check with version
@app.get("/")
def home():
    return {"message": "RAG-Based Insurance Claim API is live 🚀", "version": "1.0.0"}