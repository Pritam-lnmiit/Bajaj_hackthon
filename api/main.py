from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from graph.pipeline import run_pipeline
from pinecone import Pinecone
import logging
import time
import json
import os
from typing import Dict, List
import re

try:
    from retry import retry
except ImportError:
    logging.warning("retry module not found. Retries disabled.")
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

class QueryRequest(BaseModel):
    query: str

class VoiceQueryRequest(BaseModel):
    text: str

app = FastAPI(
    title="Insurance Claim Analyzer (RAG)",
    description="API to analyze medical insurance claims using a multi-agent RAG pipeline and provide voice-based support.",
    version="1.1.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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

def generate_embedding(text: str) -> list:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        response = client.embeddings.create(input=text, model="text-embedding-ada-002")
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding generation error: {str(e)}")
        return [0] * 1536

def extract_structured_info(text: str) -> str:
    text = text.lower().strip()

    # Remove greetings or intro
    text = re.sub(r"(hello|hi)[^,\.]*[,\.]", "", text)
    text = re.sub(r"my name is [a-z ]+", "", text)

    # Fix common ASR errors
    text = text.replace("ki surgery", "knee surgery")
    text = text.replace("key surgery", "knee surgery")
    text = re.sub(r"male", "M", text)
    text = re.sub(r"female", "F", text)
    text = re.sub(r"i am (\d+)", r"\1", text)
    text = re.sub(r"(\d+)\s*(year)?s?\s*old", r"\1", text)

    # Clean known patterns
    text = text.replace("months policy", "month policy")
    text = text.replace("policy of", "")
    text = text.replace("three", "3")
    text = text.replace("six", "6")
    text = text.replace("twelve", "12")

    return text.strip()

@app.post("/api/claim")
@retry(tries=3, delay=1, backoff=2, logger=logger)
def analyze_claim(data: QueryRequest, think_mode: bool = False):
    try:
        logger.info(f"Processing claim for query: {data.query} with think_mode: {think_mode}")
        if think_mode:
            time.sleep(2)
        result = run_pipeline(data.query, think_mode=think_mode)
        if not result:
            raise ValueError("Pipeline returned empty result")

        index = get_pinecone_index()
        query_embedding = generate_embedding(data.query)
        query_response = index.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True,
            filter={"query": data.query}
        )

        if not query_response["matches"]:
            logger.warning(f"Query '{data.query}' not found in Pinecone")
        else:
            logger.info(f"Stored in Pinecone: {query_response['matches'][0]['metadata']}")

        if not result.get("explanation") and result.get("decision"):
            result["explanation"] = f"Your claim was {result['decision']}. Amount: ₹{result['amount']}. Reason: {result['justifications']}"

        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Error processing claim: {str(e)} - Query: {data.query}")
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

@app.get("/api/stats")
def get_statistics():
    try:
        logger.info("Generating statistics from Pinecone index")
        index = get_pinecone_index()
        query_response = index.query(vector=[0]*1536, top_k=10000, include_metadata=True)
        ids = [match["id"] for match in query_response["matches"]]
        fetch_response = index.fetch(ids=ids)
        vectors = fetch_response.vectors

        rejections = 0
        total_claims = len(vectors)
        policy_durations = []
        procedures = {}
        justifications = {}
        approved_amounts = []

        for vector_id, vector_data in vectors.items():
            metadata = vector_data["metadata"]
            if metadata.get("decision") == "rejected":
                rejections += 1
            amount = metadata.get("amount", 0)
            if isinstance(amount, (int, float)) and amount > 0:
                approved_amounts.append(amount)
            try:
                parsed_query = json.loads(metadata.get("parsed_query", "{}"))
                if isinstance(parsed_query, dict):
                    policy_duration = parsed_query.get("policy_duration_months")
                    policy_durations.append(policy_duration if policy_duration is not None else 0)
                    procedure = parsed_query.get("procedure", "Unknown")
                    procedures[procedure] = procedures.get(procedure, 0) + 1
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse parsed_query: {metadata.get('parsed_query')}")
            try:
                just_list = json.loads(metadata.get("justifications", "[]"))
                if isinstance(just_list, list) and just_list:
                    key = just_list[0].get("clause_text", "No justification") if isinstance(just_list[0], dict) else str(just_list[0])
                    justifications[key] = justifications.get(key, 0) + 1
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse justifications: {metadata.get('justifications')}")
                justifications["No justification"] = justifications.get("No justification", 0) + 1

        rejection_rate = (rejections / total_claims * 100) if total_claims > 0 else 0
        avg_policy_duration = sum(policy_durations) / len(policy_durations) if policy_durations else 0
        avg_approved_amount = sum(approved_amounts) / len(approved_amounts) if approved_amounts else 0
        top_procedures = dict(sorted(procedures.items(), key=lambda x: x[1], reverse=True)[:5])
        top_justifications = dict(sorted(justifications.items(), key=lambda x: x[1], reverse=True)[:5])

        stats = {
            "rejection_rate": f"{rejection_rate:.1f}%",
            "avg_policy_duration": f"{avg_policy_duration:.1f} months",
            "avg_approved_amount": f"₹{avg_approved_amount:.2f}",
            "top_procedures": top_procedures,
            "top_justifications": top_justifications
        }
        return {"status": "success", "data": stats}
    except Exception as e:
        logger.error(f"Error generating statistics: {str(e)}")
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

@app.post("/voice-query")
async def voice_query(data: VoiceQueryRequest):
    try:
        logger.info(f"Voice query received: {data.text}")
        cleaned_query = extract_structured_info(data.text)
        logger.info(f"Cleaned voice input: {cleaned_query}")
        result = run_pipeline(cleaned_query, think_mode=False)
        if not result:
            raise ValueError("No result returned from pipeline")
        response_text = result.get("explanation") or "I'm sorry, I couldn't process your query."
        return {"response": response_text}
    except Exception as e:
        logger.error(f"Voice query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "RAG-Based Insurance Claim API is live 🚀", "version": "1.1.0"}
