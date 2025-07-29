from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from graph.pipeline import run_pipeline
from graph.faq_pipeline import run_faq_pipeline  # ✅ Corrected import
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
    text = re.sub(r"(hello|hi)[^,\.]*[,\.]", "", text)
    text = re.sub(r"my name is [a-z ]+", "", text)
    text = text.replace("ki surgery", "knee surgery")
    text = text.replace("key surgery", "knee surgery")
    text = re.sub(r"male", "M", text)
    text = re.sub(r"female", "F", text)
    text = re.sub(r"i am (\d+)", r"\1", text)
    text = re.sub(r"(\d+)\s*(year)?s?\s*old", r"\1", text)
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

@app.post("/api/faq")
def handle_faq(data: QueryRequest):
    try:
        logger.info(f"Processing FAQ for query: {data.query}")
        result = run_faq_pipeline(data.query)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"FAQ pipeline error: {str(e)}")
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

@app.post("/voice-query")
async def voice_query(data: VoiceQueryRequest):
    try:
        logger.info(f"Voice query received: {data.text}")
        cleaned_query = extract_structured_info(data.text)
        logger.info(f"Cleaned voice input: {cleaned_query}")

        # Try claim pipeline first
        result = run_pipeline(cleaned_query, think_mode=False)

        if result and result.get("decision") != "rejected":
            response_text = result.get("explanation") or "Your insurance claim is valid."
        else:
            faq_result = run_faq_pipeline(cleaned_query)
            response_text = faq_result.get("answers", ["Sorry, no answer found."])[0]

        return {"response": response_text}
    except Exception as e:
        logger.error(f"Voice query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "RAG-Based Insurance Claim API is live 🚀", "version": "1.1.0"}
