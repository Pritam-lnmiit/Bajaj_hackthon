from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from graph.pipeline import run_pipeline
from graph.faq_pipeline import  run_faq_pipeline  # ✅ FIXED: proper alias
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
    logging.warning("⚠ retry module not found. Retries disabled.")
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# ---------------------- Data Models ----------------------
class QueryRequest(BaseModel):
    query: str

class VoiceQueryRequest(BaseModel):
    text: str

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

# ---------------------- App Initialization ----------------------
app = FastAPI(
    title="Insurance Claim Analyzer (RAG)",
    description="API to analyze medical insurance claims using a multi-agent RAG pipeline and provide voice-based support.",
    version="1.1.0"
)

# ✅ Serve static files (e.g. voice assistant frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------- Pinecone & Embeddings ----------------------
def get_pinecone_index():
    """Initialize Pinecone index safely"""
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("❌ PINECONE_API_KEY missing from environment.")
        pc = Pinecone(api_key=api_key)
        return pc.Index("insurance-claims")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Pinecone index: {e}")
        return None

def generate_embedding(text: str) -> list:
    """Generate embedding for storing queries in Pinecone"""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        response = client.embeddings.create(input=text, model="text-embedding-ada-002")
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"❌ Embedding generation error: {e}")
        return [0] * 1536  # fallback

def extract_structured_info(text: str) -> str:
    """Clean voice input text into a structured format"""
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
    text = text.replace("three", "3").replace("six", "6").replace("twelve", "12")
    return text.strip()

# ---------------------- API Endpoints ----------------------

@app.post("/api/claim")
@retry(tries=3, delay=1, backoff=2, logger=logger)
def analyze_claim(data: QueryRequest, think_mode: bool = False):
    """Analyze claim queries using full multi-agent pipeline"""
    try:
        logger.info(f"🚀 Processing claim for: {data.query}")
        if think_mode:
            time.sleep(2)

        result = run_pipeline(data.query, think_mode=think_mode)
        if not result:
            raise ValueError("❌ Pipeline returned empty result")

        # ✅ Optional Pinecone storage
        index = get_pinecone_index()
        if index:
            try:
                query_embedding = generate_embedding(data.query)
                index.upsert([
                    ("query-" + str(time.time()), query_embedding, {"query": data.query})
                ])
                logger.info("✅ Query stored in Pinecone.")
            except Exception as e:
                logger.warning(f"⚠ Could not store query in Pinecone: {e}")

        if not result.get("explanation") and result.get("decision"):
            result["explanation"] = f"Your claim was {result['decision']}. Amount: ₹{result['amount']}. Reason: {result['justifications']}"

        return {"status": "success", "data": result}

    except Exception as e:
        logger.error(f"❌ Error processing claim: {e}")
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

@app.post("/api/faq")
def handle_faq(data: QueryRequest):
    """Handle general policy questions via FAQ pipeline"""
    try:
        logger.info(f"📘 FAQ query received: {data.query}")
        result = run_faq_pipeline(data.query)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"❌ FAQ pipeline error: {e}")
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})

@app.post("/voice-query")
async def voice_query(data: VoiceQueryRequest):
    """Voice assistant endpoint for natural queries"""
    try:
        logger.info(f"🎙 Voice query: {data.text}")
        cleaned_query = extract_structured_info(data.text)
        logger.info(f"✅ Cleaned voice query: {cleaned_query}")

        # ✅ First try claim pipeline
        result = run_pipeline(cleaned_query, think_mode=False)

        if result and result.get("decision") != "rejected":
            response_text = result.get("explanation") or "✅ Your insurance claim is valid."
        else:
            faq_result = run_faq_pipeline(cleaned_query)
            response_text = faq_result.get("answer", "❌ No answer found.")

        return {"response": response_text}
    except Exception as e:
        logger.error(f"❌ Voice query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    """Health check endpoint"""
    return {"message": "🚀 RAG-Based Insurance Claim API is live", "version": "1.1.0"}

@app.post("/hackrx/run")
async def hackrx_run(request: HackRxRequest):
    """HackRx endpoint for bulk question answering"""
    try:
        logger.info(f"📝 HackRx received {len(request.questions)} questions")
        logger.info(f"📄 Document source: {request.documents}")

        answers = []
        for question in request.questions:
            logger.info(f"❓ Processing: {question}")
            result = run_faq_pipeline(question)
            if isinstance(result, dict) and "answer" in result:
                answers.append(result["answer"])
            elif isinstance(result, dict) and "answers" in result:
                answers.append(result["answers"][0])
            else:
                answers.append("❌ Sorry, no answer found.")

        return {"answers": answers}

    except Exception as e:
        logger.error(f"❌ HackRx error: {e}")
        raise HTTPException(status_code=500, detail={"status": "error", "message": str(e)})
