from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from graph.pipeline import run_pipeline  # ✅ Adjust path if needed

# 📦 Define input structure
class QueryRequest(BaseModel):
    query: str

# 🚀 Initialize FastAPI
app = FastAPI(
    title="Insurance Claim Analyzer (RAG)",
    description="An API to analyze medical insurance claims using a multi-agent RAG pipeline.",
    version="1.0.0"
)

# 🌍 CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ API route
@app.post("/api/claim")
def analyze_claim(data: QueryRequest):
    return run_pipeline(data.query)

# 🧪 Optional root check
@app.get("/")
def home():
    return {"message": "RAG-Based Insurance Claim API is live 🚀"}
