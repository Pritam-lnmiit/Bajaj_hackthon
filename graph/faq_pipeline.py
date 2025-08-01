import sys
import os
import json
import logging
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from datetime import datetime

# ✅ Fix Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# ✅ Import retriever
try:
    from agents.retriever_agent import retrieve_chunks
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# ✅ Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ✅ Define pipeline state
class QAState(TypedDict):
    raw_query: str
    retrieved_chunks: List[Dict]
    final_answer: str
    timestamp: str

# ✅ STEP 1: Retrieve chunks
def retrieve_node(state: QAState) -> QAState:
    query = state["raw_query"]
    logger.info(f"🔍 Retrieving chunks for: {query}")

    try:
        chunks = retrieve_chunks(query, k=12)
        logger.info(f"✅ Retrieved {len(chunks)} chunks for query: {query}")
        return {"retrieved_chunks": chunks or []}
    except Exception as e:
        logger.error(f"❌ Retrieval error: {e}")
        return {"retrieved_chunks": []}

# ✅ STEP 2: Generate answer using GPT (with improved filtering and fallback)
def answer_node(state: QAState) -> QAState:
    query = state["raw_query"]
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        logger.warning(f"⚠ No chunks found for: {query}")
        return {"final_answer": "Sorry, I couldn’t find this in the policy database. Please contact Bajaj Allianz for details."}

    # ✅ Filter and rank high-scoring chunks
    high_quality_chunks = [c for c in chunks if c.get("score", 0) >= 0.85]
    high_quality_chunks = sorted(high_quality_chunks, key=lambda x: x["score"], reverse=True)
    top_chunks = high_quality_chunks[:6] if high_quality_chunks else chunks[:3]

    # ✅ Combine chunks into one context
    context_text = "\n\n".join([c["text"] for c in top_chunks])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    prompt = f"""
    You are an expert Bajaj Allianz Health Insurance assistant.

    User Question: {query}

    Context from the policy:
    {context_text}

    🔀 Write a clear, concise, and accurate answer using only the above policy context.
    🔀 If context partially answers the question, summarize what is available.
    🔀 If there is no relevant content, respond with:
    \"This information is not available in the policy database.\"
    """

    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()

        if "not available" in answer.lower():
            fallback_text = top_chunks[0]["text"]
            logger.info("⚠ GPT fallback triggered — returning top chunk from retriever")
            answer = f"From the policy: {fallback_text.strip().splitlines()[0]}..."

        logger.info(f"✅ Answer generated for query: {query}")
        return {"final_answer": answer}
    except Exception as e:
        logger.error(f"❌ GPT answer generation failed: {e}")
        return {"final_answer": "Failed to generate an answer at this time."}

# ✅ STEP 3: Add timestamp
def finalize_node(state: QAState) -> QAState:
    state["timestamp"] = datetime.now().isoformat()
    return state

# ✅ Build the graph
graph = StateGraph(QAState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("answer", answer_node)
graph.add_node("finalize", finalize_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "answer")
graph.add_edge("answer", "finalize")
graph.add_edge("finalize", END)

try:
    qa_app = graph.compile()
    logger.info("✅ FAQ/QA Graph compiled successfully")
except Exception as e:
    logger.error(f"❌ Graph compilation failed: {e}")
    sys.exit(1)

# ✅ Function to run the pipeline
def run_faq_pipeline(query: str) -> Dict:
    logger.info(f"🚀 Running FAQ pipeline for: {query}")
    try:
        result = qa_app.invoke({"raw_query": query})
        return {
            "query": query,
            "answers": [result.get("final_answer", "")],
            "chunks_used": result.get("retrieved_chunks", []),
            "timestamp": result.get("timestamp", "")
        }
    except Exception as e:
        logger.error(f"❌ FAQ pipeline failed: {e}")
        return {
            "query": query,
            "answers": ["Pipeline failed to process query."],
            "chunks_used": [],
            "timestamp": datetime.now().isoformat()
        }

# ✅ Self-test
if __name__ == "__main__":
    print("\n=== 🧪 FAQ PIPELINE SELF-TEST ===\n")
    test_query = "What is covered under domiciliary hospitalization and what are its conditions?"
    result = run_faq_pipeline(test_query)

    print(f"🔍 Query: {result['query']}")
    print(f"✅ Answer: {result['answers'][0]}\n")
    print("📦 Chunks Used:")
    for idx, c in enumerate(result["chunks_used"], 1):
        print(f"{idx}. (Score: {c.get('score', 0):.2f}) {c['text'][:150]}...")