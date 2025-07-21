import sys
import os
import json
import logging
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from openai import OpenAI
import uuid
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec
import time  # For think mode simulation

# Fix Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Import agents
try:
    from agents.query_parser_agent import parse_user_query
    from agents.retriever_agent import retrieve_chunks
    from agents.web_search_agent import search_policy_location
    from agents.medical_policy_agent import MedicalPolicyAgent
    from agents.decision_agent import decide_claim
    from agents.explanation_agent import explain_decision
except ImportError as e:
    print(f"Import error: {str(e)}")
    sys.exit(1)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
UPSTASH_VECTOR_URL = os.getenv("UPSTASH_VECTOR_URL")
UPSTASH_VECTOR_TOKEN = os.getenv("UPSTASH_VECTOR_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "insurance-claims"
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # Matches text-embedding-ada-002
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Adjust region if needed
    )
index = pc.Index(INDEX_NAME)

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class GraphState(TypedDict):
    raw_query: str
    parsed_query: dict
    retrieved_chunks: List[Dict]
    web_results: List[Dict]
    medical_decision: dict
    final_decision: dict
    explanation: str
    final_response: dict
    attempt_count: int

# Initialize MedicalPolicyAgent
policy_file = os.path.join(project_root, "agents", "local_policy.json")
if not os.path.exists(policy_file):
    logger.warning(f"Policy file {policy_file} not found. Using default rules.")
    policy_file = ""
medical_agent = MedicalPolicyAgent(policy_file)

def generate_embedding(text: str) -> list:
    """Generate embedding for text using OpenAI."""
    try:
        response = openai_client.embeddings.create(input=text, model="text-embedding-ada-002")
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding generation error: {str(e)}")
        return []

# Node functions
def parse_node(state: GraphState) -> GraphState:
    logger.debug("Parsing query: %s", state["raw_query"])
    try:
        parsed = parse_user_query(state["raw_query"])
        return {**state, "parsed_query": parsed or {}, "attempt_count": 0}
    except Exception as e:
        logger.error(f"Parse error: {str(e)} - Query: {state['raw_query']}")
        return {**state, "parsed_query": {}, "attempt_count": 0}

def retrieve_node(state: GraphState) -> GraphState:
    logger.debug("Retrieving chunks for query: %s", state["raw_query"])
    try:
        chunks = retrieve_chunks(state["raw_query"], url=UPSTASH_VECTOR_URL, token=UPSTASH_VECTOR_TOKEN)
        return {**state, "retrieved_chunks": chunks or []}
    except Exception as e:
        logger.error(f"Retrieve error: {str(e)} - Query: {state['raw_query']}")
        return {**state, "retrieved_chunks": []}

def web_search_node(state: GraphState) -> GraphState:
    logger.debug("Performing web search for query: %s", state["raw_query"])
    try:
        results = search_policy_location(state["raw_query"], state["parsed_query"].get("location", ""), api_key=SERPAPI_KEY)
        return {**state, "web_results": results or []}
    except Exception as e:
        logger.error(f"Web search error: {str(e)} - Query: {state['raw_query']}")
        return {**state, "web_results": []}

def medical_policy_node(state: GraphState) -> GraphState:
    logger.debug("Evaluating medical policy for query: %s", state["raw_query"])
    try:
        if not state["parsed_query"]:
            return {**state, "medical_decision": {"decision": "rejected", "reason": [{"clause_text": "No parsed data", "source": "system"}]}}
        claim_data = {
            "amount": state["parsed_query"].get("amount", 0),
            "type": "surgery",
            "condition": state["raw_query"].lower(),
            "pre_existing": state["parsed_query"].get("pre_existing", False),
            "web_info": state["web_results"]
        }
        decision = medical_agent.process_claim(json.dumps(claim_data))
        if isinstance(decision, str):
            try:
                decision = json.loads(decision)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse medical_decision string: {decision} - Query: {state['raw_query']}")
                decision = {"decision": "rejected", "reason": [{"clause_text": "Invalid medical decision format", "source": "system"}]}
        elif not isinstance(decision.get("reason"), list):
            decision["reason"] = [{"clause_text": decision.get("reason", "No reason"), "source": "system"}]
        logger.debug(f"Medical decision: {decision}")
        return {**state, "medical_decision": decision}
    except Exception as e:
        logger.error(f"Medical policy error: {str(e)} - Query: {state['raw_query']}")
        return {**state, "medical_decision": {"decision": "rejected", "reason": [{"clause_text": str(e), "source": "system"}]}}

def decision_node(state: GraphState) -> GraphState:
    logger.debug("Making final decision for query: %s", state["raw_query"])
    try:
        parsed_query = state["parsed_query"] if isinstance(state["parsed_query"], dict) else {}
        retrieved_chunks = state["retrieved_chunks"] if isinstance(state["retrieved_chunks"], list) else []
        web_results = state["web_results"] if isinstance(state["web_results"], list) else []
        logger.debug(f"Inputs to decide_claim - parsed_query: {parsed_query}, retrieved_chunks: {retrieved_chunks[:1] if retrieved_chunks else []}, web_results: {web_results[:1] if web_results else []}")

        decision = decide_claim(parsed_query, retrieved_chunks, web_results)
        # Normalize justification to list of dictionaries
        if isinstance(decision.get("justification"), str):
            decision["justification"] = [{"clause_text": decision["justification"], "source": "system"}]
        elif isinstance(decision.get("justification"), list):
            if all(isinstance(j, str) for j in decision["justification"]):
                decision["justification"] = [{"clause_text": j, "source": "system"} for j in decision["justification"]]
        elif decision.get("justification") is None:
            decision["justification"] = [{"clause_text": "No justification provided", "source": "system"}]

        # Basic amount calculation based on procedure and policy duration
        amount = 0
        if decision["decision"] == "approved":
            policy_duration = parsed_query.get("policy_duration_months", 0)
            procedure = parsed_query.get("procedure", "Unknown").lower()
            base_amounts = {
                "knee surgery": 5000,
                "heart bypass surgery": 20000,
                "appendectomy": 3000,
                "cataract surgery": 2000
            }
            amount = base_amounts.get(procedure, 1000) * min(policy_duration / 12, 1)  # Scale by policy duration (max 100% of base)
            amount = round(amount, 2)

        decision["amount"] = amount if amount > 0 else 0
        return {**state, "final_decision": decision}
    except Exception as e:
        logger.error(f"Decision error: {str(e)} - Query: {state['raw_query']} with inputs - parsed_query: {state['parsed_query']}")
        return {**state, "final_decision": {"decision": "rejected", "amount": 0, "justification": [{"clause_text": str(e), "source": "system"}]}}

def explain_node(state: GraphState) -> GraphState:
    logger.debug("Generating explanation for query: %s", state["raw_query"])
    try:
        # Simulate think mode with a delay (optional, controlled by UI)
        if getattr(state, "think_mode", False):
            time.sleep(2)  # 2-second delay to mimic thinking
        explanation = explain_decision(state["parsed_query"], state["final_decision"])
        final_response = {
            "query": state["raw_query"],
            "parsed_query": state["parsed_query"],
            "decision": state["final_decision"].get("decision", "rejected"),
            "amount": state["final_decision"].get("amount", 0),
            "justifications": state["final_decision"].get("justification", [{"clause_text": "No justification", "source": "system"}]),
            "explanation": explanation
        }
        return {**state, "explanation": explanation, "final_response": final_response}
    except Exception as e:
        logger.error(f"Explain error: {str(e)} - Query: {state['raw_query']}")
        final_response = {
            "query": state["raw_query"],
            "parsed_query": state["parsed_query"],
            "decision": "rejected",
            "amount": 0,
            "justifications": [{"clause_text": str(e), "source": "system"}],
            "explanation": f"Failed to process: {str(e)}"
        }
        return {**state, "explanation": "", "final_response": final_response}

def store_node(state: GraphState) -> GraphState:
    logger.debug("Storing user data in Pinecone for query: %s", state["raw_query"])
    try:
        final_response = state["final_response"]
        medical_decision = state["medical_decision"]
        
        # Generate embeddings
        query_embedding = generate_embedding(final_response["query"])
        explanation_embedding = generate_embedding(final_response["explanation"])
        
        # Prepare metadata with additional fields
        metadata = {
            "query": final_response["query"],
            "parsed_query": json.dumps(final_response["parsed_query"]),
            "decision": final_response["decision"],
            "amount": final_response["amount"],
            "justifications": json.dumps(final_response["justifications"]),
            "explanation": final_response["explanation"],
            "medical_decision": json.dumps(medical_decision),
            "timestamp": datetime.now().isoformat()
        }
        
        # Upsert to Pinecone
        vectors = [
            (str(uuid.uuid4()), query_embedding, metadata),  # Unique ID for query
            (str(uuid.uuid4()), explanation_embedding, metadata)  # Unique ID for explanation
        ]
        index.upsert(vectors=vectors)
        logger.info("User data stored in Pinecone successfully")
    except Exception as e:
        logger.error(f"Vector storage error: {str(e)} - Query: {state['raw_query']}")
    return state

# Build the graph
graph = StateGraph(GraphState)
graph.add_node("parse", parse_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("web_search", web_search_node)
graph.add_node("medical_policy", medical_policy_node)
graph.add_node("decision", decision_node)
graph.add_node("explain", explain_node)
graph.add_node("store", store_node)

# Define flow
graph.set_entry_point("parse")
graph.add_edge("parse", "retrieve")
graph.add_edge("retrieve", "web_search")
graph.add_edge("web_search", "medical_policy")
graph.add_edge("medical_policy", "decision")
graph.add_edge("decision", "explain")
graph.add_edge("explain", "store")
graph.add_edge("store", END)

# Compile the graph
try:
    app = graph.compile()
    logger.info("Graph compiled successfully")
except Exception as e:
    logger.error(f"Graph compilation error: {str(e)}")
    sys.exit(1)

def run_pipeline(query: str, think_mode: bool = False) -> Dict:
    logger.info(f"Processing query: {query} with think_mode: {think_mode}")
    try:
        result = app.invoke({"raw_query": query, "think_mode": think_mode})
        return result.get("final_response", {})
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)} - Query: {query}")
        return {
            "query": query,
            "parsed_query": {},
            "decision": "rejected",
            "amount": 0,
            "justifications": [{"clause_text": str(e), "source": "system"}],
            "explanation": f"Failed to process: {str(e)}"
        }

if __name__ == "__main__":
    query = "60M, heart bypass surgery, Delhi, 12-month policy, disputed pre-existing hypertension"
    result = run_pipeline(query)
    print("\n✅ FINAL OUTPUT:\n")
    print(json.dumps(result, indent=2))