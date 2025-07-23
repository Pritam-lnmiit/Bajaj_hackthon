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
import time
from langdetect import detect
from deep_translator import GoogleTranslator

# Fix Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Import agents
try:
    from agents.query_parser_agent import parse_user_query
    from agents.retriever_agent import retrieve_chunks
    from agents.medical_policy_agent import MedicalPolicyAgent
    from agents.explanation_agent import explain_decision
except ImportError as e:
    print(f"Import error: {str(e)}")
    sys.exit(1)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UPSTASH_VECTOR_URL = os.getenv("UPSTASH_VECTOR_URL")
UPSTASH_VECTOR_TOKEN = os.getenv("UPSTASH_VECTOR_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "insurance-claims"
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Translator
translator = GoogleTranslator(source='auto', target='en')

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class GraphState(TypedDict):
    raw_query: str
    parsed_query: dict
    retrieved_chunks: List[Dict]
    medical_decision: dict
    final_decision: dict
    explanation: str
    final_response: dict
    attempt_count: int
    original_language: str

# Initialize MedicalPolicyAgent
policy_file = os.path.join(project_root, "agents", "local_policy.json")
if not os.path.exists(policy_file):
    logger.warning(f"Policy file {policy_file} not found. Using default rules.")
    policy_file = ""
medical_agent = MedicalPolicyAgent(policy_file)

def generate_embedding(text: str) -> list:
    try:
        response = openai_client.embeddings.create(input=text, model="text-embedding-ada-002")
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding generation error: {str(e)}")
        return []

def translate_query(query: str) -> tuple[str, str]:
    try:
        detected_lang = detect(query)
        logger.debug(f"Detected language: {detected_lang}")
        if detected_lang != "en":
            translated = translator.translate(query)
            return translated, detected_lang
        return query, "en"
    except Exception as e:
        logger.error(f"Translation error: {str(e)} - Query: {query}")
        return query, "en"

# Node functions
def parse_node(state: GraphState) -> GraphState:
    logger.debug("Parsing query: %s", state["raw_query"])
    try:
        translated_query, original_language = translate_query(state["raw_query"])
        logger.debug(f"Translated query: {translated_query}, Original language: {original_language}")
        parsed = parse_user_query(translated_query)
        return {"parsed_query": parsed or {}, "attempt_count": 0, "original_language": original_language}
    except Exception as e:
        logger.error(f"Parse error: {str(e)} - Query: {state['raw_query']}")
        return {"parsed_query": {}, "attempt_count": 0, "original_language": "en"}

def retrieve_node(state: GraphState) -> GraphState:
    logger.debug("Retrieving chunks for query: %s", state["raw_query"])
    try:
        chunks = retrieve_chunks(state["raw_query"], k=5)
        logger.debug(f"Retrieved chunks: {chunks}")
        return {"retrieved_chunks": chunks or []}
    except Exception as e:
        logger.error(f"Retrieve error: {str(e)} - Query: {state['raw_query']}")
        return {"retrieved_chunks": []}

def medical_policy_node(state: GraphState) -> GraphState:
    logger.debug("Evaluating medical policy for query: %s", state["raw_query"])
    try:
        if not state["parsed_query"]:
            return {"medical_decision": {"decision": "rejected", "reason": [{"clause_text": "No parsed data", "source": "system"}], "amount": 0}}
        claim_data = {
            "amount": state["parsed_query"].get("amount", 0),
            "type": "surgery",
            "condition": state["raw_query"].lower(),
            "pre_existing": state["parsed_query"].get("pre_existing", False),
            "planned_treatment": "planned" in state["raw_query"].lower(),
            "submitted_days": 0,
            "pre_hosp_days": 0,
            "post_hosp_days": 0,
            "pre_authorized": "pre-authorization" in state["raw_query"].lower(),
        }
        decision = medical_agent.process_claim(json.dumps(claim_data))
        if isinstance(decision, str):
            try:
                decision = json.loads(decision)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse medical_decision string: {decision} - Query: {state['raw_query']}")
                decision = {"decision": "rejected", "reason": [{"clause_text": "Invalid medical decision format", "source": "system"}], "amount": 0}
        if not decision.get("reason"):
            decision["reason"] = [{"clause_text": "No specific reason provided by medical policy", "source": "system"}]
        logger.debug(f"Medical decision: {decision}")
        return {"medical_decision": decision}
    except Exception as e:
        logger.error(f"Medical policy error: {str(e)} - Query: {state['raw_query']}")
        return {"medical_decision": {"decision": "rejected", "reason": [{"clause_text": str(e), "source": "system"}], "amount": 0}}

def decision_node(state: GraphState) -> GraphState:
    logger.debug("Making final decision for query: %s", state["raw_query"])
    try:
        import importlib
        decision_agent = importlib.import_module("agents.decision_agent")
        decide_claim = decision_agent.decide_claim

        medical_decision = state["medical_decision"]
        chunks = state.get("retrieved_chunks", [])
        parsed_query = state["parsed_query"]

        logger.debug(f"Chunks: {chunks}, Medical decision: {medical_decision}")

        decision_from_retriever = decide_claim(parsed_query, chunks, [], medical_decision)
        decision_from_medical = medical_decision

        retriever_decision = decision_from_retriever.get("decision", "rejected")
        medical_decision_value = decision_from_medical.get("decision", "rejected")

        final_decision = decision_from_medical
        if not chunks or retriever_decision == "rejected" and "Error" in decision_from_retriever.get("justification", ""):
            final_decision = decision_from_medical
            final_decision["justification"] = (
                decision_from_medical.get("reason", [{"clause_text": "No medical reason provided", "source": "system"}])[0]["clause_text"] +
                ". Using medical policy due to lack of retriever result."
            )
        elif retriever_decision == medical_decision_value:
            final_decision = decision_from_retriever
        else:
            final_decision = decision_from_retriever
            final_decision["justification"] = (
                decision_from_retriever.get("justification", "") +
                ". Overriding medical policy due to retriever priority."
            )

        final_decision["matched_clauses"] = (
            decision_from_retriever.get("matched_clauses", []) +
            (decision_from_medical.get("reason", []) if decision_from_medical.get("reason") else [])
        )

        final_decision["amount"] = decision_from_retriever.get("amount", decision_from_medical.get("amount", 0))
        return {"final_decision": final_decision}
    except Exception as e:
        logger.error(f"Decision error: {str(e)} - Query: {state['raw_query']} with inputs - parsed_query: {state['parsed_query']}")
        return {"final_decision": {"decision": "rejected", "amount": 0, "justification": [{"clause_text": str(e), "source": "system"}]}}

def explain_node(state: GraphState) -> GraphState:
    logger.debug("Generating explanation for query: %s", state["raw_query"])
    try:
        if getattr(state, "think_mode", False):
            time.sleep(2)
        explanation = explain_decision(state["parsed_query"], state["final_decision"])
        original_language = state.get("original_language", "en")
        if original_language != "en" and explanation:
            translated_explanation = GoogleTranslator(source='en', target=original_language).translate(explanation)
        else:
            translated_explanation = explanation
        final_response = {
            "query": state["raw_query"],
            "parsed_query": state["parsed_query"],
            "decision": state["final_decision"].get("decision", "rejected"),
            "amount": state["final_decision"].get("amount", 0),
            "justifications": state["final_decision"].get("justification", [{"clause_text": "No justification", "source": "system"}]),
            "explanation": translated_explanation,
            "matched_clauses": state["final_decision"].get("matched_clauses", [])
        }
        return {"explanation": translated_explanation, "final_response": final_response}
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
        return {"explanation": "", "final_response": final_response}

def store_node(state: GraphState) -> GraphState:
    logger.debug("Storing user data in Pinecone for query: %s", state["raw_query"])
    try:
        final_response = state["final_response"]
        medical_decision = state["medical_decision"]
        
        query_embedding = generate_embedding(final_response["query"])
        explanation_embedding = generate_embedding(final_response["explanation"])
        
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
        
        vectors = [
            (str(uuid.uuid4()), query_embedding, metadata),
            (str(uuid.uuid4()), explanation_embedding, metadata)
        ]
        index.upsert(vectors=vectors)
        logger.info("User data stored in Pinecone successfully")
    except Exception as e:
        logger.error(f"Vector storage error: {str(e)} - Query: {state['raw_query']}")
    return {}

# Build the graph
graph = StateGraph(GraphState)
graph.add_node("parse", parse_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("medical_policy", medical_policy_node)
graph.add_node("decision", decision_node)
graph.add_node("explain", explain_node)
graph.add_node("store", store_node)

graph.set_entry_point("parse")
graph.add_edge("parse", "retrieve")
graph.add_edge("retrieve", "medical_policy")
graph.add_edge("medical_policy", "decision")
graph.add_edge("decision", "explain")
graph.add_edge("explain", "store")
graph.add_edge("store", END)

try:
    app = graph.compile()
    logger.info("Graph compiled successfully at 02:47 AM IST, July 23, 2025")
except Exception as e:
    logger.error(f"Graph compilation error: {str(e)}")
    sys.exit(1)

def run_pipeline(query: str, think_mode: bool = False) -> Dict:
    logger.info(f"Processing query: {query} with think_mode: {think_mode} at 02:47 AM IST, July 23, 2025")
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
    query = "I used an air ambulance to shift my father from a hospital in Mumbai to another hospital in Pune for better facilities. Why was my claim rejected?"
    result = run_pipeline(query)
    print("\n✅ FINAL OUTPUT:\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))