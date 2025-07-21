import sys
import os
import json
import logging
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
import datetime

# Fix Python path to include project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Import agents
try:
    from agents.query_parser_agent import parse_user_query
    from agents.retriever_agent import retrieve_chunks
    from agents.decision_agent import decide_claim
    from agents.web_search_agent import search_policy_location
    from agents.explanation_agent import explain_decision
    from agents.medical_policy_agent import MedicalPolicyAgent
except ImportError as e:
    print(f"Import error: {str(e)}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define the state structure for the workflow
class GraphState(TypedDict):
    """State object to track data through the LangGraph workflow."""
    raw_query: str  # Original user query
    parsed_query: dict  # Structured query data
    retrieved_chunks: List[Dict]  # Retrieved policy chunks
    web_results: List[Dict]  # Web search results
    decision: dict  # Initial decision from decision_agent
    medical_decision: dict  # Decision from MedicalPolicyAgent
    explanation: str  # Human-readable explanation
    final_response: dict  # Final structured response
    retry_count: int  # Track number of retries

# Initialize MedicalPolicyAgent with local policy data
policy_file = os.path.join(project_root, "agents", "local_policy.json")
if not os.path.exists(policy_file):
    logger.warning(f"Policy file {policy_file} not found. Using default rules.")
    policy_file = ""  # Pass empty string to avoid NoneType error
medical_agent = MedicalPolicyAgent(policy_file)

# Node functions for the workflow

def parse_node_fn(state: GraphState) -> GraphState:
    """Parse the raw query into a structured format."""
    logger.debug("Entering parse_node_fn with state: %s", json.dumps(state, indent=2))
    try:
        if not state.get("raw_query"):
            raise ValueError("raw_query is missing in state")
        parsed = parse_user_query(state["raw_query"])
        logger.info(f"Parsed query: {parsed}")
        return {**state, "parsed_query": parsed, "retry_count": state.get("retry_count", 0)}
    except Exception as e:
        logger.error(f"Parse node error: {str(e)}")
        return {**state, "parsed_query": {"error": str(e)}, "retry_count": state.get("retry_count", 0)}

def retrieve_node_fn(state: GraphState) -> GraphState:
    """Retrieve relevant policy chunks based on the query."""
    logger.debug("Entering retrieve_node_fn with state: %s", json.dumps(state, indent=2))
    try:
        if not state.get("raw_query"):
            raise ValueError("raw_query is missing in state")
        chunks = retrieve_chunks(state["raw_query"])
        if not chunks:
            logger.warning("No chunks retrieved from Upstash")
        else:
            logger.info(f"Retrieved {len(chunks)} chunks: {[c['text'][:50] for c in chunks]}")
        return {**state, "retrieved_chunks": chunks}
    except Exception as e:
        logger.error(f"Retrieve node error: {str(e)}")
        return {**state, "retrieved_chunks": []}

def web_node_fn(state: GraphState) -> GraphState:
    """Perform web search as a fallback if no chunks are retrieved."""
    logger.debug("Entering web_node_fn with state: %s", json.dumps(state, indent=2))
    try:
        if not state.get("raw_query"):
            raise ValueError("raw_query is missing in state")
        results = search_policy_location(
            query=state["raw_query"],
            location=state.get("parsed_query", {}).get("location", "")
        )
        if not results:
            logger.warning("No web results from SERPAPI")
        else:
            logger.info(f"Web search returned {len(results)} results: {[r['title'][:50] for r in results]}")
        return {**state, "web_results": results}
    except Exception as e:
        logger.error(f"Web search node error: {str(e)}")
        return {**state, "web_results": []}

def decide_node_fn(state: GraphState) -> GraphState:
    """Decide the claim outcome using the general decision agent."""
    logger.debug("Entering decide_node_fn with state: %s", json.dumps(state, indent=2))
    try:
        if not state.get("parsed_query"):
            raise ValueError("parsed_query is missing in state")
        decision = decide_claim(
            parsed_query=state["parsed_query"],
            chunks=state.get("retrieved_chunks", []),
            web_results=state.get("web_results", [])
        )
        logger.info(f"Initial decision: {decision.get('decision')}")
        return {**state, "decision": decision}
    except Exception as e:
        logger.error(f"Decision node error: {str(e)}")
        return {
            **state,
            "decision": {
                "decision": "rejected",
                "amount": 0,
                "justification": f"Decision failed: {str(e)}",
                "matched_clauses": []
            }
        }

def medical_check_node_fn(state: GraphState) -> GraphState:
    """Validate the decision with MedicalPolicyAgent."""
    logger.debug("Entering medical_check_node_fn with state: %s", json.dumps(state, indent=2))
    try:
        if not state.get("parsed_query") or not state.get("decision"):
            raise ValueError("parsed_query or decision is missing in state")
        
        # Detect if it's a medical claim
        is_medical = any(keyword in state["raw_query"].lower() for keyword in ["surgery", "hospital", "medical"])
        if is_medical and "amount" in state["parsed_query"]:
            claim_data = {
                "amount": state["parsed_query"]["amount"],
                "type": "hospitalization",  # Default; refine based on parsed_query
                "condition": state["raw_query"].lower(),
                "pre_existing": state["parsed_query"].get("pre_existing", False),
                "planned_treatment": "surgery" in state["raw_query"].lower(),
                "submitted_days": state["parsed_query"].get("submitted_days", 15),
                "pre_hosp_days": state["parsed_query"].get("pre_hosp_days", 30),
                "post_hosp_days": state["parsed_query"].get("post_hosp_days", 45),
                "pre_authorized": state["parsed_query"].get("pre_authorized", True)
            }
            medical_decision = medical_agent.process_claim(json.dumps(claim_data))
            logger.info(f"Medical decision: {medical_decision.get('decision')}")
            return {**state, "medical_decision": medical_decision, "retry_count": state["retry_count"]}
        else:
            # Non-medical claims pass through
            return {**state, "medical_decision": state["decision"], "retry_count": state["retry_count"]}
    except Exception as e:
        logger.error(f"Medical check node error: {str(e)}")
        return {
            **state,
            "medical_decision": {
                "decision": "rejected",
                "reason": [f"Medical check failed: {str(e)}"],
                "details": {}
            },
            "retry_count": state["retry_count"]
        }

def explain_node_fn(state: GraphState) -> GraphState:
    """Generate an explanation and final response based on the medical decision."""
    logger.debug("Entering explain_node_fn with state: %s", json.dumps(state, indent=2))
    try:
        if not state.get("parsed_query") or not state.get("medical_decision"):
            raise ValueError("parsed_query or medical_decision is missing in state")
        explanation = explain_decision(
            parsed_query=state["parsed_query"],
            decision=state["medical_decision"]
        )
        final_response = {
            "query": state["raw_query"],
            "parsed_query": state["parsed_query"],
            "decision": state["medical_decision"].get("decision", "rejected"),
            "amount": state["medical_decision"].get("amount", 0),
            "justifications": state["medical_decision"].get("matched_clauses", state["medical_decision"].get("reason", [])),
            "explanation": explanation
        }
        logger.info("Final response generated: %s", json.dumps(final_response, indent=2))
        return {**state, "explanation": explanation, "final_response": final_response}
    except Exception as e:
        logger.error(f"Explain node error: {str(e)}")
        final_response = {
            "query": state["raw_query"],
            "parsed_query": state.get("parsed_query", {}),
            "decision": "rejected",
            "amount": 0,
            "justifications": [{"clause_text": f"Error: {str(e)}", "source": "system"}],
            "explanation": f"Failed to generate explanation: {str(e)}"
        }
        logger.info("Fallback final response: %s", json.dumps(final_response, indent=2))
        return {**state, "explanation": "", "final_response": final_response}

# Build the LangGraph workflow
graph = StateGraph(GraphState)

# Add nodes to the graph
graph.add_node("parse", RunnableLambda(parse_node_fn))
graph.add_node("retrieve", RunnableLambda(retrieve_node_fn))
graph.add_node("web_fallback", RunnableLambda(web_node_fn))
graph.add_node("decide", RunnableLambda(decide_node_fn))
graph.add_node("medical_check", RunnableLambda(medical_check_node_fn))
graph.add_node("explain", RunnableLambda(explain_node_fn))

# Define the workflow control flow
graph.set_entry_point("parse")
graph.add_edge("parse", "retrieve")

def needs_web_search(state: GraphState) -> str:
    """Determine if web search is needed based on retrieved chunks."""
    logger.debug("Checking needs_web_search with state: %s", json.dumps(state, indent=2))
    chunks = state.get("retrieved_chunks", [])
    next_node = "web_fallback" if not chunks else "decide"
    logger.debug(f"Routing to {next_node}")
    return next_node

graph.add_conditional_edges("retrieve", needs_web_search, {"web_fallback": "web_fallback", "decide": "decide"})
graph.add_edge("web_fallback", "decide")
graph.add_edge("decide", "medical_check")

def medical_approval_check(state: GraphState) -> str:
    """Check if the medical decision is approved to proceed to explain or retry."""
    logger.debug("Checking medical_approval_check with state: %s", json.dumps(state, indent=2))
    medical_decision = state.get("medical_decision", {})
    is_approved = medical_decision.get("decision", "rejected").lower() == "approved"
    retry_count = state.get("retry_count", 0)
    next_node = "explain" if is_approved or retry_count >= 2 else "retrieve"
    new_state = {**state, "retry_count": retry_count + 1}  # Increment retry_count here
    logger.debug(f"Routing to {next_node}, Retry count: {new_state['retry_count']}")
    return next_node

graph.add_conditional_edges("medical_check", medical_approval_check, {"explain": "explain", "retrieve": "retrieve"})
graph.add_edge("explain", END)

# Compile the graph into a runnable app
try:
    app = graph.compile()  # Removed recursion_limit
    logger.info("LangGraph workflow compiled successfully")
except Exception as e:
    logger.error(f"Graph compilation error: {str(e)}")
    sys.exit(1)

def run_pipeline(query: str) -> Dict:
    """Execute the LangGraph workflow and return the final response."""
    logger.info(f"Processing query: {query}")
    # Log query for business purposes
    with open("query_log.txt", "a") as log_file:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
        log_file.write(f"{timestamp} - Query: {query}\n")
    try:
        result = app.invoke({"raw_query": query})
        logger.debug("Pipeline result: %s", json.dumps(result, indent=2))
        final_response = result.get("final_response", {})
        if not final_response:
            logger.error("Final response is empty")
            final_response = {
                "query": query,
                "parsed_query": result.get("parsed_query", {}),
                "decision": "rejected",
                "amount": 0,
                "justifications": [{"clause_text": "Pipeline failed to generate final response", "source": "system"}],
                "explanation": "Failed to process query due to internal error"
            }
        logger.info("Returning final response: %s", json.dumps(final_response, indent=2))
        return final_response
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        final_response = {
            "query": query,
            "parsed_query": {},
            "decision": "rejected",
            "amount": 0,
            "justifications": [{"clause_text": f"Pipeline failed: {str(e)}", "source": "system"}],
            "explanation": f"Failed to process query: {str(e)}"
        }
        logger.info("Fallback final response: %s", json.dumps(final_response, indent=2))
        return final_response

# CLI test
if __name__ == "__main__":
    query = "46M, knee surgery in Pune, 3-month-old policy"
    result = run_pipeline(query)
    print("\n✅ FINAL OUTPUT:\n")
    print(json.dumps(result, indent=2))