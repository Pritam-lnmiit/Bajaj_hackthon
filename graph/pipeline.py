import sys
import os
import json
import logging
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

# Fix Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

try:
    from agents.query_parser_agent import parse_user_query
    from agents.retriever_agent import retrieve_chunks
    from agents.decision_agent import decide_claim
    from agents.web_search_agent import search_policy_location
    from agents.explanation_agent import explain_decision
    from agents.chat_memory_agent import ChatMemoryAgent
except ImportError as e:
    print(f"Import error: {str(e)}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Shared chat memory instance
chat_memory = ChatMemoryAgent()

# Define LangGraph state
class GraphState(TypedDict):
    raw_query: str
    parsed_query: dict
    retrieved_chunks: List[Dict]
    web_results: List[Dict]
    decision: dict
    explanation: str
    final_response: dict

# LangGraph node functions
def parse_node_fn(state: GraphState) -> GraphState:
    logger.debug("Entering parse_node_fn with state: %s", json.dumps(state, indent=2))
    try:
        if not state.get("raw_query"):
            raise ValueError("raw_query is missing in state")
        parsed = parse_user_query(state["raw_query"])
        logger.info(f"Parsed query: {parsed}")
        return {**state, "parsed_query": parsed}
    except Exception as e:
        logger.error(f"Parse node error: {str(e)}")
        return {**state, "parsed_query": {"error": str(e)}}

def retrieve_node_fn(state: GraphState) -> GraphState:
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
    logger.debug("Entering decide_node_fn with state: %s", json.dumps(state, indent=2))
    try:
        if not state.get("parsed_query"):
            raise ValueError("parsed_query is missing in state")
        decision = decide_claim(
            parsed_query=state["parsed_query"],
            chunks=state.get("retrieved_chunks", []),
            web_results=state.get("web_results", [])
        )
        logger.info(f"Decision: {decision.get('decision')}")
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

def explain_node_fn(state: GraphState) -> GraphState:
    logger.debug("Entering explain_node_fn with state: %s", json.dumps(state, indent=2))
    try:
        if not state.get("parsed_query") or not state.get("decision"):
            raise ValueError("parsed_query or decision is missing in state")
        explanation = explain_decision(
            parsed_query=state["parsed_query"],
            decision=state["decision"]
        )
        final_response = {
            "query": state["raw_query"],
            "parsed_query": state["parsed_query"],
            "decision": state["decision"].get("decision", "rejected"),
            "amount": state["decision"].get("amount", 0),
            "justifications": state["decision"].get("matched_clauses", []),
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

# Build LangGraph
graph = StateGraph(GraphState)
graph.add_node("parse", RunnableLambda(parse_node_fn))
graph.add_node("retrieve", RunnableLambda(retrieve_node_fn))
graph.add_node("web_fallback", RunnableLambda(web_node_fn))
graph.add_node("decide", RunnableLambda(decide_node_fn))
graph.add_node("explain", RunnableLambda(explain_node_fn))

# Control flow
graph.set_entry_point("parse")
graph.add_edge("parse", "retrieve")

def needs_web_search(state: GraphState) -> str:
    logger.debug("Checking needs_web_search with state: %s", json.dumps(state, indent=2))
    chunks = state.get("retrieved_chunks", [])
    next_node = "web_fallback" if not chunks else "decide"
    logger.debug(f"Routing to {next_node}")
    return next_node

graph.add_conditional_edges("retrieve", needs_web_search, {"web_fallback": "web_fallback", "decide": "decide"})
graph.add_edge("web_fallback", "decide")
graph.add_edge("decide", "explain")
graph.add_edge("explain", END)

# Compile LangGraph app
try:
    app = graph.compile()
except Exception as e:
    logger.error(f"Graph compilation error: {str(e)}")
    sys.exit(1)

def run_pipeline(query: str) -> Dict:
    """Run the pipeline and return the structured response."""
    logger.info(f"Processing query: {query}")
    chat_memory.add_user_message(query)
    try:
        result = app.invoke({"raw_query": query})
        logger.debug("Pipeline result: %s", json.dumps(result, indent=2))
        chat_memory.add_ai_message(result.get("explanation", ""))
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