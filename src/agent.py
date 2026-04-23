"""
LangGraph agent definition for the AutoStream conversational agent.

Defines the state graph with nodes for intent classification, greeting,
RAG-powered product inquiry, and lead capture. Uses conditional routing
based on classified intent.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.state import ConversationState
from src.nodes.intent_classifier import classify_intent
from src.nodes.greeting import handle_greeting
from src.nodes.rag_responder import handle_product_inquiry
from src.nodes.lead_capture import handle_lead_capture


def _route_by_intent(state: ConversationState) -> str:
    """Route to the appropriate node based on classified intent.

    Args:
        state: Current conversation state with intent field set.

    Returns:
        Name of the next node to execute.
    """
    intent = state.get("intent", "product_inquiry")

    if intent == "greeting":
        return "greeting"
    elif intent == "high_intent_lead":
        return "lead_capture"
    else:
        return "rag_responder"


def build_agent():
    """Build and compile the AutoStream agent graph.

    Returns:
        A tuple of (compiled_graph, config) ready for invocation.
    """
    # Define the state graph
    graph = StateGraph(ConversationState)

    # Add nodes
    graph.add_node("intent_classifier", classify_intent)
    graph.add_node("greeting", handle_greeting)
    graph.add_node("rag_responder", handle_product_inquiry)
    graph.add_node("lead_capture", handle_lead_capture)

    # Set entry point
    graph.set_entry_point("intent_classifier")

    # Add conditional routing from intent classifier
    graph.add_conditional_edges(
        "intent_classifier",
        _route_by_intent,
        {
            "greeting": "greeting",
            "rag_responder": "rag_responder",
            "lead_capture": "lead_capture",
        },
    )

    # All response nodes lead to END
    graph.add_edge("greeting", END)
    graph.add_edge("rag_responder", END)
    graph.add_edge("lead_capture", END)

    # Compile with memory checkpointer for cross-turn persistence
    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)

    # Config with a thread ID for session persistence
    config = {"configurable": {"thread_id": "autostream-session-1"}}

    return compiled, config
