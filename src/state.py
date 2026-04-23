"""
Conversation state definition for the AutoStream agent.

This TypedDict defines the shared state that flows through every node
in the LangGraph state graph. It provides memory persistence across
conversation turns.
"""

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class ConversationState(TypedDict):
    """State schema for the AutoStream conversational agent.

    Attributes:
        messages: Full conversation history. Uses LangGraph's `add_messages`
                  reducer to automatically append new messages rather than
                  overwriting.
        intent: The classified intent of the latest user message.
                One of: "greeting", "product_inquiry", "high_intent_lead".
        lead_info: Dictionary holding collected lead details.
                   Keys: "name", "email", "platform" — each starts as None.
        lead_captured: Flag indicating whether mock_lead_capture has been
                       called. Prevents duplicate captures.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    intent: str
    lead_info: dict[str, Optional[str]]
    lead_captured: bool
