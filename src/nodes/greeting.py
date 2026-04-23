"""
Greeting response node for the AutoStream agent.

Handles casual greetings with a friendly, on-brand response
that naturally introduces AutoStream.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, AIMessage

from src.state import ConversationState


GREETING_SYSTEM_PROMPT = """You are a friendly customer support agent for AutoStream, a SaaS platform that provides automated video editing tools for content creators.

The user has sent a casual greeting. Respond warmly and briefly. Keep it to 1-2 sentences.
- Greet them back naturally
- Briefly mention you can help with questions about AutoStream's video editing tools, pricing, or getting started
- Be conversational, not robotic
- Do NOT dump product info unless asked"""


def handle_greeting(state: ConversationState) -> dict:
    """Generate a friendly greeting response.

    Args:
        state: Current conversation state.

    Returns:
        Partial state update with the greeting AI message.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
    )

    messages = [SystemMessage(content=GREETING_SYSTEM_PROMPT)]
    # Include recent history for natural continuation
    recent = state["messages"][-4:]
    for msg in recent:
        messages.append(msg)

    response = llm.invoke(messages)

    return {"messages": [AIMessage(content=response.content)]}
