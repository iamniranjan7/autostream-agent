"""
Intent classification node for the AutoStream agent.

Classifies the user's latest message into one of three intents:
- greeting: Casual hello, hi, how are you, etc.
- product_inquiry: Questions about pricing, features, plans, policies
- high_intent_lead: User wants to sign up, try, purchase, or start using the product
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.state import ConversationState


# System prompt for intent classification
INTENT_SYSTEM_PROMPT = """You are an intent classifier for AutoStream, a SaaS video editing platform.

Analyze the user's latest message in the context of the conversation history and classify the intent into EXACTLY ONE of these categories:

1. "greeting" — Casual greetings, small talk, hello, hi, how are you, etc.
2. "product_inquiry" — Questions about pricing, features, plans, capabilities, policies, refunds, support, or any product-related information.
3. "high_intent_lead" — The user expresses desire to sign up, try, purchase, subscribe, get started, or use AutoStream. They show clear buying intent or readiness to commit.

IMPORTANT RULES:
- If the user is providing their personal details (name, email, platform) during a lead capture flow, classify as "high_intent_lead" to continue the flow.
- If the lead has already been captured and the user asks a new question, classify based on the new message.
- When in doubt between "product_inquiry" and "high_intent_lead", lean toward "product_inquiry" unless the user explicitly expresses desire to sign up or try the product.

Respond with ONLY the intent label — one of: greeting, product_inquiry, high_intent_lead
No explanation, no punctuation, no extra text."""


def classify_intent(state: ConversationState) -> dict:
    """Classify the intent of the latest user message.

    Args:
        state: Current conversation state.

    Returns:
        Partial state update with the classified intent.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
    )

    # Build the classification prompt with conversation context
    messages = [
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
    ]

    # Include recent conversation history for context (last 6 messages)
    recent_messages = state["messages"][-6:]
    for msg in recent_messages:
        messages.append(msg)

    response = llm.invoke(messages)
    intent = response.content.strip().lower()

    # Validate the intent — fallback to product_inquiry if unrecognized
    valid_intents = {"greeting", "product_inquiry", "high_intent_lead"}
    if intent not in valid_intents:
        intent = "product_inquiry"

    return {"intent": intent}
