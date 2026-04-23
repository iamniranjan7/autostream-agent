"""
RAG-powered response node for the AutoStream agent.

Retrieves relevant product information from the FAISS vector store
and generates a grounded response to the user's product or pricing query.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

from src.state import ConversationState
from src.rag.pipeline import get_retriever


RAG_SYSTEM_PROMPT = """You are a helpful customer support agent for AutoStream, a SaaS platform providing automated video editing tools for content creators.

Answer the user's question using ONLY the context provided below. Be accurate, concise, and helpful.

RULES:
- Use the retrieved context to answer questions about pricing, features, plans, and policies
- If the context doesn't contain the answer, say you don't have that information and suggest contacting support
- Format pricing info clearly (use bullet points if listing multiple plans)
- Be conversational and friendly, not robotic
- Keep responses concise — aim for 2-4 sentences unless the question requires more detail
- If the user seems interested, you can mention they can get started anytime

RETRIEVED CONTEXT:
{context}"""


def handle_product_inquiry(state: ConversationState) -> dict:
    """Retrieve relevant context and generate a product/pricing response.

    Args:
        state: Current conversation state.

    Returns:
        Partial state update with the RAG-grounded AI message.
    """
    # Get the latest user message for retrieval query
    latest_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) or (hasattr(msg, "type") and msg.type == "human"):
            latest_message = msg.content
            break

    if not latest_message:
        return {"messages": [AIMessage(content="I'm sorry, could you repeat your question?")]}

    # Retrieve relevant chunks from the knowledge base
    retriever = get_retriever(k=4)
    docs = retriever.invoke(latest_message)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Build the prompt
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
    )

    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT.format(context=context)),
    ]

    # Include recent conversation history
    recent = state["messages"][-6:]
    for msg in recent:
        messages.append(msg)

    response = llm.invoke(messages)

    return {"messages": [AIMessage(content=response.content)]}
