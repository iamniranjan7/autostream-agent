"""
Lead capture node for the AutoStream agent.

Handles the high-intent lead flow:
1. Extracts any lead details (name, email, platform) from conversation
2. If any details are missing, asks the user for them
3. Once all details are collected, calls mock_lead_capture()
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

from src.state import ConversationState
from src.tools.lead_tool import mock_lead_capture


EXTRACTION_PROMPT = """Analyze the conversation and extract lead details the user has provided.

Extract these three fields:
1. name — The user's full name
2. email — The user's email address
3. platform — Creator platform (YouTube, Instagram, TikTok, etc.)

Only extract EXPLICITLY stated values. If not mentioned, output "MISSING".

Respond in EXACTLY this format:
name: <value or MISSING>
email: <value or MISSING>
platform: <value or MISSING>"""


LEAD_CAPTURE_PROMPT = """You are a friendly support agent for AutoStream collecting signup details.

Currently collected:
- Name: {name}
- Email: {email}
- Platform: {platform}

Ask for MISSING fields naturally in 1-3 sentences. Don't ask for already-filled fields. Be warm and encouraging."""


def _parse_extracted(text: str) -> dict:
    info = {"name": None, "email": None, "platform": None}
    for line in text.strip().split("\n"):
        line = line.strip()
        for key in info:
            if line.lower().startswith(f"{key}:"):
                val = line.split(":", 1)[1].strip()
                if val.upper() != "MISSING":
                    info[key] = val
    return info


def handle_lead_capture(state: ConversationState) -> dict:
    if state.get("lead_captured", False):
        return {
            "messages": [
                AIMessage(content="You're all set! 🎉 Your details have been captured. Anything else I can help with?")
            ]
        }

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

    # Extract details from conversation
    extraction_messages = [SystemMessage(content=EXTRACTION_PROMPT)]
    for msg in state["messages"]:
        extraction_messages.append(msg)

    extracted = _parse_extracted(llm.invoke(extraction_messages).content)

    # Merge with existing info
    current = state.get("lead_info", {"name": None, "email": None, "platform": None})
    merged = {
        "name": extracted["name"] or current.get("name"),
        "email": extracted["email"] or current.get("email"),
        "platform": extracted["platform"] or current.get("platform"),
    }

    missing = [k for k, v in merged.items() if v is None]

    if not missing:
        # All collected — call the tool
        mock_lead_capture(name=merged["name"], email=merged["email"], platform=merged["platform"])
        return {
            "messages": [
                AIMessage(
                    content=f"Awesome, you're all set! 🎉\n\n"
                    f"- **Name:** {merged['name']}\n"
                    f"- **Email:** {merged['email']}\n"
                    f"- **Platform:** {merged['platform']}\n\n"
                    f"Our team will reach out shortly. Welcome aboard! 🚀"
                )
            ],
            "lead_info": merged,
            "lead_captured": True,
        }

    # Ask for missing fields
    llm2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    display = {k: (v or "MISSING") for k, v in merged.items()}
    ask_msgs = [
        SystemMessage(content=LEAD_CAPTURE_PROMPT.format(**display)),
    ]
    for msg in state["messages"][-6:]:
        ask_msgs.append(msg)

    response = llm2.invoke(ask_msgs)
    return {"messages": [AIMessage(content=response.content)], "lead_info": merged}
