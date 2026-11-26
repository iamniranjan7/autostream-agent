"""
End-to-end test script for the AutoStream agent.
Simulates the expected conversation flow with retry logic for API rate limits.
"""

import os
import sys
import time
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Load env before imports that need the key
load_dotenv()

from src.agent import build_agent


def invoke_with_retry(agent, message, config, max_retries=5, delay=30):
    """Invoke agent with retry logic for transient API errors."""
    for attempt in range(max_retries):
        try:
            return agent.invoke(
                {"messages": [HumanMessage(content=message)]},
                config=config,
            )
        except Exception as e:
            err = str(e)
            if attempt < max_retries - 1 and ("503" in err or "429" in err or "UNAVAILABLE" in err or "RESOURCE_EXHAUSTED" in err):
                wait = delay * (attempt + 1)
                print(f"  ⏳ API rate limited, waiting {wait}s... (attempt {attempt + 2}/{max_retries})")
                time.sleep(wait)
            else:
                raise


def test_conversation():
    """Run the full conversation flow and validate each step."""
    print("=" * 60)
    print("🧪  AutoStream Agent — End-to-End Test")
    print("=" * 60)

    agent, config = build_agent()

    # Test messages simulating the expected flow
    test_messages = [
        ("Greeting + inquiry", "Hi, tell me about your pricing."),
        ("Feature question", "What features are included in the Pro plan?"),
        ("Policy question", "What's your refund policy?"),
        ("High intent", "That sounds good, I want to try the Pro plan for my YouTube channel."),
        ("Provide details", "My name is Niranjan and my email is niranjan@example.com"),
    ]

    for i, (label, msg) in enumerate(test_messages, 1):
        print(f"\n{'─' * 60}")
        print(f"Turn {i} — {label}")
        print(f"{'─' * 60}")
        print(f"You: {msg}")



        # Pause between turns to respect rate limits
        if i > 1:
            time.sleep(5)

        result = invoke_with_retry(agent, msg, config)



        ai_msg = result["messages"][-1]
        print(f"\nAgent: {ai_msg.content}")


        # Print state info
        intent = result.get("intent", "unknown")
        lead_info = result.get("lead_info", {})
        captured = result.get("lead_captured", False)
        print(f"\n  [State] intent={intent} | lead_info={lead_info} | captured={captured}")

    print(f"\n{'=' * 60}")
    print("🧪  Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_conversation()
