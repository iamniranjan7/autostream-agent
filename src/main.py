"""
AutoStream Agent — Interactive CLI Entry Point

Run with: python -m src.main
"""

import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from src.agent import build_agent


def main():
    """Run the AutoStream conversational agent in an interactive CLI loop."""
    # Load environment variables
    load_dotenv()

    # Verify API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY not set.")
        print("   Create a .env file with: GOOGLE_API_KEY=your_key_here")
        print("   Get a free key at: https://aistudio.google.com/apikey")
        sys.exit(1)

    print("=" * 60)
    print("🎬  AutoStream AI Agent")
    print("   Automated Video Editing for Content Creators")
    print("=" * 60)
    print("Type your message below. Type 'quit' or 'exit' to end.\n")

    # Build the agent graph
    agent, config = build_agent()

    # Initial state
    initial_state = {
        "messages": [],
        "intent": "",
        "lead_info": {"name": None, "email": None, "platform": None},
        "lead_captured": False,
    }

    turn = 0
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye"):
            print("\nAgent: Goodbye! Thanks for your interest in AutoStream! 👋")
            break

        turn += 1

        # Invoke the agent graph
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        # Get the last AI message
        ai_message = result["messages"][-1]
        print(f"\nAgent: {ai_message.content}\n")


if __name__ == "__main__":
    main()
