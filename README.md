# 🎬 AutoStream — Social-to-Lead Conversational AI Agent

A conversational AI agent for **AutoStream**, a SaaS platform providing automated video editing tools for content creators. Built with **LangGraph** and **Gemini 2.5 Flash**, this agent identifies user intent, answers product questions using RAG, and captures qualified leads through natural conversation.

---

## 🚀 How to Run Locally

### Prerequisites

- Python 3.9+
- A Google AI API key ([get one free here](https://aistudio.google.com/apikey))

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/autostream-agent.git
cd autostream-agent

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key
cp .env.example .env
# Edit .env and paste your GOOGLE_API_KEY

# 5. Run the agent
python -m src.main
```

### Sample Conversation

```
You: Hi, tell me about your pricing.
Agent: [Responds with pricing details from knowledge base]

You: That sounds good, I want to try the Pro plan for my YouTube channel.
Agent: [Detects high intent, asks for name and email]

You: I'm John, email is john@example.com
Agent: [Captures lead and confirms]
```

---

## 🏗️ Architecture Explanation (~200 words)

### Why LangGraph?

I chose **LangGraph** over AutoGen because the AutoStream agent follows a clear sequential workflow — classify intent, then route to the appropriate handler — which maps naturally to a **directed state graph**. AutoGen's multi-agent orchestration would be overkill for a single-agent conversational flow.

LangGraph's `StateGraph` lets me define each capability (greeting, RAG retrieval, lead capture) as an independent node with **conditional routing** between them based on classified intent. This makes the agent's behavior explicit, testable, and easy to extend.

### State Management

State is managed via a `ConversationState` TypedDict that flows through every node in the graph. It tracks:

- **messages**: Full conversation history using LangGraph's `add_messages` reducer, which automatically appends new messages without overwriting
- **intent**: The classified intent of the current turn
- **lead_info**: A dictionary tracking collected lead fields (name, email, platform)
- **lead_captured**: A boolean flag preventing duplicate tool calls

Cross-turn persistence is handled by LangGraph's **MemorySaver** checkpointer, which stores state between invocations using a `thread_id`. This ensures the agent retains context across 5–6+ conversation turns without any external database.

### RAG Pipeline

The knowledge base is stored as a Markdown file, chunked with `RecursiveCharacterTextSplitter`, embedded using Google's `embedding-001` model, and indexed in a **FAISS** vector store for fast similarity search.

---

## 📱 WhatsApp Deployment via Webhooks

To integrate this agent with WhatsApp, I would use the **Meta Cloud API** (or **Twilio WhatsApp Business API**) with a webhook-based architecture:

### Architecture

1. **Webhook Endpoint**: Deploy a FastAPI/Flask server with a POST endpoint (e.g., `/webhook`) that receives incoming WhatsApp messages from Meta's Cloud API.

2. **Message Reception**: When a user sends a WhatsApp message, Meta sends a POST request to our webhook containing the sender's phone number, message text, and metadata.

3. **Session Management**: Use the sender's phone number as the `thread_id` for LangGraph's MemorySaver. This maps each WhatsApp user to a persistent conversation session, maintaining context across turns.

4. **Agent Processing**: The webhook handler extracts the message text, passes it to the LangGraph agent with the phone-number-based thread ID, and receives the agent's response.

5. **Response Delivery**: Send the agent's response back to the user via Meta's Cloud API `messages` endpoint using an HTTP POST request with the recipient's phone number.

### Implementation Steps

```
WhatsApp User → Meta Cloud API → Webhook (FastAPI) → LangGraph Agent
                                                          ↓
WhatsApp User ← Meta Cloud API ← Webhook Response ← Agent Response
```

1. Register a WhatsApp Business Account and configure the webhook URL
2. Verify the webhook with Meta's challenge-response handshake
3. Deploy the FastAPI server on a cloud platform (e.g., Railway, AWS Lambda, GCP Cloud Run)
4. Map phone numbers to LangGraph thread IDs for session persistence
5. Handle message queuing for high-traffic scenarios using Redis or a task queue

This architecture keeps the core agent logic unchanged — only the I/O layer switches from CLI to HTTP webhooks.

---

## 📁 Project Structure

```
autostream-agent/
├── README.md
├── requirements.txt
├── .env.example
├── knowledge_base/
│   └── autostream_kb.md          # Product knowledge base for RAG
├── src/
│   ├── __init__.py
│   ├── main.py                   # CLI entry point
│   ├── agent.py                  # LangGraph StateGraph definition
│   ├── state.py                  # ConversationState TypedDict
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── intent_classifier.py  # Intent classification via LLM
│   │   ├── greeting.py           # Greeting handler
│   │   ├── rag_responder.py      # RAG-powered product Q&A
│   │   └── lead_capture.py       # Lead qualification & tool call
│   ├── tools/
│   │   ├── __init__.py
│   │   └── lead_tool.py          # mock_lead_capture function
│   └── rag/
│       ├── __init__.py
│       └── pipeline.py           # FAISS vector store & retriever
└── demo/                         # Demo video assets
```

---

## 📊 Evaluation Criteria Coverage

| Criteria | Implementation |
|---|---|
| Intent Detection | LLM-based classification into 3 categories with conversation context |
| RAG | FAISS vector store + Google embeddings over Markdown knowledge base |
| State Management | LangGraph ConversationState + MemorySaver checkpointer |
| Tool Calling | mock_lead_capture called only after all 3 fields collected |
| Code Structure | Modular architecture with separate nodes, tools, and RAG packages |
| Deployability | Clean separation of I/O (CLI) from agent logic enables easy webhook integration |
