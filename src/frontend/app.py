"""
Omni-Help Streamlit Chat UI.

A proof-of-concept enterprise chat frontend that talks to the FastAPI backend.
Demonstrates full multi-turn conversation with live routing metadata per message.

Run (with the FastAPI server already running on port 8000):
    streamlit run src/frontend/app.py
"""

import os
import uuid
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# API_BASE_URL is injected as an env var in Docker (set to http://backend:8000
# by docker-compose so the frontend container can reach the backend container
# by its service name). Falls back to localhost for local dev without Docker.
_API_BASE = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
API_URL = f"{_API_BASE}/api/v1/chat"
API_TIMEOUT_SECONDS = 30

# Maps intent values to display labels and emoji indicators
INTENT_DISPLAY = {
    "policy":       ("📋 Policy", "#4A90D9"),
    "sql":          ("🗄️ Order DB", "#27AE60"),
    "web":          ("🌐 Web Search", "#8E44AD"),
    "product_info": ("📦 Product Info", "#E67E22"),
    "complaint":    ("🚨 Escalated", "#E74C3C"),
    "unknown":      ("❓ Unknown", "#95A5A6"),
}

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Omni-Help Enterprise AI",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Session Info")
    st.code(f"Session ID:\n{st.session_state.session_id[:18]}...", language=None)
    st.metric("Queries this session", st.session_state.total_queries)

    st.markdown("---")
    st.markdown("## 🗺️ Pipeline Guide")
    for intent, (label, color) in INTENT_DISPLAY.items():
        if intent == "unknown":
            continue
        st.markdown(f"{label}")

    st.markdown("---")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.total_queries = 0
        st.rerun()

    st.markdown("---")
    st.caption("Omni-Help v1.0 · Built with LangGraph + FastAPI + Streamlit")

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------

st.markdown("# 🤖 Omni-Help Enterprise AI")
st.markdown(
    "An intelligent support assistant powered by an Adaptive RAG Router. "
    "Ask about policies, order status, or general product questions."
)
st.divider()

# ---------------------------------------------------------------------------
# Render conversation history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]
            intent_key = meta.get("intent", "unknown")
            label, color = INTENT_DISPLAY.get(intent_key, INTENT_DISPLAY["unknown"])
            confidence_pct = meta.get("confidence", 0.0) * 100
            st.caption(
                f"Routed to: **{label}** &nbsp;·&nbsp; "
                f"Confidence: **{confidence_pct:.1f}%** &nbsp;·&nbsp; "
                f"ID: `{meta.get('correlation_id', '')[:8]}...`"
            )

# ---------------------------------------------------------------------------
# Chat input + API call
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask Omni-Help..."):

    # 1. Append and display the user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call the FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                api_response = requests.post(
                    API_URL,
                    json={
                        "query": prompt,
                        "conversation_id": st.session_state.session_id,
                    },
                    timeout=API_TIMEOUT_SECONDS,
                )
                api_response.raise_for_status()
                data = api_response.json()

                ai_text: str = data.get("response", "No response received.")
                intent: str = data.get("intent", "unknown")
                confidence: float = data.get("confidence", 0.0)
                correlation_id: str = data.get("correlation_id", "")

            except requests.exceptions.ConnectionError:
                ai_text = (
                    "⚠️ **Cannot connect to the Omni-Help backend.** "
                    "Please ensure the FastAPI server is running:\n\n"
                    "```bash\nuvicorn api.main:app --host 0.0.0.0 --port 8000 "
                    "--reload --app-dir src\n```"
                )
                intent, confidence, correlation_id = "unknown", 0.0, ""

            except requests.exceptions.Timeout:
                ai_text = "⚠️ **Request timed out.** The server took too long to respond. Please try again."
                intent, confidence, correlation_id = "unknown", 0.0, ""

            except Exception as exc:
                ai_text = f"⚠️ **Unexpected error:** {exc}"
                intent, confidence, correlation_id = "unknown", 0.0, ""

        # 3. Render response + routing metadata badge
        st.markdown(ai_text)
        if intent != "unknown":
            label, color = INTENT_DISPLAY.get(intent, INTENT_DISPLAY["unknown"])
            confidence_pct = confidence * 100
            st.caption(
                f"Routed to: **{label}** &nbsp;·&nbsp; "
                f"Confidence: **{confidence_pct:.1f}%** &nbsp;·&nbsp; "
                f"ID: `{correlation_id[:8]}...`"
            )

    # 4. Persist to session history
    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_text,
        "metadata": {
            "intent": intent,
            "confidence": confidence,
            "correlation_id": correlation_id,
        },
    })
    st.session_state.total_queries += 1
