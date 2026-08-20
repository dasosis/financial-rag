"""
app.py — Streamlit chat interface for the Financial Report Q&A Engine.

Usage:
    streamlit run app.py

Requires the FastAPI server to be running:
    uvicorn api:app --reload --port 8000
"""

import os

import httpx
import streamlit as st

# Overridden in Docker, where the API is reachable by service name, not localhost.
API_URL = os.getenv("API_URL", "http://localhost:8000/query")

st.set_page_config(
    page_title="Financial Report Q&A",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Financial Report Q&A Engine")
st.caption("Ask questions about 10-K filings — powered by RAG + GPT-4o-mini")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Source citations", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f"**{src['source_file']}** — Page {src['page']}\n\n"
                        f"> {src['excerpt']}"
                    )

# Chat input
if question := st.chat_input("Ask a question about the financial reports …"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Call the API
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant passages and generating answer …"):
            try:
                response = httpx.post(
                    API_URL,
                    json={"question": question},
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()
                answer = data["answer"]
                sources = data.get("sources", [])
            except httpx.ConnectError:
                answer = (
                    "Could not connect to the API server. "
                    "Make sure `uvicorn api:app --reload --port 8000` is running."
                )
                sources = []
            except httpx.HTTPStatusError as exc:
                detail = exc.response.json().get("detail", str(exc))
                answer = f"API error ({exc.response.status_code}): {detail}"
                sources = []
            except Exception as exc:
                answer = f"Unexpected error: {exc}"
                sources = []

        st.markdown(answer)

        if sources:
            with st.expander("Source citations", expanded=True):
                for src in sources:
                    st.markdown(
                        f"**{src['source_file']}** — Page {src['page']}\n\n"
                        f"> {src['excerpt']}"
                    )

    # Persist assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
