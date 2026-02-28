import streamlit as st
from chatbot_rag_backend import (
    chatbot,
    retrieve_all_threads,
    ingest_pdf,
    thread_document_metadata,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import uuid
import os
from dotenv import load_dotenv

# ---------------------------------------------------------
# Load environment variables (.env file)
# ---------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------
# Set LangSmith project name
# Each run will be tracked under this project
# ---------------------------------------------------------
os.environ["LANGCHAIN_PROJECT"] = "Chatbot_rag_implementation"

############################################################
# ---------------- Utility Functions -----------------------
############################################################

# ---------------------------------------------------------
# Generate a new thread ID
# Always return STRING (important for consistency)
# ---------------------------------------------------------
def generate_thread_id():
    return str(uuid.uuid4())


# ---------------------------------------------------------
# Add thread to session list if not already present
# ---------------------------------------------------------
def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


# ---------------------------------------------------------
# Reset chat:
# 1. Create new thread
# 2. Add to thread list
# 3. Clear message history
# 4. Initialize document store for that thread
# ---------------------------------------------------------
def reset_chat():
    new_thread = generate_thread_id()
    st.session_state["thread_id"] = new_thread
    add_thread(new_thread)
    st.session_state["message_history"] = []
    st.session_state["ingested_docs"].setdefault(new_thread, {})


# ---------------------------------------------------------
# Load conversation messages from LangGraph state
# Fetches database-backed state using thread_id
# ---------------------------------------------------------
def load_conversation(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": str(thread_id)}}
    )

    # If no state or no messages exist yet, return empty list
    if not state or "messages" not in state.values:
        return []

    return state.values["messages"]


# ---------------------------------------------------------
# Generate sidebar preview for a thread
# Uses first HumanMessage as conversation title
# ---------------------------------------------------------
def get_thread_preview(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": str(thread_id)}}
    )

    if not state or "messages" not in state.values:
        return "New Chat"

    messages = state.values["messages"]

    # Use FIRST user message as preview (like ChatGPT)
    for msg in messages:
        if isinstance(msg, HumanMessage):
            preview = msg.content.strip()
            return preview[:40] + ("..." if len(preview) > 40 else "")

    return "New Chat"


############################################################
# ---------------- Session Setup ---------------------------
############################################################

# ---------------------------------------------------------
# Initialize session variables only once
# ---------------------------------------------------------

# Stores UI message history
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Stores current active thread ID
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

# Stores all existing threads (loaded from backend)
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

# Stores per-thread PDF ingestion metadata
if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

# Ensure current thread exists in thread list
add_thread(st.session_state["thread_id"])

# Current thread key (string)
thread_key = st.session_state["thread_id"]

# Get documents associated with this thread
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})

############################################################
# ---------------- Sidebar UI ------------------------------
############################################################

st.sidebar.title("LangGraph Chatbot")

# Show active thread ID
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

# ---------------------------------------------------------
# New Chat Button
# ---------------------------------------------------------
if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

# ---------------------------------------------------------
# Show currently indexed PDF for this thread
# ---------------------------------------------------------
if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

# ---------------------------------------------------------
# PDF Upload Section (Thread-specific)
# ---------------------------------------------------------
uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF for this chat", type=["pdf"]
)

if uploaded_pdf:
    # Prevent duplicate ingestion
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed.")
    else:
        # Show indexing progress
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )

            # Save metadata per thread
            thread_docs[uploaded_pdf.name] = summary

            status_box.update(
                label="✅ PDF indexed",
                state="complete",
                expanded=False,
            )

# ---------------------------------------------------------
# Past Conversations (Preview Titles)
# ---------------------------------------------------------
st.sidebar.subheader("Past Conversations")

for t_id in st.session_state["chat_threads"][::-1]:

    preview = get_thread_preview(t_id)

    # Each preview is clickable
    if st.sidebar.button(preview, key=f"thread-{t_id}"):

        # Switch active thread
        st.session_state["thread_id"] = t_id

        # Load conversation from backend
        messages = load_conversation(t_id)

        # Convert LangChain messages to UI-friendly format
        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append(
                {"role": role, "content": msg.content}
            )

        st.session_state["message_history"] = temp_messages
        st.session_state["ingested_docs"].setdefault(t_id, {})
        st.rerun()

############################################################
# ---------------- Main Chat UI ----------------------------
############################################################

st.title("Multi Utility Chatbot")

# ---------------------------------------------------------
# Render chat history
# ---------------------------------------------------------
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

# Chat input box
user_input = st.chat_input("Ask about your document or use tools")

# ---------------------------------------------------------
# LangGraph Configuration
# thread_id ensures conversation persistence
# ---------------------------------------------------------
CONFIG = {
    "configurable": {"thread_id": thread_key},
    "metadata": {"thread_id": thread_key},
    "run_name": "chat_turn",
}

if user_input:

    # Add user message to UI state
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.text(user_input)

    # -----------------------------------------------------
    # Assistant Streaming Block
    # -----------------------------------------------------
    with st.chat_message("assistant"):

        # Used to display tool usage status
        status_holder = {"box": None}

        def ai_only_stream():

            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):

                # If tool is triggered
                if isinstance(message_chunk, ToolMessage):

                    tool_name = getattr(message_chunk, "name", "tool")

                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …",
                            expanded=True,
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        # Stream assistant output token by token
        ai_message = st.write_stream(ai_only_stream())

        # Mark tool as finished if used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished",
                state="complete",
                expanded=False,
            )

    # Save assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    # -----------------------------------------------------
    # Show document metadata below response
    # -----------------------------------------------------
    doc_meta = thread_document_metadata(thread_key)

    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

# Visual separator
st.divider()