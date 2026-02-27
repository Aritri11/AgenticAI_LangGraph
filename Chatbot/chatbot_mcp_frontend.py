#It is recomended not to use streamlit when using MCP client server because streamlit is inherently synchronous

import streamlit as st
from chatbot_mcp_backend import chatbot, retrieve_all_threads, submit_async_task
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, ToolMessage
import uuid
import os
import queue
from dotenv import load_dotenv
load_dotenv()
#setting up project name inside the code (method 2)
os.environ['LANGCHAIN_PROJECT'] = 'Chatbot_modified'

######################### Utility functions ######################
# Generate a unique ID for each new chat thread
def generate_thread_id():
     thread_id=uuid.uuid4()
     return thread_id

# Reset the chat by:
# 1. Creating a new thread ID
# 2. Adding it to session thread list
# 3. Clearing message history in UI
def reset_chat():
    thread_id=generate_thread_id()
    st.session_state['thread_id']=thread_id
    add_thread(thread_id)
    st.session_state['message_history']=[]

# Add thread to session list if not already present
def add_thread (thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

# Load conversation messages from LangGraph state (database-backed)
def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get("messages", [])

# Create a short preview title for sidebar using first user message
def get_thread_preview(thread_id):
    state = chatbot.get_state(
        config={'configurable': {'thread_id': thread_id}}
    )
    # If no state or no messages exist yet
    if not state or 'messages' not in state.values:
        return "New Chat"

    messages = state.values['messages']

    # # Find first HumanMessage and use it as preview
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            preview = msg.content.strip()
            return preview[:40] + ("..." if len(preview) > 40 else "")

    return "New Chat"

###############################################################


####################### Session Setup #######################
# Initialize session state variables if not already present
#st.session_state -> dict
# Stores current conversation messages for UI rendering
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Stores currently active thread ID
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

# Stores all available threads (loaded from database)
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

# Ensure current thread is in thread list
add_thread(st.session_state['thread_id'])

#############################################################

######################### Sidebar UI #######################

st.sidebar.title('LangGraph Chatbot')
# Button to start new conversation
if st.sidebar.button('New Chat'):
    reset_chat()
st.sidebar.header('My Conversations')

# Display all saved threads in reverse order (latest first)
for thread_id in st.session_state['chat_threads'][::-1]:
    # Generate preview text for each conversation
    preview = get_thread_preview(thread_id)

    # Each preview acts as a clickable button
    if st.sidebar.button(preview, key=str(thread_id)):
        # Switch active thread
        st.session_state['thread_id'] = thread_id
        # Load messages from backend
        messages = load_conversation(thread_id)

        # Convert LangChain message objects into UI-friendly format
        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})
        # Replace current UI history with loaded conversation
        st.session_state['message_history'] = temp_messages

#############################################################

####################### Main UI #############################

#loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])
# Chat input box at bottom of UI
user_input=st.chat_input('Type here')

# LangGraph config: thread_id determines conversation context
# CONFIG = {'configurable': {'thread_id':st.session_state['thread_id'] }}

#To integrate the concept of LANGSMITH in this code, we need to replace the above config with a new config such that each time a new chat with a new thread id is initiated, a new project gets created in langsmith for easy and organised access
CONFIG = {'configurable': {'thread_id':st.session_state['thread_id'] },
          "metadata": {
              'thread_id': st.session_state['thread_id']
          },
          "run_name": "chat_turn",
          }

if user_input:
    #first add the message to the message_history
    st.session_state['message_history'].append({'role':'user','content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    #Initial implementations (Without streaming feature)
    # response=chatbot.invoke({'messages': [HumanMessage(content=user_input)]},config=CONFIG)
    # ai_message=response['messages'][-1].content
    # # first add the message to the message_history
    # st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

    # with st.chat_message('assistant'):
    #     ai_message=st.write_stream( #specific feature of streamlit to implement token-wise display
    #         message_chunk.content for message_chunk, metadata in chatbot.stream({'messages': [HumanMessage(content=user_input)]}, #.stream in langraph is used to implement the streaming feature
    #                        config=CONFIG,
    #                        stream_mode='messages'
    #
    #                                                                              )
    #     )
    # #Store assistant response in UI session state
    # st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

    # Assistant streaming block
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        #Extra coding to make streamlit asynchronous forcefully
        def ai_only_stream():
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    async for message_chunk, metadata in chatbot.astream(
                            {"messages": [HumanMessage(content=user_input)]},
                            config=CONFIG,
                            stream_mode="messages",
                    ):
                        event_queue.put((message_chunk, metadata))
                except Exception as exc:
                    event_queue.put(("error", exc))
                finally:
                    event_queue.put(None)

            submit_async_task(run_stream())

            while True:
                item = event_queue.get()
                if item is None:
                    break
                message_chunk, metadata = item
                if message_chunk == "error":
                    raise metadata
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                             f"🔧 Using `{tool_name}` …", expanded=True
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


        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )





