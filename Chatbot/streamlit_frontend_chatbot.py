import streamlit as st
from langgraph_backend_chatbot import chatbot
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
import uuid

######################### Utility functions ######################
def generate_thread_id():
     thread_id=uuid.uuid4()
     return thread_id

def reset_chat():
    thread_id=generate_thread_id()
    st.session_state['thread_id']=thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history']=[]

def add_thread (thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    return chatbot.get_state(config={'configurable': {'thread_id':thread_id}}).values['messages']

def get_thread_preview(thread_id):
    state = chatbot.get_state(
        config={'configurable': {'thread_id': thread_id}}
    )

    if not state or 'messages' not in state.values:
        return "New Chat"

    messages = state.values['messages']

    # Find first human message
    for msg in messages:
        if isinstance(msg, HumanMessage):
            preview = msg.content.strip()
            return preview[:40] + ("..." if len(preview) > 40 else "")

    return "New Chat"

###############################################################


####################### Session Setup #######################

#st.session_state -> dict
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

add_thread(st.session_state['thread_id'])

#############################################################

######################### Sidebar UI #######################

st.sidebar.title('LangGraph Chatbot')
if st.sidebar.button('New Chat'):
    reset_chat()
st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    preview = get_thread_preview(thread_id)

    if st.sidebar.button(preview, key=str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages

#############################################################

####################### Main UI #############################

#loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input=st.chat_input('Type here')

CONFIG = {'configurable': {'thread_id':st.session_state['thread_id'] }}
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

    with st.chat_message('assistant'):
        ai_message=st.write_stream( #specific feature of streamlit to implement token-wise display
            message_chunk.content for message_chunk, metadata in chatbot.stream({'messages': [HumanMessage(content=user_input)]}, #.stream in langraph is used to implement the streaming feature
                           config=CONFIG,
                           stream_mode='messages'

                                                                                 )
        )
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})




















