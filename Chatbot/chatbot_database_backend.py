from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]


def chat_node(state: ChatState):
    #take user query from state
    messages=state['messages']

    #send to llm
    response=llm.invoke(messages)

    #response store state
    return {'messages':[response]}

#Adding a sqlite connection after creating the database
conn=sqlite3.connect(database='chatbot.db', check_same_thread=False)
#Checkpointer
checkpointer=SqliteSaver(conn=conn)


graph=StateGraph(ChatState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node', END)

chatbot=graph.compile(checkpointer=checkpointer)

#To extract unique no of threads that is used in the frontend code to retain the threads that were previously present
def retrieve_all_threads():
    all_threads=set()
    for checkpoint in checkpointer.list(None): #the checkpointer.list func gives all the details of each checkpointer
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)



