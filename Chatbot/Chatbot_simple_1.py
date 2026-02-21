from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from patsy import state
import operator
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


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

#concept of 'checkpointer' helps to bring persistence
checkpointer = MemorySaver()
graph=StateGraph(ChatState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node', END)

chatbot=graph.compile(checkpointer=checkpointer)

# #to see the workflow
# from IPython.display import Image
# Image(workflow.get_graph().draw_mermaid_png())

#looping strategy:
thread_id='1'
while True:
    user_message=input("Type here: ")

    print('User:',user_message)

    if user_message.strip().lower() in ['exit', 'quit,''bye']:
        break
    config={'configurable': {'thread_id':thread_id}}
    response=chatbot.invoke({'messages': [HumanMessage(content=user_message)]},config=config)
    print('AI: ', response['messages'][-1].content)


