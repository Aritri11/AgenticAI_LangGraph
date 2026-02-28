#Type 1: States are isolated for both the graphs and are joined through invoke
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
load_dotenv()

#setting up project name inside the code (method 2)
os.environ['LANGCHAIN_PROJECT'] = 'Subgraphs'

#This is for the subgraph
class Substate(TypedDict):
    input_text: str
    translated_text: str

subgraph_llm =  ChatOllama(
    model="llama3.1:8b",
    temperature=0
)

def translate_text(state: Substate):
    prompt=f"""
Translate the following text to Hindi.
Keep it natural and clear. Do not add extra content.
Text: {state["input_text"]}""".strip()

    translated_text= subgraph_llm.invoke(prompt).content
    return {'translated_text':translated_text}

#building the subgraph
subgraph_builder=StateGraph(Substate)

subgraph_builder.add_node('translate_text', translate_text)

subgraph_builder.add_edge(START, 'translate_text')
subgraph_builder.add_edge('translate_text', END)

subgraph= subgraph_builder.compile()

#This is for the parent graph
class Parentstate(TypedDict):
    question: str
    answer_eng: str
    answer_hin: str

parent_llm =  ChatOllama(
    model="llama3.1:8b",
    temperature=0
)

def generate_answer(state: Parentstate):
    answer= parent_llm.invoke(f"You are a helpful assistant. Answer clearly. \n\nQuestion: {state['question']}").content
    return {'answer_eng':answer}

def translate_answer(state: Parentstate):

    #calling of the subgraph
    result=subgraph.invoke({'input_text':state["answer_eng"]})
    return {'answer_hin': result['translated_text']}

parent_builder= StateGraph(Parentstate)

parent_builder.add_node('answer', generate_answer)
parent_builder.add_node('translate', translate_answer)

parent_builder.add_edge(START, 'answer')
parent_builder.add_edge('answer', 'translate')
parent_builder.add_edge('translate', END)

graph= parent_builder.compile()

final_result=graph.invoke({'question':'What is a cell?'})
print(final_result)
