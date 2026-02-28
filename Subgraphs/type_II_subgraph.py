#Type 1: There is only 1 state which is shared by both parent and the child graphs
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
load_dotenv()

#setting up project name inside the code (method 2)
os.environ['LANGCHAIN_PROJECT'] = 'Subgraphs'

class Parentstate(TypedDict):
    question: str
    answer_eng: str
    answer_hin: str

parent_llm =  ChatOllama(
    model="llama3.1:8b",
    temperature=0
)

subgraph_llm =  ChatOllama(
    model="llama3.1:8b",
    temperature=0
)

def translate_text(state: Parentstate):
    prompt=f"""
Translate the following text to Hindi.
Keep it natural and clear. Do not add extra content.
Text: {state["answer_eng"]}""".strip()

    translated_text= subgraph_llm.invoke(prompt).content
    return {'answer_hin':translated_text}

#building the subgraph that inherits the state from parent
subgraph_builder=StateGraph(Parentstate)

subgraph_builder.add_node('translate_text', translate_text)

subgraph_builder.add_edge(START, 'translate_text')
subgraph_builder.add_edge('translate_text', END)

subgraph= subgraph_builder.compile()


def generate_answer(state: Parentstate):
    answer= parent_llm.invoke(f"You are a helpful assistant. Answer clearly. \n\nQuestion: {state['question']}").content
    return {'answer_eng':answer}

parent_builder= StateGraph(Parentstate)

parent_builder.add_node('answer', generate_answer)
parent_builder.add_node('translate', subgraph) #using subgraph as a node

parent_builder.add_edge(START, 'answer')
parent_builder.add_edge('answer', 'translate')
parent_builder.add_edge('translate', END)

graph= parent_builder.compile()

final_result=graph.invoke({'question':'What is a cell?'})
print(final_result)