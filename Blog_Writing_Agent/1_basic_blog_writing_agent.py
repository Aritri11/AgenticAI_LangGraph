from __future__ import annotations
import operator
from typing import List, TypedDict, Annotated
from pydantic import BaseModel, Field
from langgraph.types import Send
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

class Task(BaseModel):
    id: str
    title: str
    brief: str= Field(..., description='what to cover')

class Plan(BaseModel):
    blog_title: str
    tasks: List[Task]

class State(TypedDict):
    topic: str
    plan: Plan
    #reducer - results from workers get concatenated automatically
    sections: Annotated[List[str],operator.add]
    final: str

llm=ChatOllama(
    model="llama3.1:8b",
    temperature=0
)

def orchestrator(state: State) -> dict:

    plan = llm.with_structured_output(Plan).invoke(
        [
            SystemMessage(
                content=(
                    "Create a blog plan with 5-7 sections on the following topic."
                )
            ),
            HumanMessage(content=f"Topic: {state['topic']}"),
        ]
    )
    return {"plan": plan}

#Before going to the worker node, we don't know how many tasks are created and how many workers are required to complete each of them, so this fanout function bridges the gap between the two
def fanout(state: State):
    return [Send("worker", {"task": task, "topic": state["topic"], "plan": state["plan"]})
            for task in state["plan"].tasks]


def worker(payload: dict) -> dict:

    # payload contains what we sent
    task = payload["task"]
    topic = payload["topic"]
    plan = payload["plan"]

    blog_title = plan.blog_title

    section_md = llm.invoke(
        [
            SystemMessage(content="Write one clean Markdown section."),
            HumanMessage(
                content=(
                    f"Blog: {blog_title}\n"
                    f"Topic: {topic}\n\n"
                    f"Section: {task.title}\n"
                    f"Brief: {task.brief}\n\n"
                    "Return only the section content in Markdown."
                )
            ),
        ]
    ).content.strip()

    return {"sections": [section_md]}

def reducer(state: State) -> dict:
    title = state["plan"].blog_title
    body = "\n\n".join(state["sections"]).strip()

    final_md = f"# {title}\n\n{body}\n"

    # Save to file
    filename = "".join(c if c.isalnum() or c in (" ", "_", "-") else "" for c in title)
    filename = filename.strip().lower().replace(" ", "_") + ".md"
    Path(filename).write_text(final_md, encoding="utf-8")

    return {"final": final_md}

graph=StateGraph(State)
graph.add_node('orchestrator', orchestrator)
graph.add_node('worker', worker)
graph.add_node('reducer', reducer)

graph.add_edge(START, "orchestrator")
graph.add_conditional_edges("orchestrator", fanout, ["worker"])
graph.add_edge("worker","reducer")
graph.add_edge('reducer', END)

app=graph.compile()

out = app.invoke({
    "topic": "Write a blog on Cross Attention",
    "sections": []
})

print(out)


for p in Path(".").glob("*.md"):
    print("Found:", p, "size:", p.stat().st_size, "bytes")