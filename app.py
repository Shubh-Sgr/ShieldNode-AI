import warnings
import os

# 1. SILENCE THE WARNING: Do this before importing anything else
warnings.filterwarnings("ignore", message="The default value of `allowed_objects` will change")
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"

from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

# 2. Configuration
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "your_api_key_here")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

class AgentState(TypedDict):
    task: str
    report: str
    feedback: str
    iterations: int

def researcher_node(state: AgentState):
    print("--- 🔍 Researcher Drafting ---")
    prompt = f"Task: {state['task']}\nFeedback: {state.get('feedback', '')}\nWrite a report:"
    response = llm.invoke(prompt)
    return {"report": response.content, "iterations": state.get("iterations", 0) + 1}

def reviewer_node(state: AgentState):
    print("--- ⚖️ Reviewer Checking ---")
    prompt = f"Review this: {state['report']}\nIf perfect, say 'APPROVED'. Else, list 3 fixes."
    response = llm.invoke(prompt)
    return {"feedback": response.content}

def should_continue(state: AgentState):
    if "APPROVED" in state["feedback"].upper() or state["iterations"] >= 3:
        return END
    return "researcher"

# Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher_node)
workflow.add_node("reviewer", reviewer_node)
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "reviewer")
workflow.add_conditional_edges("reviewer", should_continue)

app = workflow.compile()