import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START
from vectorstore import retrieve_relevant_chunks
from google import genai

load_dotenv()  

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment")

# Init Gemini chat client
genai_client = genai.Client(api_key=GEMINI_KEY)

def query_agent(state: dict) -> dict:
    state["chunks"] = retrieve_relevant_chunks(state["query"])
    return state

def summarization_agent(state: dict) -> dict:
    prompt = (
        "Convert the following legal excerpts into a clear, step-by-step summary and then provide a simple, concise and accurate answer:\n\n"
        + "\n\n".join(state["chunks"])
    )
    resp = genai_client.chats.create(model="gemini-2.0-flash").send_message(prompt)
    state["summary"] = resp.text
    return state

# Build a stateful graph
graph_builder = StateGraph(dict)
graph_builder.add_node("QueryAgent", query_agent)
graph_builder.add_node("SummarizationAgent", summarization_agent)
graph_builder.add_edge(START, "QueryAgent")
graph_builder.add_edge("QueryAgent", "SummarizationAgent")

# Compile into an executable graph
compiled_graph = graph_builder.compile()