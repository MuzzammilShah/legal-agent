# import os
# from dotenv import load_dotenv
# from langgraph.graph import StateGraph, START
# from vectorstore import retrieve_relevant_chunks
# from google import genai
# from google.genai import types

# def load_keys():
#     load_dotenv()
#     gemini_key = os.getenv("GEMINI_API_KEY")
#     if not gemini_key:
#         raise RuntimeError("GEMINI_API_KEY not set")
#     genai.configure(api_key=gemini_key)
#     return genai

# def trim_chunks(chunks, max_chars=1000, top_k=5):
#     # Limit chunk length and count to control context size
#     return [c[:max_chars] for c in chunks][:top_k]

# # Helper to call Gemini with token & context limits
# def call_gemini(prompt: str, max_tokens: int = 200, stop: list = None):
#     cfg = types.GenerateContentConfig(
#         max_output_tokens=max_tokens,
#         stop_sequences=stop or []
#     )
#     resp = genai.models.generate_content(
#         model="gemini-2.0-flash",
#         contents=prompt,
#         config=cfg
#     )
#     return resp.text.strip()

# # Agents

# def query_agent(state: dict) -> dict:
#     user_q = state.get("query", "")
#     # retrieve and trim to avoid context overflow
#     chunks = retrieve_relevant_chunks(user_q)
#     state["chunks"] = trim_chunks(chunks)
#     return state


# def summarization_agent(state: dict) -> dict:
#     chunks = state.get("chunks", [])
#     if not chunks:
#         state["summary"] = "No relevant documents found."
#         return state
#     prompt = (
#         "Convert the following legal excerpts into a simple, numbered guide, max 4 steps:\n\n"
#         + "\n\n".join(chunks)
#     )
#     state["summary"] = call_gemini(prompt, max_tokens=150, stop=["\n\n"])
#     return state


# def followup_agent(state: dict) -> dict:
#     # Maintain conversational history context
#     history = state.get("history", [])
#     last_resp = state.get("summary", "")
#     # Append last summary to history
#     history.append({"role": "assistant", "content": last_resp})
#     state["history"] = history
#     return state


# def response_agent(state: dict) -> dict:
#     # Build a concise reply and invite follow-ups
#     user_q = state.get("query", "")
#     summary = state.get("summary", "")
#     prompt = (
#         f"You are a helpful legal assistant. The user asked: '{user_q}'. "
#         f"Based on this summary: {summary}\n"  
#         "Provide a one-sentence answer and then ask 'Anything else?'."
#     )
#     state["response"] = call_gemini(prompt, max_tokens=50)
#     return state

# # Assemble the graph
# load_keys()
# builder = StateGraph(dict)
# builder.add_node("QueryAgent", query_agent)
# builder.add_node("SummarizationAgent", summarization_agent)
# builder.add_node("FollowupAgent", followup_agent)
# builder.add_node("ResponseAgent", response_agent)

# builder.add_edge(START, "QueryAgent")
# builder.add_edge("QueryAgent", "SummarizationAgent")
# builder.add_edge("SummarizationAgent", "FollowupAgent")
# builder.add_edge("FollowupAgent", "ResponseAgent")

# compiled_graph = builder.compile()
# ==================================

# FINAL MAIN IMPLEMENATION
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START
from vectorstore import retrieve_relevant_chunks
from google import genai
from google.genai import types

# 1. Load API key & init client
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not set")
# genai.configure(api_key=api_key)
# client = genai.Client()
client = genai.Client(api_key=api_key)

# 2. Helpers
def trim_chunks(chunks, max_chars=1000, top_k=5):
    return [c[:max_chars] for c in chunks][:top_k]

def call_gemini(prompt: str, max_tokens: int = 150, stop: list[str] = None) -> str:
    cfg = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        stop_sequences=stop or []
    )
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=cfg
    )
    return resp.text.strip()

# 3. Agents
def query_agent(state: dict) -> dict:
    q = state.get("query", "")
    chunks = retrieve_relevant_chunks(q)
    state["chunks"] = trim_chunks(chunks)
    return state

def summarization_agent(state: dict) -> dict:
    chunks = state.get("chunks", [])
    if not chunks:
        state["summary"] = "No relevant documents found."
        return state
    prompt = (
        "Convert these legal excerpts into a plain-language, numbered guide (max 4 steps):\n\n"
        + "\n\n".join(chunks)
    )
    state["summary"] = call_gemini(prompt, max_tokens=150, stop=["\n\n"])
    return state

def response_agent(state: dict) -> dict:
    q = state.get("query", "")
    summary = state.get("summary", "")
    prompt = (
        f"You are a helpful legal assistant. The user asked: '{q}'.\n"
        f"Based on this summary: {summary}\n"
        "Give a one-sentence answer and then ask 'Anything else?'."
    )
    state["response"] = call_gemini(prompt, max_tokens=50)
    return state

# 4. Build the StateGraph
builder = StateGraph(dict)
builder.add_node("QueryAgent", query_agent)
builder.add_node("SummarizationAgent", summarization_agent)
builder.add_node("ResponseAgent", response_agent)
builder.add_edge(START, "QueryAgent")
builder.add_edge("QueryAgent", "SummarizationAgent")
builder.add_edge("SummarizationAgent", "ResponseAgent")
compiled_graph = builder.compile()