# import os
# from dotenv import load_dotenv
# from langgraph.graph import StateGraph, START
# from vectorstore import retrieve_relevant_chunks
# from google import genai

# load_dotenv()  

# GEMINI_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_KEY:
#     raise RuntimeError("GEMINI_API_KEY not set in environment")

# # Init Gemini chat client
# genai_client = genai.Client(api_key=GEMINI_KEY)

# def query_agent(state: dict) -> dict:
#     state["chunks"] = retrieve_relevant_chunks(state["query"])
#     return state

# def summarization_agent(state: dict) -> dict:
#     prompt = (
#         "Convert the following legal excerpts into a clear, step-by-step summary and then provide a simple, concise and accurate answer:\n\n"
#         + "\n\n".join(state["chunks"])
#     )
#     resp = genai_client.chats.create(model="gemini-2.0-flash").send_message(prompt)
#     state["summary"] = resp.text
#     return state

# # Build a stateful graph
# graph_builder = StateGraph(dict)
# graph_builder.add_node("QueryAgent", query_agent)
# graph_builder.add_node("SummarizationAgent", summarization_agent)
# graph_builder.add_edge(START, "QueryAgent")
# graph_builder.add_edge("QueryAgent", "SummarizationAgent")

# # Compile into an executable graph
# compiled_graph = graph_builder.compile()



# # MORE DETAILED AGENT BREAKDOWN
# import os
# from dotenv import load_dotenv
# from langgraph.graph import StateGraph, START
# from vectorstore import retrieve_relevant_chunks
# from google import genai

# load_dotenv()

# GEMINI_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_KEY:
#     raise RuntimeError("GEMINI_API_KEY not set in environment")

# # Init Gemini chat client
# genai_client = genai.Client(api_key=GEMINI_KEY)

# # Agent definitions

# def query_agent(state: dict) -> dict:
#     """
#     Retrieves relevant legal text chunks based on the user query.
#     """
#     state["chunks"] = retrieve_relevant_chunks(state.get("query", ""))
#     return state


# def extraction_agent(state: dict) -> dict:
#     """
#     Extracts procedural steps or key points from the retrieved text.
#     """
#     # Build extraction prompt
#     extract_prompt = (
#         "Extract the procedural steps or key points from the following legal text excerpts:\n\n"
#         + "\n\n".join(state.get("chunks", []))
#     )
#     extract_resp = genai_client.chats.create(model="gemini-2.0-flash").send_message(
#         extract_prompt
#     )
#     state["steps_raw"] = extract_resp.text
#     return state


# def summarization_agent(state: dict) -> dict:
#     """
#     Converts raw steps into a clear, concise, human-friendly summary.
#     """
#     summary_prompt = (
#         "Convert the following list of legal procedural steps into a simple, numbered, plain-language guide, "
#         "preserving accuracy and completeness:\n\n"
#         + state.get("steps_raw", "")
#     )
#     summary_resp = genai_client.chats.create(model="gemini-2.0-flash").send_message(
#         summary_prompt
#     )
#     state["summary"] = summary_resp.text
#     return state


# def response_agent(state: dict) -> dict:
#     """
#     Formats the final response to the user, offering follow-up.
#     """
#     response_prompt = (
#         "You are a helpful assistant. The user asked: '" + state.get("query", "") + "'. "
#         "Based on the following summary, craft a concise response (no more than 2 lines) and invite further questions if needed:\n\n"
#         + state.get("summary", "")
#     )
#     resp = genai_client.chats.create(model="gemini-2.0-flash").send_message(
#         response_prompt
#     )
#     state["response"] = resp.text
#     return state

# # Build a stateful graph
# graph_builder = StateGraph(dict)
# graph_builder.add_node("QueryAgent", query_agent)
# graph_builder.add_node("ExtractionAgent", extraction_agent)
# graph_builder.add_node("SummarizationAgent", summarization_agent)
# graph_builder.add_node("ResponseAgent", response_agent)

# # Define edges for the flow
# graph_builder.add_edge(START, "QueryAgent")
# graph_builder.add_edge("QueryAgent", "ExtractionAgent")
# graph_builder.add_edge("ExtractionAgent", "SummarizationAgent")
# graph_builder.add_edge("SummarizationAgent", "ResponseAgent")

# # Compile into an executable graph
# compiled_graph = graph_builder.compile()


# STICKING TO MY QUOTA LIMITS - TO HAVE EFFECTIVE FOLLOW UPS
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START
from vectorstore import retrieve_relevant_chunks
from google import genai
from google.api_core import retry as g_retry

load_dotenv()
genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Retry policy
retry_policy = g_retry.Retry(
    predicate=g_retry.if_transient_error,
    initial=1.0, multiplier=2.0, maximum=8.0
)

# Helpers
def trim_chunks(chunks, max_chars=500, top_k=3):
    return [c[:max_chars] for c in chunks][:top_k]

def send_flash(prompt, max_tokens):
    return genai_client.chats.create(model="gemini-2.0-flash") \
        .send_message(
            prompt,
            max_output_tokens=max_tokens,
            stop_sequences=["\n\n"],
            timeout=30,
            retry=retry_policy
        ).text

# Agents
def query_agent(state):
    q = state.get("query", "")
    # embed only once
    if "query_emb" not in state:
        state["query_emb"] = q  # your vectorstore will re-embed here
    chunks = retrieve_relevant_chunks(q)
    state["chunks"] = trim_chunks(chunks)
    return state

def extraction_agent(state):
    if not state["chunks"]:
        state["steps_raw"] = ""
        return state
    prompt = (
        "Extract procedural steps/key points from these excerpts:\n\n"
        + "\n\n".join(state["chunks"])
    )
    state["steps_raw"] = send_flash(prompt, max_tokens=30)
    return state

def summarization_agent(state):
    raw = state.get("steps_raw", "")
    prompt = (
        "Turn this into a numbered plain-language guide (max 3 steps):\n\n" + raw
    )
    state["summary"] = send_flash(prompt, max_tokens=20)
    return state

def response_agent(state):
    summary = state.get("summary", "")
    prompt = (
        f"User asked: '{state['query']}'. Based on this summary, give a 1-line answer and ask ‘Anything else?’:\n\n"
        + summary
    )
    state["response"] = send_flash(prompt, max_tokens=20)
    return state

# Build graph
builder = StateGraph(dict)
builder.add_node("QueryAgent", query_agent)
builder.add_node("ExtractionAgent", extraction_agent)
builder.add_node("SummarizationAgent", summarization_agent)
builder.add_node("ResponseAgent", response_agent)
builder.add_edge(START, "QueryAgent")
builder.add_edge("QueryAgent", "ExtractionAgent")
builder.add_edge("ExtractionAgent", "SummarizationAgent")
builder.add_edge("SummarizationAgent", "ResponseAgent")
compiled_graph = builder.compile()