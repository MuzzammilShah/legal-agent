from langgraph import Graph, Node
from vectorstore import retrieve_relevant_chunks
from google import genai

# Init Gemini chat client
genai_client = genai.Client(api_key=None)  # key loaded via .env

def query_agent(state: dict) -> dict:
    state["chunks"] = retrieve_relevant_chunks(state["query"])
    return state

def summarization_agent(state: dict) -> dict:
    prompt = (
        "Convert the following legal excerpts into a clear, step-by-step summary:\n\n"
        + "\n\n".join(state["chunks"])
    )
    resp = genai_client.chats.create(model="gemini-2.0-flash").send_message(prompt)
    state["summary"] = resp.text
    return state

# Build graph
graph = Graph()
graph.add_node(Node(run=query_agent, name="QueryAgent"))
graph.add_node(Node(run=summarization_agent, name="SummarizationAgent"))
graph.add_edge("QueryAgent", "SummarizationAgent")