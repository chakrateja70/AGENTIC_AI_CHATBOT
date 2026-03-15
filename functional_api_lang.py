from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper

from langgraph.graph import add_messages
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    ToolCall,
)
from langchain_core.messages import BaseMessage
from langgraph.func import entrypoint, task

from src.settings import settings

model = init_chat_model(
    "gpt-4o-mini", 
    api_key=settings.openai_api_key,
    model_provider="openai"
    )

@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    print("[TOOL] multiply is being used with a=", a, "b=", b)
    return a * b

@tool
def tavily_tool_run(query: str) -> str:
    """
    Use this tool when the LLM needs fresh or real-time web information.
    It searches the web with Tavily and returns the top results for the query as a string.
    """
    print("[TOOL] tavily_tool_run is being used with query=", query)
    tavily_run = TavilySearch(max_results=2, tavily_api_key=settings.tavily_api_key)
    return tavily_run.invoke(query)

@tool
def arxiv_tool_run(query: str) -> str:
    """
    Use this tool when the LLM needs academic or research-paper sources.
    It searches arXiv for papers matching the query and returns top results as a string.
    """
    arxiv_wrapper = ArxivAPIWrapper( #type: ignore
        top_k_results=2,
        doc_content_chars_max=1024
    )
    arxiv_run = ArxivQueryRun(api_wrapper=arxiv_wrapper)
    return arxiv_run.invoke(query)

tools = [multiply, tavily_tool_run, arxiv_tool_run]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

@task
def call_llm(messages: list[BaseMessage]):
    """LLM decides whether to call a tool or not"""
    return model_with_tools.invoke(
        [
            SystemMessage(
                content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
            )
        ]
        + messages
    )

@task
def call_tool(tool_call: ToolCall):
    """Performs the tool call"""
    print(f"[AGENT] Calling tool: {tool_call['name']} with args: {tool_call['args']}")
    tool = tools_by_name[tool_call["name"]]
    return tool.invoke(tool_call)

@entrypoint()
def agent(messages: list[BaseMessage]):
    model_response = call_llm(messages).result()

    while True:
        if not model_response.tool_calls:
            break

        # Execute tools
        tool_result_futures = [
            call_tool(tool_call) for tool_call in model_response.tool_calls
        ]
        tool_results = [fut.result() for fut in tool_result_futures]
        messages = add_messages(messages, [model_response, *tool_results])
        model_response = call_llm(messages).result()

    messages = add_messages(messages, model_response)
    return messages

# Invoke
messages = [HumanMessage(content="Attention is all you need.")]
for chunk in agent.stream(messages, stream_mode="updates"):
    print(chunk)
    print("\n")