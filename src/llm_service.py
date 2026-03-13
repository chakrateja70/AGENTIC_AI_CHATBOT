from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from src.tools.wikipedia_tool import wikipedia_tool_run
from src.tools.arxiv_tool import arxiv_tool_run
from src.tools.tavily_tool import tavily_tool_run

from src.settings import settings


class QueryInput(BaseModel):
    query: str = Field(description="Search query text")

tools = [
    StructuredTool.from_function(
        name="wikipedia_tool_run",
        description="Use for timeless encyclopedic facts and definitions.",
        func=wikipedia_tool_run,
        args_schema=QueryInput,
    ),
    StructuredTool.from_function(
        name="arxiv_tool_run",
        description="Use for academic/research-paper search and scientific topics.",
        func=arxiv_tool_run,
        args_schema=QueryInput,
    ),
    StructuredTool.from_function(
        name="tavily_tool_run",
        description="Use for web search, latest updates, recent news, and current events.",
        func=tavily_tool_run,
        args_schema=QueryInput,
    ),
]

tools_map = {tool.name: tool for tool in tools}


def _extract_query(args: object) -> str:
    if isinstance(args, dict):
        for key in ("query", "search_query", "question", "input"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return str(args)
    return str(args)


def llm_run(query: str) -> str:
    llm = ChatGroq(api_key=settings.groq_api_key, model=settings.groq_llm_model)  # type: ignore
    llm_with_tools = llm.bind_tools(tools=tools)
    system_prompt = (
        "You are a precise research assistant. Choose tools as follows: "
        "tavily_tool_run for latest/recent/current/news queries, "
        "wikipedia_tool_run for stable factual knowledge, "
        "arxiv_tool_run for research papers. "
        "Always use only the provided tools."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    while getattr(response, "tool_calls", None):
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool = tools_map.get(tool_name)
            if tool is None:
                tool_result = f"Unsupported tool requested: {tool_name}"
            else:
                tool_result = tool.invoke({"query": _extract_query(tool_call.get("args", {}))})

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                )
            )

        response = llm_with_tools.invoke(messages)
        messages.append(response)

    return str(response.content)

if __name__ == "__main__":
    print("running llm run")
    print(llm_run("What is the latest news about github copilot student pack?"))