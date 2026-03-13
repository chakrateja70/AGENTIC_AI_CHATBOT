from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from src.tools.wikipedia_tool import wikipedia_tool_run
from src.tools.arxiv_tool import arxiv_tool_run
from src.tools.tavily_tool import tavily_tool_run
import re

from src.settings import settings

tools = [
    wikipedia_tool_run,
    arxiv_tool_run,
    tavily_tool_run,
]

tools_map = {tool.name: tool for tool in tools}


def _needs_external_lookup(query: str) -> bool:
    """
    Return True only when the query likely requires external or up-to-date information.
    """
    text = query.lower()
    lookup_patterns = [
        r"\blatest\b",
        r"\brecent\b",
        r"\bnews\b",
        r"\bcurrent\b",
        r"\btoday\b",
        r"\bnow\b",
        r"\bprice\b",
        r"\bstock\b",
        r"\bweather\b",
        r"\btemperature\b",
        r"\btrend(s)?\b",
        r"\bupdate(s)?\b",
        r"\brelease(d|s)?\b",
        r"\bnew\b",
        r"\bwhat happened\b",
        r"\bsearch\b",
        r"\bfind\b",
        r"\blook up\b",
        r"\barxiv\b",
        r"\bpaper(s)?\b",
        r"\bresearch\b",
        r"\bstudy\b",
        r"\bscientific\b",
        r"\bcitation(s)?\b",
        r"\bsource(s)?\b",
        r"\b20[2-9][0-9]\b",
    ]
    return any(re.search(pattern, text) for pattern in lookup_patterns)


def _select_required_tool(query: str) -> str:
    """
    Pick the most appropriate tool for the user's query.
    """
    text = query.lower()

    arxiv_patterns = [
        r"\barxiv\b",
        r"\bresearch\b",
        r"\bpaper(s)?\b",
        r"\bstudy\b",
        r"\bscientific\b",
        r"\bjournal\b",
        r"\bmethodolog(y|ies)\b",
        r"\battention is all you need\b",
        r"\btransformer(s)?\b",
    ]
    tavily_patterns = [
        r"\blatest\b",
        r"\brecent\b",
        r"\bnews\b",
        r"\bcurrent\b",
        r"\btoday\b",
        r"\bupdate(s)?\b",
        r"\brelease(d|s)?\b",
        r"\bthis week\b",
        r"\bthis month\b",
        r"\bnow\b",
        r"\b20[2-9][0-9]\b",
    ]

    if any(re.search(pattern, text) for pattern in arxiv_patterns):
        return "arxiv_tool_run"
    if any(re.search(pattern, text) for pattern in tavily_patterns):
        return "tavily_tool_run"
    return "wikipedia_tool_run"


def _run_required_tool(query: str, required_tool_name: str) -> str:
    tool = tools_map[required_tool_name]
    try:
        return str(tool.invoke({"query": query}))
    except Exception as exc:
        return f"Tool execution failed for {required_tool_name}: {exc}"


def llm_run(query: str) -> str:
    llm = ChatGroq(api_key=settings.groq_api_key, model=settings.groq_llm_model)  # type: ignore
    use_tool = _needs_external_lookup(query)

    if use_tool:
        required_tool_name = _select_required_tool(query)
        print(f"Tool selected: {required_tool_name}")
        tool_result = _run_required_tool(query, required_tool_name)
        system_prompt = "You are a precise research assistant. Base your answer only on the provided tool output."

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"User query: {query}\n"
                    f"Tool used: {required_tool_name}\n"
                    f"Tool output:\n{tool_result}\n\n"
                    "Answer clearly and concisely. If the tool output is empty or uncertain, say so."
                )
            ),
        ]
        response = llm.invoke(messages)
        return str(response.content)

    print("Tool selected: none (LLM knowledge)")
    messages = [
        SystemMessage(content="You are a precise assistant. Answer from general knowledge and be concise."),
        HumanMessage(content=query),
    ]

    response = llm.invoke(messages)

    return str(response.content)

if __name__ == "__main__":
    print("running llm run")
    print(llm_run("what is deepagents"))