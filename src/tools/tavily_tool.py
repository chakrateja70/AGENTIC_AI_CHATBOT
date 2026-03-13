from langchain_tavily import TavilySearch
from ..settings import settings
from langchain.tools import tool

@tool
def tavily_tool_run(query: str) -> str:
    """
    Use this tool when the LLM needs fresh or real-time web information.
    It searches the web with Tavily and returns the top results for the query as a string.
    """
    tavily_run = TavilySearch(max_results=2, tavily_api_key=settings.tavily_api_key)
    return tavily_run.invoke(query)

if __name__ == "__main__":
    print("running tavily run")
    print(tavily_tool_run("give me recent github student copilot news")) # type: ignore