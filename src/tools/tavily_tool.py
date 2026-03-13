from langchain_tavily import TavilySearch
from ..settings import settings


def tavily_tool_run(query: str) -> str:
    tavily_run = TavilySearch(max_results=2, tavily_api_key=settings.tavily_api_key)
    return tavily_run.invoke(query)

if __name__ == "__main__":
    print("running tavily run")
    print(tavily_tool_run("give me recent github student copilot news"))