from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool

@tool
def wikipedia_tool_run(query: str) -> str:
    """
    Use this tool when the LLM needs general background or encyclopedic knowledge.
    It searches Wikipedia for the query and returns the top results as a string.
    """
    wikipedia_wrapper = WikipediaAPIWrapper( #type: ignore
        top_k_results=2,
        lang="en",
        doc_content_chars_max=1024
    )
    wikipedia_run = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)
    return wikipedia_run.invoke(query)

if __name__ == "__main__":
    print("running wikipedia run")
    print(wikipedia_tool_run("Attention is all you need")) # type: ignore