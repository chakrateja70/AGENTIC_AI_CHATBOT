from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain.tools import tool

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

if __name__ == "__main__":
    print("running arxiv run")
    print(arxiv_tool_run("Attention is all you need")) # type: ignore