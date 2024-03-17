from langchain_community.tools import DuckDuckGoSearchResults, WikipediaQueryRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper


def wikipedia(top_k_results: int = 4, doc_content_chars_max: int = 600) -> WikipediaQueryRun:
    """Return a new Wikipedia searcher."""
    return WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=top_k_results, doc_content_chars_max=doc_content_chars_max)
    )


def duckduckgo(max_results: int = 4) -> DuckDuckGoSearchResults:
    """Return a new DuckDuckGo searcher."""
    return DuckDuckGoSearchResults(api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=max_results))
