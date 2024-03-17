from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults, WikipediaQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper


def wikipedia(top_k_results: int = 3, doc_content_chars_max: int = 800) -> WikipediaQueryRun:
    """Return a new Wikipedia searcher."""
    return WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=top_k_results, doc_content_chars_max=doc_content_chars_max),
        description="""A wrapper around Wikipedia. \
        Useful for when you need to answer general questions about people, places, companies, facts, \
        historical events, or other subjects. \
        Input should be a search query.""",
    )


def duckduckgo(max_results: int = 5) -> DuckDuckGoSearchResults:
    """Return a new DuckDuckGo searcher."""
    return DuckDuckGoSearchResults(
        api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=max_results),
        description="""A wrapper around Duck Duck Go Search. \
        Useful for when you need to get a broader perspective on a particular topic. \
        Input should be a search query. \
        Output is a JSON array of the query results""",
    )


def tavily() -> TavilySearchResults:
    """Return a new Tavily searcher."""
    return TavilySearchResults(
        # read api key from env TAVILY_API_KEY
        # no need to pass API key to APIWrapper `tavily_api_key` argument
        api_wrapper=TavilySearchAPIWrapper(),  # type: ignore
        description="""A search engine optimized for comprehensive, accurate, and trusted results. \
        Useful for when you need to answer questions about current events. \
        Input should be a search query.""",
    )
