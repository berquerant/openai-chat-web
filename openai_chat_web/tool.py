from datetime import datetime
from typing import Dict, List, Optional, Type, cast

from bs4 import BeautifulSoup
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import DuckDuckGoSearchResults, WikipediaQueryRun
from langchain_community.tools.requests.tool import RequestsGetTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, TextRequestsWrapper, WikipediaAPIWrapper
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.tools import tool as core_tool

from openai_chat_web import token


@core_tool
def now() -> str:
    """Get the current time as a string in %Y-%m-%d %H:%M:%S format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def wikipedia(top_k_results: int = 3, doc_content_chars_max: int = 1000) -> WikipediaQueryRun:
    """Return a new Wikipedia searcher."""
    return WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=top_k_results, doc_content_chars_max=doc_content_chars_max),  # type: ignore
        description="""A wrapper around Wikipedia. \
        Useful for when you need to answer general questions about people, places, companies, facts, \
        historical events, or other subjects. \
        Input should be a search query.""",
    )


class CustomDuckDuckGoSearchAPIWrapper(DuckDuckGoSearchAPIWrapper):
    """Customized wrapper for DuckDuckGo Search API."""

    content_chars_max: int

    def run(self, query: str) -> str:
        """Run query through DuckDuckGo and return concatenated results."""
        return super().run(query)[: self.content_chars_max]

    def results(self, query: str, max_results: int, source: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Run query through DuckDuckGo and return metadata.

        Args:
        ----
            query: The query to search for.
            max_results: The number of results to return.
            source: The source to look from.

        Returns:
        -------
            A list of dictionaries with the following keys:
                snippet - The description of the result.
                title - The title of the result.
                link - The link to the result.

        """
        r = super().results(query=query, max_results=max_results, source=source)
        for x in r:
            x["snippet"] = x["snippet"][: self.content_chars_max]
        return r


def duckduckgo(max_results: int = 3, content_chars_max: int = 1000) -> DuckDuckGoSearchResults:
    """Return a new DuckDuckGo searcher."""
    return DuckDuckGoSearchResults(
        api_wrapper=CustomDuckDuckGoSearchAPIWrapper(max_results=max_results, content_chars_max=content_chars_max),
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


class HTTPGetInput(BaseModel):
    url: str = Field(description="should be a url (i.e. https://www.google.com)")


class CustomHTTPGet(BaseTool):
    name: str = "custom_http_get"
    description: str = """A portal to the internet. \
    Use this when you need to get specific content from a website. \
    You should use this only when a user explicitly specifies a URL \
    and you need the contents of the web page at that URL to answer a question from the user. \
    Input should be a url (i.e. https://www.google.com). \
    The output will be the concatenated text part of the body element of the response html of the GET request."""
    args_schema: Type[BaseModel] = HTTPGetInput
    max_tokens: int = 3000
    model_name: str = "gpt-3.5-turbo"

    @property
    def __requests_get(self) -> TextRequestsWrapper:
        get_tool = cast(RequestsGetTool, load_tools(["requests_get"], allow_dangerous_tools=True)[0])
        wrapper = cast(TextRequestsWrapper, get_tool.requests_wrapper)
        return wrapper

    def __get_text(self, html: str) -> str:
        text = BeautifulSoup(html, "html.parser").find("body").get_text(separator="\n", strip=True)
        tokens = token.List.new(text, self.model_name)
        return tokens.read(n=self.max_tokens)

    def _run(self, url: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        html = cast(str, self.__requests_get.get(url))
        return self.__get_text(html)

    async def _arun(self, url: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        html = cast(str, await self.__requests_get.aget(url))
        return self.__get_text(html)
