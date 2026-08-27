import asyncio
import concurrent.futures
import json
import threading
from contextlib import contextmanager

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


class MCPToolClient:
    """A thin wrapper around one or more MCP servers: to add an integration,
    add an entry to `servers`
    Every call opens a fresh session inside the loop that made it, unless
    one is being held open by `session()`
    """

    def __init__(self, servers: dict):
        self._servers = servers
        self._open = None       # (loop, tools) while a session is held open

    @contextmanager
    def session(self):
        """Holds one session open for a run of calls.
        """
        if self._open:                  # already inside one, nothing to do
            yield self
            return

        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        started = concurrent.futures.Future()

        async def hold():
            stop = asyncio.Event()
            try:
                client = MultiServerMCPClient(self._servers)
                async with client.session(next(iter(self._servers))) as session:
                    tools = await load_mcp_tools(session)
                    started.set_result(({tool.name: tool for tool in tools}, stop))
                    await stop.wait()
            except Exception as error:
                if not started.done():
                    started.set_exception(error)

        stop = keeper = None
        try:
            keeper = asyncio.run_coroutine_threadsafe(hold(), loop)
            tools, stop = started.result()
            self._open = (loop, tools)
        except Exception:
            # Not being able to hold one open costs speed, not the run: the
            # calls fall back to opening their own
            pass

        try:
            yield self
        finally:
            self._open = None
            if stop is not None:
                loop.call_soon_threadsafe(stop.set)
                try:
                    keeper.result(timeout=10)
                except Exception:
                    pass
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5)
            loop.close()

    @staticmethod
    def _normalize(result):
        """MCP returns content blocks, not the payload itself. Pull the text
        out and keep unwrapping while it is still JSON in a string, so callers
        get real Python data instead of the envelope"""
        if isinstance(result, list) and result and all(
            isinstance(item, dict) and item.get("type") == "text" for item in result
        ):
            result = "".join(item.get("text", "") for item in result)

        while isinstance(result, str):
            try:
                parsed = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                break
            if parsed == result:
                break
            result = parsed

        return result

    async def acall(self, tool_name: str, **kwargs):
        client = MultiServerMCPClient(self._servers)
        tools = await client.get_tools()
        tools_by_name = {tool.name: tool for tool in tools}
        if tool_name not in tools_by_name:
            raise ValueError(f"MCP tool '{tool_name}' not found. Available: {sorted(tools_by_name)}")
        result = await tools_by_name[tool_name].ainvoke(kwargs)
        return self._normalize(result)

    def call(self, tool_name: str, **kwargs):
        """Blocking wrapper, fine for the one-call-at-a-time nodes here. If
        they ever go async, call `acall` directly"""
        if self._open:
            loop, tools = self._open
            # A tool the open session does not serve falls through to a call
            # of its own, which is also where 'no such tool' gets reported
            if tool_name in tools:
                return self._normalize(asyncio.run_coroutine_threadsafe(
                    tools[tool_name].ainvoke(kwargs), loop).result())
        return asyncio.run(self.acall(tool_name, **kwargs))