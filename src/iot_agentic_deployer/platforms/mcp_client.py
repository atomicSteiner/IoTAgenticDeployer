import asyncio
import json

from langchain_mcp_adapters.client import MultiServerMCPClient


class MCPToolClient:
    """A thin wrapper around one or more MCP servers.

    To add an integration beyond ThingsBoard, add an entry to `servers` -
    nothing else changes.

    One thing to be careful about: every call opens a fresh session, inside
    the same event loop that made the call. MCP sessions, and the httpx/anyio
    transport under them, belong to the loop that created them. Cache tool
    objects across separate `asyncio.run()` calls and you end up reusing
    things whose loop is long closed - which is exactly where the flaky
    connection failures were coming from: a different error every time, none
    of them reproducible.
    """

    def __init__(self, servers: dict):
        self._servers = servers

    @staticmethod
    def _normalize(result):
        """MCP hands results back as raw content blocks - something like
        [{'type': 'text', 'text': '<json>', 'id': '...'}] - rather than the
        payload itself. Pull the text out, then keep unwrapping while it is
        still JSON in a string, so callers get real Python data instead of
        MCP's envelope."""
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
        """Blocking convenience wrapper. Fine for the one-call-at-a-time
        nodes here; if they ever go async, call `acall` directly instead and
        save re-listing the tools every time."""
        return asyncio.run(self.acall(tool_name, **kwargs))