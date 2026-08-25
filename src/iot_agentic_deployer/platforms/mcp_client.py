import asyncio
import json

from langchain_mcp_adapters.client import MultiServerMCPClient


class MCPToolClient:
    """A thin wrapper around one or more MCP servers: to add an integration,
    add an entry to `servers`
    Every call opens a fresh session inside the loop that made it
    """

    def __init__(self, servers: dict):
        self._servers = servers

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
        return asyncio.run(self.acall(tool_name, **kwargs))