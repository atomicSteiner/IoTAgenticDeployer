"""ThingsBoard adapter

Carries out the operations planning produced, through the official
ThingsBoard MCP server. Everything ThingsBoard-specific stops here (tool
names, payload shapes, the ways an id comes back), so nothing above knows it
"""

import json
import os

from iot_agentic_deployer.platforms.base import PlatformAdapter, register_adapter
from iot_agentic_deployer.platforms.mcp_client import MCPToolClient

# What ThingsBoard calls a containment relation.
CONTAINS = "Contains"


class ThingsBoardError(RuntimeError):
    """An operation the platform refused. Raised so realisation stops there
    with the reason, instead of marking it done with an id it never got"""


@register_adapter
class ThingsBoardAdapter(PlatformAdapter):

    name = "thingsboard"

    def __init__(self, mcp_url: str | None = None, **_ignored):
        self.mcp_url = mcp_url or os.getenv("THINGSBOARD_MCP_URL",
                                            "http://localhost:8000/sse")
        self._mcp = MCPToolClient({
            "thingsboard": {"url": self.mcp_url, "transport": "sse"},
        })

    # -- MCP plumbing -----------------------------------------------------

    @staticmethod
    def _check(tool: str, result):
        """The MCP tools do not fail on error, they return the problem: a
        dict with status ERROR, or a bare string where an entity should have
        been. Miss it and a refused operation is marked completed with a
        None id, so every shape becomes an exception here"""
        if isinstance(result, dict) and str(result.get("status", "")).upper() == "ERROR":
            raise ThingsBoardError(f"{tool}: {result.get('message') or result}")
        if isinstance(result, str):
            raise ThingsBoardError(f"{tool}: {result}")
        return result

    def _call(self, tool: str, **kwargs):
        return self._check(tool, self._mcp.call(tool, **kwargs))

    def _lookup(self, tool: str, **kwargs):
        """A call where failing just means not there, rather than refused."""
        try:
            result = self._mcp.call(tool, **kwargs)
        except Exception:
            return None
        if isinstance(result, dict) and str(result.get("status", "")).upper() != "ERROR":
            return result
        return None

    # -- contract ---------------------------------------------------------

    def check_reachability(self) -> tuple[bool, str]:
        try:
            self._call("getTenantDevices", pageSize=1, page=0)
        except Exception as e:
            return False, (
                f"The ThingsBoard MCP server at {self.mcp_url} did not answer: "
                f"{type(e).__name__}: {e}\n\n"
                "Start it with `docker compose up -d` from "
                "src/iot_agentic_deployer, and check that ThingsBoard itself is "
                "running at the URL that compose file points at."
            )
        return True, f"ThingsBoard reachable through the MCP server at {self.mcp_url}."

    def create_asset(self, name: str, asset_type: str) -> str:
        # Devices get upserted, a duplicate asset name is refused. Reusing
        # the existing one is what lets an interrupted run be confirmed again
        # without colliding with the assets it already created
        existing = self._lookup("getTenantAsset", assetName=name)
        if existing:
            return existing["id"]["id"]

        asset = self._call("saveAsset",
                           assetJson=json.dumps({"name": name, "type": asset_type}))
        return asset["id"]["id"]

    def create_device(self, name: str, device_type: str, label: str | None = None) -> str:
        payload = {"name": name, "type": device_type}
        if label:
            payload["label"] = label
        return self._call("createOrUpsertDevice", **payload)["deviceId"]

    def set_attributes(self, entity_id: str, entity_type: str, scope: str,
                       attributes: dict) -> None:
        if not attributes:
            return
        self._call("saveEntityAttributesV2", entityType=entity_type,
                   entityIdStr=entity_id, scope=scope,
                   jsonBody=json.dumps(attributes))

    def create_relation(self, from_id: str, from_type: str,
                        to_id: str, to_type: str) -> None:
        self._call("saveRelation", relationJson=json.dumps({
            "from": {"entityType": from_type, "id": from_id},
            "to": {"entityType": to_type, "id": to_id},
            "type": CONTAINS,
            "typeGroup": "COMMON",
        }))

    # -- reading ----------------------------------------------------------
    # Not provisioning: just for looking at what ended up on the platform

    def get_all_devices(self) -> list:
        devices, page = [], 0
        while True:
            data = self._call("getTenantDevices", pageSize=100, page=page)
            devices.extend(data["data"])
            if not data["hasNext"]:
                return devices
            page += 1

    def get_device_attributes(self, device_id: str):
        return self._call("getAttributes", entityType="DEVICE", entityId=device_id)

    def get_device_telemetry(self, device_id: str):
        return self._call("getLatestTimeseries", entityType="DEVICE", entityId=device_id)
