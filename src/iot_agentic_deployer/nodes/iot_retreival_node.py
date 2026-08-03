from langchain_core.messages import AIMessage

from iot_agentic_deployer.state import IoTDeploymentState


class IoTRetrievalNode:
    """Retrieve all the devices from ThingsBoard."""

    def __init__(self, tb_client):
        self.tb_client = tb_client

    def __call__(self, state: IoTDeploymentState) -> dict:
        devices = self.tb_client.get_all_devices()

        dt_assets = [
            {
                "id": d["id"]["id"],
                "name": d["name"],
                "type": d["type"],
                "label": d.get("label"),
                "created_time": d.get("createdTime"),
                "attributes": self.tb_client.get_device_attributes(d["id"]["id"]),
                "telemetry": self.tb_client.get_device_telemetry(d["id"]["id"]),
            }
            for d in devices
        ]

        summary = f"[Retrieval] Found {len(dt_assets)} device(s) on ThingsBoard: " + ", ".join(
            d["name"] for d in dt_assets
        ) if dt_assets else "[Retrieval] No devices found on ThingsBoard."

        return {
            "dt_data": dt_assets,
            "messages": [AIMessage(content=summary)],
        }
