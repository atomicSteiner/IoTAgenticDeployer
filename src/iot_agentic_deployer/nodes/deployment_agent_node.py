from langchain_core.messages import AIMessage

from iot_agentic_deployer.state import IoTDeploymentState


class DeploymentAgentNode:
    """Implements the IoT Deployment Agent for final execution on ThingsBoard."""

    def __init__(self, tb_client):
        self.tb_client = tb_client

    def __call__(self, state: IoTDeploymentState) -> dict:
        device_name = state["device_name"]

        device = self.tb_client.create_device(
            name=device_name,
            device_type=state.get("device_type", "default"),
        )

        attributes = state.get("attributes")
        if attributes:
            self.tb_client.push_attributes(device["id"]["id"], "SHARED_SCOPE", attributes)

        return {
            "messages": [AIMessage(content=f"[Actuator] Device '{device_name}' deployed on ThingsBoard.")]
        }


