"""Summary agent: shows the configuration so far as a table and
a diagram, both read straight off the model."""


from langchain_core.messages import AIMessage

from iot_agentic_deployer.domain.models import Installation
from iot_agentic_deployer.domain.state import IoTDeploymentState, trace
from iot_agentic_deployer.rendering.formatting import installation_to_markdown, installation_to_mermaid


class SummaryNode:

    def __call__(self, state: IoTDeploymentState) -> dict:
        inst = Installation(**state.get("installation", {}))

        parts = ["[Summary] Current configuration:", "", installation_to_markdown(inst)]
        diagram = installation_to_mermaid(inst)
        if diagram:
            parts += ["", "```mermaid", diagram, "```"]

        return {"messages": [AIMessage(content="\n".join(parts))],
                "trace": trace("Summary", "render")}