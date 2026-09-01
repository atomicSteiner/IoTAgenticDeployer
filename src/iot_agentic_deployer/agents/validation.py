"""Validation agent: checks the model against the minimal
topology rules and whatever profile is in force, then says whether it can go
to deployment"""


from langchain_core.messages import AIMessage

from iot_agentic_deployer.domain.models import Installation
from iot_agentic_deployer.domain.state import IoTDeploymentState, trace
from iot_agentic_deployer.domain.validation import validate_installation
from iot_agentic_deployer.rendering.formatting import validation_to_markdown


class ValidationNode:

    def __call__(self, state: IoTDeploymentState) -> dict:
        inst = Installation(**state.get("installation", {}))
        report = validate_installation(inst)

        return {
            "validation_report": report,
            "messages": [AIMessage(content="[Validation] " + validation_to_markdown(report))],
            "trace": trace("Validation", "evaluate",
                           f"{report['errors']} error(s), {report['warnings']} warning(s)"),
        }