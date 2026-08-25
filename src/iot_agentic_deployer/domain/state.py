from datetime import datetime
from typing import Annotated, Literal, Type

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, create_model
from typing_extensions import TypedDict


def append(left: list, right: list) -> list:
    """Trace entries pile up instead of replacing each other."""
    return (left or []) + (right or [])


def trace(agent: str, action: str, detail: str = "") -> list[dict]:
    """One line for the activity trace, ready to drop into state['trace']

    Every agent writes down what it was asked for and what came of it, in the
    same shape, so the trace reads as a single account of the session rather
    than six agents keeping their own notes"""
    return [{"agent": agent, "action": action, "detail": detail,
             "at": datetime.utcnow().isoformat()}]


class IoTDeploymentState(TypedDict, total=False):
    """Everything the agents share as they work."""
    messages: Annotated[list, add_messages]

    installation: dict          # models.Installation, serialised
    validation_report: dict     # whatever the validation engine last said
    deployment_plan: list       # the operations, in the order they run
    plan_confirmed: bool        # the architect said go ahead

    # Every hand-off and every tool call stays visible (R7).
    trace: Annotated[list, append]

    next_node: str


def build_routing_decision(valid_destinations: list[str]) -> Type[BaseModel]:
    return create_model(
        "RoutingDecision",
        response_to_user=(str, Field(description="Message shown to the IoT architect.")),
        next_node=(
            Literal[tuple(valid_destinations + ["WaitUser"])],
            Field(description="Agent to delegate to. 'WaitUser' to stop and await the architect."),
        ),
    )
