from typing import TypedDict, Annotated, Literal, Type
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, create_model

class IoTDeploymentState(TypedDict):
    """Maintains the global state and shared memory of the ecosystem."""
    messages: Annotated[list, add_messages]
    context_constraints: str
    dt_data: str
    next_node: str


def build_routing_decision(valid_destinations: list[str]) -> Type[BaseModel]:
    """Builds a RoutingDecision model constrained to exactly the node names
    that exist in the graph (+ "WaitUser").
    """
    return create_model(
        "RoutingDecision",
        response_to_user=(str, Field(description="Message to display to the industrial operator.")),
        next_node=(
            Literal[tuple(valid_destinations + ["WaitUser"])],
            Field(description="The next module to activate. Use 'WaitUser' to request confirmation or stop."),
        ),
    )
