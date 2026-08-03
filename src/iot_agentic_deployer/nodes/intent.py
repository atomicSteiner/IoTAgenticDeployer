

from langchain_core.messages import AIMessage

from iot_agentic_deployer.state import IoTDeploymentState, build_routing_decision


class OrchestratorNode:
    """Implements the IoT Deployment Orchestrator."""

    def __init__(self, llm, valid_destinations: list[str]):
        self.llm = llm
        self.valid_destinations = valid_destinations

        # RoutingDecision.next_node is constrained to exactly the node names
        # that exist in the graph (+ "WaitUser"). If the LLM tries to route
        # anywhere else, structured-output validation rejects it before it
        # ever reaches the router - instead of crashing the graph later.
        self.RoutingDecision = build_routing_decision(valid_destinations)
        self.orchestrator_llm = self.llm.with_structured_output(self.RoutingDecision)

        agents_list = "\n".join(f"- {name}" for name in valid_destinations)
        self.system_prompt = (
            "You are the IoT Deployment Orchestrator for an industrial environment.\n"
            "Analyze the operator's requests and the system state.\n"
            f"Available agents:\n{agents_list}\n"
            "If critical data is missing, select 'WaitUser' and ask for clarification.\n"
        )

    def __call__(self, state: IoTDeploymentState) -> dict:
        messages = [{"role": "system", "content": self.system_prompt}] + state["messages"]
        decision = self.orchestrator_llm.invoke(messages)

        return {
            "messages": [AIMessage(content=decision.response_to_user)],
            "next_node": decision.next_node,
        }
