"""Orchestration graph (thesis 5.5).

The supervisor and the specialised agents as nodes of a stateful graph: this
decides who runs when, and where the architect gets consulted. The flow goes
in circles on purpose - asking for clarification, or validating, both send
the activity back to a stage it has already been through (thesis 5.6.2) - and
whatever was built up in the meantime survives the trip back.
"""

import os
import sqlite3

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph



# Importing this runs its @register_adapter, which is what makes the platform
# selectable by name through get_adapter/available_platforms.
from iot_agentic_deployer.platforms import thingsboard  # noqa: F401


from iot_agentic_deployer.agents.conceptualization import ConceptualisationNode
from iot_agentic_deployer.agents.configuration import ConfigurationNode
from iot_agentic_deployer.agents.deployment import DeploymentNode
from iot_agentic_deployer.agents.summary import SummaryNode
from iot_agentic_deployer.agents.supervisor import SupervisorNode
from iot_agentic_deployer.agents.validation import ValidationNode
from iot_agentic_deployer.domain.state import IoTDeploymentState


class IoTAgenticWorkflow:

    def __init__(self, db_path: str = "iot_agentic.db"):
        load_dotenv()
        self.llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            model=os.getenv("OPENAI_MODEL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            # Everything we ask for is a routing decision, an intent, or a
            # topology - a few thousand tokens at most. Leave this unset and
            # the client asks for the model's full 65536-token ceiling every
            # time, which OpenRouter refuses outright once the balance cannot
            # cover it, even for a call that would have taken 200 tokens.
            max_tokens=4096,
        )

        # Where the configuration lives: keeps the model and its checkpoints
        # across steps, and across sessions (R7).
        conn = sqlite3.connect(db_path, check_same_thread=False)
        self.checkpointer = SqliteSaver(conn)

        adapter_options = {"mcp_url": os.getenv("THINGSBOARD_MCP_URL",
                                                "http://localhost:8000/sse")}

        # The one list of specialised agents. The routing map, the edges back
        # to the supervisor, and the destinations it is allowed to pick all
        # come from this dictionary.
        self.agents = {
            "Conceptualisation": ConceptualisationNode(llm=self.llm),
            "Configuration": ConfigurationNode(llm=self.llm),
            "Summary": SummaryNode(),
            "Validation": ValidationNode(),
            "Deployment": DeploymentNode(adapter_options=adapter_options),
        }
        self.nodes = {
            "Supervisor": SupervisorNode(llm=self.llm, valid_destinations=list(self.agents)),
            **self.agents,
        }

        self.workflow = StateGraph(IoTDeploymentState)
        self._setup_nodes()
        self._setup_edges()
        self.system = self.workflow.compile(checkpointer=self.checkpointer)

    def _setup_nodes(self):
        for name, node in self.nodes.items():
            self.workflow.add_node(name, node)

    def _router(self, state: IoTDeploymentState) -> str:
        destination = state.get("next_node", "WaitUser")
        return END if destination == "WaitUser" else destination

    def _setup_edges(self):
        self.workflow.set_entry_point("Supervisor")
        routing_map = {name: name for name in self.agents}
        routing_map[END] = END
        self.workflow.add_conditional_edges("Supervisor", self._router, routing_map)
        for name in self.agents:
            self.workflow.add_edge(name, "Supervisor")

    # -- session API ------------------------------------------------------

    def _config(self, session_id: str) -> dict:
        return {"configurable": {"thread_id": session_id}}

    def run_turn(self, session_id: str, user_input: str) -> str:
        final = None
        for event in self.system.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=self._config(session_id), stream_mode="values",
        ):
            final = event
        return final["messages"][-1].content

    def confirm_plan(self, session_id: str) -> None:
        """Notes that the architect confirmed. Going from planning to actually
        doing it is a step they take deliberately (thesis 5.4)."""
        self.system.update_state(self._config(session_id), {"plan_confirmed": True})

    def get_state_values(self, session_id: str) -> dict:
        snapshot = self.system.get_state(self._config(session_id))
        return snapshot.values if snapshot else {}

    def get_history(self, session_id: str) -> list:
        return self.get_state_values(session_id).get("messages", [])

    def delete_session(self, session_id: str) -> None:
        """Throws a configuration away for good - model, history and
        checkpoints. No undo, so the UI asks first."""
        self.checkpointer.delete_thread(session_id)