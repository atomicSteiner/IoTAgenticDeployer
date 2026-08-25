"""Orchestration graph

The supervisor and the specialised agents as nodes of a stateful graph: this
decides who runs when, and where the architect gets consulted. The flow goes
in circles on purpose and whatever was built up in the meantime survives the trip back.
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

    #build the graph once: agents, nodes, edges and the store they all share
    def __init__(self, db_path: str = "iot_agentic.db"):
        load_dotenv()
        self.llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            model=os.getenv("OPENAI_MODEL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=4096,
        )

        # Where the configuration lives: keeps the model and its checkpoints
        # across steps, and across sessions.
        conn = sqlite3.connect(db_path, check_same_thread=False)
        self.checkpointer = SqliteSaver(conn)

        adapter_options = {"mcp_url": os.getenv("THINGSBOARD_MCP_URL",
                                                "http://localhost:8000/sse")}

        # The only centralized list of specialised agents
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

        #the graph itself, then compiled with the checkpointer that makes it resumable
        self.workflow = StateGraph(IoTDeploymentState)
        self._setup_nodes()
        self._setup_edges()
        self.system = self.workflow.compile(checkpointer=self.checkpointer)

    #every agent becomes a node, addressed by its own name
    def _setup_nodes(self):
        for name, node in self.nodes.items():
            self.workflow.add_node(name, node)

    #where to go after the supervisor: an agent, or stop and wait for the architect
    def _router(self, state: IoTDeploymentState) -> str:
        destination = state.get("next_node", "WaitUser")
        return END if destination == "WaitUser" else destination

    #supervisor -> agent (chosen by the router), and every agent back to the supervisor
    def _setup_edges(self):
        self.workflow.set_entry_point("Supervisor")
        routing_map = {name: name for name in self.agents}
        routing_map[END] = END
        self.workflow.add_conditional_edges("Supervisor", self._router, routing_map)
        for name in self.agents:
            self.workflow.add_edge(name, "Supervisor")

    # -- session API ------------------------------------------------------

    #a session is just a thread_id: this is how the checkpointer is told which one
    def _config(self, session_id: str) -> dict:
        return {"configurable": {"thread_id": session_id}}

    #one full turn: feed the message in, let the graph run to a stop, return the last reply
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
        doing it is a step they take deliberately."""
        self.system.update_state(self._config(session_id), {"plan_confirmed": True})

    #the whole stored state of a session, or {} if it has never run
    def get_state_values(self, session_id: str) -> dict:
        snapshot = self.system.get_state(self._config(session_id))
        return snapshot.values if snapshot else {}

    #just the conversation out of that state
    def get_history(self, session_id: str) -> list:
        return self.get_state_values(session_id).get("messages", [])

    def delete_session(self, session_id: str) -> None:
        """Throws a configuration away(model, history and checkpoints"""
        self.checkpointer.delete_thread(session_id)