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
from langgraph.types import Command



# Importing this runs its @register_adapter, which is what makes the platform
# selectable by name through get_adapter/available_platforms.
from iot_agentic_deployer.platforms import openremote, thingsboard  # noqa: F401


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
            # The largest thing we ever ask for is a topology: ~2k tokens for
            # five floors, ~9k for a very large building. You pay for what is
            # generated, not for the ceiling, so this only has to be high
            # enough not to cut an extraction in half
            max_tokens=8192,
        )

        # Routing is one small call per turn, and now breaks a message into
        # an ordered list of stages: worth a model of its own when the cheap
        # one struggles. Unset, it is the same model as everything else
        self.router_llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            model=os.getenv("ROUTER_MODEL") or os.getenv("OPENAI_MODEL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=1024,
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
            "Deployment": DeploymentNode(llm=self.llm, adapter_options=adapter_options),
        }
        self.nodes = {
            "Supervisor": SupervisorNode(llm=self.llm, router_llm=self.router_llm,
                                         valid_destinations=list(self.agents)),
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
    def run_turn(self, session_id: str, user_input: str, on_step=None) -> str:
        return self._drain(self.system.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=self._config(session_id), stream_mode="values"), on_step)

    @staticmethod
    def _drain(events, on_step) -> str:
        """Runs the graph to its next stop, handing each step to `on_step` as
        it lands: a turn is several agents long now, and the caller has to be
        able to say where it has got to"""
        final, reported = None, None
        for final in events:
            steps = final.get("trace") or []
            if on_step:
                # The first event is the state before anything ran, so its
                # last entry is the previous turn's. Report what this turn
                # adds, and all of it: one agent can log several steps at once
                reported = len(steps) if reported is None else reported
                for entry in steps[reported:]:
                    on_step(entry)
                reported = len(steps)
        return final["messages"][-1].content

    #answers the interrupt the deployment agent is waiting on, and lets the graph carry on
    def resume(self, session_id: str, value, on_step=None) -> str:
        return self._drain(self.system.stream(Command(resume=value),
                                              config=self._config(session_id),
                                              stream_mode="values"), on_step)

    #the payload of the interrupt waiting on the architect, if there is one
    def pending_approval(self, session_id: str) -> dict | None:
        snapshot = self.system.get_state(self._config(session_id))
        return snapshot.interrupts[0].value if snapshot.interrupts else None

    def message_times(self, session_id: str) -> dict[str, str]:
        """When each message first appeared, taken from the checkpoint that
        first held it. Nothing is written down at the time: conversations
        saved before this existed get their times too"""
        times = {}
        for snapshot in reversed(list(self.system.get_state_history(self._config(session_id)))):
            for message in snapshot.values.get("messages", []):
                times.setdefault(message.id, snapshot.created_at)
        return times

    #the whole stored state of a session, or {} if it has never run
    def get_state_values(self, session_id: str) -> dict:
        snapshot = self.system.get_state(self._config(session_id))
        return snapshot.values if snapshot else {}

    def delete_session(self, session_id: str) -> None:
        """Throws a configuration away(model, history and checkpoints"""
        self.checkpointer.delete_thread(session_id)