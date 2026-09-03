"""Supervisor: works out which stage of the activity the architect is asking
for, hands off to it and turns the result back into one reply. The only agent
that talks to the architect directly."""

from typing import Literal, Optional, get_args

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from iot_agentic_deployer.domain.models import Installation, SpaceType
from iot_agentic_deployer.domain.state import IoTDeploymentState, build_routing_decision, trace
from iot_agentic_deployer.domain.validation import missing_topology_information
from iot_agentic_deployer.platforms.base import available_platforms

# The whole room vocabulary, so an answer about it cannot be invented
SPACE_TYPES = tuple(t for t in get_args(SpaceType) if t != "unknown")

# A message asks for one or two things; a longer list is the model running away
# with it rather than reading it
MAX_STAGES = 3

AGENT_PURPOSE = {
    "Conceptualisation": "translates the description of the environment into the topology "
                         "model and asks for what is missing. It also owns the building's "
                         "name: naming or renaming it goes here, never anywhere else.",
    "Configuration": "holds the device catalogue and answers what it offers, applies the "
                     "use-case profile, associates devices with spaces or access points, "
                     "models a door on a room that already exists, and sets which platform "
                     "the installation is to be deployed onto.",
    "Summary": "renders the configuration built so far, in tabular and diagrammatic form.",
    "Validation": "evaluates the topology and profile rules and pronounces on whether the "
                  "model may proceed to deployment.",
    "Deployment": "verifies platform reachability, derives the provisioning operations and "
                  "executes them once confirmed.",
}

GUIDANCE = (
    "Stages of the activity: describe, rename or correct the building -> Conceptualisation; "
    "choose, add, remove or change the use case, the devices or the target platform "
    "-> Configuration; review -> Summary; "
    "Doors, entrances and passages belong to both: a description of the building that "
    "mentions them is Conceptualisation, but adding or removing one on rooms already "
    "modelled ('give every classroom a back door') is Configuration, which does it "
    "without re-reading the whole building. "
    "check before deploying -> "
    "Validation; deploy -> Deployment.\n"
    "Asking to deploy is a request for Deployment, and for Deployment alone: it checks the "
    "rules itself, derives the plan, shows it and waits there for the architect to confirm "
    "before writing anything. Do not send a Validation ahead of it - that is the architect's "
    "to ask for, and it is not what they asked for. If the message also names a target "
    "platform other than the current one, Configuration goes first to set it, then "
    "Deployment: deploying is done on the platform of the model, not on one named in "
    "passing.\n"
    "Do not re-delegate a stage merely to repeat what it has already produced. A message "
    "asking that stage for something new is a new request, and goes to it like any other: "
    "more devices, a different profile, another target platform. That a stage has run "
    "before is no reason to withhold what is being asked of it now.\n"
    "Delegate to exactly the stage(s) the architect's message actually asks for, in the "
    "order they run - never a stage they did not request, even if it would logically come "
    "next (e.g. describing the building is not a request to also pick a use case or assign "
    "devices). Most messages ask for one stage; name a second only when the message asks "
    "for that too ('assign the sensors, then validate'). When in doubt, choose WaitUser "
    "and ask, rather than guessing ahead on the architect's behalf.\n"
    "A question about the catalogue - which devices exist, which are available, what the "
    "profile recommends, what a given device measures or what it is installed on - is a "
    "request for Configuration, which holds the catalogue and answers out of it. Never "
    "answer one yourself: the Model status below does not list the devices, so anything "
    "you say about them is invented.\n"
    "If the architect is asking a question about the process itself - what to do next, "
    "which stage comes next, for advice or an opinion - rather than describing, deciding "
    "or requesting a stage, that is not a delegation: choose WaitUser and answer the "
    "question directly, grounded in the Model status below.\n"
)


# Which language the architect writes in. Established rather than asked for,
# and read by the agent that composes the reply
class MessageLanguage(BaseModel):
    # A Literal: left free, 'switch the platform to openremote' came back as
    # language 'OpenRemote'. Anything outside the list falls back to English
    language: Literal[
        "English", "Italian", "Spanish", "French", "German", "Portuguese", "Dutch"
    ] = Field(description="The language this message is written in.")


class SupervisorNode:

    def __init__(self, llm, valid_destinations: list[str], router_llm=None):
        self.llm = llm
        self.valid_destinations = valid_destinations
        self.supervisor_llm = (router_llm or llm).with_structured_output(
            build_routing_decision(valid_destinations), method="function_calling")

        agents = "\n".join(f"- {n}: {AGENT_PURPOSE.get(n, '')}" for n in valid_destinations)
        self.system_prompt = (
            "You are the supervisor of an assistant that helps an IoT architect conceptualise "
            "a building and configure an IoT installation on it. You guide the decisions of "
            "the architect, you do not replace them.\n\n"
            f"Specialised agents:\n{agents}\n\n{GUIDANCE}"
            "If critical information is missing, choose 'WaitUser' and ask for it.\n"
            "Write response_to_user in the language the architect is writing in.\n\n"
            "You never perform a stage yourself and you never describe an agent's outcome in "
            "response_to_user - only the agent that actually runs may report what it did, in "
            "its own message, on a later step. Your response_to_user must describe only what "
            "is true right now: if next_nodes names an agent, say that you are handing off to "
            "it, not what it will find or produce. Never write 'I will now create/select/"
            "configure...' or 'I have created/configured...' - you do not create or configure "
            "anything.\n"
            "The 'Model status' message below is ground truth from the actual stored model, "
            "not from what earlier messages in this conversation claimed. If it says the "
            "building is not yet modelled, delegate to Conceptualisation even if an earlier "
            "assistant message claimed a topology had already been built - that claim was "
            "never backed by an actual model update.\n"
        )

    @staticmethod
    def _model_status(state: IoTDeploymentState) -> str:
        inst = Installation(**(state.get("installation") or {}))
        if not inst.building:
            return "Model status: no building has been modelled yet."
        spaces = sum(1 for _ in inst.iter_spaces())
        return (
            f"Model status: building '{inst.building.name}' modelled with "
            f"{len(inst.building.floors)} floor(s) and {spaces} space(s); "
            f"use case: {inst.use_case or 'not selected'}; "
            f"{len(inst.all_devices())} device(s) assigned; "
            # Nothing else states the target, and asked about it the
            # supervisor invents one
            f"target platform: {inst.target.platform} "
            f"(adapters available: {', '.join(available_platforms())}). "
            # Asked which room types exist it answered 'reception, restroom,
            # storage', none of which the model has
            f"Room types that exist: {', '.join(SPACE_TYPES)}. There are no others."
        )

    @staticmethod
    def _next_step_hint(state: IoTDeploymentState) -> Optional[str]:
        """One suggestion for what to do next, read straight off the stored
        state like Model status is"""
        inst = Installation(**(state.get("installation") or {}))

        if not inst.building:
            return "Describe the building to get started."
        if missing_topology_information(inst):
            return None  # Conceptualisation's own message already asks

        if not inst.use_case:
            return ("Choose a use case (e.g. wellness, seismic safety) so Configuration can "
                     "recommend devices.")
        if not inst.all_devices():
            return "Assign devices to spaces, e.g. 'add temperature sensors to all classrooms'."

        report = state.get("validation_report")
        if not report:
            return "Ask for a validation check before deploying."
        if not report.get("deployable"):
            return f"Resolve the {report.get('errors', 0)} error(s) above before deploying."

        plan = state.get("deployment_plan")
        if not plan:
            return "The configuration is deployable - ask to deploy when ready."
        if any(op["status"] == "pending" for op in plan):
            return "Review the planned operations and confirm to execute them."

        return None

    def _answer(self, history: list, status: str) -> str:
        """A message that asks for no stage still deserves an answer, and only
        the model can give one. A prompt of its own, not the routing one, which
        orders it to announce a hand-off - and it announced one here too"""
        return self.llm.invoke(
            [{"role": "system", "content": (
                "You help an IoT architect conceptualise a building and configure an IoT "
                "installation on it. Answer their last message directly, in the language "
                "they wrote it in, from this conversation and the model status below. "
                "Nothing whatever happens as a result of this message: no agent runs, "
                "nothing is created, changed or removed. So do not announce handing over "
                "to anyone, do not report anything as done, and do not promise anything "
                "either - no 'I will remove...', no 'I am adding...'. If they asked for "
                "something to be done, say plainly that it has not been done and ask them "
                "to say it again.")}]
            + history + [{"role": "system", "content": status}]).content

    @staticmethod
    def _queue(decision) -> list[str]:
        """The stages to run, in order, out of a decision that may be missing
        altogether. Deployment always ends the queue: it stops on its own
        interrupt, so anything behind it would run off the confirmation"""
        if decision is None:
            return []
        queue = [node for node in decision.next_nodes if node != "WaitUser"][:MAX_STAGES]
        if "Deployment" in queue:
            queue = queue[:queue.index("Deployment") + 1]
        return queue

    def __call__(self, state: IoTDeploymentState) -> dict:
        history = state["messages"]

        # If the last message is an agent's rather than the architect's, an
        # agent has just run and we're on the way back
        if history and not isinstance(history[-1], HumanMessage):
            # Whatever else the message asked for was decided on the way out and
            # waits here: deciding twice is how work that never ran got reported
            pending = state.get("pending_nodes") or []
            if pending:
                return {"next_node": pending[0], "pending_nodes": pending[1:],
                        "trace": trace("Supervisor", "delegate", pending[0])}

            result = {
                "next_node": "WaitUser",
                "trace": trace("Supervisor", "delegate", "WaitUser"),
            }
            # No message of its own here: Response composes the whole turn,
            # the hint included, once every agent it asked for has run
            return result

        status = self._model_status(state)
        messages = (
            [{"role": "system", "content": self.system_prompt}]
            + history
            + [{"role": "system", "content": status}]
        )
        decision = self.supervisor_llm.invoke(messages)
        queue = self._queue(decision)

        # No destination is sometimes right and sometimes a miss: one message
        # in six that plainly asks for a stage came back empty. Worth a retry
        if not queue:
            second = self.supervisor_llm.invoke(messages)
            if self._queue(second):
                decision, queue = second, self._queue(second)

        # Still nobody to delegate to, so response_to_user cannot be shown: it
        # announces a hand-off that is not happening. Answer afresh instead
        if not queue:
            return {
                # Named, unlike the hand-off line below: Response knows an
                # agent's prose by that prefix and drops anything without one
                "messages": [AIMessage(
                    content=f"[Supervisor] {self._answer(history, status)}")],
                "next_node": "WaitUser",
                "trace": trace("Supervisor", "answer", "no stage requested"),
            }

        return {
            # Kept for the trace and for what follows to read; Response takes
            # it out of the transcript, so there is nothing to translate
            "messages": [AIMessage(content=decision.response_to_user)],
            "next_node": queue[0],
            "pending_nodes": queue[1:],
            "trace": trace("Supervisor", "delegate", " -> ".join(queue)),
        }
