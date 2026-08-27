"""Supervisor

Works out which stage of the activity the architect is asking for, hands off
to it, and turns the result back into one reply. The only agent that talks to
the architect directly.
"""

from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from iot_agentic_deployer.domain.models import Installation
from iot_agentic_deployer.domain.state import IoTDeploymentState, build_routing_decision, trace
from iot_agentic_deployer.domain.validation import missing_topology_information

# A message asks for one or two things; a longer list is the model running away
# with it rather than reading it
MAX_STAGES = 3

AGENT_PURPOSE = {
    "Conceptualisation": "translates the description of the environment into the topology "
                         "model and asks for what is missing.",
    "Configuration": "applies the use-case profile, consults the catalogue and associates "
                     "devices with spaces or access points.",
    "Summary": "renders the configuration built so far, in tabular and diagrammatic form.",
    "Validation": "evaluates the topology and profile rules and pronounces on whether the "
                  "model may proceed to deployment.",
    "Deployment": "verifies platform reachability, derives the provisioning operations and "
                  "executes them once confirmed.",
}

GUIDANCE = (
    "Stages of the activity: describe the building -> Conceptualisation; choose the use case "
    "and the devices -> Configuration; review -> Summary; check before deploying -> "
    "Validation; deploy -> Deployment.\n"
    "Asking to deploy is a request for Deployment, and for Deployment alone: it checks the "
    "rules itself, derives the plan, shows it and waits there for the architect to confirm "
    "before writing anything. Do not send a Validation ahead of it - that is the architect's "
    "to ask for, and it is not what they asked for.\n"
    "Do not re-delegate a stage that already produced a result, unless the architect asks "
    "to change or refresh something.\n"
    "Delegate to exactly the stage(s) the architect's message actually asks for, in the "
    "order they run - never a stage they did not request, even if it would logically come "
    "next (e.g. describing the building is not a request to also pick a use case or assign "
    "devices). Most messages ask for one stage; name a second only when the message asks "
    "for that too ('assign the sensors, then validate'). When in doubt, choose WaitUser "
    "and ask, rather than guessing ahead on the architect's behalf.\n"
    "If the architect is asking a question about the process itself - what to do next, what "
    "options they have, for advice or an opinion - rather than describing, deciding or "
    "requesting a stage, that is not a delegation: choose WaitUser and answer the question "
    "directly, grounded in the Model status below.\n"
)


# Naming the language and translating are two calls, not one: shown the
# architect's message and the suggestion together, the model rewrites the
# suggestion into their request, or answers it outright. Split, the sentence
# being translated never sees the message at all
class MessageLanguage(BaseModel):
    language: str = Field(description=(
        "The language this message is written in, named in English "
        "('Italian', 'Spanish', 'English')."))


# Plain text, not a structured field
TRANSLATOR = ("You are a translator. Reply with the translation and nothing else: no "
              "preamble, no alternatives, no quotation marks, no explanation.")


class SupervisorNode:

    def __init__(self, llm, valid_destinations: list[str], router_llm=None):
        self.llm = llm
        self.language_llm = llm.with_structured_output(
            MessageLanguage, method="function_calling")
        self.valid_destinations = valid_destinations
        self.supervisor_llm = (router_llm or llm).with_structured_output(
            build_routing_decision(valid_destinations), method="function_calling")

        agents = "\n".join(f"- {n}: {AGENT_PURPOSE.get(n, '')}" for n in valid_destinations)
        self.system_prompt = (
            "You are the supervisor of an assistant that helps an IoT architect conceptualise "
            "a building and configure an IoT installation on it. You guide the decisions of "
            "the architect, you do not replace them.\n\n"
            f"Specialised agents:\n{agents}\n\n{GUIDANCE}"
            "If critical information is missing, choose 'WaitUser' and ask for it.\n\n"
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
            f"{len(inst.all_devices())} device(s) assigned."
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

    def _phrase(self, hint: str, history: list) -> str:
        """Which step comes next is still read off the stored model; only the
        wording is the LLM's, so the suggestion comes back in whatever
        language the architect is writing in"""
        asked = next((m for m in reversed(history) if isinstance(m, HumanMessage)), None)
        try:
            language = self.language_llm.invoke(
                [{"role": "user", "content": asked.content if asked else "hello"}]).language
            if language.strip().lower().startswith("english"):
                return hint
            said = self.llm.invoke([
                {"role": "system", "content": TRANSLATOR},
                {"role": "user", "content": f"Translate into {language}: {hint}"},
            ]).content.strip()
            return said or hint
        except Exception:
            return hint     # a hint is not worth losing the turn over

    def __call__(self, state: IoTDeploymentState) -> dict:
        history = state["messages"]

        # If the last message is an agent's rather than the architect's, an
        # agent has just run and we're on the way back
        if history and not isinstance(history[-1], HumanMessage):
            # An agent has just finished. Whatever else the architect's message
            # asked for was decided on the way out and is waiting here, so take
            # it rather than ask a model again: deciding twice about the same
            # message is how the supervisor used to report work that never ran
            pending = state.get("pending_nodes") or []
            if pending:
                return {"next_node": pending[0], "pending_nodes": pending[1:],
                        "trace": trace("Supervisor", "delegate", pending[0])}

            result = {
                "next_node": "WaitUser",
                "trace": trace("Supervisor", "delegate", "WaitUser"),
            }
            hint = self._next_step_hint(state)
            if hint:
                result["messages"] = [AIMessage(
                    content=f"💡 Next: {self._phrase(hint, history)}")]
            return result

        messages = (
            [{"role": "system", "content": self.system_prompt}]
            + history
            + [{"role": "system", "content": self._model_status(state)}]
        )
        decision = self.supervisor_llm.invoke(messages)

        queue = [node for node in decision.next_nodes if node != "WaitUser"][:MAX_STAGES]
        # Deployment stops on its own interrupt: anything queued behind it would
        # run off the back of the architect's confirmation, unasked
        if "Deployment" in queue:
            queue = queue[:queue.index("Deployment") + 1]

        return {
            "messages": [AIMessage(content=decision.response_to_user)],
            "next_node": queue[0] if queue else "WaitUser",
            "pending_nodes": queue[1:],
            "trace": trace("Supervisor", "delegate", " -> ".join(queue) or "WaitUser"),
        }