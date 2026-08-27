"""Deployment agent.

Two phases, following the SAIBA split:

  planning     check the platform answers, work out the operations, show
               them, and stop there;
  realisation  once the architect confirms, run them and report back.

Execution can be picked up again where it left off: anything already done is
not repeated, so one failure can be sorted out and retried without throwing
away the work that succeeded
"""


from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from iot_agentic_deployer.domain.models import Installation
from iot_agentic_deployer.domain.state import IoTDeploymentState, trace
from iot_agentic_deployer.domain.validation import validate_installation
from iot_agentic_deployer.platforms.base import PlatformOperation, get_adapter
from iot_agentic_deployer.platforms.planning import derive_plan, plan_to_markdown
from iot_agentic_deployer.rendering.formatting import validation_to_markdown


class ConfirmationAnswer(BaseModel):
    # Worded flatly on purpose. Phrased as "'yes' ONLY when..." followed by a
    # list of refusals, a cautious model reads the caveats and answers no even
    # to the word 'confirm'
    answer: Literal["yes", "no"] = Field(description=(
        "yes = the message approves executing the plan now, in any language "
        "('confirm', 'yes', 'ok', 'go ahead', 'procedi'). "
        "no = it does not, because it refuses, postpones, asks something, or "
        "attaches a condition ('yes, but change the gateway first')."))


class DeploymentNode:

    def __init__(self, llm=None, adapter_options: dict | None = None):
        # Options passed to the adapter factory (endpoint, credentials).
        self.adapter_options = adapter_options or {}
        # The one thing this agent asks a model: whether the answer to its own
        # interruption point was a go-ahead. Planning, validation and execution
        # stay plain code. Two Literals, so the answer cannot be a third thing
        self.confirm_llm = llm and llm.with_structured_output(
            ConfirmationAnswer, method="function_calling")

    def __call__(self, state: IoTDeploymentState) -> dict:
        inst = Installation(**state.get("installation", {}))

        # Everything above the interrupt runs again when the architect
        # answers, so rules and operations are always re-established against
        # the model as it stands, not as it stood when the plan was shown
        planned = self._plan(inst)
        if "deployment_plan" not in planned:      # not valid, or unreachable
            return planned

        plan = self._carry_completed(
            [PlatformOperation(**op) for op in state.get("deployment_plan", [])],
            [PlatformOperation(**op) for op in planned["deployment_plan"]])

        # Suspends here with the plan; what comes back is whatever the
        # architect wrote next, and reading it is this agent's own business
        answer = interrupt({"operations": [op.model_dump() for op in plan],
                            "platform": inst.target.platform})
        if not self._confirms(answer):
            return {**planned, "messages": [AIMessage(content=(
                "[Deployment] Not executed. Nothing has been written.\n\n"
                + plan_to_markdown(plan)))],
                "trace": trace("Deployment", "not_executed", f"{len(plan)} operation(s)")}
        return {**planned, **self._realise(inst, plan)}

    def _confirms(self, answer) -> bool:
        """Anything that is not a plain go-ahead leaves the platform untouched,
        so an unreadable answer is a refusal rather than a gamble."""
        if not isinstance(answer, str) or not self.confirm_llm:
            return False
        return self.confirm_llm.invoke([
            {"role": "system", "content": (
                "The architect was shown a deployment plan and asked to confirm it. "
                "Their reply follows. Does it approve executing the plan now?")},
            {"role": "user", "content": answer},
        ]).answer == "yes"

    @staticmethod
    def _carry_completed(previous: list[PlatformOperation],
                         current: list[PlatformOperation]) -> list[PlatformOperation]:
        """Because op_ids come from the model, anything created under an
        earlier plan keeps the same identity in a freshly derived one: so
        work that's already done carries over instead of being attempted
        twice"""
        completed = {op.op_id: op for op in previous if op.status == "completed"}
        for op in current:
            if op.op_id in completed:
                op.status = "completed"
                op.result_ref = completed[op.op_id].result_ref
        return current

    # -- behaviour planning ----------------------------------------------

    def _plan(self, inst: Installation) -> dict:
        # Permission via validation is established against the model as it stands now
        report = validate_installation(inst)
        if not report["deployable"]:
            return {
                "validation_report": report,
                "messages": [AIMessage(content=(
                    "[Deployment] Not planned: the configuration does not satisfy the "
                    "rules in force.\n\n" + validation_to_markdown(report)))],
                "trace": trace("Deployment", "plan_refused", "configuration not valid"),
            }

        #then try to reach the platform
        adapter = get_adapter(inst.target.platform, **self.adapter_options)
        reachable, diagnostic = adapter.check_reachability()
        if not reachable:
            return {
                "messages": [AIMessage(content=(
                    f"[Deployment] Target platform '{inst.target.platform}' is not "
                    f"reachable, so nothing was planned.\n\n{diagnostic}"))],
                "trace": trace("Deployment", "reachability_failed", diagnostic),
            }

        #then derive the deployment plan
        plan = derive_plan(inst)
        return {
            "deployment_plan": [op.model_dump() for op in plan],
            # Record the check that let this plan through, not only the ones
            # that block it
            "validation_report": report,
            "messages": [AIMessage(content=(
                f"[Deployment] Planned operations on **{inst.target.platform}**:\n\n"
                f"{plan_to_markdown(plan)}\n\n"
                "Nothing has been written yet. Confirm to execute, or ask for changes."))],
            "trace": trace("Deployment", "plan_derived", f"{len(plan)} operation(s)"),
        }

    # -- behaviour realisation -------------------------------------------

    def _realise(self, inst: Installation, plan: list[PlatformOperation]) -> dict:
        adapter = get_adapter(inst.target.platform, **self.adapter_options)

        # Rebuild the reference table from what already completed, so a retry
        # can still resolve entities created on an earlier run
        refs = {op.op_id: op.result_ref for op in plan if op.status == "completed"}
        failure = None

        # One connection for the whole run, instead of one per operation
        with adapter.batch():
            for op in plan:
                #skip the completed operations
                if op.status == "completed":
                    continue
                try:
                    op.result_ref = adapter.execute(op, refs)
                    op.status = "completed"
                    op.error = None
                    refs[op.op_id] = op.result_ref
                except Exception as e:
                    op.status = "failed"
                    op.error = f"{type(e).__name__}: {e}"
                    failure = op
                    break   # stop at the first refusal; the rest stay pending

        done = sum(1 for op in plan if op.status == "completed")

        if failure is not None:
            content = (
                f"[Deployment] Interrupted after {done} of {len(plan)} operation(s).\n\n"
                f"Rejected: {failure.description}\n> {failure.error}\n\n"
                f"{plan_to_markdown(plan)}\n\n"
                "The entities already created were left untouched. Resolve the conflict "
                "(a different name, or reuse of the existing entity) and confirm again: "
                "execution resumes from the operation that failed."
            )
            status = "partial"
        else:
            content = (f"[Deployment] Completed on **{inst.target.platform}**: "
                       f"{done} operation(s) performed.\n\n{plan_to_markdown(plan)}")
            status = "success"

        return {
            "deployment_plan": [op.model_dump() for op in plan],
            "messages": [AIMessage(content=content)],
            "trace": trace("Deployment", "realise", f"{done}/{len(plan)} completed, status={status}"),
        }

