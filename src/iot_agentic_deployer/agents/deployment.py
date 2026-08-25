"""Deployment agent.

Two phases, following the SAIBA split:

  planning     check the platform answers, work out the operations, show
               them, and stop there;
  realisation  once the architect confirms, run them and report back.

Execution can be picked up again where it left off: anything already done is
not repeated, so one failure can be sorted out and retried without throwing
away the work that succeeded
"""


from langchain_core.messages import AIMessage

from iot_agentic_deployer.domain.models import Installation
from iot_agentic_deployer.domain.state import IoTDeploymentState, trace
from iot_agentic_deployer.domain.validation import validate_installation
from iot_agentic_deployer.platforms.base import PlatformOperation, get_adapter
from iot_agentic_deployer.platforms.planning import derive_plan, plan_to_markdown
from iot_agentic_deployer.rendering.formatting import validation_to_markdown


class DeploymentNode:

    def __init__(self, adapter_options: dict | None = None):
        # Options passed to the adapter factory (endpoint, credentials).
        self.adapter_options = adapter_options or {}

    def __call__(self, state: IoTDeploymentState) -> dict:
        inst = Installation(**state.get("installation", {}))
        plan = [PlatformOperation(**op) for op in state.get("deployment_plan", [])]
        confirmed = state.get("plan_confirmed", False)

        #there is a plan and confirmed
        if plan and confirmed:
            # What was confirmed is the plan the architect saw, not whatever
            # the model says by now.
            # Still it's necessary to check the rules and re-derive the operations
            # against the model as it stands. _plan already works this way
            # it just has to hold here too, where the writing actually happens
            report = validate_installation(inst)
            if not report["deployable"]:
                return {
                    "validation_report": report,
                    "plan_confirmed": False,
                    "messages": [AIMessage(content=(
                        "[Deployment] Not executed: the configuration no longer satisfies "
                        "the rules in force.\n\n" + validation_to_markdown(report)))],
                    "trace": trace("Deployment", "execution_refused", "configuration not valid"),
                }

            current = derive_plan(inst)
            if [op.op_id for op in current] != [op.op_id for op in plan]:
                return self._replan_stale(plan, current)

            return self._realise(inst, plan)

        #if there is no plan we formulate it
        if plan:
            return self._awaiting_confirmation(inst, plan)
        return self._plan(inst)

    # -- a plan exists but has not been confirmed -------------------------

    def _awaiting_confirmation(self, inst: Installation,
                               plan: list[PlatformOperation]) -> dict:
        """We're back at Deployment while a plan is still waiting.

        Confirming is a step the architect takes deliberately,
        and it goes through the confirm control rather than through the
        conversation"""
        report = validate_installation(inst)
        if not report["deployable"]:
            return {
                "validation_report": report,
                "messages": [AIMessage(content=(
                    "[Deployment] The pending plan can no longer be executed: the "
                    "configuration no longer satisfies the rules in force.\n\n"
                    + validation_to_markdown(report)))],
                "trace": trace("Deployment", "plan_refused", "configuration not valid"),
            }

        current = derive_plan(inst)
        if [op.op_id for op in current] == [op.op_id for op in plan]:
            done = sum(1 for op in plan if op.status == "completed")
            detail = (f"{len(plan)} operation(s)"
                      + (f", {done} already completed" if done else ""))
            return {
                "validation_report": report,
                "messages": [AIMessage(content=(
                    f"[Deployment] A plan of {detail} is ready and unchanged. Nothing was "
                    f"re-planned and nothing has been written.\n\n"
                    "Use the **▶️ Confirm and execute** button above the chat to run it — "
                    "execution is confirmed there rather than in conversation, so that "
                    "writing to the platform is always a deliberate step."))],
                "trace": trace("Deployment", "plan_pending", detail),
            }

        current = self._carry_completed(plan, current)
        return {
            "deployment_plan": [op.model_dump() for op in current],
            "validation_report": report,
            "plan_confirmed": False,
            "messages": [AIMessage(content=(
                f"[Deployment] The configuration changed since this plan was derived "
                f"({len(plan)} operation(s) then, {len(current)} now), so it has been "
                f"re-derived.\n\n{plan_to_markdown(current)}\n\n"
                "Nothing has been written. Use the **▶️ Confirm and execute** button "
                "above the chat to run it."))],
            "trace": trace("Deployment", "plan_rederived",
                                  f"{len(plan)} -> {len(current)} operation(s)"),
        }

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

    def _replan_stale(self, confirmed_plan: list[PlatformOperation],
                      current: list[PlatformOperation]) -> dict:
        """The model moved after the plan was confirmed. Nothing runs: show
        what the configuration means now and ask again, so whatever ends up
        written is always something they actually looked at."""
        done = sum(1 for op in confirmed_plan if op.status == "completed")
        note = ""
        if done:
            current = self._carry_completed(confirmed_plan, current)
            note = (f"\n\nThe {done} operation(s) already performed are preserved and will "
                    f"not be repeated.")

        return {
            "deployment_plan": [op.model_dump() for op in current],
            "plan_confirmed": False,
            "messages": [AIMessage(content=(
                f"[Deployment] Not executed: the configuration changed after this plan was "
                f"confirmed ({len(confirmed_plan)} operation(s) then, {len(current)} now).\n\n"
                f"{plan_to_markdown(current)}{note}\n\n"
                "Nothing has been written. Confirm again to execute the plan above."))],
            "trace": trace("Deployment", "plan_stale",
                                  f"{len(confirmed_plan)} -> {len(current)} operation(s)"),
        }

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
            "plan_confirmed": False,
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
                        "plan_confirmed": False,     # retrying means confirming again
            "messages": [AIMessage(content=content)],
            "trace": trace("Deployment", "realise", f"{done}/{len(plan)} completed, status={status}"),
        }

