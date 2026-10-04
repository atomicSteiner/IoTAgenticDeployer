"""Response agent: turns a whole turn into the single reply the architect
reads, in their language, with the code's renderings left untouched. It is
given facts (trace, model, validation) rather than asked to summarise prose."""

from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from pydantic import BaseModel, Field

from iot_agentic_deployer.agents.supervisor import MessageLanguage, SupervisorNode
from iot_agentic_deployer.domain.models import Installation
from iot_agentic_deployer.domain.state import IoTDeploymentState, trace
from iot_agentic_deployer.domain.validation import missing_topology_information


# The line each of the code's renderings opens with; from there on it is the
# model itself. A rendering missing here gets paraphrased by an LLM instead
RENDERED = ("**Installation:", "**Pre-deployment check:", "_No operation is required",
            "✅ All pre-deployment checks passed")


class Reply(BaseModel):
    account: str = Field(description=(
        "What you HAVE DONE this turn, in the language the architect wrote in. Say it "
        "in the first person and the active voice, as a colleague reporting back: 'I "
        "set the wellness profile and put a sensor in each classroom'. Never 'I will' "
        "or 'I am going to' - it has already happened - and never the passive 'the "
        "profile was set', which in most languages reads as officialese. Only what the "
        "facts state: no count, room or device that is not among them, and nothing "
        "about steps that did not run. Two or three sentences, no lists and no tables "
        "- the tables are added after you."))
    next_step: str = Field(description=(
        "What the ARCHITECT can usefully do next, worked out from the facts and the "
        "order of the stages below. One sentence, spoken to them: 'You can now "
        "assign devices to the classrooms' or 'Ask for a validation check when you "
        "are ready'. It is their move, never yours: no 'I can', no 'I will'. Never "
        "a stage the facts rule out: no deploying while there are errors, no "
        "devices before a use case. When there is genuinely nothing to suggest, say "
        "the configuration is where they left it."))


class ResponseNode:
    """Runs once per turn, after every agent the turn asked for."""

    def __init__(self, llm):
        self.reply_llm = llm.with_structured_output(Reply, method="function_calling")
        # Which language, established rather than asked for: told to answer
        # 'in the language the architect wrote in', the model answers in English
        self.language_llm = llm.with_structured_output(
            MessageLanguage, method="function_calling")

    def _language(self, asked: Optional[HumanMessage]) -> str:
        try:
            return self.language_llm.invoke(
                [{"role": "user", "content": asked.content if asked else "hello"}]).language
        except Exception:
            return "English"

    # -- what the turn produced -------------------------------------------

    @staticmethod
    def _turn(messages: list) -> list:
        """Everything said since the architect last spoke."""
        for position in range(len(messages) - 1, -1, -1):
            if isinstance(messages[position], HumanMessage):
                return messages[position + 1:]
        return list(messages)

    @staticmethod
    def _split(content: str) -> tuple[str, str]:
        """Prose, and what the code rendered. The rendering is the model itself
        - table, plan, findings - and never goes to a language model: it lands
        under the reply exactly as the code wrote it"""
        def opens_the_rendering(paragraph: str) -> bool:
            # An agent glues its own [Name] onto what it renders, so the marker
            # is not at the start: read as prose it cost the findings themselves
            named = paragraph.split("] ", 1)[-1] if paragraph.startswith("[") else paragraph
            return "|" in paragraph or named.startswith(RENDERED)

        paragraphs = (content or "").split("\n\n")
        rendered = next((i for i, p in enumerate(paragraphs)
                         if opens_the_rendering(p)), len(paragraphs))
        return ("\n\n".join(paragraphs[:rendered]).strip(),
                "\n\n".join(paragraphs[rendered:]).strip())

    # -- the facts --------------------------------------------------------

    @staticmethod
    def _facts(state: IoTDeploymentState, prose: list[str], steps: list[dict],
               answered: list[str] = ()) -> str:
        """Everything true about this turn, and nothing else: whatever is
        missing here the model supplies on its own, plausibly and wrongly -
        platforms with no adapter, room types that do not exist"""
        inst = Installation(**(state.get("installation") or {}))
        lines = [SupervisorNode._model_status(state)]

        if steps:
            lines.append("What ran this turn:")
            lines += [f"- {s['agent']} {s['action']}: {s['detail']}" for s in steps]
        else:
            # Spelled out: 'no stage ran' was read as room to report what the
            # architect asked for as though it had actually happened
            lines.append("No stage ran this turn: nothing was created, changed, removed "
                         "or deployed. Do not write that anything was - say what was "
                         "understood and what is in the model, and nothing more.")

        # Which rooms there are, by name: told only how many, the model invents
        # the ones it cannot name (a Classroom 3 that never existed)
        if inst.building:
            for floor in inst.building.floors:
                named = ", ".join(f"{s.name} ({s.type})" for s in floor.spaces[:15])
                lines.append(f"Rooms on {floor.name}: {named or 'none'}."
                             + (" And no others." if len(floor.spaces) <= 15 else ""))

        # Which spaces are still bare. Left unsaid it was invented: with every
        # space equipped, the architect was sent to 'the remaining spaces'
        if inst.building:
            bare = [space.name for _, space in inst.iter_spaces() if not space.devices]
            lines.append(f"{len(bare)} space(s) have no device"
                         + (f": {', '.join(bare[:8])}." if bare
                            else " - every space is equipped."))

        report = state.get("validation_report")
        if report:
            lines.append(f"Last check: {report['errors']} error(s), "
                         f"{report['warnings']} warning(s); "
                         f"deployable: {report['deployable']}.")

        plan = state.get("deployment_plan") or []
        waiting = [op for op in plan if op["status"] == "pending"]
        if waiting:
            lines.append(f"{len(waiting)} operation(s) planned on "
                         f"{inst.target.platform}, waiting to be confirmed.")

        if prose:
            lines.append("What the agents reported, to be said again, not added to:")
            lines += [f"- {p}" for p in prose]

        # Kept apart from the agents' reports: listed with them, 'noted' in an
        # answer that changed nothing was reported back as work done
        if answered:
            lines.append("What the supervisor answered. Nothing in the model changed "
                         "because of it, so none of it may be reported as done:")
            lines += [f"- {a}" for a in answered]

        # What is still missing, never what to do about it: the move is the
        # one thing this agent is now left to work out for itself
        missing = missing_topology_information(inst)
        if missing:
            lines.append("The description is still incomplete: " + "; ".join(missing))
        return "\n".join(lines)

    # -- the reply --------------------------------------------------------

    def _compose(self, state: IoTDeploymentState, asked: Optional[HumanMessage],
                 prose: list[str], steps: list[dict],
                 answered: list[str] = ()) -> Optional[Reply]:
        language = self._language(asked)
        try:
            return self.reply_llm.invoke([
                {"role": "system", "content": (
                    "You write the single reply an IoT architect reads at the end of a "
                    "turn. Speak from the facts below and from nothing else: they are "
                    "the stored model, not a claim made earlier in the conversation. "
                    "Tables and plans are added after you, so write none. "
                    f"Write both fields in {language}, whatever language the facts "
                    f"below happen to be written in - and write it as someone would "
                    f"say it, not as a word-for-word rendering of the facts. "
                    "The stages run in this order, each needing the one before "
                    "it: describe the building; choose a use case; assign devices "
                    "to spaces; ask for a validation check; deploy, which shows a "
                    "plan and waits for confirmation.")},
                {"role": "user", "content": f"They said: {asked.content if asked else ''}"},
                {"role": "user", "content": self._facts(state, prose, steps, answered)},
            ])
        except Exception:
            return None

    @staticmethod
    def _clean(said: str) -> str:
        """Models escape apostrophes that never needed escaping, and the
        backslash is what the architect ends up reading"""
        return said.strip().replace("\\'", "'")

    def __call__(self, state: IoTDeploymentState) -> dict:
        messages = state.get("messages") or []
        spoken = self._turn(messages)
        if not spoken:
            return {}

        prose, answered, rendered = [], [], []
        for message in spoken:
            head, tail = self._split(message.content)
            # The supervisor's answer is the model's from its first word to its
            # last: a table in it looks like a rendering and is not one, and
            # pasted under the reply it states a model the code never built
            if message.content.startswith("[Supervisor]"):
                if head:
                    answered.append(head)
                continue
            # Only what an agent reported, marked by its [Name] prefix: the
            # supervisor's hand-off line predates the work and misreports it
            if head.startswith("["):
                prose.append(head)
            if tail:
                rendered.append(tail)

        # Everything since this agent last ran is this turn - exact, where
        # counting messages backwards reached into the turn before
        history = state.get("trace") or []
        mine = max((i for i, s in enumerate(history) if s["agent"] == "Response"),
                   default=-1)
        steps = [s for s in history[mine + 1:]
                 if s["agent"] not in ("Supervisor", "Response")]

        reply = self._compose(state, next((m for m in reversed(messages)
                                           if isinstance(m, HumanMessage)), None),
                              prose, steps, answered)
        if reply is None:
            # A failed reply is no reason to lose what the agents said: leave
            # the turn as it is and add the next step they would have got
            hint = SupervisorNode._next_step_hint(state)
            return {"messages": [AIMessage(content=f"💡 Next: {hint}")]} if hint else {}

        # Only the last rendering: the earlier ones are the same model, older
        body = "\n\n".join(
            [self._clean(reply.account)] + rendered[-1:]
            + [f"💡 {self._clean(reply.next_step)}"])
        return {
            "messages": [RemoveMessage(id=m.id) for m in spoken] + [AIMessage(content=body)],
            "trace": trace("Response", "compose",
                           f"{len(spoken)} message(s) into one"),
        }
