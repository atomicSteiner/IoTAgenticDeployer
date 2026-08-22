"""Conceptualisation agent (thesis 5.4).

Turns the architect's description of a building into the topology model. Each
new description refines the same model rather than starting a new one, and
whatever the completeness rules say is still missing becomes a question to
ask back.
"""

import re
from typing import Literal, Optional, get_args

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from iot_agentic_deployer.domain.models import (
    AccessPoint, Accessibility, Building, Floor, Installation, Priority, Space, SpaceType,
)
from iot_agentic_deployer.domain.state import IoTDeploymentState, trace
from iot_agentic_deployer.domain.validation import missing_topology_information
from iot_agentic_deployer.rendering.formatting import installation_to_markdown


# Shapes used only for extraction, with no `devices` field on purpose.
# Equipment belongs to Configuration, which checks it against the catalogue
# (thesis 5.3, R2). Give the topology schema somewhere to put devices and a
# room description that happens to mention a TV, a microscope or a door will
# put them there - inventing equipment nobody asked for, under names that
# aren't in the catalogue, which Validation can then only reject.
class TopologyAccessPoint(BaseModel):
    name: str = Field(description="What this door or entrance is called, e.g. 'Main entrance'.")


class TopologySpace(BaseModel):
    name: str = Field(description="What this room is called, e.g. 'Classroom 1'.")
    type: SpaceType = "unknown"
    intended_use: Optional[str] = None
    accessibility: Accessibility = "unknown"
    priority: Priority = "normal"
    access_points: list[TopologyAccessPoint] = Field(default_factory=list)


class TopologyFloor(BaseModel):
    name: str = Field(description="What this floor is called, e.g. 'Ground floor'.")
    level: int = 0
    spaces: list[TopologySpace] = Field(default_factory=list)


class TopologyBuilding(BaseModel):
    # `building_name`, not `name`. Asked for a bare `name` the model answers
    # "name of what?" and fills in whatever is nearest to hand - the schema's
    # own title, or the use case the architect happened to mention - while
    # the real name ends up in location. Saying which name we mean in the
    # field itself is what stops that.
    building_name: str = Field(description=(
        "Just the building's name, as the architect said it - for example "
        "'Computer Science Department'. Not the installation, not the use "
        "case, not the name of this schema."))
    location: Optional[str] = Field(default=None, description=(
        "Where the building is: a town, campus or address. Leave empty unless "
        "the architect actually mentioned a place. Not the building's name."))
    installation_name: Optional[str] = Field(default=None, description=(
        "Only when the architect gives the installation itself a name of its "
        "own, separate from the building's - 'call the installation Wellness "
        "Pilot'. Leave empty otherwise; naming the building is not naming the "
        "installation."))
    floors: list[TopologyFloor] = Field(default_factory=list)


# What counts as the architect having mentioned a door, or a kind of room.
# The extractor is told to model only what it was given, and mostly ignores
# that: on wording that mentions no door at all it still hangs one off every
# room about five times in six. Since an access point is where a people
# counter ends up being installed, that means planning hardware onto doorways
# nobody has. So the text gets the final say (thesis 5.3, R2).
ACCESS_POINT_EVIDENCE = (
    r"doors?", r"doorways?", r"entrances?", r"entry", r"entries", r"exits?",
    r"passages?", r"gates?", r"access\s*points?", r"turnstiles?",
)

TYPE_EVIDENCE = {
    "classroom": (r"class\s*rooms?", r"lecture\s*(hall|room)s?"),
    "office": (r"offices?",),
    "laboratory": (r"laborator(y|ies)", r"\blabs?\b"),
    "meeting_room": (r"meeting\s*rooms?", r"conference\s*rooms?"),
    "corridor": (r"corridors?", r"hallways?"),
    "technical_room": (r"technical\s*rooms?", r"server\s*rooms?", r"plant\s*rooms?"),
    "common_area": (r"common\s*(area|room)s?", r"lounges?", r"canteens?",
                    r"foyers?", r"atriums?"),
    "outdoor": (r"outdoors?", r"outside", r"courtyards?", r"gardens?"),
}


def _mentioned(text: str, patterns) -> bool:
    return any(re.search(rf"\b{p}", text) for p in patterns)


def _ground(extracted: TopologyBuilding, said: str,
            previous: Optional[Building]) -> tuple[TopologyBuilding, int, int]:
    """Throws away anything the architect never actually said.

    Only their own words count, plus whatever the model already held - an
    earlier turn's structure is established, not invented. Returns the
    cleaned extraction and how much was dropped, so the trace can show it.
    """
    said = said.lower()
    known_types, known_aps = {}, set()
    if previous:
        for floor in previous.floors:
            for space in floor.spaces:
                known_types[(floor.name, space.name)] = space.type
                if space.access_points:
                    known_aps.add((floor.name, space.name))

    dropped_types = dropped_aps = 0
    for floor in extracted.floors:
        for space in floor.spaces:
            here = (floor.name, space.name)

            if (space.type != "unknown"
                    and not _mentioned(said, TYPE_EVIDENCE.get(space.type, ()))
                    and known_types.get(here) != space.type):
                space.type = "unknown"
                dropped_types += 1

            if (space.access_points
                    and not _mentioned(said, ACCESS_POINT_EVIDENCE)
                    and here not in known_aps):
                dropped_aps += len(space.access_points)
                space.access_points = []

    return extracted, dropped_types, dropped_aps


def _derived_installation_name(building_name: str) -> str:
    """What the installation gets called when nobody has named it."""
    return f"{building_name} installation"


def _to_topology(building: Building) -> TopologyBuilding:
    """The model as the extractor expects to see it - the spatial part only,
    with no devices, which is exactly what we want it refining."""
    return TopologyBuilding(
        building_name=building.name,
        location=building.location,
        floors=[
            TopologyFloor(name=f.name, level=f.level, spaces=[
                TopologySpace(
                    name=s.name, type=s.type, intended_use=s.intended_use,
                    accessibility=s.accessibility, priority=s.priority,
                    access_points=[TopologyAccessPoint(name=ap.name) for ap in s.access_points],
                )
                for s in f.spaces
            ])
            for f in building.floors
        ],
    )


def _to_building(extracted: TopologyBuilding, previous: Optional[Building]) -> Building:
    """Builds the real Building back out of the topology-only extraction,
    keeping whatever devices Configuration had already placed.

    Since extraction can't see devices, let alone emit them, describing the
    building again can't invent equipment or throw any away. All it can touch
    is the spatial structure."""
    prev_devices, prev_ap_devices = {}, {}
    if previous:
        for floor in previous.floors:
            for space in floor.spaces:
                prev_devices[(floor.name, space.name)] = space.devices
                for ap in space.access_points:
                    prev_ap_devices[(floor.name, space.name, ap.name)] = ap.devices

    return Building(
        name=extracted.building_name,
        # Models sometimes answer the literal string "null" rather than
        # leaving the field out, which would otherwise show up as a location.
        location=None if (extracted.location or "").strip().lower() in {"", "null", "none"}
                 else extracted.location,
        floors=[
            Floor(
                name=tf.name,
                level=tf.level,
                spaces=[
                    Space(
                        name=ts.name,
                        type=ts.type,
                        intended_use=ts.intended_use,
                        accessibility=ts.accessibility,
                        priority=ts.priority,
                        devices=prev_devices.get((tf.name, ts.name), []),
                        access_points=[
                            AccessPoint(
                                name=tap.name,
                                devices=prev_ap_devices.get((tf.name, ts.name, tap.name), []),
                            )
                            for tap in ts.access_points
                        ],
                    )
                    for ts in tf.spaces
                ],
            )
            for tf in extracted.floors
        ],
    )


class ConceptualisationIntent(BaseModel):
    action: Literal["describe_topology", "list_space_types"] = Field(description=(
        "'describe_topology' when the architect is describing, correcting or refining the "
        "building itself (floors, rooms, corridors, quantities). 'list_space_types' when "
        "they are only asking what room/space types or vocabulary is available, without "
        "describing anything new about their own building."
    ))


class ConceptualisationNode:

    def __init__(self, llm):
        # A cheap check on what the message is even asking, before we touch
        # the model. Extracting a topology rewrites every field of every
        # space from the whole conversation, so it must not run on a message
        # that wasn't describing the building at all - ask "what room types
        # can I have?" and it will happily go and re-guess types for rooms
        # the architect had already settled.
        # function_calling instead of json_schema for the reason spelled out
        # in supervisor.py.
        self.intent_llm = llm.with_structured_output(
            ConceptualisationIntent, method="function_calling")
        self.intent_prompt = (
            "Classify what the architect's latest message is doing with respect to the "
            "building topology."
        )

        # Constrained generation: no free text where a structure is expected
        # (thesis 5.3). Space types are a Literal, so the schema itself holds
        # the vocabulary - and with no `devices` field on TopologyBuilding,
        # equipment mentioned in passing has nowhere to land.
        self.modeller = llm.with_structured_output(TopologyBuilding, method="function_calling")
        self.system_prompt = (
            "Extract a structured building topology from the description given by the "
            "architect. You are modelling spatial structure only - never equipment: any "
            "furniture, fixture or device mentioned (a TV, a desk, a microscope...) is out "
            "of scope here and must be ignored.\n"
            "- Expand quantities into individually named spaces ('12 classrooms' -> 12 spaces).\n"
            "- Assign each space a type from the allowed vocabulary; use 'unknown' only when "
            "the description is genuinely ambiguous, never a plausible guess.\n"
            "- Model doors, entrances and passages as access points of their space.\n"
            "- Preserve the structure already established: refine the existing model rather "
            "than replacing it.\n"
        )

    def _extract(self, context: list, had_floors: bool) -> Optional[TopologyBuilding]:
        """Runs the extraction, and refuses to hand back a result that would
        throw the model away.

        The extractor is not reliable: given identical input it will sometimes
        return the building with every floor intact and sometimes return no
        floors at all. Writing that straight into the model wipes a topology
        the architect had already built - which is what made conceptualisation
        look like it kept coming back empty. So an empty answer is retried
        once, and if it comes back empty again the caller keeps what it had.

        Only applies when there was something to lose: on a first description
        an empty result is a real answer, and the completeness rules will ask
        about the floors soon enough."""
        extracted = self.modeller.invoke(context)
        if extracted.floors or not had_floors:
            return extracted

        extracted = self.modeller.invoke(context)
        if extracted.floors:
            return extracted
        return None

    def __call__(self, state: IoTDeploymentState) -> dict:
        inst = Installation(**state.get("installation", {}))

        intent = self.intent_llm.invoke(
            [{"role": "system", "content": self.intent_prompt}] + state["messages"])
        if intent.action == "list_space_types":
            types = ", ".join(f"`{t}`" for t in get_args(SpaceType) if t != "unknown")
            return {
                "messages": [AIMessage(content=(
                    "[Conceptualisation] Available space types: " + types + "."))],
                "trace": trace("Conceptualisation", "list_space_types"),
            }

        context = [{"role": "system", "content": self.system_prompt}] + state["messages"]
        if inst.building:
            # Show the current model in the shape we want back, field names
            # and all, so refining it doesn't mean translating between two
            # slightly different vocabularies.
            context.append({"role": "system", "content": "Model to refine:\n" + _to_topology(
                inst.building).model_dump_json()})

        extracted = self._extract(context, had_floors=bool(inst.building and inst.building.floors))
        if extracted is None:
            # Nothing usable came back and there is a model already worth
            # keeping. Say so and leave it alone.
            return {
                "messages": [AIMessage(content=(
                    "[Conceptualisation] I couldn't read a building out of that, so the model "
                    "is unchanged.\n\nCould you say it again - which floors, and which rooms "
                    "on each?\n\n" + installation_to_markdown(inst)))],
                "trace": trace("Conceptualisation", "extraction_failed", "model left as it was"),
            }

        # Only the architect's own words can justify a door or a room type.
        said = " ".join(m.content for m in state["messages"]
                        if isinstance(m, HumanMessage))
        extracted, dropped_types, dropped_aps = _ground(extracted, said, inst.building)

        previous_building_name = inst.building.name if inst.building else None
        inst.building = _to_building(extracted, inst.building)

        # The installation is named after the building until the architect
        # gives it a name of its own; from then on it keeps that name, and a
        # later building rename leaves it alone. "Still the derived name"
        # is how we tell the two apart - there is no flag to consult, and
        # deriving it once and freezing it (what used to happen here) left
        # the installation stuck with the building's old name for good.
        still_derived = {"unnamed-installation"}
        if previous_building_name:
            still_derived.add(_derived_installation_name(previous_building_name))

        if extracted.installation_name:
            inst.name = extracted.installation_name
        elif inst.name in still_derived:
            inst.name = _derived_installation_name(inst.building.name)

        # Rules decide what's missing, not the model's judgement - otherwise
        # a plausible-sounding default quietly stands in for a decision the
        # architect never actually made (thesis 5.3, R2).
        questions = missing_topology_information(inst)
        if questions:
            body = "\n".join(f"- {q}" for q in questions)
            content = ("[Conceptualisation] The description is not yet complete:\n\n"
                       f"{body}\n\n{installation_to_markdown(inst)}")
        else:
            content = ("[Conceptualisation] Topology model updated.\n\n"
                       + installation_to_markdown(inst))

        return {
            "installation": inst.model_dump(),
            "messages": [AIMessage(content=content)],
            "trace": trace("Conceptualisation", "update_topology",
                           f"{len(questions)} clarification(s) pending"
                           + (f", dropped {dropped_types} unstated type(s)" if dropped_types else "")
                           + (f", dropped {dropped_aps} unstated access point(s)" if dropped_aps else "")),
        }