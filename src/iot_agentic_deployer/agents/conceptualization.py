"""Conceptualisation agent

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
    building_name: str = Field(description=(
        "Just the building's name, as the architect said it - for example "
        "'Computer Science Department'. Not the installation, not the use "
        "case, not the name of this schema."))
    location: Optional[str] = Field(default=None, description=(
        "The town, campus or address, when the architect names one: 'the CS "
        "Building in Camerino' has location 'Camerino'. Empty only when no "
        "place is mentioned at all. Not the building's name."))
    installation_name: Optional[str] = Field(default=None, description=(
        "Only when the architect gives the installation itself a name of its "
        "own, separate from the building's - 'call the installation Wellness "
        "Pilot'. Leave empty otherwise; naming the building is not naming the "
        "installation."))
    floors: list[TopologyFloor] = Field(default_factory=list)


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


def _same_shape(extracted: TopologyBuilding, previous: Optional[Building]) -> bool:
    """True when the extraction has the same floors, and the same rooms per
    floor, as the model already held."""
    if not previous or len(previous.floors) != len(extracted.floors):
        return False
    return all(len(pf.spaces) == len(ef.spaces)
               for pf, ef in zip(previous.floors, extracted.floors))


def _ground(extracted: TopologyBuilding, said: str,
            previous: Optional[Building]) -> tuple[TopologyBuilding, int, int]:
    """Throws away anything the architect never actually said.

    Only their own words count, plus whatever the model already held(no inventions)
    Returns the cleaned extraction and how much was dropped, so the trace can show it.
    """

    #search for known types and access points, indexed by name AND by position
    said = said.lower()
    known_types, known_aps = {}, set()
    if previous:
        for fi, floor in enumerate(previous.floors):
            for si, space in enumerate(floor.spaces):
                for key in ((floor.name, space.name), (fi, si)):
                    known_types[key] = space.type
                    if space.access_points:
                        known_aps.add(key)
    by_position = _same_shape(extracted, previous)

    dropped_types = dropped_aps = 0
    for fi, floor in enumerate(extracted.floors):
        for si, space in enumerate(floor.spaces):
            #the name first; if it no longer matches it was renamed, so fall back on where it sits
            here = (floor.name, space.name)
            if here not in known_types and by_position:
                here = (fi, si)
            #a type different from unknown survive only if is mentioned in text
            if (space.type != "unknown"
                    and not _mentioned(said, TYPE_EVIDENCE.get(space.type, ()))
                    and known_types.get(here) != space.type):
                space.type = "unknown"
                dropped_types += 1
            #also the access points
            if (space.access_points
                    and not _mentioned(said, ACCESS_POINT_EVIDENCE)
                    and here not in known_aps):
                dropped_aps += len(space.access_points)
                space.access_points = []

    return extracted, dropped_types, dropped_aps


def _derived_installation_name(building_name: str) -> str:
    """What the installation gets called when nobody has named it"""
    return f"{building_name} installation"


def _to_topology(building: Building) -> TopologyBuilding:
    """The model as the extractor expects to see it: the spatial part only,
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
    keeping whatever devices Configuration had already placed"""
    prev_devices, prev_ap_devices = {}, {}
    if previous:
        #indexed by name AND by position, so a rename does not lose the room's devices
        for fi, floor in enumerate(previous.floors):
            for si, space in enumerate(floor.spaces):
                for key in ((floor.name, space.name), (fi, si)):
                    prev_devices[key] = space.devices
                    for ai, ap in enumerate(space.access_points):
                        prev_ap_devices[(*key, ap.name)] = ap.devices
                        prev_ap_devices[(*key, ai)] = ap.devices
    by_position = _same_shape(extracted, previous)

    def carried(fi, si, tf, ts, ai=None, tap=None):
        """The devices the previous model held here: found by name, or by
        position when the name changed under a rename."""
        for key in ((tf.name, ts.name), (fi, si) if by_position else None):
            if key is None:
                continue
            if tap is None:
                if key in prev_devices:
                    return prev_devices[key]
            else:
                for leaf in (tap.name, ai):
                    if (*key, leaf) in prev_ap_devices:
                        return prev_ap_devices[(*key, leaf)]
        return []

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
                        devices=carried(fi, si, tf, ts),
                        access_points=[
                            AccessPoint(name=tap.name,
                                        devices=carried(fi, si, tf, ts, ai, tap))
                            for ai, tap in enumerate(ts.access_points)
                        ],
                    )
                    for si, ts in enumerate(tf.spaces)
                ],
            )
            for fi, tf in enumerate(extracted.floors)
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
        # the model
        self.intent_llm = llm.with_structured_output(
            ConceptualisationIntent, method="function_calling")
        self.intent_prompt = (
            "Classify what the architect's latest message is doing with respect to the "
            "building topology."
        )

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
        The extraction is repeated for better results"""
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
            # Show the current model in the shape we want back
            context.append({"role": "system", "content": "Model to refine:\n" + _to_topology(
                inst.building).model_dump_json()})

        extracted = self._extract(context, had_floors=bool(inst.building and inst.building.floors))
        if extracted is None:
            # Nothing usable came back so let's keep the previous model
            return {
                "messages": [AIMessage(content=(
                    "[Conceptualisation] I couldn't read a building out of that, so the model "
                    "is unchanged.\n\nCould you say it again - which floors, and which rooms "
                    "on each?\n\n" + installation_to_markdown(inst)))],
                "trace": trace("Conceptualisation", "extraction_failed", "model left as it was"),
            }

        # Only the architect's own words can justify a door or a room type
        said = " ".join(m.content for m in state["messages"]
                        if isinstance(m, HumanMessage))
        extracted, dropped_types, dropped_aps = _ground(extracted, said, inst.building)

        previous_building_name = inst.building.name if inst.building else None
        devices_before = len(inst.all_devices())
        inst.building = _to_building(extracted, inst.building)
        # Devices are meant to survive a re-description untouched. If any did
        # not, old and new could not be matched up - a rename the position
        # fallback could not cover - and that must not pass in silence, which
        # is exactly how this went unnoticed for so long
        orphaned = devices_before - len(inst.all_devices())

        # The installation is named after the building until the architect
        # gives it a name of its own; from then on it keeps that name, and a
        # later building rename leaves it alone.
        still_derived = {"unnamed-installation"}
        if previous_building_name:
            still_derived.add(_derived_installation_name(previous_building_name))

        if extracted.installation_name:
            inst.name = extracted.installation_name
        elif inst.name in still_derived:
            inst.name = _derived_installation_name(inst.building.name)

        # Rules decide what's missing, not the model's judgement
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
                           + (f", dropped {dropped_aps} unstated access point(s)" if dropped_aps else "")
                           + (f", {orphaned} device(s) orphaned by a rename" if orphaned > 0 else "")),
        }