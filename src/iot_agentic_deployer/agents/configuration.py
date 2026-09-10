"""Configuration agent: applies the use-case profile, reads the catalogue and
puts devices into spaces."""

from typing import Literal, Optional, Type

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field, ValidationError, create_model

from iot_agentic_deployer.domain.catalog.defaults import (
    device_instance_name, find_gateway, resolve_default_metadata,
)
from iot_agentic_deployer.domain.catalog.loader import (
    load_device_catalog, load_use_cases, devices_for_use_case, profile_rules,
)
from iot_agentic_deployer.domain.catalog.matching import match_device_types
from iot_agentic_deployer.domain.models import AccessPoint, Installation, Exclusion, Device
from iot_agentic_deployer.domain.state import IoTDeploymentState, trace
from iot_agentic_deployer.platforms.base import available_platforms
from iot_agentic_deployer.rendering.formatting import installation_to_markdown

ACTIONS = (
    "list_catalog", "select_use_case", "assign_devices", "remove_devices",
    "set_metadata", "record_exclusion", "select_platform",
    "add_access_point", "remove_access_point",
)

# What a door gets called when the architect asks for one without naming it
DEFAULT_ACCESS_POINT = "Entrance"

# A message asks the configuration for one or two things; a longer list is the
# model reading intent into the conversation rather than into the message
MAX_ACTIONS = 3

# A room takes a few devices of a kind, not fifty: a larger number is a quantity
# read out of the building rather than out of the message
MAX_PER_SPACE = 20


def metadata_keys() -> list[str]:
    """Read from the catalogue, so a new device type brings its own fields."""
    return sorted({key for spec in load_device_catalog().values()
                   for key in (list(spec.get("required_metadata", []))
                               + list(spec.get("optional_metadata", [])))})


def build_configuration_intent(device_ids: list[str], use_cases: list[str]) -> Type[BaseModel]:
    """Return the model that represent the schema of the configuration intent,
     built at runtime taking in input use cases and the devices
    """
    return create_model(
        "ConfigurationIntent",
        # A Literal rather than a bare str: matching none of these, the model
        # had nothing to settle on and ran until the token cap cut it off
        action=(Literal[ACTIONS], Field(description=(
            "assign_devices to add a device type to spaces; remove_devices to remove devices "
            "already configured - a specific device_type_id if named, otherwise every device - "
            "from the targeted spaces (or every space, if none is named); "
            "add_access_point and remove_access_point to model or unmodel a door, entrance "
            "or passage of the targeted spaces, which is what an access-point device is "
            "installed on; list_catalog for anything else about what's available."
        ))),
        use_case=(Optional[Literal[tuple(use_cases)]], Field(default=None)),
        device_type_id=(Optional[Literal[tuple(device_ids)]], Field(
            default=None,
            description="The catalogue id of the device concerned. Match the architect's own "
                        "words to the catalogue entry whose name or capability they describe "
                        "('an environmental sensor' -> generic.environmental_sensor). Null "
                        "only when the request genuinely names no device.")),
        target_space_types=(list[str], Field(default_factory=list,
            description="Space types to apply to, e.g. ['classroom'] for 'all classrooms'. "
                        "Leave this and target_space_names empty to apply to every space.")),
        quantity=(int, Field(default=1, description=(
            "How many devices of this type the message is about, read with quantity_mode. "
            "'three sensors in every classroom' -> 3; 'add two more' -> 2. Leave at 1 "
            "unless the message itself names a number."))),
        quantity_mode=(Optional[Literal["total", "additional"]], Field(
            default=None,
            description=(
                "'additional' when the message asks for more ON TOP OF what is already "
                "there - 'another sensor', 'one more in Classroom 1', 'add two more', "
                "'un altro sensore', 'uno in piu'. 'total' when it says how many the "
                "space is to have - 'three sensors in every classroom', 'each office "
                "needs two'. A plain assignment with no number and no word meaning "
                "'more' is a total of one: leave this unset."))),
        access_point_name=(Optional[str], Field(default=None, description=(
            "The door, entrance or passage concerned, by name. For add_access_point and "
            "remove_access_point, the one to model or unmodel ('give every classroom a back "
            "door' -> 'Back door'); null on a removal means every one the space has. For "
            "assign_devices, it confines the assignment to that one way in ('a counter on "
            "the main entrance' -> 'Main entrance'); null lets the devices spread over the "
            "ways in the space already has."))),
        target_space_names=(list[str], Field(default_factory=list)),
        target_floor_names=(list[str], Field(default_factory=list,
            description="Floors to restrict to, e.g. ['Second floor'] for 'the offices on "
                        "the second floor'. Narrows the spaces above; empty means any floor.")),
        # A Literal: left a bare string, 'rename the building to X' arrived
        # here as metadata_key 'building_name', written onto a device
        metadata_key=(Optional[Literal[tuple(metadata_keys()) or ("physical_label",)]], Field(
            default=None,
            description="Which metadata field of the devices to set. These belong to "
                        "devices, never to the building, a floor or a room.")),
        metadata_value=(Optional[str], Field(default=None, description="Value to set it to")),
        # A Literal for the same reason: a bare string was answered with a
        # platform nobody has an adapter for
        platform=(Optional[Literal[tuple(available_platforms()) or ("thingsboard",)]], Field(
            default=None,
            description="The platform to deploy onto, when the architect names one "
                        "('deploy on openremote' -> openremote).")),
        exclusion_rule=(Optional[str], Field(default=None, description="Rule being excluded")),
        exclusion_scope=(Optional[str], Field(
            default=None, description="Space name, or 'installation'")),
        justification=(Optional[str], Field(default=None)),
    )


def build_configuration_intents(device_ids: list[str], use_cases: list[str]) -> Type[BaseModel]:
    """A list of intents rather than one, so a single message can both pick a
    profile and equip the rooms: they are two actions of the same agent, and
    routing to it twice would only repeat the first"""
    return create_model(
        "ConfigurationIntents",
        actions=(list[build_configuration_intent(device_ids, use_cases)], Field(description=(
            "One entry per thing the architect asks of the configuration, in the order "
            "they are to be applied - most messages ask for one. Give a second only when "
            "the message really asks for it too ('use the wellness profile and add "
            "environmental sensors to the classrooms'). Never empty."))),
    )


class ConfigurationNode:

    def __init__(self, llm):
        # the output will be structure as the intent model we defined
        self.intents_model = build_configuration_intents(
            list(load_device_catalog()), list(load_use_cases()))
        # include_raw: a value outside a Literal comes back to be mended,
        # instead of an exception that costs the whole turn
        self.intent_llm = llm.with_structured_output(
            self.intents_model, method="function_calling", include_raw=True)

    def _repair(self, result: dict, named: list[str]):
        """What the model answered, with invented values put right: an unknown
        device becomes the one the architect named, but only for an assignment
        (an empty removal already means every device). No action, no entry."""
        calls = getattr(result.get("raw"), "tool_calls", None) or []
        if not calls:
            return []
        allowed = {"use_case": tuple(load_use_cases()),
                   "device_type_id": tuple(load_device_catalog()),
                   "platform": tuple(available_platforms()),
                   "quantity_mode": ("total", "additional"),
                   "metadata_key": tuple(metadata_keys())}

        mended = []
        for entry in calls[0].get("args", {}).get("actions", []):
            if entry.get("action") not in ACTIONS:
                continue
            for field, values in allowed.items():
                if entry.get(field) is not None and entry[field] not in values:
                    guessable = (field == "device_type_id" and named
                                 and entry["action"] == "assign_devices")
                    entry[field] = named[0] if guessable else None
            # A quantity that is not a positive number is dropped, not mended:
            # the default of one is the only safe thing to assume
            try:
                if int(entry.get("quantity", 1)) < 1:
                    raise ValueError
            except (TypeError, ValueError):
                entry.pop("quantity", None)
            mended.append(entry)
        try:
            return self.intents_model(actions=mended).actions
        except ValidationError:
            return []

    def __call__(self, state: IoTDeploymentState) -> dict:
        inst = Installation(**state.get("installation", {}))
        catalog = load_device_catalog()

        # Describe the catalogue by name and capability, not by bare id: 'an
        # environmental sensor' can only land on an id the model can connect it to.
        entries = "\n".join(
            f"- {device_id}: {spec['display_name']}"
            f" (measures: {', '.join(spec.get('telemetry_model', [])) or 'nothing'};"
            f" capabilities: {', '.join(spec.get('capabilities', [])) or 'none'};"
            f" also called: {', '.join(spec.get('aliases', [])) or '-'};"
            f" installed on {spec.get('installation_target', 'space')})"
            for device_id, spec in catalog.items()
        )
        prompt = [{"role": "system", "content": (
            "Recognise what the architect wants regarding the equipment of the installation.\n"
            f"Device catalogue:\n{entries}\n"
            f"Use cases: {', '.join(load_use_cases())}\n"
            "'all classrooms' -> target_space_types=['classroom'].\n"
            "'the offices on the second floor' -> target_space_types=['office'] plus "
            "target_floor_names=['Second floor'].\n"
            "'give every classroom a back door' -> add_access_point with "
            "access_point_name='Back door'.\n"
            "'three sensors in every classroom' -> quantity=3, quantity_mode='total'.\n"
            "'add another sensor to Classroom 1' -> quantity=1, "
            "quantity_mode='additional'.\n"
            "'set the physical label of the sensors to X' -> set_metadata with "
            "metadata_key and metadata_value.\n"
            "A decision not to cover a space, stated with a reason, is a record_exclusion.\n"
            "'remove/clear/delete the devices' -> remove_devices; 'remove all devices' means "
            "no device_type_id and no target spaces (every device, everywhere).\n"
        )}] + state["messages"]
        # The devices the architect named: what settles which assignment was
        # meant, and what a value outside the catalogue is mended with
        said = next((m.content for m in reversed(state["messages"])
                     if isinstance(m, HumanMessage)), "")
        named = [device_id for device_id, _ in match_device_types(said)]

        #now we have the intents, in the order they are to be applied
        result = self.intent_llm.invoke(prompt)
        actions = result["parsed"].actions if result["parsed"] else self._repair(result, named)
        if not actions:
            return {"messages": [AIMessage(content=(
                "[Configuration] I could not tell which catalogue device you meant. "
                "Name it as the catalogue does, or ask to see what is available."))],
                "trace": trace("Configuration", "intent_unreadable", "")}

        # One of each kind: the model emits assign_devices several times for one
        # request, so keep the assignment whose device the architect named
        chosen = {}
        for intent in actions:
            held = chosen.get(intent.action)
            if held is None or (intent.action == "assign_devices"
                                and held.device_type_id not in named
                                and intent.device_type_id in named):
                chosen[intent.action] = intent
        intents = list(chosen.values())[:MAX_ACTIONS]

        handlers = {
            "select_use_case": self._select_use_case,
            "assign_devices": self._assign,
            "remove_devices": self._remove_devices,
            "set_metadata": self._set_metadata,
            "record_exclusion": self._record_exclusion,
            "select_platform": self._select_platform,
            "add_access_point": self._add_access_point,
            "remove_access_point": self._remove_access_point,
        }

        # Applied one after the other to the same installation: each handler
        # already works on it in place, so the second sees what the first did
        merged, messages, traces = {}, [], []
        for intent in intents:
            # Last line of defence when the model leaves the field empty. Only
            # with one action: it reads the whole message, so it would misassign
            interpretation = None
            if (len(intents) == 1 and not intent.device_type_id
                    and intent.action in ("assign_devices", "remove_devices")):
                #search any match in the last human message
                last_human = next((m for m in reversed(state["messages"])
                                   if isinstance(m, HumanMessage)), None)
                matches = match_device_types(last_human.content) if last_human else []
                if matches:
                    intent.device_type_id, reason = matches[0]
                    interpretation = (f"Understood **{catalog[intent.device_type_id]['display_name']}** "
                                      f"({reason}).")
                    # Say what else would have fitted, so it's obvious the
                    # choice was of the system and not of the user
                    others = ", ".join(catalog[d]["display_name"] for d, _ in matches[1:4])
                    if others:
                        interpretation += (f" Also matching: {others} — name one of those instead "
                                           f"if you meant it.")

            #now the respective handlers will concretize the intent
            handler = handlers.get(intent.action, self._list_catalog)
            result = handler(inst, intent)

            if interpretation and result.get("messages"):
                message = result["messages"][0]
                message.content = (f"[Configuration] {interpretation}\n\n"
                                   + message.content.replace("[Configuration] ", "", 1))

            messages += result.pop("messages", [])
            traces += result.pop("trace", trace("Configuration", intent.action,
                                                intent.device_type_id or intent.use_case or ""))
            merged.update(result)

        # One reply for the turn, not one per action: what the architect reads
        # should be the account of what their message did
        if messages:
            merged["messages"] = [AIMessage(content="\n\n".join(m.content for m in messages))]
        merged["trace"] = traces
        return merged

    # -- handlers ---------------------------------------------------------

    @staticmethod
    def _targeted(inst, intent):
        """The spaces an intent is about. The floor narrows the spaces rather
        than adding to them: 'the offices on the second floor' is floor AND
        type, not floor OR type"""
        types, names = set(intent.target_space_types), set(intent.target_space_names)
        floors = set(intent.target_floor_names)
        scoped = bool(types or names)          # neither named: every space
        for floor, space in inst.iter_spaces():
            if floors and floor.name not in floors:
                continue
            if scoped and not (space.type in types or space.name in names):
                continue
            yield floor, space

    def _add_access_point(self, inst, intent) -> dict:
        """A door on a room already modelled, without going back through the
        whole description: it is what an access-point device is installed on,
        and asking for one twice must not give the room two"""
        name = (intent.access_point_name or "").strip() or DEFAULT_ACCESS_POINT
        added, already = [], 0
        for floor, space in self._targeted(inst, intent):
            if any(ap.name.lower() == name.lower() for ap in space.access_points):
                already += 1
                continue
            space.access_points.append(AccessPoint(name=name))
            added.append(f"{space.name} ({floor.name})")

        if not added:
            reason = (f"those {already} space(s) already have a '{name}'"
                      if already else "no space matched")
            return {"messages": [AIMessage(content=(
                f"[Configuration] Nothing to do: {reason}.")),],
                "trace": trace("Configuration", "add_access_point_none", reason)}

        return {"installation": inst.model_dump(),
                "messages": [AIMessage(content=(
                    f"[Configuration] **{name}** modelled on {len(added)} space(s): "
                    f"{', '.join(added)}.\n\n" + installation_to_markdown(inst)))],
                "trace": trace("Configuration", "add_access_point",
                               f"'{name}' on {len(added)} space(s)")}

    def _remove_access_point(self, inst, intent) -> dict:
        """Unmodelling a way in takes whatever was mounted on it, so the count
        is reported rather than left for the architect to discover"""
        name = (intent.access_point_name or "").strip()
        removed, lost = 0, 0
        for _floor, space in self._targeted(inst, intent):
            kept = [ap for ap in space.access_points
                    if name and ap.name.lower() != name.lower()]
            lost += sum(len(ap.devices) for ap in space.access_points
                        if ap not in kept)
            removed += len(space.access_points) - len(kept)
            space.access_points = kept

        if not removed:
            return {"messages": [AIMessage(content=(
                "[Configuration] No access point matched, so nothing was removed."))],
                "trace": trace("Configuration", "remove_access_point_none", name or "any")}

        went = f", and the {lost} device(s) mounted on them" if lost else ""
        return {"installation": inst.model_dump(),
                "messages": [AIMessage(content=(
                    f"[Configuration] Removed {removed} access point(s){went}.\n\n"
                    + installation_to_markdown(inst)))],
                "trace": trace("Configuration", "remove_access_point",
                               f"{removed} removed, {lost} device(s) with them")}

    def _list_catalog(self, inst, intent) -> dict:
        devices = devices_for_use_case(inst.use_case)
        lines = ["[Configuration] Devices available"
                 + (f" for the {inst.use_case} profile:" if inst.use_case else ":"), "",
                 "| Id | Name | Category | Capabilities | Installed on |",
                 "| --- | --- | --- | --- | --- |"]
        for d in devices:
            lines.append(f"| `{d['device_type_id']}` | {d['display_name']} | {d['category']} "
                         f"| {', '.join(d.get('capabilities', [])) or '—'} "
                         f"| {d.get('installation_target', '—')} |")
        return {"messages": [AIMessage(content="\n".join(lines))],
                "trace": trace("Configuration", "list_catalog",
                               f"{len(devices)} device(s) available")}

    def _select_use_case(self, inst, intent) -> dict:
        profiles = load_use_cases()
        if intent.use_case not in profiles:
            return {"messages": [AIMessage(content=(
                f"[Configuration] Unknown use case. Available: {', '.join(profiles)}."))]}

        inst.use_case = intent.use_case
        profile = profiles[intent.use_case]
        recommended = profile.get("recommended_devices") or ["(entire catalogue)"]
        return {"installation": inst.model_dump(), "messages": [AIMessage(content=(
            f"[Configuration] Use case set to **{profile['display_name']}**.\n\n"
            f"{profile['description'].strip()}\n\n"
            f"Recommended devices: {', '.join(recommended)}"))]}

    def _select_platform(self, inst, intent) -> dict:
        if intent.platform not in available_platforms():
            return {"messages": [AIMessage(content=(
                f"[Configuration] No adapter for '{intent.platform}'. "
                f"Available: {', '.join(available_platforms())}."))]}
        previous = inst.target.platform
        inst.target.platform = intent.platform
        result = {"installation": inst.model_dump(), "messages": [AIMessage(content=(
            f"[Configuration] Deployment target set to **{intent.platform}**."))]}
        # op_ids come from the model and not from the platform, which is what lets
        # an interrupted run resume. Left in place across a change of target, the
        # plan of the old platform marks every operation of the new one completed
        # and nothing is written, so a new target is a new plan
        if intent.platform != previous:
            result["deployment_plan"] = []
        return result

    def _record_exclusion(self, inst, intent) -> dict:
        """A decision the architect actually took, written into the model
        instead of quietly ignored"""
        if not intent.exclusion_rule or not intent.justification:
            return {"messages": [AIMessage(content=(
                "[Configuration] An exclusion needs both the rule it concerns and a "
                "justification, which is what makes it reviewable afterwards."))]}

        inst.exclusions.append(Exclusion(
            rule=intent.exclusion_rule,
            scope=intent.exclusion_scope or "installation",
            justification=intent.justification,
        ))
        return {"installation": inst.model_dump(), "messages": [AIMessage(content=(
            f"[Configuration] Exclusion recorded for `{intent.exclusion_rule}` on "
            f"'{intent.exclusion_scope or 'installation'}': {intent.justification}\n\n"
            "It will appear in validation as a documented exclusion rather than an omission."))]}

    def _set_metadata(self, inst, intent) -> dict:
        """Fills in what the catalogue asked for and the defaults could not
        work out, or corrects a default that does not match the site."""
        if not intent.metadata_key or intent.metadata_value is None:
            return {"messages": [AIMessage(content=(
                "[Configuration] Tell me which field to set and to what value, "
                "e.g. 'set gateway_id to gw-1 for the environmental sensors'."))]}

        updated = 0
        for _floor, space in self._targeted(inst, intent):
            for device in list(space.devices) + [d for ap in space.access_points
                                                 for d in ap.devices]:
                if intent.device_type_id and device.device_type_id != intent.device_type_id:
                    continue
                device.metadata[intent.metadata_key] = intent.metadata_value
                updated += 1

        if not updated:
            return {"messages": [AIMessage(content=(
                "[Configuration] No device matched, so nothing was set."))]}
        return {"installation": inst.model_dump(), "messages": [AIMessage(content=(
            f"[Configuration] `{intent.metadata_key}` set to "
            f"'{intent.metadata_value}' on {updated} device(s).\n\n"
            + installation_to_markdown(inst)))]}

    def _remove_devices(self, inst, intent) -> dict:
        def keep(device: Device) -> bool:
            return intent.device_type_id is not None and device.device_type_id != intent.device_type_id

        removed = 0
        # The same targeting as every other handler: removal was the one that
        # never read the floor, so one named floor emptied all of them
        for _floor, space in self._targeted(inst, intent):
            kept = [d for d in space.devices if keep(d)]
            removed += len(space.devices) - len(kept)
            space.devices = kept

            for ap in space.access_points:
                kept_ap = [d for d in ap.devices if keep(d)]
                removed += len(ap.devices) - len(kept_ap)
                ap.devices = kept_ap

        if removed == 0:
            return {"messages": [AIMessage(content="[Configuration] No matching devices to remove.")]}

        scope = f" of type `{intent.device_type_id}`" if intent.device_type_id else ""
        parts = [f"[Configuration] Removed {removed} device(s){scope}.",
                 "\n" + installation_to_markdown(inst)]
        return {"installation": inst.model_dump(), "messages": [AIMessage(content="\n".join(parts))]}

    def _ensure_gateway(self, inst) -> str | None:
        """Drops in the gateway the current profile insists on, if there
        isn't one already"""
        if not profile_rules(inst.use_case).get("require_gateway") or find_gateway(inst):
            return None

        spec = load_device_catalog()["dipme.edge_gateway"]
        compatible = spec.get("compatible_space_types", [])
        candidates = [(f, s) for f, s in inst.iter_spaces() if s.type in compatible]
        if not candidates:
            return None

        # A technical room is where this really belongs
        floor, space = next((c for c in candidates if c[1].type == "technical_room"),
                            candidates[0])
        instance = device_instance_name(spec, floor, space)
        metadata, _ = resolve_default_metadata(spec, inst, floor, space)
        space.devices.append(Device(device_type_id=spec["device_type_id"],
                                    instance_name=instance, metadata=metadata))

        # Devices placed before there was a gateway were left without a
        # gateway_id, because it could not be worked out yet
        catalog = load_device_catalog()
        for _f, _s, _ap, device in inst.iter_devices():
            needs = catalog.get(device.device_type_id, {}).get("required_metadata", [])
            if "gateway_id" in needs and not device.metadata.get("gateway_id"):
                device.metadata["gateway_id"] = instance

        return (f"**{spec['display_name']}** placed in '{space.name}' ({floor.name}): the "
                f"{inst.use_case} profile requires a gateway.")

    def _assign(self, inst, intent) -> dict:
        catalog = load_device_catalog()
        spec = catalog.get(intent.device_type_id)
        #if the spec is not avaiable we will tell the possible real alternatives
        if spec is None:
            available = ", ".join(f"{spec['display_name']} (`{i}`)"
                                  for i, spec in catalog.items())
            named = (f"'{intent.device_type_id}' is not in the catalogue"
                     if intent.device_type_id else
                     "I could not tell which device you meant")
            return {"messages": [AIMessage(content=(
                f"[Configuration] {named}. Available: {available}."))],
                "trace": trace("Configuration", "assign_refused",
                               "no catalogue device matched the request")}

        #ensure if there is a gateway
        gateway_note = self._ensure_gateway(inst)

        on_access_point = spec.get("installation_target") == "access_point"
        door_named = (intent.access_point_name or "").strip().lower()
        quantity = max(1, min(intent.quantity or 1, MAX_PER_SPACE))
        adding = intent.quantity_mode == "additional"
        assigned, skipped, no_such_door, already, placed = [], [], [], 0, 0
        defaulted, undefaulted = set(), set()

        def held(where) -> int:
            return sum(1 for d in where.devices
                       if d.device_type_id == spec["device_type_id"])

        for floor, space in self._targeted(inst, intent):
            if on_access_point:
                if not space.access_points:
                    # Validation is what enforces the profile rule;
                    # here there's simply nothing to attach the device to
                    skipped.append(space.name)
                    continue
                targets = [ap for ap in space.access_points
                           if not door_named or ap.name.lower() == door_named]
                if not targets:
                    no_such_door.append(space.name)
                    continue
            else:
                targets = [space]

            # Counted over the room and not over one door: the quantity is what
            # the space is to hold, wherever in it the devices hang
            present = sum(held(t) for t in targets)
            # 'another sensor' adds to what is there; a bare number is what the
            # room is to end up with, which is what keeps a repeat idempotent
            wanted = min(present + quantity, MAX_PER_SPACE) if adding else quantity
            if present >= wanted:
                already += 1
                continue

            for _ in range(wanted - present):
                # The emptiest way in, so counters spread over the doors of a
                # room instead of piling onto whichever was modelled first
                target = min(targets, key=held)
                access_point = target if on_access_point else None
                # The next free index rather than a count: removing the second
                # of three must not make the next assignment collide
                taken = {d.instance_name for d in target.devices}
                index = 1
                while device_instance_name(spec, floor, space, access_point,
                                           index) in taken:
                    index += 1
                instance = device_instance_name(spec, floor, space, access_point, index)
                #resolve the default metadata
                metadata, unresolved = resolve_default_metadata(
                    spec, inst, floor, space, access_point, index)
                defaulted.update(metadata)
                undefaulted.update(unresolved)

                target.devices.append(Device(device_type_id=spec["device_type_id"],
                                             instance_name=instance, metadata=metadata))
                placed += 1
            assigned.append(f"{space.name} ({floor.name})")

        if not assigned and not skipped and not no_such_door:
            # Still return the model: _ensure_gateway may have placed one above,
            # and dropping it here would leave the profile rule unsatisfiable
            if not already:
                reason = "no space matched that assignment"
            elif adding:
                reason = (f"those {already} space(s) already hold the most of that type "
                          f"a room may have ({MAX_PER_SPACE})")
            else:
                reason = f"those {already} space(s) already hold the device(s) asked for"
            return {"installation": inst.model_dump(),
                    "messages": [AIMessage(content=(
                        f"[Configuration] Nothing to do: {reason}."
                        + (f"\n\n{gateway_note}" if gateway_note else "")))],
                    "trace": trace("Configuration", "assign_none", reason)}

        # The name goes on the front once, not on whichever part comes first:
        # Response knows an agent's prose by that prefix and drops the rest
        parts = []
        if gateway_note:
            parts.append(f"{gateway_note}\n")
        if assigned:
            # The count only when it is not one per space, so the ordinary
            # assignment reads as it always did
            total = f" ({placed} device(s) in all)" if placed != len(assigned) else ""
            parts.append(f"**{spec['display_name']}** associated with "
                         f"{len(assigned)} space(s){total}: {', '.join(assigned)}.")
            # Spell the defaults out rather than assuming them quietly, so
            # it's clear which fields the architect never chose
            if defaulted:
                parts.append(f"\nDefaults applied: {', '.join(sorted(defaulted))}. "
                             f"Ask to change any of them if they do not match the site.")
            if undefaulted:
                parts.append(f"\n⚠️ Still to be provided: {', '.join(sorted(undefaulted))}.")
        if skipped:
            parts.append(f"\n⚠️ Skipped {', '.join(skipped)}: this device is installed on "
                         f"an access point, and none has been modelled there yet - ask for "
                         f"one and this goes through.")
        if no_such_door:
            parts.append(f"\n⚠️ Skipped {', '.join(no_such_door)}: no access point "
                         f"called '{intent.access_point_name}' is modelled there.")
        parts.append("\n" + installation_to_markdown(inst))

        # The outcome, never the request: traced as assign_devices whatever
        # happened, a turn that equipped nothing was reported as an assignment
        detail = (f"{spec['display_name']}: {placed} device(s) in "
                  f"{len(assigned)} space(s)"
                  + (f", {len(skipped)} skipped for want of an access point"
                     if skipped else "")
                  + (f", {len(no_such_door)} with no such access point"
                     if no_such_door else ""))
        return {"installation": inst.model_dump(),
                "messages": [AIMessage(
                    content="[Configuration] " + "\n".join(parts))],
                "trace": trace("Configuration",
                               "assign_devices" if assigned else "assign_skipped",
                               detail)}