"""Configuration agent.

Applies the use-case profile, reads the catalogue, and puts devices into
spaces
The language model only works out what was being asked for; looking devices up
and placing them is code.
"""

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
from iot_agentic_deployer.domain.models import Installation, Exclusion, Device
from iot_agentic_deployer.domain.state import IoTDeploymentState, trace
from iot_agentic_deployer.platforms.base import available_platforms
from iot_agentic_deployer.rendering.formatting import installation_to_markdown

ACTIONS = (
    "list_catalog", "select_use_case", "assign_devices", "remove_devices",
    "set_metadata", "record_exclusion", "select_platform",
)

# A message asks the configuration for one or two things; a longer list is the
# model reading intent into the conversation rather than into the message
MAX_ACTIONS = 3


def build_configuration_intent(device_ids: list[str], use_cases: list[str]) -> Type[BaseModel]:
    """Return the model that represent the schema of the configuration intent,
     built at runtime taking in input use cases and the devices
    """
    return create_model(
        "ConfigurationIntent",
        # A Literal rather than a bare str. When a request matched none of
        # these, the model had nothing valid to settle on and would just keep
        # going until the token cap cut it off mid-object
        action=(Literal[ACTIONS], Field(description=(
            "assign_devices to add a device type to spaces; remove_devices to remove devices "
            "already configured - a specific device_type_id if named, otherwise every device - "
            "from the targeted spaces (or every space, if none is named); list_catalog for "
            "anything else about what's available."
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
        target_space_names=(list[str], Field(default_factory=list)),
        target_floor_names=(list[str], Field(default_factory=list,
            description="Floors to restrict to, e.g. ['Second floor'] for 'the offices on "
                        "the second floor'. Narrows the spaces above; empty means any floor.")),
        metadata_key=(Optional[str], Field(default=None,
            description="Metadata field to set, e.g. 'physical_label' or 'gateway_id'")),
        metadata_value=(Optional[str], Field(default=None, description="Value to set it to")),
        platform=(Optional[str], Field(default=None)),
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
        self.intent_llm = llm.with_structured_output(
            build_configuration_intents(list(load_device_catalog()), list(load_use_cases())),
            method="function_calling")

    def __call__(self, state: IoTDeploymentState) -> dict:
        inst = Installation(**state.get("installation", {}))
        catalog = load_device_catalog()

        # Describe the catalogue by name and capability, not by bare id. The
        # architect says 'an environmental sensor'; for that to land on
        # generic.environmental_sensor, the model has to be able to see the
        # connection in what it was given.
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
            "'set the physical label of the sensors to X' -> set_metadata with "
            "metadata_key and metadata_value.\n"
            "A decision not to cover a space, stated with a reason, is a record_exclusion.\n"
            "'remove/clear/delete the devices' -> remove_devices; 'remove all devices' means "
            "no device_type_id and no target spaces (every device, everywhere).\n"
        )}] + state["messages"]
        #now we have the intents, in the order they are to be applied
        # The catalogue id is a Literal, and nested inside a list the model
        # misses it more often than it did on its own, which takes the whole
        # turn down with a schema error. One more go, then say so
        try:
            actions = self.intent_llm.invoke(prompt).actions
        except ValidationError:
            try:
                actions = self.intent_llm.invoke(prompt).actions
            except ValidationError:
                return {"messages": [AIMessage(content=(
                    "[Configuration] I could not tell which catalogue device you meant. "
                    "Name it as the catalogue does, or ask to see what is available."))],
                    "trace": trace("Configuration", "intent_unreadable", "")}

        # One of each kind, kept in the order they came. Asked for a list, the
        # model will gladly emit assign_devices twice for a single request, the
        # second time with the targets left empty, and equip every room in the
        # building: doing less than was asked can be asked for again, doing more
        # is the architect undoing it by hand
        seen, intents = set(), []
        for intent in actions:
            if intent.action not in seen:
                seen.add(intent.action)
                intents.append(intent)
        intents = intents[:MAX_ACTIONS]

        handlers = {
            "select_use_case": self._select_use_case,
            "assign_devices": self._assign,
            "remove_devices": self._remove_devices,
            "set_metadata": self._set_metadata,
            "record_exclusion": self._record_exclusion,
            "select_platform": self._select_platform,
        }

        # Applied one after the other to the same installation: each handler
        # already works on it in place, so the second sees what the first did
        merged, messages, traces = {}, [], []
        for intent in intents:
            # Last line of defence. The request can name a device perfectly
            # clearly, or say exactly what it should measure, and the model can
            # still leave the field empty. Only when the turn holds one action,
            # though: it reads the whole message, so given 'remove all devices,
            # then add a structural sensor' it would hand the removal the
            # sensor named for the assignment, and remove nothing
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
        return {"messages": [AIMessage(content="\n".join(lines))]}

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
        inst.target.platform = intent.platform
        return {"installation": inst.model_dump(), "messages": [AIMessage(content=(
            f"[Configuration] Deployment target set to **{intent.platform}**."))]}

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

        wanted_types = set(intent.target_space_types)
        wanted_names = set(intent.target_space_names)
        scoped = bool(wanted_types or wanted_names)
        wanted_floors = set(intent.target_floor_names)

        updated = 0
        for floor, space in inst.iter_spaces():
            if wanted_floors and floor.name not in wanted_floors:
                continue
            if scoped and not (space.type in wanted_types or space.name in wanted_names):
                continue
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
        wanted_types = set(intent.target_space_types)
        wanted_names = set(intent.target_space_names)
        scoped = bool(wanted_types or wanted_names)

        def keep(device: Device) -> bool:
            return intent.device_type_id is not None and device.device_type_id != intent.device_type_id

        removed = 0
        for floor, space in inst.iter_spaces():
            if scoped and not (space.type in wanted_types or space.name in wanted_names):
                continue

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
                f"[Configuration] {named}. Available: {available}."))]}

        #ensure if there is a gateway
        gateway_note = self._ensure_gateway(inst)

        wanted_types = set(intent.target_space_types)
        wanted_names = set(intent.target_space_names)
        scoped = bool(wanted_types or wanted_names)   # neither named: every space
        wanted_floors = set(intent.target_floor_names)
        on_access_point = spec.get("installation_target") == "access_point"
        assigned, skipped, already = [], [], 0
        defaulted, undefaulted = set(), set()

        for floor, space in inst.iter_spaces():
            # the floor narrows the spaces rather than adding to them: 'the
            # offices on the second floor' is floor AND type, not floor OR type
            if wanted_floors and floor.name not in wanted_floors:
                continue
            if scoped and not (space.type in wanted_types or space.name in wanted_names):
                continue

            if on_access_point:
                if not space.access_points:
                    # Validation is what enforces the profile rule;
                    # here there's simply nothing to attach the device to
                    skipped.append(space.name)
                    continue
                target = space.access_points[0]
            else:
                target = space
            instance = device_instance_name(
                spec, floor, space, target if on_access_point else None)
            #skip the duplicates
            if any(d.instance_name == instance for d in target.devices):
                already += 1
                continue
            #resolve the default metadata
            metadata, unresolved = resolve_default_metadata(
                spec, inst, floor, space, target if on_access_point else None)
            defaulted.update(metadata)
            undefaulted.update(unresolved)

            target.devices.append(Device(device_type_id=spec["device_type_id"],
                                         instance_name=instance, metadata=metadata))
            assigned.append(f"{space.name} ({floor.name})")

        if not assigned and not skipped:
            # Still return the model: _ensure_gateway may have placed a gateway
            # above, and dropping the installation here would throw that away
            # and leave the profile rule unsatisfiable
            reason = (f"those {already} device(s) are already in place"
                      if already else "no space matched that assignment")
            return {"installation": inst.model_dump(),
                    "messages": [AIMessage(content=(
                        f"[Configuration] Nothing to do: {reason}."
                        + (f"\n\n{gateway_note}" if gateway_note else "")))]}

        parts = []
        if gateway_note:
            parts.append(f"[Configuration] {gateway_note}\n")
        if assigned:
            head = "" if gateway_note else "[Configuration] "
            parts.append(f"{head}**{spec['display_name']}** associated with "
                         f"{len(assigned)} space(s): {', '.join(assigned)}.")
            # Spell the defaults out rather than assuming them quietly, so
            # it's clear which fields the architect never chose
            if defaulted:
                parts.append(f"\nDefaults applied: {', '.join(sorted(defaulted))}. "
                             f"Ask to change any of them if they do not match the site.")
            if undefaulted:
                parts.append(f"\n⚠️ Still to be provided: {', '.join(sorted(undefaulted))}.")
        if skipped:
            parts.append(f"\n⚠️ Skipped {', '.join(skipped)}: this device is installed on an "
                         f"access point, and none has been modelled there yet.")
        parts.append("\n" + installation_to_markdown(inst))

        return {"installation": inst.model_dump(),
                "messages": [AIMessage(content="\n".join(parts))]}