"""Configuration agent.

Applies the use-case profile, reads the catalogue, and puts devices into
spaces
The language model only works out what was being asked for; looking devices up
and placing them is code.
"""

from typing import Literal, Optional, Type

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field, create_model

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
    "record_exclusion", "select_platform",
)


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
        platform=(Optional[str], Field(default=None)),
        exclusion_rule=(Optional[str], Field(default=None, description="Rule being excluded")),
        exclusion_scope=(Optional[str], Field(
            default=None, description="Space name, or 'installation'")),
        justification=(Optional[str], Field(default=None)),
    )


class ConfigurationNode:

    def __init__(self, llm):
        # the output will be structure as the intent model we defined
        self.intent_llm = llm.with_structured_output(
            build_configuration_intent(list(load_device_catalog()), list(load_use_cases())),
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
            "A decision not to cover a space, stated with a reason, is a record_exclusion.\n"
            "'remove/clear/delete the devices' -> remove_devices; 'remove all devices' means "
            "no device_type_id and no target spaces (every device, everywhere).\n"
        )}] + state["messages"]
        #now we have the intent
        intent = self.intent_llm.invoke(prompt)

        # Last line of defence. The request can name a device perfectly
        # clearly, or say exactly what it should measure, and the model can
        # still leave the field empty.
        interpretation = None
        if intent.action in ("assign_devices", "remove_devices") and not intent.device_type_id:
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

        handlers = {
            "select_use_case": self._select_use_case,
            "assign_devices": self._assign,
            "remove_devices": self._remove_devices,
            "record_exclusion": self._record_exclusion,
            "select_platform": self._select_platform,
        }
        #now the respective handlers will concretize the intent
        handler = handlers.get(intent.action, self._list_catalog)
        result = handler(inst, intent)

        if interpretation and result.get("messages"):
            message = result["messages"][0]
            message.content = (f"[Configuration] {interpretation}\n\n"
                               + message.content.replace("[Configuration] ", "", 1))

        result.setdefault("trace", trace("Configuration", intent.action,
                                         intent.device_type_id or intent.use_case or ""))
        return result

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
        on_access_point = spec.get("installation_target") == "access_point"
        assigned, skipped = [], []
        defaulted, undefaulted = set(), set()

        for floor, space in inst.iter_spaces():
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
            return {"messages": [AIMessage(content=(
                "[Configuration] No space matched that assignment."))]}

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