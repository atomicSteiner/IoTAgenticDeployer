"""Validation engine, plain code on purpose: completeness and consistency are
decided against the catalogue and the profile, never left to the model. What
comes out as missing is what conceptualisation turns into questions."""
from iot_agentic_deployer.domain.catalog.loader import load_device_catalog, load_use_cases, GATEWAY_CATEGORIES
from iot_agentic_deployer.domain.models import Installation


class Finding:
    def __init__(self, level: str, message: str, rule: str = "", scope: str = "installation"):
        self.level = level          # "error" | "warning"
        self.message = message
        self.rule = rule            # rule id, so an exclusion can reference it
        self.scope = scope

    @property
    def blocking(self) -> bool:
        return self.level == "error"

    def to_dict(self) -> dict:
        return {"level": self.level, "message": self.message,
                "rule": self.rule, "scope": self.scope}


# ---------------------------------------------------------- topology rules

def check_topology(inst: Installation) -> list[Finding]:
    """The bare minimum: a building needs a floor, a floor needs a space,
    and every space needs a type from the controlled vocabulary"""
    findings = []

    if not inst.building:
        return [Finding("error", "No building has been described yet.", "building_present")]

    if not inst.building.floors:
        findings.append(Finding(
            "error", f"Building '{inst.building.name}' contains no floor.", "floor_per_building"))

    for floor in inst.building.floors:
        if not floor.spaces:
            findings.append(Finding(
                "error", f"Floor '{floor.name}' contains no space.",
                "space_per_floor", floor.name))
        for space in floor.spaces:
            if space.type == "unknown":
                findings.append(Finding(
                    "error", f"Space '{space.name}' has no type assigned.",
                    "space_type_assigned", space.name))
    return findings


# ------------------------------------------------------------ device rules

def _check_devices(inst: Installation) -> list[Finding]:
    findings = []
    catalog = load_device_catalog()

    for floor, space, access_point, device in inst.iter_devices():
        spec = catalog.get(device.device_type_id)
        location = f"'{access_point.name}' of '{space.name}'" if access_point else f"'{space.name}'"

        if spec is None:
            findings.append(Finding(
                "error", f"Device type '{device.device_type_id}' is not in the catalogue.",
                "device_in_catalogue", space.name))
            continue

        if space.type not in spec.get("compatible_space_types", []):
            findings.append(Finding(
                "warning",
                f"'{spec['display_name']}' in {location} ({space.type}) is outside "
                f"its compatible space types.",
                "device_space_compatible", space.name))

        missing = [f for f in spec.get("required_metadata", []) if f not in device.metadata]
        if missing:
            findings.append(Finding(
                "error",
                f"'{device.instance_name}' in {location} lacks required metadata: "
                f"{', '.join(missing)}.",
                "device_metadata_complete", space.name))

        # A device meant for an access point has to actually be on one
        if spec.get("installation_target") == "access_point" and access_point is None:
            findings.append(Finding(
                "error",
                f"'{device.instance_name}' must be installed on an access point, "
                f"but is associated directly with '{space.name}'.",
                "device_on_access_point", space.name))

    return findings


# ------------------------------------------------------------ profile rules

def _space_capabilities(space, catalog) -> set:
    caps = set()
    devices = list(space.devices) + [d for ap in space.access_points for d in ap.devices]
    for device in devices:
        caps.update(catalog.get(device.device_type_id, {}).get("capabilities", []))
    return caps


def _check_profile(inst: Installation) -> list[Finding]:
    findings = []
    if not inst.use_case:
        return [Finding("warning", "No use case has been selected yet.", "use_case_selected")]

    profile = load_use_cases().get(inst.use_case)
    if profile is None:
        return [Finding("error", f"Unknown use case '{inst.use_case}'.", "use_case_known")]

    rules = profile.get("rules", {})
    typical = profile.get("typical_space_types", [])
    catalog = load_device_catalog()
    specs = [catalog[d.device_type_id] for d in inst.all_devices() if d.device_type_id in catalog]

    if rules.get("require_gateway") and not any(
        s.get("category") in GATEWAY_CATEGORIES for s in specs
    ):
        findings.append(Finding(
            "error", "No gateway or coordinator is present in the installation.",
            "gateway_present"))

    # A capability every selected space needs, unless excluded on the record
    needed = rules.get("require_capability_per_selected_space")
    if needed:
        for floor, space in inst.iter_spaces():
            if typical and space.type not in typical:
                continue
            if needed in _space_capabilities(space, catalog):
                continue
            rule_id = f"capability:{needed}"
            if rules.get("capability_exclusion_admitted") and inst.has_exclusion(rule_id, space.name):
                findings.append(Finding(
                    "warning",
                    f"Space '{space.name}' has no '{needed}' source; covered by a "
                    f"documented exclusion.",
                    rule_id, space.name))
            else:
                level = "error" if rules.get("require_full_coverage") else "warning"
                findings.append(Finding(
                    level, f"Space '{space.name}' has no '{needed}' source.",
                    rule_id, space.name))

    if rules.get("require_full_coverage"):
        for floor, space in inst.iter_spaces():
            if space.type == "outdoor" or space.devices or space.access_points:
                continue
            if inst.has_exclusion("coverage", space.name):
                findings.append(Finding(
                    "warning", f"Space '{space.name}' is uncovered; documented exclusion.",
                    "coverage", space.name))
            else:
                findings.append(Finding(
                    "error", f"Space '{space.name}' is not covered by any device.",
                    "coverage", space.name))

    if rules.get("people_counter_requires_access_point"):
        for floor, space in inst.iter_spaces():
            direct = [d for d in space.devices
                      if catalog.get(d.device_type_id, {}).get("category") == "people_counter"]
            if direct and not space.access_points:
                findings.append(Finding(
                    "error",
                    f"A people counter is configured in '{space.name}' but no access "
                    f"point has been modelled there.",
                    "people_counter_access_point", space.name))

    return findings


def validate_installation(inst: Installation) -> dict:
    findings = check_topology(inst)
    # Device and profile rules presuppose a well-formed topology.
    if not any(f.blocking for f in findings):
        findings += _check_devices(inst) + _check_profile(inst)

    return {
        "findings": [f.to_dict() for f in findings],
        "errors": sum(1 for f in findings if f.level == "error"),
        "warnings": sum(1 for f in findings if f.level == "warning"),
        "deployable": not any(f.blocking for f in findings),
    }


def missing_topology_information(inst: Installation) -> list[str]:
    """What conceptualisation still has to ask about: from the topology
    rules, not from the model's own opinion"""
    questions = []
    for f in check_topology(inst):
        if f.rule == "building_present":
            questions.append("Which building is being configured, and where is it located?")
        elif f.rule == "floor_per_building":
            questions.append(f"How many floors does '{inst.building.name}' have?")
        elif f.rule == "space_per_floor":
            questions.append(f"Which rooms are present on floor '{f.scope}'?")
        elif f.rule == "space_type_assigned":
            questions.append(f"What kind of space is '{f.scope}'?")
    return questions