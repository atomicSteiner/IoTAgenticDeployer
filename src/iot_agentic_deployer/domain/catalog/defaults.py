"""Starting values worked out from the catalogue and the model
"""

from iot_agentic_deployer.domain.catalog.loader import GATEWAY_CATEGORIES, load_device_catalog
from iot_agentic_deployer.domain.models import Installation


def device_instance_name(spec: dict, floor, space, access_point=None) -> str:
    """What a device instance gets called. Built from where it sits, so it
    comes out the same every time the plan is derived again."""
    parts = [spec["category"], floor.name, space.name]
    if access_point is not None:
        parts.append(access_point.name)
    return "-".join(parts).replace(" ", "_")


def find_gateway(inst: Installation) -> str | None:
    """Name of the gateway serving this installation, if one has been placed
    yet. It's what devices point at in their gateway_id metadata."""
    catalog = load_device_catalog()
    for _floor, _space, _ap, device in inst.iter_devices():
        if catalog.get(device.device_type_id, {}).get("category") in GATEWAY_CATEGORIES:
            return device.instance_name
    return None


def resolve_default_metadata(spec: dict, inst: Installation, floor, space,
                             access_point=None) -> tuple[dict, list[str]]:
    """Fills in the catalogue's default_metadata templates from the model.

    Hands back what it managed to work out, and separately the keys it
    couldn't. Those get left empty rather than stuffed with a placeholder -
    usually gateway_id, when there's no gateway yet: so validation still
    reports the gap instead of the model carrying a value that means nothing"""
    values = {
        "building": inst.building.name if inst.building else "",
        "floor": floor.name,
        "space": space.name,
    }
    if access_point is not None:
        values["access_point"] = access_point.name
    gateway = find_gateway(inst)
    if gateway:
        values["gateway"] = gateway

    resolved, unresolved = {}, []
    for key, template in (spec.get("default_metadata") or {}).items():
        try:
            resolved[key] = str(template).format(**values)
        except KeyError:
            # Nothing to put in the placeholder, so leave the field empt
            unresolved.append(key)
    return resolved, unresolved
