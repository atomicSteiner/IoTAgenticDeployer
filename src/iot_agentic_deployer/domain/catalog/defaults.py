"""Starting values worked out from the catalogue and the model
"""

from iot_agentic_deployer.domain.catalog.loader import GATEWAY_CATEGORIES, load_device_catalog
from iot_agentic_deployer.domain.models import Installation


def device_instance_name(spec: dict, floor, space, access_point=None,
                         index: int = 1) -> str:
    """What a device instance gets called, built from where it sits so it comes
    out the same every time. `index` tells apart several of one type in one
    room; the first keeps the bare name, so nothing already deployed moves"""
    parts = [spec["device_type_id"].replace(".", "_"), floor.name, space.name]
    if access_point is not None:
        parts.append(access_point.name)
    name = "-".join(parts).replace(" ", "_")
    return name if index <= 1 else f"{name}-{index}"


def find_gateway(inst: Installation) -> str | None:
    """Name of the gateway serving this installation, if one has been placed
    yet. It's what devices point at in their gateway_id metadata."""
    catalog = load_device_catalog()
    for _floor, _space, _ap, device in inst.iter_devices():
        if catalog.get(device.device_type_id, {}).get("category") in GATEWAY_CATEGORIES:
            return device.instance_name
    return None


def resolve_default_metadata(spec: dict, inst: Installation, floor, space,
                             access_point=None, index: int = 1) -> tuple[dict, list[str]]:
    """Fills the catalogue's default_metadata templates from the model, handing
    back what it resolved and, separately, the keys it could not. Those stay
    empty rather than hold a placeholder, so validation still reports the gap"""
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

    # The label is what a human reads off the device, so where a room holds
    # several of one type it has to say which of them this is
    if index > 1 and "physical_label" in resolved:
        resolved["physical_label"] = f"{resolved['physical_label']} {index}"
    return resolved, unresolved
