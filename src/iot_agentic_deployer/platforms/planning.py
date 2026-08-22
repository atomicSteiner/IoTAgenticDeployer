"""Behaviour planning (thesis 5.4, 5.6.3).

Works out the operations that would turn the configuration model into a real
installation, without running any of them. Planning is kept apart from doing,
so the architect can read what is about to happen before anything is written
(R3).

None of this is generated: it is a straight reading of the model and the
catalogue. Platform-specific naming comes from each entry's
`platform_mapping` instead of being written in here, so supporting another
platform is a catalogue edit rather than a change to this file (R4, R6).
"""

from iot_agentic_deployer.domain.catalog.loader import load_device_catalog
from iot_agentic_deployer.domain.models import Installation
from iot_agentic_deployer.platforms.base import PlatformOperation

# Asset types for the containment hierarchy. Devices bring their own type
# from the catalogue; the spatial levels do not, so they are named here.
BUILDING_ASSET_TYPE = "building"
FLOOR_ASSET_TYPE = "floor"
SPACE_ASSET_TYPE = "space"
ACCESS_POINT_ASSET_TYPE = "access_point"

STATUS_ICONS = {"pending": "·", "completed": "✅", "failed": "❌", "skipped": "⏭️"}


def _ref(op_id: str) -> dict:
    """Points at an id some earlier operation will produce. Filled in at
    execution time by PlatformAdapter.execute."""
    return {"$ref": op_id}


def _relation(op_id: str, parent_op: str, parent_kind: str,
              child_op: str, child_kind: str, description: str) -> PlatformOperation:
    return PlatformOperation(
        op_id=op_id,
        kind="create_relation",
        description=description,
        params={
            "from_id": _ref(parent_op), "from_type": parent_kind,
            "to_id": _ref(child_op), "to_type": child_kind,
        },
    )


def derive_plan(inst: Installation) -> list[PlatformOperation]:
    """Turns the installation into an ordered list of platform operations.

    They come out in the model's own containment order - building, floors,
    spaces, access points, devices - so nothing ever refers to a parent that
    has not been created yet. op_ids come from the model rather than being
    generated, which means planning the same configuration twice gives the
    same ids, and a retry can still find entities created during an earlier
    half-finished run (R7)."""
    if not inst.building:
        return []

    catalog = load_device_catalog()
    platform = inst.target.platform
    ops: list[PlatformOperation] = []

    building_op = "asset:building"
    ops.append(PlatformOperation(
        op_id=building_op,
        kind="create_asset",
        description=f"Create building asset '{inst.building.name}'",
        params={"name": inst.building.name, "asset_type": BUILDING_ASSET_TYPE},
    ))

    for floor in inst.building.floors:
        floor_op = f"asset:floor:{floor.name}"
        floor_name = f"{inst.building.name} / {floor.name}"
        ops.append(PlatformOperation(
            op_id=floor_op,
            kind="create_asset",
            description=f"Create floor asset '{floor_name}'",
            params={"name": floor_name, "asset_type": FLOOR_ASSET_TYPE},
        ))
        ops.append(_relation(
            f"rel:{floor_op}", building_op, "ASSET", floor_op, "ASSET",
            f"Relate '{floor.name}' to '{inst.building.name}'"))

        for space in floor.spaces:
            space_op = f"asset:space:{floor.name}:{space.name}"
            space_name = f"{floor_name} / {space.name}"
            ops.append(PlatformOperation(
                op_id=space_op,
                kind="create_asset",
                description=f"Create {space.type} asset '{space_name}'",
                params={"name": space_name, "asset_type": SPACE_ASSET_TYPE},
            ))
            ops.append(_relation(
                f"rel:{space_op}", floor_op, "ASSET", space_op, "ASSET",
                f"Relate '{space.name}' to '{floor.name}'"))

            for device in space.devices:
                ops += _device_operations(device, space_op, "ASSET", space_name,
                                          catalog, platform)

            # Access points are entities in their own right. A device on a
            # door belongs to the door, not to the room, and the deployed
            # hierarchy should say so (thesis 5.1).
            for access_point in space.access_points:
                ap_op = f"asset:ap:{floor.name}:{space.name}:{access_point.name}"
                ap_name = f"{space_name} / {access_point.name}"
                ops.append(PlatformOperation(
                    op_id=ap_op,
                    kind="create_asset",
                    description=f"Create access point asset '{ap_name}'",
                    params={"name": ap_name, "asset_type": ACCESS_POINT_ASSET_TYPE},
                ))
                ops.append(_relation(
                    f"rel:{ap_op}", space_op, "ASSET", ap_op, "ASSET",
                    f"Relate '{access_point.name}' to '{space.name}'"))

                for device in access_point.devices:
                    ops += _device_operations(device, ap_op, "ASSET", ap_name,
                                              catalog, platform)

    return ops


def _device_operations(device, parent_op: str, parent_kind: str, parent_name: str,
                       catalog: dict, platform: str) -> list[PlatformOperation]:
    """Everything needed to create one device and hang it off whatever it is
    installed in."""
    spec = catalog.get(device.device_type_id, {})
    mapping = spec.get("platform_mapping", {}).get(platform, {})
    display_name = spec.get("display_name", device.device_type_id)

    # The catalogue decides whether this ends up a device or an asset, which
    # is what lets one model deploy onto platforms that disagree about the
    # distinction (R6).
    as_asset = mapping.get("entity") == "asset"
    device_op = f"device:{device.instance_name}"

    if as_asset:
        create = PlatformOperation(
            op_id=device_op,
            kind="create_asset",
            description=f"Create {display_name} asset '{device.instance_name}' in {parent_name}",
            params={"name": device.instance_name,
                    "asset_type": mapping.get("asset_type", device.device_type_id)},
        )
        entity_kind = "ASSET"
    else:
        create = PlatformOperation(
            op_id=device_op,
            kind="create_device",
            description=f"Create {display_name} '{device.instance_name}' in {parent_name}",
            params={"name": device.instance_name,
                    "device_type": mapping.get("type", device.device_type_id),
                    "label": display_name},
        )
        entity_kind = "DEVICE"

    ops = [create]

    # Carry the required metadata across too, so what lands on the platform
    # can still be traced back to the decisions in the model instead of
    # turning up anonymous (R7).
    attributes = dict(device.metadata)
    attributes.setdefault("device_type_id", device.device_type_id)
    ops.append(PlatformOperation(
        op_id=f"attrs:{device.instance_name}",
        kind="set_attributes",
        description=f"Set attributes of '{device.instance_name}'",
        params={"entity_id": _ref(device_op), "entity_type": entity_kind,
                "scope": "SERVER_SCOPE", "attributes": attributes},
    ))

    ops.append(_relation(
        f"rel:{device_op}", parent_op, parent_kind, device_op, entity_kind,
        f"Relate '{device.instance_name}' to '{parent_name}'"))
    return ops


def plan_to_markdown(plan: list[PlatformOperation]) -> str:
    """Renders the plan for the architect. Built in code like every other
    view of the model, so what gets confirmed is exactly what was
    derived."""
    if not plan:
        return "_No operation is required for this configuration._"

    lines = ["| # | | Operation |", "| --- | --- | --- |"]
    lines += [
        f"| {i} | {STATUS_ICONS.get(op.status, '·')} | {op.description} |"
        for i, op in enumerate(plan, 1)
    ]

    done = sum(1 for op in plan if op.status == "completed")
    lines.append("")
    lines.append(f"_{len(plan)} operation(s), {done} completed._")
    return "\n".join(lines)
