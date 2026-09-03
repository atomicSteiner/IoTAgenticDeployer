"""Rendering helpers, all plain code. Nothing here goes near the LLM: tables
and diagrams are built in Python so they always come out well-formed.
"""
from collections import Counter

from iot_agentic_deployer.domain.catalog.loader import load_device_catalog
from iot_agentic_deployer.domain.models import Installation


# ------------------------------------------------------- installation model


def _sanitize(text: str) -> str:
    """Mermaid node ids will not take spaces or punctuation."""
    return "".join(c if c.isalnum() else "_" for c in text)


def installation_to_mermaid(inst: Installation) -> str:
    """Draws the building topology as a Mermaid graph (rationale §7,
    "Configuration summary")."""
    if not inst.building:
        return ""

    lines = ["graph TD"]
    b_id = f"B_{_sanitize(inst.building.name)}"
    lines.append(f'    {b_id}["🏢 {inst.building.name}"]')

    for floor in inst.building.floors:
        f_id = f"F_{_sanitize(floor.name)}"
        lines.append(f'    {f_id}["🧱 {floor.name}"]')
        lines.append(f"    {b_id} --> {f_id}")

        for space in floor.spaces:
            s_id = f"S_{_sanitize(floor.name)}_{_sanitize(space.name)}"
            lines.append(f'    {s_id}["{space.name}<br/><i>{space.type}</i>"]')
            lines.append(f"    {f_id} --> {s_id}")

            for dev in space.devices:
                lines += _device_node(dev, s_id)

            # Access points hang off their space, and devices installed on one
            # hang off the door, not the room - as model and plan do (5.1).
            for ap in space.access_points:
                ap_id = f"A_{_sanitize(floor.name)}_{_sanitize(space.name)}_{_sanitize(ap.name)}"
                lines.append(f'    {ap_id}{{{{"🚪 {ap.name}"}}}}')
                lines.append(f"    {s_id} --> {ap_id}")

                for dev in ap.devices:
                    lines += _device_node(dev, ap_id)

    return "\n".join(lines)


def _device_node(device, parent_id: str) -> list[str]:
    """A device, drawn hanging off whatever it is installed in."""
    d_id = f"D_{_sanitize(device.instance_name)}"
    return [f'    {d_id}("📡 {device.instance_name}")',
            f"    {parent_id} --> {d_id}"]


def installation_to_markdown(inst: Installation) -> str:
    """A short written summary of the configuration so far."""
    if not inst.building:
        return "No building configured yet."

    catalog = load_device_catalog()
    lines = [
        f"**Installation:** {inst.name}  ",
        f"**Building:** {inst.building.name}"
        + (f" ({inst.building.location})" if inst.building.location else "") + "  ",
        f"**Use case:** {inst.use_case or '_not selected_'}",
        "",
        "| Floor | Space | Type | Devices | Ways in |",
        "| --- | --- | --- | --- | --- |",
    ]

    for floor, space in inst.iter_spaces():
        # The ways in are a column of their own, devices and all: a counter on
        # a door was in the model and in the diagram, and nowhere in the table
        ways = ", ".join(
            ap.name + (f" ({_device_names(ap.devices, catalog)})" if ap.devices else "")
            for ap in space.access_points
        )
        lines.append(f"| {floor.name} | {space.name} | {space.type} "
                     f"| {_device_names(space.devices, catalog) or '_none_'} "
                     f"| {ways or '_none_'} |")

    total_spaces = sum(1 for _ in inst.iter_spaces())
    ways_in = sum(len(space.access_points) for _, space in inst.iter_spaces())
    lines.append("")
    lines.append(f"_{total_spaces} space(s), {ways_in} way(s) in, "
                 f"{len(inst.all_devices())} device(s) configured._")
    return "\n".join(lines)


def _device_names(devices, catalog: dict) -> str:
    """Counted, not listed: three of one type in a room would otherwise read as
    the same name written out three times"""
    counted = Counter(
        catalog.get(d.device_type_id, {}).get("display_name", d.device_type_id)
        for d in devices
    )
    return ", ".join(n if c == 1 else f"{n} x{c}" for n, c in counted.items())


def validation_to_markdown(report: dict) -> str:
    """Lays out a report from validation.validate_installation."""
    if not report.get("findings"):
        return "✅ All pre-deployment checks passed."

    icons = {"error": "❌", "warning": "⚠️"}
    lines = [
        f"**Pre-deployment check:** {report['errors']} error(s), "
        f"{report['warnings']} warning(s)",
        "",
    ]
    lines += [
        f"- {icons.get(f['level'], '•')} {f['message']}" for f in report["findings"]
    ]
    lines.append("")
    lines.append(
        "✅ Configuration is deployable."
        if report["deployable"]
        else "🚫 Deployment blocked until the errors above are resolved."
    )
    return "\n".join(lines)

def trace_to_markdown(trace: list) -> str:
    """Everything that was handed off stays visible to the architect (R7)."""
    if not trace:
        return "_No activity recorded yet._"
    lines = ["| Agent | Action | Detail |", "| --- | --- | --- |"]
    lines += [f"| {e.get('agent', '')} | {e.get('action', '')} | {e.get('detail', '')} |"
              for e in trace]
    return "\n".join(lines)