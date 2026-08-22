"""Rendering helpers, all plain code. Nothing here goes near the LLM: tables
and diagrams are built in Python so they always come out well-formed.
"""
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

            # Access points hang off their space, and anything installed on
            # one hangs off the door rather than the room. A people counter
            # belongs to the doorway it watches, and the picture should say
            # so - the same distinction the model and the deployment plan
            # both make (thesis 5.1).
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
        "| Floor | Space | Type | Devices |",
        "| --- | --- | --- | --- |",
    ]

    for floor, space in inst.iter_spaces():
        if space.devices:
            names = ", ".join(
                catalog.get(d.device_type_id, {}).get("display_name", d.device_type_id)
                for d in space.devices
            )
        else:
            names = "_none_"
        lines.append(f"| {floor.name} | {space.name} | {space.type} | {names} |")

    total_spaces = sum(1 for _ in inst.iter_spaces())
    lines.append("")
    lines.append(f"_{total_spaces} space(s), {len(inst.all_devices())} device(s) configured._")
    return "\n".join(lines)


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