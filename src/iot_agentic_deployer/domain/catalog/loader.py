"""Reads the device catalogue and the use-case profiles off disk"""

from functools import lru_cache
from pathlib import Path

import yaml

# loader.py lives in catalog/, so the YAML files are right next to it.
CATALOG_DIR = Path(__file__).parent

GATEWAY_CATEGORIES = {"gateway", "coordinator"}


@lru_cache(maxsize=1)
def load_device_catalog() -> dict:
    data = yaml.safe_load((CATALOG_DIR / "devices.yaml").read_text(encoding="utf-8"))
    return {d["device_type_id"]: d for d in data.get("devices", [])}


@lru_cache(maxsize=1)
def load_use_cases() -> dict:
    data = yaml.safe_load((CATALOG_DIR / "use_cases.yaml").read_text(encoding="utf-8"))
    return data.get("use_cases", {})


def profile_rules(use_case: str | None) -> dict:
    return (load_use_cases().get(use_case) or {}).get("rules", {})


def devices_for_use_case(use_case: str | None) -> list[dict]:
    """Limits what the system is allowed to suggest, so proposals come from
    equipment that actually exists rather than from whatever the language
    model happens to remember"""
    catalog = load_device_catalog()
    if not use_case:
        return list(catalog.values())
    recommended = (load_use_cases().get(use_case) or {}).get("recommended_devices") or []
    if not recommended:
        return list(catalog.values())
    return [catalog[d] for d in recommended if d in catalog]


def load_platform_mappings() -> dict:
    """How each platform names what the model holds, indexed by platform. Its
    own file rather than a block per catalogue entry: that grew as devices
    times platforms, and a third target meant editing all of them"""
    data = yaml.safe_load(
        (CATALOG_DIR / "platform_mappings.yaml").read_text(encoding="utf-8"))
    return data.get("platforms", {})


def platform_mapping(device_type_id: str, platform: str) -> dict:
    """What this platform calls this device, or nothing when it has no name
    for it - which is a configuration gap, not something to guess at"""
    return load_platform_mappings().get(platform, {}).get("devices", {}).get(device_type_id, {})


def spatial_mapping(platform: str) -> dict:
    """What this platform calls a building, a floor, a space and a door."""
    return load_platform_mappings().get(platform, {}).get("spatial", {})