"""Working out which catalogue entry the architect meant

What a device is called and what an architect calls it are both facts about
the catalogue, so they live next to it and no agent has to be updated
"""

import re
from collections import Counter
from functools import lru_cache

from iot_agentic_deployer.domain.catalog.loader import load_device_catalog


NAME_SIGNAL = 10     # the architect named the device
DATA_SIGNAL = 1      # the architect described what it measures or does


# Words for the category rather than the product: no entry gets to own these
GENERIC_WORDS = {"device", "devices", "sensor", "sensors", "unit", "units",
                 "system", "systems", "module", "modules"}


@lru_cache(maxsize=1)
def _distinctive_tokens() -> dict[str, set[str]]:
    """Words from ids, display names and aliases that point at exactly one
    catalogue entry. Worked out rather than listed, so adding an entry
    re-sorts the vocabulary on its own"""
    catalog = load_device_catalog()
    per_device = {
        device_id: set(re.findall(r"[a-z0-9]{3,}", " ".join([
            device_id.replace(".", " ").replace("_", " "),
            spec["display_name"],
            *spec.get("aliases", []),
        ]).lower()))
        for device_id, spec in catalog.items()
    }
    counts = Counter(token for tokens in per_device.values() for token in tokens)
    return {
        device_id: {t for t in tokens if counts[t] == 1 and t not in GENERIC_WORDS}
        for device_id, tokens in per_device.items()
    }


def match_device_types(text: str) -> list[tuple[str, str]]:
    """Ranks catalogue entries against what the architect wrote, best first,
    as (device_type_id, reason).

    Matches on what a device is called and on what it does: telemetry_model
    and capabilities are what make 'a temperature sensor' find every entry
    that measures temperature. Naming a device beats describing one"""
    lowered = text.lower()
    distinctive = _distinctive_tokens()
    ranked = []

    for device_id, spec in load_device_catalog().items():
        score, reasons = 0, []

        phrases = [spec["display_name"].lower(),
                   device_id.split(".", 1)[-1].replace("_", " ").lower()]
        phrases += [a.lower() for a in spec.get("aliases", [])]
        named = device_id.lower() in lowered or any(p in lowered for p in phrases)
        tokens = sorted(t for t in distinctive[device_id]
                        if re.search(rf"\b{re.escape(t)}\b", lowered))
        if named or tokens:
            score += NAME_SIGNAL
            reasons.append("named explicitly" if named else f"you said '{tokens[0]}'")

        described = sorted({
            term for term in
            list(spec.get("telemetry_model", [])) + list(spec.get("capabilities", []))
            if re.search(rf"\b{re.escape(term.replace('_', ' '))}\b", lowered)
        })
        if described:
            score += DATA_SIGNAL * len(described)
            reasons.append("measures " + ", ".join(described))

        if score:
            # Ties go to the simpler device: 'temperature sensor' gets the
            # plain sensor, not the multisensor
            ranked.append((-score, len(spec.get("telemetry_model", [])), device_id,
                           "; ".join(reasons)))

    ranked.sort()
    return [(device_id, reason) for _s, _n, device_id, reason in ranked]
