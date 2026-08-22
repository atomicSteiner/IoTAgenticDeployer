"""Working out which catalogue entry the architect meant (thesis 5.2, R4).

Deterministic, and kept out of the agents on purpose. What a device is
called and what an architect calls it are both facts about the catalogue, so
they live next to it: declare a new entry and it is understood straight away,
with no agent to update.
"""

import re
from collections import Counter
from functools import lru_cache

from iot_agentic_deployer.domain.catalog.loader import load_device_catalog


NAME_SIGNAL = 10     # the architect named the device
DATA_SIGNAL = 1      # the architect described what it measures or does


# Words for the category rather than the product. No entry gets to own
# these, however few of them happen to spell one out.
GENERIC_WORDS = {"device", "devices", "sensor", "sensors", "unit", "units",
                 "system", "systems", "module", "modules"}


@lru_cache(maxsize=1)
def _distinctive_tokens() -> dict[str, set[str]]:
    """Words from ids, display names and aliases that point at exactly one
    catalogue entry.

    Worked out rather than listed, so adding an entry re-sorts the vocabulary
    on its own: 'multisensor' turns up once and so identifies its entry,
    while 'dipme' runs across the whole vendor range and identifies nothing.
    Counting alone doesn't quite do it, though - a word can be generic and
    still appear in only one display name - which is what GENERIC_WORDS is
    there to catch."""
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
    """Ranks catalogue entries against what the architect actually wrote,
    best match first, as (device_type_id, reason).

    People ask for what a device does about as often as for what it's called,
    and the catalogue already knows both: `telemetry_model` and `capabilities`
    are what make 'a temperature sensor' mean every entry that measures
    temperature. Reading them lets the request and the catalogue meet in the
    middle, rather than the catalogue having to be dumbed down.

    Naming a device beats describing one, which is why 'a multisensor,
    specifically a temperature sensor' lands on the multisensor it names
    instead of on everything that happens to measure temperature."""
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
            # Ties go to the simpler device. Ask for a temperature sensor and
            # you get the plain sensor, not the multisensor that measures
            # temperature along with four other things.
            ranked.append((-score, len(spec.get("telemetry_model", [])), device_id,
                           "; ".join(reasons)))

    ranked.sort()
    return [(device_id, reason) for _s, _n, device_id, reason in ranked]
