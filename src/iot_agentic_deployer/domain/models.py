"""Conceptual model of an IoT installation 
Things sit inside each other like this:
    Tenant -> Installation -> Building -> Floor -> Space -> AccessPoint
and a device belongs to whichever Space or AccessPoint it is installed in.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

SpaceType = Literal[
    "classroom", "office", "laboratory", "meeting_room",
    "corridor", "technical_room", "common_area", "outdoor",
    "unknown",
]

Accessibility = Literal["public", "restricted", "staff_only", "unknown"]
Priority = Literal["low", "normal", "high", "critical"]


class Device(BaseModel):
    """One real device, of a type taken from the catalogue."""
    device_type_id: str
    instance_name: str
    metadata: dict = Field(default_factory=dict)


class AccessPoint(BaseModel):
    """A door, passage or entrance belonging to a space - somewhere a people
    counter and the like can be mounted (Table 5.1)."""
    name: str
    devices: list[Device] = Field(default_factory=list)


class Space(BaseModel):
    name: str
    type: SpaceType = "unknown"
    # Details deployment leans on later (thesis 4.2, "Enrich spaces").
    intended_use: Optional[str] = None
    accessibility: Accessibility = "unknown"
    priority: Priority = "normal"
    devices: list[Device] = Field(default_factory=list)
    access_points: list[AccessPoint] = Field(default_factory=list)


class Floor(BaseModel):
    name: str
    level: int = 0
    spaces: list[Space] = Field(default_factory=list)


class Building(BaseModel):
    name: str
    location: Optional[str] = None
    floors: list[Floor] = Field(default_factory=list)


class Exclusion(BaseModel):
    """A deliberate, written-down decision not to meet a profile rule
    (thesis 5.6.2). Kept in the model rather than quietly dropped, so
    validation can tell a decision apart from an oversight."""
    rule: str
    scope: str            # space name, or "installation" for a global rule
    justification: str


class PlatformTarget(BaseModel):
    """Where the installation gets deployed. The architect picks this
    (thesis 5.6.1, step 8) - it is not fixed up front."""
    platform: str = "thingsboard"
    endpoint: Optional[str] = None
    authenticated: bool = False


class Installation(BaseModel):
    """The root of it all - the shared configuration every agent works on.

    Agents read and change this, it is what gets checkpointed between steps,
    and it is what the architect is shown when they ask (thesis 5.1).
    """
    tenant: str = "default-tenant"
    name: str = "unnamed-installation"
    use_case: Optional[str] = None
    building: Optional[Building] = None
    exclusions: list[Exclusion] = Field(default_factory=list)
    target: PlatformTarget = Field(default_factory=PlatformTarget)

    def iter_spaces(self):
        if not self.building:
            return
        for floor in self.building.floors:
            for space in floor.spaces:
                yield floor, space

    def iter_devices(self):
        """Walks every device, as (floor, space, access_point or None, device)."""
        for floor, space in self.iter_spaces():
            for device in space.devices:
                yield floor, space, None, device
            for ap in space.access_points:
                for device in ap.devices:
                    yield floor, space, ap, device

    def all_devices(self) -> list[Device]:
        return [d for *_, d in self.iter_devices()]

    def has_exclusion(self, rule: str, scope: str = "installation") -> bool:
        return any(e.rule == rule and e.scope == scope for e in self.exclusions)