"""Platform adapters

Each supported platform exposes what it can provision as a set of typed
operations. Keeping the platform-specific parts behind that boundary is what
lets another platform be added without redesigning the configuration layer
above
"""

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Literal, Optional

from pydantic import BaseModel, Field

OperationKind = Literal["create_asset", "create_device", "set_attributes", "create_relation"]


class PlatformOperation(BaseModel):
    """A single step in the ordered list of operations planning produces."""
    op_id: str
    kind: OperationKind
    description: str                       # shown to the architect in the plan
    params: dict = Field(default_factory=dict)
    # Filled during realisation:
    status: Literal["pending", "completed", "failed", "skipped"] = "pending"
    error: Optional[str] = None
    result_ref: Optional[str] = None       # id assigned by the platform


class PlatformAdapter(ABC):
    """What every target platform has to be able to do."""

    name: str = "abstract"

    @abstractmethod
    def check_reachability(self) -> tuple[bool, str]:
        """Returns (reachable, something a human can read)."""

    @abstractmethod
    def create_asset(self, name: str, asset_type: str) -> str: ...

    @abstractmethod
    def create_device(self, name: str, device_type: str, label: str | None = None) -> str: ...

    @abstractmethod
    def set_attributes(self, entity_id: str, entity_type: str, scope: str, attributes: dict) -> None: ...

    @abstractmethod
    def create_relation(self, from_id: str, from_type: str, to_id: str, to_type: str) -> None: ...

    def batch(self):
        """Somewhere to hold a connection open for a run of operations. An
        adapter with nothing to keep open need not override it"""
        return nullcontext()

    # -- dispatch ---------------------------------------------------------

    def execute(self, op: PlatformOperation, refs: dict) -> str | None:
        """Runs one operation. `refs` maps op_id to the platform id it
        produced, which is how an operation points at something an earlier
        one created."""
        p = dict(op.params)

        # Fill in any {"$ref": "<op_id>"} placeholders
        for key, value in list(p.items()):
            if isinstance(value, dict) and "$ref" in value:
                p[key] = refs.get(value["$ref"])

        if op.kind == "create_asset":
            return self.create_asset(p["name"], p["asset_type"])
        if op.kind == "create_device":
            return self.create_device(p["name"], p["device_type"], p.get("label"))
        if op.kind == "set_attributes":
            self.set_attributes(p["entity_id"], p.get("entity_type", "DEVICE"),
                                p.get("scope", "SERVER_SCOPE"), p["attributes"])
            return None
        if op.kind == "create_relation":
            self.create_relation(p["from_id"], p["from_type"], p["to_id"], p["to_type"])
            return None
        raise ValueError(f"Unsupported operation kind: {op.kind}")


_REGISTRY: dict[str, type[PlatformAdapter]] = {}


def register_adapter(cls: type[PlatformAdapter]):
    _REGISTRY[cls.name] = cls
    return cls


def get_adapter(platform: str, **kwargs) -> PlatformAdapter:
    if platform not in _REGISTRY:
        raise ValueError(f"No adapter registered for platform '{platform}'. "
                         f"Available: {sorted(_REGISTRY)}")
    return _REGISTRY[platform](**kwargs)


def available_platforms() -> list[str]:
    return sorted(_REGISTRY)