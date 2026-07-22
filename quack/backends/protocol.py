"""Provider protocol for Quack operator backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from quack.backends.target import Target


@dataclass(frozen=True, slots=True)
class Support:
    """A provider support decision with an actionable rejection reason."""

    supported: bool
    reason: str = ""

    @classmethod
    def yes(cls) -> "Support":
        return cls(True)

    @classmethod
    def no(cls, reason: str) -> "Support":
        if not reason:
            raise ValueError("an unsupported decision requires a reason")
        return cls(False, reason)

    def __bool__(self) -> bool:
        return self.supported


@runtime_checkable
class BackendProvider(Protocol):
    """Operator-level provider contract.

    Providers are modules with these callables; they remain unloaded until the
    registry selects their operator/vendor pair.
    """

    def is_available(self) -> bool: ...

    def supports(self, spec: Any, target: Target) -> Support: ...

    def load_op(self) -> Callable: ...
