"""Lazy, in-tree registry for operator backend providers."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from types import ModuleType
from typing import Any, Mapping

from quack.backends.target import Target

_STATIC_PROVIDERS = {
    ("rmsnorm", "nvidia"): "quack.backends.cute.rmsnorm_provider",
    ("rmsnorm", "amd"): "quack.backends.flydsl.rmsnorm_provider",
}


class BackendRegistry:
    """Map ``(operator, vendor)`` to a lazily imported provider module."""

    def __init__(self, providers: Mapping[tuple[str, str], str]):
        self._providers = dict(providers)

    def provider_path(self, operator: str, vendor: str) -> str:
        try:
            return self._providers[(operator, vendor)]
        except KeyError:
            raise LookupError(
                f"no Quack provider is registered for operator={operator!r}, vendor={vendor!r}"
            ) from None

    @lru_cache(maxsize=None)
    def load_provider(self, operator: str, vendor: str) -> ModuleType:
        module = import_module(self.provider_path(operator, vendor))
        missing = [
            name
            for name in ("is_available", "supports", "load_op")
            if not callable(getattr(module, name, None))
        ]
        if missing:
            raise TypeError(
                f"backend provider {module.__name__!r} is missing callables: {', '.join(missing)}"
            )
        return module

    def resolve(self, operator: str, spec: Any, target: Target):
        """Load and validate only the provider selected by ``target.vendor``."""

        provider = self.load_provider(operator, target.vendor)
        if not provider.is_available():
            raise RuntimeError(f"provider {provider.__name__!r} is unavailable for {operator!r}")
        support = provider.supports(spec, target)
        if not support:
            raise NotImplementedError(support.reason)
        return provider.load_op()


REGISTRY = BackendRegistry(_STATIC_PROVIDERS)

__all__ = ["BackendRegistry", "REGISTRY"]
