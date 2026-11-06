# Copyright (c) 2026, Tri Dao.

"""Torch build facts that are safe before either DSL runtime is imported."""

import torch

IS_ROCM_BUILD = torch.version.hip is not None

__all__ = ["IS_ROCM_BUILD"]
