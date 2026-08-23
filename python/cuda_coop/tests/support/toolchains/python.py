# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scoped loaders for repository-local Python tool modules."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


def load_tool_module(module_name: str, tools_root: Path) -> ModuleType:
    """Import a tool without leaving its directory on ``sys.path``."""

    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(tools_root))
        importlib.invalidate_caches()
        return importlib.import_module(module_name)
    finally:
        sys.path[:] = original_path


__all__ = ["load_tool_module"]
