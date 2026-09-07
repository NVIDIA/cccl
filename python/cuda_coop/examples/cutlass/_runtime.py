# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime import helpers for CUTLASS examples."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_COOP_PATH = SOURCE_ROOT / "cuda" / "coop"
COOP_PARENT_SCOPE = "cuda.coop"
ROOT_SCOPE = "cuda.coop.cutlass"
_INSTALLED_COOP_ENV = "CUDA_COOP_EXAMPLES_USE_INSTALLED_CUDA_COOP"


def _drop_stale_cuda_coop_editable_finders() -> None:
    kept = []
    for finder in sys.meta_path:
        if finder.__class__.__module__ != "_cuda_coop_editable":
            kept.append(finder)
            continue

        finder_path = getattr(finder, "path", None)
        if finder_path is None:
            # Source-tree examples must prefer this checkout over opaque
            # cuda-coop editable redirectors from other worktrees.
            continue
        try:
            resolved = Path(finder_path).resolve()
        except OSError:
            # Keep only editables whose target can be proven to be this tree.
            continue
        if resolved == SOURCE_ROOT or SOURCE_ROOT in resolved.parents:
            kept.append(finder)
    sys.meta_path[:] = kept


def _prepend_source_cuda_coop_path() -> None:
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))

    coop_parent = importlib.import_module(COOP_PARENT_SCOPE)

    source_coop_path = str(SOURCE_COOP_PATH)
    resolved_source_coop_path = str(SOURCE_COOP_PATH.resolve())
    parent_paths = list(coop_parent.__path__)
    if (
        parent_paths
        and str(Path(parent_paths[0]).resolve()) == resolved_source_coop_path
    ):
        return

    coop_parent.__path__ = [
        source_coop_path,
        *(
            path
            for path in parent_paths
            if str(Path(path).resolve()) != resolved_source_coop_path
        ),
    ]
    spec_locations = getattr(coop_parent.__spec__, "submodule_search_locations", None)
    if spec_locations is not None:
        coop_parent.__spec__.submodule_search_locations = list(coop_parent.__path__)


def _use_installed_cuda_coop() -> bool:
    value = os.environ.get(_INSTALLED_COOP_ENV)
    if value not in (None, "0", "1"):
        raise RuntimeError(f"{_INSTALLED_COOP_ENV} must be 0 or 1 when set")
    return value == "1"


def _path_is_in_active_environment(path: str | Path) -> bool:
    return Path(path).resolve().is_relative_to(Path(sys.prefix).resolve())


def _reject_cuda_coop_editable_finders() -> None:
    if any(
        finder.__class__.__module__ == "_cuda_coop_editable" for finder in sys.meta_path
    ):
        raise RuntimeError(
            f"{_INSTALLED_COOP_ENV}=1 rejects cuda-coop editable import finders"
        )


def _remove_source_root_from_sys_path() -> None:
    source_root = SOURCE_ROOT.resolve()
    retained = []
    for entry in sys.path:
        try:
            resolved = Path(entry or ".").resolve()
        except OSError:
            retained.append(entry)
            continue
        if resolved != source_root:
            retained.append(entry)
    sys.path[:] = retained


def _prepare_installed_cuda_coop_parent() -> None:
    _reject_cuda_coop_editable_finders()
    # Repository examples still come from SOURCE_ROOT, but the package under
    # qualification must resolve from the active wheel environment.
    _remove_source_root_from_sys_path()
    coop_parent = importlib.import_module(COOP_PARENT_SCOPE)
    parent_file = getattr(coop_parent, "__file__", None)
    if parent_file is None or not _path_is_in_active_environment(parent_file):
        raise RuntimeError(
            f"{COOP_PARENT_SCOPE} did not resolve from the active installed "
            f"environment {Path(sys.prefix).resolve()}: {parent_file}"
        )
    installed_paths = [
        path for path in coop_parent.__path__ if _path_is_in_active_environment(path)
    ]
    if not installed_paths:
        raise RuntimeError(
            f"{COOP_PARENT_SCOPE} has no package path in the active installed "
            f"environment {Path(sys.prefix).resolve()}"
        )
    coop_parent.__path__ = installed_paths
    spec_locations = getattr(coop_parent.__spec__, "submodule_search_locations", None)
    if spec_locations is not None:
        coop_parent.__spec__.submodule_search_locations = list(installed_paths)


def _import_installed_cuda_coop_cutlass() -> Any:
    _prepare_installed_cuda_coop_parent()

    loaded_modules = {
        module_name: module
        for module_name, module in sys.modules.items()
        if module_name == ROOT_SCOPE or module_name.startswith(f"{ROOT_SCOPE}.")
    }
    for module_name, module in loaded_modules.items():
        module_file = getattr(module, "__file__", None)
        if module_file is None or not _path_is_in_active_environment(module_file):
            raise RuntimeError(
                f"{module_name} was already loaded outside the active installed "
                f"environment {Path(sys.prefix).resolve()}: {module_file}"
            )

    loaded = sys.modules.get(ROOT_SCOPE)
    if loaded is not None:
        return loaded

    importlib.invalidate_caches()
    coop = importlib.import_module(ROOT_SCOPE)
    module_file = getattr(coop, "__file__", None)
    environment_prefix = Path(sys.prefix).resolve()
    if module_file is None or not _path_is_in_active_environment(module_file):
        raise RuntimeError(
            f"{ROOT_SCOPE} did not resolve from the active installed environment "
            f"{environment_prefix}: {module_file}"
        )
    return coop


def _import_cuda_coop_cutlass() -> Any:
    if _use_installed_cuda_coop():
        return _import_installed_cuda_coop_cutlass()

    _drop_stale_cuda_coop_editable_finders()
    _prepend_source_cuda_coop_path()

    loaded = sys.modules.get(ROOT_SCOPE)
    if loaded is not None:
        loaded_file = getattr(loaded, "__file__", None)
        if loaded_file is not None and Path(loaded_file).resolve().is_relative_to(
            SOURCE_ROOT
        ):
            return loaded

    for module_name in list(sys.modules):
        if module_name == "cuda.coop.cutlass" or module_name.startswith(
            "cuda.coop.cutlass."
        ):
            del sys.modules[module_name]

    importlib.invalidate_caches()
    coop = importlib.import_module(ROOT_SCOPE)
    module_file = Path(coop.__file__).resolve()
    if not module_file.is_relative_to(SOURCE_ROOT):
        raise RuntimeError(
            f"{ROOT_SCOPE} resolved outside this source tree: {module_file}"
        )
    return coop


def require_runtime(
    *,
    include_int32: bool = False,
) -> tuple[Any, ...]:
    """Return CUTLASS, torch, DLPack, and selected coop runtime handles."""

    try:
        import cutlass
        import cutlass.cute as cute
        import torch
        from cutlass.cute.runtime import from_dlpack

        if include_int32:
            from cutlass.base_dsl.typing import Int32
    except ImportError as exc:
        raise RuntimeError(
            "cuda.coop.cutlass examples require the CUTLASS Python DSL and torch."
        ) from exc

    coop = _import_cuda_coop_cutlass()
    if include_int32:
        return cutlass, cute, torch, from_dlpack, coop, Int32
    return cutlass, cute, torch, from_dlpack, coop
