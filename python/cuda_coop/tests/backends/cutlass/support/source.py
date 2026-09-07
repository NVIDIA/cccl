# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source-tree subprocess helpers for CUTLASS backend tests."""

from __future__ import annotations

import os
import subprocess
import sys

from ....support.paths import PACKAGE_ROOT

SOURCE_ROOT = PACKAGE_ROOT

_CUTLASS_RUNTIME_STUB = """
import sys as _cuda_coop_test_sys
import types as _cuda_coop_test_types
import importlib as _cuda_coop_test_importlib

_cuda_coop_test_cutlass = _cuda_coop_test_types.ModuleType("cutlass")
_cuda_coop_test_cutlass.__path__ = []
_cuda_coop_test_dsl = _cuda_coop_test_types.ModuleType("cutlass.cutlass_dsl")

class _CudaCoopTestCuTeDSL:
    _instance = None

    @classmethod
    def _get_dsl(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.trace_context_factories = []

    def register_trace_context_factory(self, factory):
        if factory not in self.trace_context_factories:
            self.trace_context_factories.append(factory)

    def register_trace_finalize_hook(self, hook):
        self.trace_finalize_hook = hook

_cuda_coop_test_dsl.CuTeDSL = _CudaCoopTestCuTeDSL
_cuda_coop_test_cute = _cuda_coop_test_types.ModuleType("cutlass.cute")
_cuda_coop_test_cute._get_launch_facts = lambda: {}
_cuda_coop_test_base_dsl = _cuda_coop_test_types.ModuleType("cutlass.base_dsl")
_cuda_coop_test_base_dsl.__path__ = []
_cuda_coop_test_compiler = _cuda_coop_test_types.ModuleType(
    "cutlass.base_dsl.compiler"
)
_cuda_coop_test_compiler.LinkLibraries = type(
    "LinkLibraries",
    (),
    {"_option_name": "link-libraries"},
)
_cuda_coop_test_compiler.GPUArch = type("GPUArch", (), {})
_cuda_coop_test_cutlass.cutlass_dsl = _cuda_coop_test_dsl
_cuda_coop_test_cutlass.cute = _cuda_coop_test_cute
_cuda_coop_test_cutlass.base_dsl = _cuda_coop_test_base_dsl
_cuda_coop_test_base_dsl.compiler = _cuda_coop_test_compiler
_cuda_coop_test_sys.modules["cutlass"] = _cuda_coop_test_cutlass
_cuda_coop_test_sys.modules["cutlass.cutlass_dsl"] = _cuda_coop_test_dsl
_cuda_coop_test_sys.modules["cutlass.cute"] = _cuda_coop_test_cute
_cuda_coop_test_sys.modules["cutlass.base_dsl"] = _cuda_coop_test_base_dsl
_cuda_coop_test_sys.modules[
    "cutlass.base_dsl.compiler"
] = _cuda_coop_test_compiler
_cuda_coop_test_real_import_module = _cuda_coop_test_importlib.import_module

def _cuda_coop_test_import_module(name, package=None):
    if name == "cutlass.cutlass_dsl":
        module = _cuda_coop_test_sys.modules.get(name)
        if module is None:
            module = _cuda_coop_test_types.ModuleType(name)
            _cuda_coop_test_sys.modules[name] = module
        parent = _cuda_coop_test_sys.modules.get("cutlass")
        if parent is not None:
            parent.cutlass_dsl = module
        return module
    return _cuda_coop_test_real_import_module(name, package)

_cuda_coop_test_importlib.import_module = _cuda_coop_test_import_module
"""


def run_python_with_source(script: str) -> subprocess.CompletedProcess[str]:
    """Run an isolated Python process against this cuda.coop source tree."""

    env = os.environ.copy()
    python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{SOURCE_ROOT}{os.pathsep}{python_path}" if python_path else str(SOURCE_ROOT)
    )

    return subprocess.run(
        [sys.executable, "-S", "-B", "-c", _CUTLASS_RUNTIME_STUB + script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
