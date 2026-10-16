# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Record what the typed pipeline did, for tests and for reading.

Call resolution reports each kernel it resolves to ``record``. Two consumers
read those records:

* ``capture()`` collects them while a test compiles a kernel, so the test
  can check the plan and the typed IR with FileCheck;
* ``CUDA_COOP_NUMBA_MLIR_PIPELINE_PRINT`` prints them to standard error as
  each kernel compiles. It takes a comma-separated list of ``plan`` and
  ``ir``, in the spirit of LLVM's ``-print-after``.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from . import _bridges as nb
from ._plan import KernelPlan, format_plan

PRINT_ENVIRONMENT_VARIABLE = "CUDA_COOP_NUMBA_MLIR_PIPELINE_PRINT"


_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")
_MODULE_PATH = re.compile(r"<module '([^']+)' from '[^']*'>")


def _stable(text: str) -> str:
    """Drop object addresses and module file paths from a repr."""

    return _MODULE_PATH.sub(r"<module '\1'>", _ADDRESS.sub("", text))


def format_typed_ir(name: str, func_ir: Any, typemap: dict) -> str:
    """Render ``func_ir`` with the type of every assigned variable."""

    lines = [f"typed-ir {name}"]
    for label in sorted(func_ir.blocks):
        lines.append(f"label {label}:")
        for stmt in func_ir.blocks[label].body:
            if isinstance(stmt, nb.ir.Del):
                continue
            text = str(stmt)
            if isinstance(stmt, nb.ir.Assign):
                ty = typemap.get(stmt.target.name)
                if ty is not None:
                    text += f"  :: {ty}"
            lines.append(f"    {_stable(text)}")
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class KernelRecord:
    """One kernel as call resolution left it, after its last rewrite."""

    plan: KernelPlan
    plan_text: str
    typed_ir_text: str


@dataclass
class Capture:
    """Kernels recorded while a ``capture()`` block was active."""

    kernels: list[KernelRecord] = field(default_factory=list)

    def only(self) -> KernelRecord:
        if len(self.kernels) != 1:
            names = [k.plan.name for k in self.kernels]
            raise AssertionError(f"expected one recorded kernel, got {names}")
        return self.kernels[0]


_active: list[Capture] = []


@contextlib.contextmanager
def capture() -> Iterator[Capture]:
    """Collect the records of every kernel planned inside the block."""

    cap = Capture()
    _active.append(cap)
    try:
        yield cap
    finally:
        _active.remove(cap)


def _print_requests() -> set[str]:
    value = os.environ.get(PRINT_ENVIRONMENT_VARIABLE, "")
    return {part.strip() for part in value.split(",") if part.strip()}


def record(plan: KernelPlan, func_ir: Any, typemap: dict) -> None:
    """Report a kernel to active captures and to the print switch.

    Call this once call resolution has finished rewriting ``func_ir``.
    """

    requests = _print_requests()
    if not _active and not requests:
        return
    rec = KernelRecord(
        plan,
        format_plan(plan),
        format_typed_ir(plan.name, func_ir, typemap),
    )
    for cap in _active:
        cap.kernels.append(rec)
    if "plan" in requests:
        print(rec.plan_text, file=sys.stderr, end="")
    if "ir" in requests:
        print(rec.typed_ir_text, file=sys.stderr, end="")


__all__ = [
    "PRINT_ENVIRONMENT_VARIABLE",
    "Capture",
    "KernelRecord",
    "capture",
    "format_typed_ir",
    "record",
]
