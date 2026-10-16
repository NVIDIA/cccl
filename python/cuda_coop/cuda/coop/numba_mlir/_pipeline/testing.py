# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check what the typed pipeline produced, in the style of LLVM's FileCheck.

A test compiles one small kernel and checks one textual dump of it:

* ``plan`` -- call resolution's decisions (``format_plan``);
* ``typed_ir`` -- the Numba IR with types, after call resolution;
* ``mlir`` -- the MLIR module after lowering.

Checks use FileCheck syntax (``CHECK``, ``CHECK-NEXT``, ``CHECK-NOT``,
``CHECK-LABEL``, ``[[NAME:regex]]`` captures). One kernel can carry checks
for several configurations under different prefixes, for example
``STATIC:`` and ``RUNTIME:`` for a control bound either way.

This module needs the ``filecheck`` package, a test-only dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from textwrap import dedent

from ._debug import capture


def filecheck(
    checks: str,
    text: str,
    *,
    prefixes: tuple[str, ...] = ("CHECK",),
) -> None:
    """Assert that ``text`` matches the FileCheck ``checks``."""

    from filecheck.finput import FInput
    from filecheck.matcher import Matcher
    from filecheck.options import Options
    from filecheck.parser import Parser, pattern_for_opts

    opts = Options(match_filename="-", check_prefixes=list(prefixes))
    parser = Parser(opts, StringIO(dedent(checks)), *pattern_for_opts(opts))
    matcher = Matcher(opts, FInput(fname="-", content=text), parser)
    matcher.stderr = StringIO()
    if matcher.run() != 0:
        raise AssertionError(
            f"FileCheck failed:\n{matcher.stderr.getvalue()}\n"
            f"--- input ---\n{text}"
        )


@dataclass(frozen=True)
class Compiled:
    """The dumps of one compiled kernel."""

    plan: str
    typed_ir: str
    mlir: str


def compile_kernel(
    kernel, signature, *, block=None, **targetoptions
) -> Compiled:
    """Compile ``kernel`` for ``signature`` without launching it.

    ``block`` stands in for the launch's block dimensions, which coop's
    block groups take from the launch. Pass it and an explicit ``chip`` so
    the result depends neither on a launch nor on the local GPU.

    ``kernel`` may be a dispatcher or a plain function; it is compiled
    through a fresh dispatcher with exactly ``targetoptions``, because
    ``compile_mlir`` ignores target options for an existing dispatcher.
    """

    from numba_cuda_mlir import compiler, cuda

    dispatcher = cuda.jit(**targetoptions)(getattr(kernel, "py_func", kernel))
    if block is not None:
        dims = (block,) if isinstance(block, int) else tuple(block)
        dispatcher.targetoptions["__launch_config__"] = {
            "grid": (1, 1, 1),
            "block": dims + (1,) * (3 - len(dims)),
            "sharedmem": 0,
        }
    with capture() as cap:
        mlir = compiler.compile_mlir(dispatcher, signature)
    record = cap.only()
    return Compiled(record.plan_text, record.typed_ir_text, str(mlir))


def check(
    kernel,
    signature,
    checks: str,
    *,
    dump: str = "plan",
    prefixes: tuple[str, ...] = ("CHECK",),
    block=None,
    **targetoptions,
) -> Compiled:
    """Compile ``kernel`` and FileCheck one of its dumps against ``checks``."""

    compiled = compile_kernel(kernel, signature, block=block, **targetoptions)
    filecheck(checks, getattr(compiled, dump), prefixes=prefixes)
    return compiled


__all__ = ["Compiled", "check", "compile_kernel", "filecheck"]
