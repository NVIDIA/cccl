# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run CUTLASS compile and link probes in an isolated Python process.

CUTLASS compiler state and its environment-derived artifact settings are
process-wide. The child process therefore provides a cold compiler session for
the real example, exercises repeated tracing in one process, and lets the
parent verify that temporary provider LTO-IR is removed when the process exits.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = PACKAGE_ROOT / "examples" / "cutlass" / "block_load_store.py"

# Bounded child entry point: compile the executable example with fake tensors,
# without allocating device memory or launching a kernel.
_COMPILE_SCRIPT = r"""
import importlib.util
from pathlib import Path

from cutlass import cute
from cutlass.base_dsl.typing import Int32
from cutlass.cute import runtime

example_path = Path(__EXAMPLE_PATH__)
spec = importlib.util.spec_from_file_location("cuda_coop_block_load_store", example_path)
assert spec is not None and spec.loader is not None
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)

run, *_ = example.make_runner()
source = runtime.make_fake_compact_tensor(
    Int32,
    (example.INPUT_OFFSET + example.LOAD_VALID_ITEMS,),
)
destination = runtime.make_fake_compact_tensor(
    Int32,
    (example.OUTPUT_OFFSET + example.TILE_ITEMS + 4,),
)
compiled = cute.compile(run, source, destination, options="--gpu-arch sm_120")
assert callable(compiled)
compiled_again = cute.compile(
    run,
    source,
    destination,
    options="--gpu-arch sm_120",
)
assert callable(compiled_again)
print("COMPILED_TWICE")
"""


def compile_example(
    *, dump_dir: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Compile the example twice for SM120 in a fresh CUTLASS process."""

    env = os.environ.copy()
    if dump_dir is not None:
        env["CUTE_DSL_KEEP"] = "all"
        env["CUTE_DSL_DUMP_DIR"] = str(dump_dir)
    script = _COMPILE_SCRIPT.replace("__EXAMPLE_PATH__", repr(str(EXAMPLE_PATH)))
    return subprocess.run(
        [sys.executable, "-B", "-c", script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def assert_compiled(result: subprocess.CompletedProcess[str]) -> None:
    """Assert that an isolated compiler invocation completed successfully."""

    assert result.returncode == 0, (
        f"CUTLASS compile exited with {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "COMPILED_TWICE" in result.stdout


__all__ = ["assert_compiled", "compile_example"]
