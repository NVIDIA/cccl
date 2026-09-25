# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Radix runtime guards, storage reuse, failure recovery, and final code."""

import os
import re
import shutil
import subprocess
import sys
from dataclasses import replace

import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass.base_dsl.common import DSLRuntimeError, get_current_env_manager
from cutlass.base_dsl.compiler import DumpDir, KeepCUBIN

from cuda.coop import cutlass as cutlass_coop
from cuda.coop._core.api._dispatch import _backend_module_name
from cuda.coop.cutlass._compiler import _bundle, _cache
from tests.backends.cutlass.runtime.test_radix import _run_rank, _run_sort
from tests.support.paths import PACKAGE_ROOT

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
@pytest.mark.parametrize("auto_sync", (False, True))
def test_sort_storage_reuse(sharing, auto_sync):
    _run_sort(
        reuse=True, sharing=sharing, auto_sync=auto_sync, alignment=128, chain=True
    )


@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
def test_storage_alignment(sharing):
    _run_sort(reuse=True, sharing=sharing, capacity=16384, alignment=1)


@pytest.mark.parametrize("bits", (4, 8))
def test_rank_reuse(bits):
    _run_rank(cutlass_coop, radix_bits=bits, prefix=True, reuse=True, chain=True)


def test_undersized_storage():
    with pytest.raises((ValueError, DSLRuntimeError), match="capacity"):
        _run_sort(sharing="shared", capacity=1)


@pytest.mark.parametrize(
    "begin,end,control",
    (
        (-1, 4, "Int64"),
        (3, 3, "Int64"),
        (7, 3, "Int64"),
        (0, 33, "Int64"),
        (1 << 32, 32, "Int64"),
        (0, (1 << 32) + 4, "Int64"),
        ((1 << 32) - 1, 32, "Uint32"),
    ),
)
def test_invalid_runtime_bits_trap(begin, end, control):
    script = (
        "import cutlass\n"
        "from tests.backends.cutlass.runtime.test_radix import _run_sort\n"
        f"_run_sort(begin_bit={begin}, end_bit={end}, bounds='runtime', control_type=cutlass.{control})\n"
        "raise AssertionError('invalid bit interval did not trap')\n"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(PACKAGE_ROOT), environment.get("PYTHONPATH")))
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode != 0, output
    assert any(
        error in output
        for error in (
            "ILLEGAL_INSTRUCTION",
            "LAUNCH_FAILED",
            "illegal instruction",
            "launch failed",
            "CUDA Driver call failed: 715",
            "CUDA Driver call failed: 719",
        )
    ), output


@pytest.mark.parametrize("operation", ("sort", "rank4", "rank8"))
def test_final_cubin(tmp_path, operation):
    tool = shutil.which("cuobjdump")
    if tool is None:
        pytest.skip("cuobjdump is required for final linked code inspection")
    options = (KeepCUBIN(True), DumpDir(str(tmp_path)))
    if operation == "sort":
        _run_sort(
            cutlass_coop,
            bounds="runtime",
            begin_bit=7,
            end_bit=31,
            compile_options=options,
        )
    else:
        _run_rank(
            cutlass_coop,
            radix_bits=int(operation[-1]),
            prefix=True,
            compile_options=options,
        )
    cubins = list(tmp_path.rglob("*.cubin"))
    assert cubins
    for cubin in cubins:
        sass = subprocess.check_output([tool, "--dump-sass", str(cubin)], text=True)
        assert "cuda_coop_cutlass_radix_" not in sass
        assert re.search(r"\bCALL\b", sass) is None


@pytest.mark.parametrize("failure", ("compile", "link"))
def test_provider_failure_retry(monkeypatch, tmp_path, failure):
    original = _bundle.compile_bundle_source_with_layouts
    malformed = tmp_path / "malformed-radix.ltoir"
    malformed.write_bytes(b"not NVIDIA LTO IR\n")
    attempted = []

    def fail(*args, **kwargs):
        attempted.append(True)
        if failure == "compile":
            raise RuntimeError("injected radix provider compilation failure")
        compiled = original(*args, **kwargs)
        _cache.add_managed_bundle_path(str(malformed))
        return replace(compiled, path=str(malformed))

    with monkeypatch.context() as patch:
        patch.setattr(_bundle, "compile_bundle_source_with_layouts", fail)
        with pytest.raises(Exception) as caught:
            _run_rank(cutlass_coop, prefix=True)
    assert attempted
    message = str(caught.value).lower()
    assert any(
        token in message for token in ("compile", "compilation", "link", "lto", "nvvm")
    ), message
    assert get_current_env_manager() is None
    assert _backend_module_name() is None
    _run_rank(cutlass_coop, prefix=True)
