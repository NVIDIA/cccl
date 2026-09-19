# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""TopK wide count guards, scratch reuse, mixed bundles, and final code."""

import os
import re
import shutil
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute
from cutlass.base_dsl.common import DSLRuntimeError, get_current_env_manager
from cutlass.base_dsl.compiler import DumpDir, KeepCUBIN

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from cuda.coop._core.api._dispatch import _backend_module_name
from cuda.coop.cutlass._compiler import _bundle, _cache
from tests.backends.cutlass.runtime.test_topk import _run
from tests.backends.cutlass.support import device_array
from tests.support.paths import PACKAGE_ROOT

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
@pytest.mark.parametrize("auto_sync", (False, True))
def test_scratch_reuse(sharing, auto_sync):
    _run(
        reuse=True,
        valid_items=91,
        sharing=sharing,
        auto_sync=auto_sync,
        alignment=128,
        chain=True,
    )


@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
def test_alignment_minimum(sharing):
    _run(reuse=True, sharing=sharing, capacity=16384, alignment=1)


@pytest.mark.parametrize("k,count", ((0, 128), (17, 0), (0, 0)))
def test_empty_selection_reuse(k, count):
    _run(reuse=True, sharing="shared", k=k, valid_items=count)


def test_undersized_storage():
    with pytest.raises((ValueError, DSLRuntimeError), match="capacity"):
        _run(sharing="shared", capacity=1)


@pytest.mark.parametrize("name", ("k", "valid_items"))
@pytest.mark.parametrize("value", (-1, 129, 1 << 32))
def test_invalid_runtime_counts_trap(name, value):
    args = dict(k=17, valid_items=91)
    args[name] = value
    script = (
        "from tests.backends.cutlass.runtime.test_topk import _run\n"
        f"_run(k={args['k']}, valid_items={args['valid_items']})\n"
        "raise AssertionError('invalid count did not trap')\n"
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


@pytest.mark.parametrize("pairs", (False, True))
@pytest.mark.parametrize("mode", ("min", "max"))
def test_final_cubin(tmp_path, pairs, mode):
    tool = shutil.which("cuobjdump")
    if tool is None:
        pytest.skip("cuobjdump is required for final linked code inspection")
    _run(
        cutlass_coop,
        pairs=pairs,
        mode=mode,
        valid_items=91,
        compile_options=(KeepCUBIN(True), DumpDir(str(tmp_path))),
    )
    cubins = list(tmp_path.rglob("*.cubin"))
    assert cubins
    for cubin in cubins:
        sass = subprocess.check_output([tool, "--dump-sass", str(cubin)], text=True)
        assert "cuda_coop_cutlass_topk_" not in sass
        assert re.search(r"\bCALL\b", sass) is None


@pytest.mark.parametrize("failure", ("compile", "link"))
def test_provider_failure_retry(monkeypatch, tmp_path, failure):
    original = _bundle.compile_bundle_source_with_layouts
    malformed = tmp_path / "malformed-topk.ltoir"
    malformed.write_bytes(b"not NVIDIA LTO IR\n")
    attempted = []

    def fail(*args, **kwargs):
        attempted.append(True)
        if failure == "compile":
            raise RuntimeError("injected TopK provider compilation failure")
        compiled = original(*args, **kwargs)
        _cache.add_managed_bundle_path(str(malformed))
        return replace(compiled, path=str(malformed))

    with monkeypatch.context() as patch:
        patch.setattr(_bundle, "compile_bundle_source_with_layouts", fail)
        with pytest.raises(Exception) as caught:
            _run(valid_items=91)
    assert attempted
    message = str(caught.value).lower()
    assert any(
        token in message for token in ("compile", "compilation", "link", "lto", "nvvm")
    ), message
    assert get_current_env_manager() is None
    assert _backend_module_name() is None
    _run(valid_items=91)


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
def test_load_topk_sort_store_loop(api):
    threads, items, tiles, selected = 32, 2, 4, 11
    tile = threads * items

    @cute.kernel
    def kernel(source: cute.Pointer, output: cute.Pointer, repeats: cutlass.Int32):
        scratch = api.TempStorage(alignment=128)
        group = api.this_block()
        for index in range(repeats):
            data = api.ThreadData(items, dtype=cutlass.Int32)
            api.load(
                group,
                source,
                data,
                offset=index * tile,
                algorithm="transpose",
                temp_storage=scratch,
            )
            chosen = api.topk_min_keys(group, data, k=selected, temp_storage=scratch)
            ordered = api.merge_sort_keys(
                group,
                chosen,
                valid_items=selected,
                oob_default=(1 << 31) - 1,
                temp_storage=scratch,
            )
            api.store(
                group,
                output,
                ordered,
                offset=index * tile,
                valid_items=selected,
                algorithm="transpose",
                temp_storage=scratch,
            )

    @cute.jit
    def launch(source: cute.Pointer, output: cute.Pointer, repeats: cutlass.Int32):
        kernel(source, output, repeats).launch(grid=1, block=threads)

    source = np.random.default_rng(24).integers(
        -101, 102, size=tile * tiles, dtype=np.int32
    )
    observed = np.full_like(source, -999)
    with device_array(source) as src, device_array(observed) as out:
        compiled = cute.compile(launch, src, out, cutlass.Int32(tiles))
        compiled(src, out, cutlass.Int32(tiles))
    for start in range(0, tile * tiles, tile):
        np.testing.assert_array_equal(
            observed[start : start + selected],
            np.sort(source[start : start + tile])[:selected],
        )
        np.testing.assert_array_equal(observed[start + selected : start + tile], -999)
