# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Link and launch captured LTO-IR with its CUB scratch layout on a new run."""

import json
import os
import subprocess
import sys

import pytest

from tests.support.paths import PACKAGE_ROOT

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


def test_relocated_pack_launches_without_provider_compilation(tmp_path):
    example = PACKAGE_ROOT / "examples" / "block_storage.py"
    workload = ["--", sys.executable, str(example)]

    def run_cli(*args, cache, forbid_compilation=False):
        # Keep the same package provenance as the parent, including when the
        # CI runner imports an installed wheel with Python's isolated mode.
        setup = f"import sys\nsys.path[:] = {sys.path!r}\n"
        if forbid_compilation:
            setup += (
                "from cuda.coop.cutlass._compiler import _nvrtc\n"
                "def fail(*args, **kwargs):\n"
                "    raise AssertionError('pack hit reached provider JIT')\n"
                "_nvrtc.resolve_compile_context = fail\n"
            )
        setup += "from cuda.coop.cutlass._aot_cli import main\nsys.exit(main())"
        result = subprocess.run(
            [sys.executable, "-I", "-c", setup, *map(str, args)],
            cwd=tmp_path,
            env={
                **os.environ,
                "CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR": str(cache),
                "CUTE_DSL_NO_CACHE": "1",
            },
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    original = tmp_path / "capture"
    run_cli("capture", "--output", original, *workload, cache=tmp_path / "jit")
    installed = original.rename(tmp_path / "relocated")
    cache = tmp_path / "consumer-cache"
    cache.mkdir()
    info = json.loads(run_cli("inspect", installed, cache=cache))
    assert info["artifact_bytes"] > 0
    assert any(entry["layouts"] for entry in info["entries"])
    run_cli(
        "run",
        "--pack",
        installed,
        "--mode",
        "required",
        *workload,
        cache=cache,
        forbid_compilation=True,
    )
