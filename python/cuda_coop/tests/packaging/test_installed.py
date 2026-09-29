# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check that the installed distribution supplies the CUTLASS adapter.

The child interpreter ignores an intentionally supplied checkout PYTHONPATH
and runs from a temporary directory. Module origins must lie under the
installed distribution. Registration must succeed while leaving no active
compiler backend in ordinary host code.
"""

import importlib.metadata
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def test_isolated_cutlass_backend_uses_installed_modules(tmp_path):
    """Probe installed imports in isolated mode from a temporary directory."""

    try:
        importlib.metadata.distribution("cuda-coop")
        importlib.metadata.distribution("nvidia-cutlass-dsl")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip(
            "requires installed cuda-coop and CUTLASS DSL distributions"
        )

    probe = textwrap.dedent(
        """
        import importlib.metadata
        from pathlib import Path

        from cuda import coop
        import cuda.coop.cutlass as cutlass_coop
        from cuda.coop.cutlass._compiler import _bundle
        from cuda.coop._core.api import _dispatch

        installed = Path(
            importlib.metadata.distribution("cuda-coop").locate_file("")
        ).resolve()
        for module in (coop, cutlass_coop, _bundle):
            assert Path(module.__file__).resolve().is_relative_to(installed)
        assert cutlass_coop.this_block().kind == "block"
        assert callable(_bundle.compile_bundle_source)
        assert "cuda.coop.cutlass" in _dispatch._COMPILER_CONTEXT_PROBES
        assert _dispatch._backend_module_name() is None
        """
    )
    environment = os.environ.copy()
    environment.pop("CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION", None)
    # Isolated mode must ignore a source checkout on PYTHONPATH.
    environment["PYTHONPATH"] = str(Path(__file__).parents[2])
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
