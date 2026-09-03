# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).parents[2].resolve()


def test_isolated_python_uses_only_the_installed_wheel(tmp_path: Path) -> None:
    try:
        importlib.metadata.distribution("cuda-coop")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("cuda-coop is not installed in this interpreter")

    probe = textwrap.dedent(
        """
        import importlib.metadata
        import importlib.util
        import os
        from pathlib import Path

        from cuda import coop
        from cuda.coop._headers import resolve_include_paths

        distribution_root = Path(
            importlib.metadata.distribution("cuda-coop").locate_file("")
        ).resolve()
        source_root = Path(os.environ["CUDA_COOP_SOURCE_ROOT"]).resolve()
        module_file = Path(coop.__file__).resolve()

        assert module_file.is_relative_to(distribution_root), (
            module_file,
            distribution_root,
        )
        assert not module_file.is_relative_to(source_root), (module_file, source_root)
        assert importlib.util.find_spec("cuda.coop.cutlass") is None

        required = {
            "Hierarchy",
            "TempStorage",
            "TempStorageLike",
            "ThreadData",
            "ThreadDataLike",
            "ThreadGroup",
            "ThreadHierarchy",
            "load",
            "store",
            "this_block",
            "this_cluster",
            "this_grid",
            "this_thread",
            "this_warp",
        }
        assert required <= set(coop.__all__)
        assert {"reduce", "scan", "sum"}.isdisjoint(coop.__all__)

        paths = resolve_include_paths(
            start=Path.cwd(),
            required_headers=(
                "cub/block/block_load.cuh",
                "cub/block/block_store.cuh",
                "cuda/experimental/coop.cuh",
                "thrust/detail/raw_pointer_cast.h",
                "cuda/std/cstdint",
            ),
        )
        assert paths.origin == "cuda-coop wheel header bundle"
        assert all(path.resolve().is_relative_to(distribution_root) for path in paths.cccl)
        """
    )

    environment = os.environ.copy()
    environment["CUDA_COOP_SOURCE_ROOT"] = str(_PACKAGE_ROOT)
    # Isolated mode must ignore this deliberate source-tree contamination.
    environment["PYTHONPATH"] = str(_PACKAGE_ROOT)
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
