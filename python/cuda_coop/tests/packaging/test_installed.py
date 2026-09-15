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
        import inspect
        import os
        import sys
        from pathlib import Path

        from cuda import coop
        import cuda.coop.numba_mlir as qualified_coop
        from cuda.coop._headers import resolve_include_paths

        public_reduce_module = "cuda.coop.numba_mlir._group_reduce"
        compiler_reduce_module = "cuda.coop.numba_mlir._compiler._group_reduce"
        public_scan_module = "cuda.coop.numba_mlir._group_scan"
        compiler_scan_module = "cuda.coop.numba_mlir._compiler._group_scan"
        assert public_reduce_module not in sys.modules
        assert compiler_reduce_module not in sys.modules
        assert public_scan_module not in sys.modules
        assert compiler_scan_module not in sys.modules

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
            "exchange",
            "exclusive_scan",
            "exclusive_sum",
            "inclusive_scan",
            "inclusive_sum",
            "load",
            "reduce",
            "scan",
            "shuffle",
            "store",
            "sum",
            "this_block",
            "this_cluster",
            "this_grid",
            "this_thread",
            "this_warp",
        }
        assert required <= set(coop.__all__)
        scan_names = {
            "exclusive_scan",
            "exclusive_sum",
            "inclusive_scan",
            "inclusive_sum",
            "scan",
        }
        assert {"reduce", "sum", *scan_names} <= set(coop.__all__)
        assert {"exchange", "reduce", "shuffle", "sum", *scan_names} <= set(
            qualified_coop.__all__
        )
        assert "BlockScanAlgorithm" not in qualified_coop.__all__
        assert not hasattr(qualified_coop, "BlockScanAlgorithm")
        assert callable(qualified_coop.exchange)
        assert callable(qualified_coop.shuffle)
        assert callable(qualified_coop.reduce)
        assert callable(qualified_coop.sum)
        assert all(callable(getattr(qualified_coop, name)) for name in scan_names)
        assert public_reduce_module in sys.modules
        assert compiler_reduce_module not in sys.modules
        assert public_scan_module in sys.modules
        assert compiler_scan_module not in sys.modules
        for name in scan_names:
            parameters = inspect.signature(getattr(qualified_coop, name)).parameters
            assert "prefix_op" not in parameters
            assert "block_prefix_callback_op" not in parameters
        assert importlib.util.find_spec("cuda.coop.numba_mlir._scan_op") is None
        assert importlib.util.find_spec("cuda.coop.numba_mlir._stateful_function") is None

        from cuda.coop.numba_mlir._compiler._operations import group_primitive

        assert group_primitive("reduce") is not None
        assert group_primitive("sum") is not None
        for name in scan_names:
            assert group_primitive(name) is not None
        assert compiler_reduce_module in sys.modules
        assert compiler_scan_module in sys.modules

        paths = resolve_include_paths(
            start=Path.cwd(),
            required_headers=(
                "cub/block/block_exchange.cuh",
                "cub/block/block_load.cuh",
                "cub/block/block_reduce.cuh",
                "cub/block/block_scan.cuh",
                "cub/block/block_shuffle.cuh",
                "cub/block/block_store.cuh",
                "cub/warp/warp_exchange.cuh",
                "cub/warp/warp_reduce.cuh",
                "cub/warp/warp_scan.cuh",
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
