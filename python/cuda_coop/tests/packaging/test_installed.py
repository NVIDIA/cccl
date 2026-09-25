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
        cutlass_spec = importlib.util.find_spec("cuda.coop.cutlass")
        assert cutlass_spec is not None and cutlass_spec.origin is not None
        cutlass_file = Path(cutlass_spec.origin).resolve()
        assert cutlass_file.is_relative_to(distribution_root), cutlass_file
        assert not cutlass_file.is_relative_to(source_root), cutlass_file
        assert "cutlass" not in sys.modules
        assert "cuda.coop.cutlass" not in sys.modules

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
        assert "StatefulFunction" not in coop.__all__
        assert not hasattr(coop, "StatefulFunction")

        common_scan_parameters = {
            "scan": (
                "group",
                "value",
                "mode",
                "scan_op",
                "initial_value",
                "algorithm",
                "temp_storage",
            ),
            "exclusive_scan": (
                "group",
                "value",
                "scan_op",
                "initial_value",
                "algorithm",
                "temp_storage",
            ),
            "inclusive_scan": (
                "group",
                "value",
                "scan_op",
                "algorithm",
                "temp_storage",
            ),
            "exclusive_sum": ("group", "value", "algorithm", "temp_storage"),
            "inclusive_sum": ("group", "value", "algorithm", "temp_storage"),
        }
        for name, common_parameters in common_scan_parameters.items():
            assert (
                tuple(inspect.signature(getattr(coop, name)).parameters)
                == common_parameters
            )

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
                "cub/warp/warp_load.cuh",
                "cub/warp/warp_reduce.cuh",
                "cub/warp/warp_scan.cuh",
                "cub/warp/warp_store.cuh",
                "cuda/experimental/coop/algorithm",
                "cuda/experimental/coop/group",
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


def test_isolated_cutlass_backend_uses_installed_modules(tmp_path: Path) -> None:
    try:
        importlib.metadata.distribution("cuda-coop")
        importlib.metadata.distribution("nvidia-cutlass-dsl")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("requires installed cuda-coop and CUTLASS DSL distributions")

    probe = textwrap.dedent(
        """
        import importlib.metadata
        import sys
        from pathlib import Path

        import cutlass
        from cuda import coop
        import cuda.coop.cutlass as cutlass_coop
        from cuda.coop.cutlass._compiler import _bundle
        from cuda.coop._core.api import _dispatch

        distribution_root = Path(
            importlib.metadata.distribution("cuda-coop").locate_file("")
        ).resolve()
        for name, module in tuple(sys.modules.items()):
            if name == "cuda.coop" or name.startswith("cuda.coop."):
                origin = getattr(module, "__file__", None)
                assert origin is not None, name
                assert Path(origin).resolve().is_relative_to(distribution_root), (
                    name, origin, distribution_root
                )
        assert callable(coop.load) and callable(cutlass_coop.load)
        assert "this_warp" in cutlass_coop.__all__
        assert cutlass_coop.this_warp().kind == "warp"
        assert "cuda.coop.cutlass" in _dispatch._COMPILER_CONTEXT_PROBES
        assert _dispatch._backend_module_name() is None
        """
    )
    environment = os.environ.copy()
    environment.pop("CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION", None)
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
