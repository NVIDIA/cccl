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
        stateful_function_module = "cuda.coop.numba_mlir._stateful_function"
        assert public_reduce_module not in sys.modules
        assert compiler_reduce_module not in sys.modules
        assert public_scan_module not in sys.modules
        assert compiler_scan_module not in sys.modules
        assert stateful_function_module not in sys.modules

        distribution_root = Path(
            importlib.metadata.distribution("cuda-coop").locate_file("")
        ).resolve()
        source_root = Path(os.environ["CUDA_COOP_SOURCE_ROOT"]).resolve()
        portable_module_file = Path(coop.__file__).resolve()
        qualified_module_file = Path(qualified_coop.__file__).resolve()

        for module_file in (portable_module_file, qualified_module_file):
            assert module_file.is_relative_to(distribution_root), (
                module_file,
                distribution_root,
            )
            assert not module_file.is_relative_to(source_root), (
                module_file,
                source_root,
            )
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
        assert "StatefulFunction" not in coop.__all__
        assert {
            "StatefulFunction",
            "exchange",
            "reduce",
            "shuffle",
            "sum",
            *scan_names,
        } <= set(qualified_coop.__all__)
        assert not hasattr(coop, "StatefulFunction")
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
        assert stateful_function_module not in sys.modules

        public_scan_module_file = Path(
            sys.modules[public_scan_module].__file__
        ).resolve()
        assert public_scan_module_file.is_relative_to(distribution_root)
        assert not public_scan_module_file.is_relative_to(source_root)

        portable_scan_parameters = {
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
        for name, portable_parameters in portable_scan_parameters.items():
            assert (
                tuple(inspect.signature(getattr(coop, name)).parameters)
                == portable_parameters
            )
            qualified_scan = getattr(qualified_coop, name)
            assert qualified_scan.__module__ == public_scan_module
            qualified_parameters = inspect.signature(
                qualified_scan
            ).parameters
            assert tuple(qualified_parameters) == (
                *portable_parameters[:2],
                "prefix_state",
                *portable_parameters[2:],
                "valid_items",
                "aggregate_output",
                "prefix_op",
            )
            assert (
                qualified_parameters["prefix_state"].kind
                is inspect.Parameter.POSITIONAL_ONLY
            )
            assert (
                qualified_parameters["prefix_op"].kind
                is inspect.Parameter.KEYWORD_ONLY
            )
            assert qualified_parameters["prefix_op"].default is None
            assert qualified_parameters["prefix_state"].default is None

        assert importlib.util.find_spec("cuda.coop.numba_mlir._scan_op") is None
        stateful_spec = importlib.util.find_spec(stateful_function_module)
        assert stateful_spec is not None
        assert stateful_spec.origin is not None
        stateful_spec_file = Path(stateful_spec.origin).resolve()
        assert stateful_spec_file.is_relative_to(distribution_root)
        assert not stateful_spec_file.is_relative_to(source_root)

        StatefulFunction = qualified_coop.StatefulFunction
        assert stateful_function_module in sys.modules
        stateful_module = sys.modules[stateful_function_module]
        stateful_module_file = Path(stateful_module.__file__).resolve()
        assert stateful_module_file == stateful_spec_file
        assert StatefulFunction is stateful_module.StatefulFunction
        assert StatefulFunction.__module__ == stateful_function_module
        assert tuple(inspect.signature(StatefulFunction).parameters) == (
            "op",
            "dtype",
            "name",
        )

        def running_prefix(state, block_aggregate):
            del state
            return block_aggregate

        state_dtype = object()
        descriptor = StatefulFunction(
            running_prefix,
            state_dtype,
            name="installed_running_prefix",
        )
        assert descriptor.op is running_prefix
        assert descriptor.dtype is state_dtype
        assert descriptor.name == "installed_running_prefix"

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
