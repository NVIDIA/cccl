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
                "cuda/functional",
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
        import inspect
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
        logical = cutlass_coop.this_warp().group_by(8)
        assert isinstance(logical, cutlass_coop.ThreadGroup)
        assert logical.kind == "threads_within_warp"
        assert {"this_thread", "this_cluster", "this_grid", "reduce", "sum"} <= set(
            cutlass_coop.__all__
        )
        for kind in ("thread", "cluster", "grid"):
            assert getattr(cutlass_coop, "this_" + kind)().kind == kind
        assert tuple(inspect.signature(cutlass_coop.reduce).parameters) == (
            "group", "value", "binary_op", "broadcast", "valid_items", "algorithm"
        )
        assert tuple(inspect.signature(cutlass_coop.sum).parameters) == (
            "group", "value", "broadcast", "valid_items", "algorithm"
        )
        scan_names = {
            "scan", "exclusive_scan", "inclusive_scan", "exclusive_sum", "inclusive_sum"
        }
        assert scan_names <= set(cutlass_coop.__all__)
        for name in scan_names:
            common_parameters = tuple(inspect.signature(getattr(coop, name)).parameters)
            parameters = inspect.signature(getattr(cutlass_coop, name)).parameters
            assert tuple(parameters) == common_parameters + (
                "valid_items", "aggregate_output"
            )
            for index, parameter in enumerate(parameters.values()):
                expected = (
                    inspect.Parameter.POSITIONAL_ONLY
                    if index < 2
                    else inspect.Parameter.KEYWORD_ONLY
                )
                assert parameter.kind is expected
        assert {"exchange", "shuffle"} <= set(cutlass_coop.__all__)
        for name, extra_parameters in (
            ("exchange", ("ranks", "valid_flags", "warp_time_slicing")),
            ("shuffle", ()),
        ):
            common_parameters = tuple(inspect.signature(getattr(coop, name)).parameters)
            parameters = inspect.signature(getattr(cutlass_coop, name)).parameters
            assert tuple(parameters) == common_parameters + extra_parameters
            for index, parameter in enumerate(parameters.values()):
                expected = (
                    inspect.Parameter.POSITIONAL_ONLY
                    if index < 2
                    else inspect.Parameter.KEYWORD_ONLY
                )
                assert parameter.kind is expected
        merge_sort_names = {"merge_sort_keys", "merge_sort_pairs"}
        assert merge_sort_names <= set(cutlass_coop.__all__)
        for name in merge_sort_names:
            parameters = inspect.signature(getattr(cutlass_coop, name)).parameters
            assert tuple(parameters) == tuple(
                inspect.signature(getattr(coop, name)).parameters
            )
            input_count = 3 if name.endswith("pairs") else 2
            for index, parameter in enumerate(parameters.values()):
                expected = (
                    inspect.Parameter.POSITIONAL_ONLY
                    if index < input_count
                    else inspect.Parameter.KEYWORD_ONLY
                )
                assert parameter.kind is expected
        radix_names = {"radix_sort_keys", "radix_sort_pairs", "radix_rank"}
        assert radix_names <= set(cutlass_coop.__all__)
        for name in radix_names:
            parameters = inspect.signature(getattr(cutlass_coop, name)).parameters
            extension = (
                "exclusive_digit_prefix" if name == "radix_rank"
                else "blocked_to_striped"
            )
            assert tuple(parameters) == tuple(
                inspect.signature(getattr(coop, name)).parameters
            ) + (extension,)
            input_count = 3 if name.endswith("pairs") else 2
            for index, parameter in enumerate(parameters.values()):
                expected = (
                    inspect.Parameter.POSITIONAL_ONLY
                    if index < input_count
                    else inspect.Parameter.KEYWORD_ONLY
                )
                assert parameter.kind is expected
        topk_names = {
            "topk_min_keys", "topk_min_pairs", "topk_max_keys", "topk_max_pairs"
        }
        assert topk_names <= set(cutlass_coop.__all__)
        for name in topk_names:
            parameters = inspect.signature(getattr(cutlass_coop, name)).parameters
            assert tuple(parameters) == tuple(
                inspect.signature(getattr(coop, name)).parameters
            )
            input_count = 3 if name.endswith("pairs") else 2
            for index, parameter in enumerate(parameters.values()):
                expected = (
                    inspect.Parameter.POSITIONAL_ONLY
                    if index < input_count
                    else inspect.Parameter.KEYWORD_ONLY
                )
                assert parameter.kind is expected
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
