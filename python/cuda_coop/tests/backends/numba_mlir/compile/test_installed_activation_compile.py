# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Installed-wheel activation and compilation regressions."""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]

_PACKAGE_ROOT = Path(__file__).parents[4].resolve()
_IMPORT_ORDERS = (
    pytest.param("numba-first-common", id="numba-first-common"),
    pytest.param(
        "numba-first-explicit-qualified-alias",
        id="numba-first-explicit-qualified-alias",
    ),
    pytest.param(
        "root-first-explicit-qualified",
        id="root-first-explicit-qualified",
    ),
    pytest.param("root-first-register", id="root-first-register"),
)

_COMPILE_PROBE = textwrap.dedent(
    """
    import importlib.metadata
    import importlib.util
    import inspect
    import os
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    import_order = os.environ["CUDA_COOP_IMPORT_ORDER"]

    if import_order == "numba-first-common":
        import numba_cuda_mlir
        from numba_cuda_mlir import cuda, types
        from cuda import coop

        assert "cuda.coop.numba_mlir" in sys.modules
    elif import_order == "numba-first-explicit-qualified-alias":
        import numba_cuda_mlir
        from numba_cuda_mlir import cuda, types

        compiler_cuda = cuda
        import cuda.coop as coop
        import cuda.coop.numba_mlir as numba_coop

        assert cuda is compiler_cuda
        assert numba_coop is sys.modules["cuda.coop.numba_mlir"]
    elif import_order == "root-first-explicit-qualified":
        from cuda import coop

        assert "numba_cuda_mlir" not in sys.modules
        assert "cuda.coop.numba_mlir" not in sys.modules

        import cuda.coop.numba_mlir  # noqa: F401
        from numba_cuda_mlir import cuda, types
    elif import_order == "root-first-register":
        from cuda import coop

        assert "numba_cuda_mlir" not in sys.modules
        assert "cuda.coop.numba_mlir" not in sys.modules

        assert coop.register("numba-cuda-mlir") is None
        from numba_cuda_mlir import cuda, types
    else:
        raise AssertionError(f"unexpected import order: {import_order!r}")

    distribution = importlib.metadata.distribution("cuda-coop")
    distribution_root = Path(distribution.locate_file("")).resolve()
    expected_module = Path(
        distribution.locate_file("cuda/coop/__init__.py")
    ).resolve()
    source_root = Path(os.environ["CUDA_COOP_SOURCE_ROOT"]).resolve()
    module_file = Path(coop.__file__).resolve()
    qualified_file = Path(sys.modules["cuda.coop.numba_mlir"].__file__).resolve()

    assert module_file == expected_module
    assert module_file.is_relative_to(distribution_root)
    assert qualified_file.is_relative_to(distribution_root)
    assert not module_file.is_relative_to(source_root)
    assert not qualified_file.is_relative_to(source_root)

    from cuda.coop._headers import resolve_include_paths

    include_paths = resolve_include_paths(
        start=Path.cwd(),
        required_headers=(
            "cub/block/block_load.cuh",
            "cuda/std/cstdint",
        ),
    )
    assert include_paths.origin == "cuda-coop wheel header bundle"
    assert all(
        path.resolve().is_relative_to(distribution_root)
        for path in include_paths.cccl
    )

    numba_coop = sys.modules["cuda.coop.numba_mlir"]
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

    scan_names = {
        "exclusive_scan",
        "exclusive_sum",
        "inclusive_scan",
        "inclusive_sum",
        "scan",
    }
    assert {
        "StatefulFunction",
        "exchange",
        "reduce",
        "shuffle",
        "sum",
        *scan_names,
    } <= set(numba_coop.__all__)
    assert "BlockScanAlgorithm" not in numba_coop.__all__
    assert not hasattr(numba_coop, "BlockScanAlgorithm")
    assert callable(numba_coop.exchange)
    assert callable(numba_coop.shuffle)
    assert callable(numba_coop.reduce)
    assert callable(numba_coop.sum)
    assert all(callable(getattr(numba_coop, name)) for name in scan_names)
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
        qualified_scan = getattr(numba_coop, name)
        assert qualified_scan.__module__ == public_scan_module
        qualified_parameters = inspect.signature(
            qualified_scan
        ).parameters
        assert tuple(qualified_parameters) == (
            *common_parameters[:2],
            "prefix_state",
            *common_parameters[2:],
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

    StatefulFunction = numba_coop.StatefulFunction
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

    # Numba-CUDA-MLIR 0.5 has no public GPU-free entry point that carries
    # configured launch metadata. Fix only its current-device queries; the
    # compiler, NVRTC provider compilation, and nvJitLink remain real.
    import numba_cuda_mlir.tools as numba_mlir_tools

    fixed_device = SimpleNamespace(compute_capability=(9, 0))

    def fixed_compute_capability(as_type=str):
        assert as_type in (str, tuple)
        return (9, 0) if as_type is tuple else "sm_90"

    numba_mlir_tools.get_gpu_compute_capability = fixed_compute_capability
    cuda.get_current_device = lambda: fixed_device

    @cuda.jit(chip="sm_90")
    def block_load(source, destination):
        thread = cuda.threadIdx.x
        payload = coop.ThreadData(2)
        coop.load(
            coop.this_block(),
            source,
            payload,
            algorithm="direct",
        )
        for item in range(2):
            destination[thread * 2 + item] = payload[item]

    signature = types.void(types.int32[::1], types.int32[::1])
    launch_config_key = (
        ("grid", (1, 1, 1)),
        ("block", (32, 1, 1)),
        ("sharedmem", 0),
        ("cluster", None),
    )
    result = block_load._compile_launch_config_signature(
        signature,
        launch_config_key,
    )

    assert isinstance(result.metadata["ltoir"], bytes)
    assert result.metadata["ltoir"]
    assert isinstance(result.metadata["cubin"], bytes)
    assert result.metadata["cubin"]

    linked_ltoir = tuple(
        Path(path) for path in result.metadata["linked_external_link_items"]
    )
    assert linked_ltoir
    assert all(path.suffix == ".ltoir" and path.stat().st_size for path in linked_ltoir)

    ptx_by_specialization = block_load.inspect_lto_ptx()
    assert len(ptx_by_specialization) == 1
    ptx = next(iter(ptx_by_specialization.values()))
    assert isinstance(ptx, str)
    assert ".visible .entry" in ptx
    """
)


@pytest.mark.parametrize("import_order", _IMPORT_ORDERS)
def test_installed_wheel_import_order_compiles_block_load(
    import_order: str,
    tmp_path: Path,
) -> None:
    try:
        distribution = importlib.metadata.distribution("cuda-coop")
    except importlib.metadata.PackageNotFoundError:
        pytest.fail("the Numba-CUDA-MLIR compile stage requires an installed wheel")

    expected_module = Path(distribution.locate_file("cuda/coop/__init__.py")).resolve()
    assert expected_module.is_file()

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["CUDA_COOP_ENABLE_CACHE"] = "0"
    environment["CUDA_COOP_IMPORT_ORDER"] = import_order
    environment["CUDA_COOP_SOURCE_ROOT"] = str(_PACKAGE_ROOT)
    # Isolated mode must ignore this deliberate source-tree contamination.
    environment["PYTHONPATH"] = str(_PACKAGE_ROOT)
    environment.pop("CUDA_COOP_CCCL_ROOT", None)
    environment.pop("CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION", None)
    environment.pop("CUDA_COOP_SOURCE_DUMP_DIR", None)

    result = subprocess.run(
        [sys.executable, "-I", "-c", _COMPILE_PROBE],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
