# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""GPU-free compilation contracts for physical and logical Warp Load/Store."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import types

from cuda.coop._core import ArgumentBinding, SynchronizationScope
from cuda.coop._core.warp import make_warp_load_spec, make_warp_store_spec
from cuda.coop.numba_mlir import _types
from cuda.coop.numba_mlir._compiler import _nvrtc
from cuda.coop.numba_mlir._compiler._operations import StorageABI
from cuda.coop.numba_mlir._lowering._core import NumbaMlirCoreAdapter
from cuda.coop.numba_mlir._lowering._load_store import _load_store_value_abis

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]

_FIXED_COMPUTE_CAPABILITY = (9, 0)
_BLOCK_THREADS = 64
_WARP_THREADS = 32
_ITEMS_PER_THREAD = 2
_ALGORITHMS = ("direct", "striped", "vectorize", "transpose")
_DTYPES = (
    pytest.param(types.int8, id="int8"),
    pytest.param(types.uint8, id="uint8"),
    pytest.param(types.int16, id="int16"),
    pytest.param(types.uint16, id="uint16"),
    pytest.param(types.int32, id="int32"),
    pytest.param(types.uint32, id="uint32"),
    pytest.param(types.int64, id="int64"),
    pytest.param(types.uint64, id="uint64"),
    pytest.param(types.float32, id="float32"),
    pytest.param(types.float64, id="float64"),
)


@pytest.fixture(autouse=True)
def _fixed_current_device(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Hide runtime discovery while leaving NVRTC and nvJitLink real."""

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", (
        "the Numba-CUDA-MLIR compile stage must hide all CUDA devices"
    )
    queries: list[tuple[int, int]] = []

    def current_device() -> SimpleNamespace:
        queries.append(_FIXED_COMPUTE_CAPABILITY)
        return SimpleNamespace(compute_capability=_FIXED_COMPUTE_CAPABILITY)

    monkeypatch.setattr(_types.cuda, "get_current_device", current_device)
    return queries


@pytest.fixture(scope="module")
def compile_context() -> _nvrtc.CompileContext:
    return _nvrtc.resolve_compile_context()


def _algorithm(
    context: _nvrtc.CompileContext,
    *,
    operation: str,
    algorithm: str,
    dtype=types.int32,
    threads_in_warp: int = _WARP_THREADS,
) -> _types.Algorithm:
    valid_items = ArgumentBinding.runtime()
    oob_default = (
        ArgumentBinding.runtime() if operation == "load" else ArgumentBinding.omitted()
    )
    adapter = NumbaMlirCoreAdapter(
        value_abis=_load_store_value_abis(
            dtype=dtype,
            items_per_thread=_ITEMS_PER_THREAD,
            threads_in_warp=threads_in_warp,
            valid_items=valid_items,
            oob_default=oob_default,
        )
    )
    factory = make_warp_load_spec if operation == "load" else make_warp_store_spec
    spec = factory(
        dtype=adapter.core_dtype(dtype),
        items_per_thread=_ITEMS_PER_THREAD,
        threads_in_warp=threads_in_warp,
        algorithm=algorithm,
        valid_items=valid_items,
        oob_default=oob_default,
        include_pointer_offset=True,
    )
    storage_free = algorithm != "transpose"
    specialization = adapter.materialize(
        spec.specialization,
        storage_abi=(StorageABI.NONE if storage_free else StorageABI.LEADING_POINTER),
        execution_scope=SynchronizationScope.WARP,
        synchronization_scope=(
            SynchronizationScope.NONE if storage_free else SynchronizationScope.WARP
        ),
        extra_type_definitions=(_types.numba_type_to_wrapper(dtype),),
    )
    specialization._compile_context = context
    specialization.threads = threads_in_warp
    specialization.block_threads = _BLOCK_THREADS
    return specialization


def _source(algorithm: _types.Algorithm) -> str:
    source, _support_lto_irs, _storage_symbols, _udf_declarations = (
        algorithm._source_code()
    )
    return source


def _production_compile_environment(monkeypatch: pytest.MonkeyPatch):
    import numba_cuda_mlir.tools as numba_mlir_tools
    from numba_cuda_mlir import cuda as compiler_cuda

    fixed_device = SimpleNamespace(compute_capability=_FIXED_COMPUTE_CAPABILITY)

    def fixed_compute_capability(as_type=str):
        assert as_type in (str, tuple)
        return _FIXED_COMPUTE_CAPABILITY if as_type is tuple else "sm_90"

    monkeypatch.setattr(
        numba_mlir_tools,
        "get_gpu_compute_capability",
        fixed_compute_capability,
    )
    monkeypatch.setattr(compiler_cuda, "get_current_device", lambda: fixed_device)
    return compiler_cuda


def _production_launch_config_key() -> tuple[tuple[str, object], ...]:
    return (
        ("grid", (1, 1, 1)),
        ("block", (_BLOCK_THREADS, 1, 1)),
        ("sharedmem", 0),
        ("cluster", None),
    )


@pytest.mark.parametrize("threads_in_warp", (32, 8), ids=("physical", "logical-8"))
def test_all_warp_algorithms_compile_with_scope_owned_storage(
    compile_context: _nvrtc.CompileContext,
    _fixed_current_device: list[tuple[int, int]],
    threads_in_warp: int,
) -> None:
    algorithms = {
        (operation, name): _algorithm(
            compile_context,
            operation=operation,
            algorithm=name,
            threads_in_warp=threads_in_warp,
        )
        for operation in ("load", "store")
        for name in _ALGORITHMS
    }

    for (operation, name), specialization in algorithms.items():
        source = _source(specialization)
        assert f"::cub::WARP_{operation.upper()}_{name.upper()}" in source
        tile_items = threads_in_warp * _ITEMS_PER_THREAD
        assert f"if (param_2 < 0 || param_2 > {tile_items})" in source
        assert "(param_0 + param_" in source
        assert "__syncthreads" not in source
        assert "bar.sync" not in source
        if name == "transpose":
            assert specialization.storage_abi is StorageABI.LEADING_POINTER
            assert "TempStorage" in source
            instances = _BLOCK_THREADS // threads_in_warp
            assert f"temp_storages[{instances}]" in source
            assert f"__coop_thread_rank / {threads_in_warp}" in source
            if threads_in_warp == 32:
                assert "__syncwarp();" in source
            else:
                assert "__syncwarp(" in source
                assert "__syncwarp();" not in source
            assert f"(*temp_storage).{operation.title()}(" in source
        else:
            assert specialization.storage_abi is StorageABI.NONE
            assert "TempStorage" not in source
            assert "temp_storages" not in source
            assert "__syncwarp" not in source
            assert f"().{operation.title()}(" in source

    bundle = _types.prepare_ltoir_bundle(
        list(algorithms.values()),
        bundle_name=(
            f"cuda_coop_numba_mlir_all_warp_load_store_algorithms_{threads_in_warp}"
        ),
    )
    assert isinstance(bundle, bytes)
    assert bundle
    ptx = _types._ltoir_to_ptx(
        bundle,
        name="warp_load_store_synchronization_scope",
        cc=10 * _FIXED_COMPUTE_CAPABILITY[0] + _FIXED_COMPUTE_CAPABILITY[1],
    )
    assert "bar.sync" not in ptx
    assert _fixed_current_device
    for (_operation, name), specialization in algorithms.items():
        if name == "transpose":
            assert specialization.temp_storage_bytes > 0
            assert specialization.temp_storage_alignment > 0
        else:
            assert specialization.temp_storage_bytes == 0
            assert specialization.temp_storage_alignment == 1


@pytest.mark.parametrize("dtype", _DTYPES)
def test_direct_multi_item_load_store_compiles_for_every_supported_dtype(
    compile_context: _nvrtc.CompileContext,
    dtype,
) -> None:
    algorithms = [
        _algorithm(
            compile_context,
            operation=operation,
            algorithm="direct",
            dtype=dtype,
        )
        for operation in ("load", "store")
    ]

    for operation, specialization in zip(("load", "store"), algorithms):
        source = _source(specialization)
        assert f"::cub::WARP_{operation.upper()}_DIRECT" in source
        assert f"().{operation.title()}(" in source
        assert specialization.storage_abi is StorageABI.NONE

    bundle = _types.prepare_ltoir_bundle(
        algorithms,
        bundle_name=f"cuda_coop_numba_mlir_warp_load_store_{dtype}",
    )
    assert isinstance(bundle, bytes)
    assert bundle
    assert all(specialization.temp_storage_bytes == 0 for specialization in algorithms)
    assert all(
        specialization.temp_storage_alignment == 1 for specialization in algorithms
    )


def test_logical_warp_widths_have_distinct_specializations_and_cache_keys(
    compile_context: _nvrtc.CompileContext,
) -> None:
    algorithms = [
        _algorithm(
            compile_context,
            operation="load",
            algorithm="transpose",
            threads_in_warp=width,
        )
        for width in (1, 2, 4, 8, 16, 32)
    ]

    assert [algorithm.threads for algorithm in algorithms] == [1, 2, 4, 8, 16, 32]
    assert (
        len(
            {
                algorithm._make_lto_ir_cache_key(
                    threads=algorithm.threads,
                    block_threads=_BLOCK_THREADS,
                )
                for algorithm in algorithms
            }
        )
        == 6
    )
    for width, algorithm in zip((1, 2, 4, 8, 16, 32), algorithms):
        source = _source(algorithm)
        assert f", {width}>" in source
        assert f"temp_storages[{_BLOCK_THREADS // width}]" in source

    bundle = _types.prepare_ltoir_bundle(
        algorithms,
        bundle_name="cuda_coop_numba_mlir_logical_warp_widths",
    )
    assert bundle


def test_production_routes_compile_portable_and_qualified_warp_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cuda.coop.numba_mlir as qualified_coop
    from cuda import coop as root_coop

    compiler_cuda = _production_compile_environment(monkeypatch)

    def qualified_kernel(algorithm: str):
        @compiler_cuda.jit(chip="sm_90")
        def kernel(
            load_source,
            store_source,
            observed,
            destination,
            valid_items,
            offset,
        ):
            thread = compiler_cuda.threadIdx.x
            loaded = qualified_coop.load(
                qualified_coop.this_warp(),
                load_source,
                qualified_coop.ThreadData(
                    _ITEMS_PER_THREAD,
                    dtype=types.int32,
                ),
                algorithm=algorithm,
                valid_items=valid_items,
                oob_default=types.int32(-17),
                offset=offset,
            )
            payload = qualified_coop.ThreadData(
                _ITEMS_PER_THREAD,
                dtype=types.int32,
            )
            for item in range(_ITEMS_PER_THREAD):
                index = thread * _ITEMS_PER_THREAD + item
                observed[index] = loaded[item]
                payload[item] = store_source[index]
            qualified_coop.store(
                qualified_coop.this_warp(),
                destination,
                payload,
                algorithm=algorithm,
                valid_items=valid_items,
                offset=offset,
            )

        return kernel

    def portable_kernel(algorithm: str):
        @compiler_cuda.jit(chip="sm_90")
        def kernel(
            load_source,
            store_source,
            observed,
            destination,
            valid_items,
            offset,
        ):
            thread = compiler_cuda.threadIdx.x
            loaded = root_coop.load(
                root_coop.this_warp(),
                load_source,
                root_coop.ThreadData(
                    _ITEMS_PER_THREAD,
                    dtype=types.int32,
                ),
                algorithm=algorithm,
                valid_items=valid_items,
                oob_default=types.int32(-17),
                offset=offset,
            )
            payload = root_coop.ThreadData(
                _ITEMS_PER_THREAD,
                dtype=types.int32,
            )
            for item in range(_ITEMS_PER_THREAD):
                index = thread * _ITEMS_PER_THREAD + item
                observed[index] = loaded[item]
                payload[item] = store_source[index]
            root_coop.store(
                root_coop.this_warp(),
                destination,
                payload,
                algorithm=algorithm,
                valid_items=valid_items,
                offset=offset,
            )

        return kernel

    dispatchers = [
        (qualified, algorithm, factory(algorithm))
        for qualified, factory in (
            (False, portable_kernel),
            (True, qualified_kernel),
        )
        for algorithm in ("direct", "transpose")
    ]

    signature = types.void(
        types.int32[::1],
        types.int32[::1],
        types.int32[::1],
        types.int32[::1],
        types.int32,
        types.int64,
    )
    launch_config_key = _production_launch_config_key()
    for qualified, algorithm, dispatcher in dispatchers:
        result = dispatcher._compile_launch_config_signature(
            signature,
            launch_config_key,
        )
        assert isinstance(result.metadata["ltoir"], bytes)
        assert result.metadata["ltoir"]
        assert isinstance(result.metadata["cubin"], bytes)
        assert result.metadata["cubin"]
        assert result.metadata["linked_external_link_items"]

        ptx = next(iter(dispatcher.inspect_lto_ptx().values()))
        assert ".visible .entry" in ptx
        if algorithm == "transpose":
            assert ".shared" in ptx
        else:
            assert ".shared" not in ptx


def test_production_routes_compile_portable_and_qualified_logical_warp_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cuda.coop.numba_mlir as qualified_coop
    from cuda import coop as root_coop

    compiler_cuda = _production_compile_environment(monkeypatch)

    def qualified_kernel(algorithm: str):
        load_algorithm = qualified_coop.WarpLoadAlgorithm[algorithm.upper()]
        store_algorithm = qualified_coop.WarpStoreAlgorithm[algorithm.upper()]

        @compiler_cuda.jit(chip="sm_90")
        def kernel(source, destination, valid_items, offset):
            group = qualified_coop.this_warp().group_by(8)
            payload = qualified_coop.ThreadData(
                _ITEMS_PER_THREAD,
                dtype=types.int32,
            )
            loaded = qualified_coop.load(
                group,
                source,
                payload,
                algorithm=load_algorithm,
                valid_items=valid_items,
                oob_default=types.int32(-17),
                offset=offset,
            )
            qualified_coop.store(
                group,
                destination,
                loaded,
                algorithm=store_algorithm,
                valid_items=valid_items,
                offset=offset,
            )

        return kernel

    def portable_kernel(algorithm: str):
        @compiler_cuda.jit(chip="sm_90")
        def kernel(source, destination, valid_items, offset):
            group = root_coop.this_warp().group_by(8)
            payload = root_coop.ThreadData(
                _ITEMS_PER_THREAD,
                dtype=types.int32,
            )
            loaded = root_coop.load(
                group,
                source,
                payload,
                algorithm=algorithm,
                valid_items=valid_items,
                oob_default=types.int32(-17),
                offset=offset,
            )
            root_coop.store(
                group,
                destination,
                loaded,
                algorithm=algorithm,
                valid_items=valid_items,
                offset=offset,
            )

        return kernel

    signature = types.void(
        types.int32[::1],
        types.int32[::1],
        types.int32,
        types.int64,
    )
    launch_config_key = _production_launch_config_key()
    for algorithm in ("direct", "transpose"):
        for factory in (portable_kernel, qualified_kernel):
            dispatcher = factory(algorithm)
            result = dispatcher._compile_launch_config_signature(
                signature,
                launch_config_key,
            )
            assert result.metadata["ltoir"]
            assert result.metadata["cubin"]
            assert result.metadata["linked_external_link_items"]
            ptx = next(iter(dispatcher.inspect_lto_ptx().values()))
            assert "bar.sync" not in ptx
            if algorithm == "transpose":
                assert ".shared" in ptx
            else:
                assert ".shared" not in ptx


@pytest.mark.parametrize("qualified", (False, True), ids=("portable", "qualified"))
def test_warp_scalar_literal_store_compiles_with_destination_dtype(
    monkeypatch: pytest.MonkeyPatch,
    qualified: bool,
) -> None:
    import cuda.coop.numba_mlir as qualified_coop
    from cuda import coop as root_coop

    compiler_cuda = _production_compile_environment(monkeypatch)
    if qualified:

        @compiler_cuda.jit(chip="sm_90")
        def kernel(destination):
            qualified_coop.store(
                qualified_coop.this_warp(),
                destination,
                23,
                algorithm="direct",
            )

    else:

        @compiler_cuda.jit(chip="sm_90")
        def kernel(destination):
            root_coop.store(
                root_coop.this_warp(),
                destination,
                23,
                algorithm="direct",
            )

    result = kernel._compile_launch_config_signature(
        types.void(types.int32[::1]),
        _production_launch_config_key(),
    )
    assert result.metadata["ltoir"]
    assert result.metadata["cubin"]


@pytest.mark.parametrize("qualified", (False, True), ids=("portable", "qualified"))
def test_warp_scalar_runtime_expression_rejects_implicit_narrowing(
    monkeypatch: pytest.MonkeyPatch,
    qualified: bool,
) -> None:
    import cuda.coop.numba_mlir as qualified_coop
    from cuda import coop as root_coop

    compiler_cuda = _production_compile_environment(monkeypatch)
    if qualified:

        @compiler_cuda.jit(chip="sm_90")
        def kernel(source, destination):
            thread = compiler_cuda.threadIdx.x
            qualified_coop.store(
                qualified_coop.this_warp(),
                destination,
                source[thread] + 1,
                algorithm="direct",
            )

    else:

        @compiler_cuda.jit(chip="sm_90")
        def kernel(source, destination):
            thread = compiler_cuda.threadIdx.x
            root_coop.store(
                root_coop.this_warp(),
                destination,
                source[thread] + 1,
                algorithm="direct",
            )

    signature = types.void(types.int32[::1], types.int32[::1])
    with pytest.raises(TypeError, match="does not match payload dtype"):
        kernel._compile_launch_config_signature(
            signature,
            _production_launch_config_key(),
        )
