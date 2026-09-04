# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""GPU-free, real-toolchain compilation contracts for hierarchy Reduce."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")

import numba_cuda_mlir.tools as numba_mlir_tools
from numba_cuda_mlir import cuda, types

from cuda.coop._core import ArgumentBinding, SynchronizationScope
from cuda.coop._core.thread_group import (
    ThreadHierarchy,
    this_block,
    this_cluster,
    this_thread,
    this_warp,
)
from cuda.coop.numba_mlir import _types
from cuda.coop.numba_mlir._compiler import _nvrtc
from cuda.coop.numba_mlir._compiler._operations import StorageABI
from cuda.coop.numba_mlir._lowering import _reduce

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]

_FIXED_COMPUTE_CAPABILITY = (9, 0)
_FIXED_CC = 90
_BLOCK_THREADS = 64


@pytest.fixture(autouse=True)
def _fixed_compiler_target(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Hide devices while giving both compiler paths an exact target."""

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", (
        "the Numba-CUDA-MLIR compile stage must hide all CUDA devices"
    )
    queries: list[tuple[int, int]] = []

    def current_device() -> SimpleNamespace:
        queries.append(_FIXED_COMPUTE_CAPABILITY)
        return SimpleNamespace(compute_capability=_FIXED_COMPUTE_CAPABILITY)

    def compute_capability(as_type=str):
        assert as_type in (str, tuple)
        return _FIXED_COMPUTE_CAPABILITY if as_type is tuple else "sm_90"

    monkeypatch.setattr(_types.cuda, "get_current_device", current_device)
    monkeypatch.setattr(
        numba_mlir_tools,
        "get_gpu_compute_capability",
        compute_capability,
    )
    _types._DEVICE_LTOIR_CACHE.clear()
    return queries


@pytest.fixture(scope="module")
def compile_context() -> _nvrtc.CompileContext:
    return _nvrtc.resolve_compile_context()


def _resolved_group(kind: str):
    if kind == "mapped_warps":
        hierarchy = ThreadHierarchy._resolved(block_dim=128)
        return this_block().group_by(2).with_hierarchy(hierarchy)
    if kind == "cluster":
        hierarchy = ThreadHierarchy._resolved(
            block_dim=_BLOCK_THREADS,
            grid_dim=2,
            cluster_dim=2,
        )
        return this_cluster().with_hierarchy(hierarchy)

    hierarchy = ThreadHierarchy._resolved(block_dim=_BLOCK_THREADS)
    group = {
        "thread": this_thread(),
        "warp": this_warp(),
        "logical_warp": this_warp().group_by(8),
        "block": this_block(),
    }[kind]
    return group.with_hierarchy(hierarchy)


def _group_factory(scope: SynchronizationScope):
    return {
        SynchronizationScope.NONE: _reduce.group_reduce_none,
        SynchronizationScope.WARP: _reduce.group_reduce_warp,
        SynchronizationScope.BLOCK: _reduce.group_reduce_block,
        SynchronizationScope.GROUP: _reduce.group_reduce_group,
    }[scope]


_CUDAX_CASES = (
    pytest.param(
        "thread",
        SynchronizationScope.NONE,
        types.int32,
        1,
        "scalar",
        None,
        True,
        id="thread-scalar-broadcast-sum",
    ),
    pytest.param(
        "warp",
        SynchronizationScope.WARP,
        types.float32,
        1,
        "array",
        "max",
        False,
        id="warp-array-one-root-max",
    ),
    pytest.param(
        "logical_warp",
        SynchronizationScope.WARP,
        types.uint32,
        3,
        "array",
        "bit_xor",
        True,
        id="logical-warp-array-broadcast-xor",
    ),
    pytest.param(
        "block",
        SynchronizationScope.BLOCK,
        types.int64,
        1,
        "scalar",
        "multiplies",
        False,
        id="block-scalar-root-multiplies",
    ),
    pytest.param(
        "mapped_warps",
        SynchronizationScope.GROUP,
        types.float64,
        2,
        "array",
        "min",
        True,
        id="mapped-warps-array-broadcast-min",
    ),
    pytest.param(
        "cluster",
        SynchronizationScope.GROUP,
        types.int16,
        1,
        "scalar",
        "sum",
        False,
        id="cluster-scalar-root-sum",
    ),
)


@pytest.mark.parametrize(
    (
        "group_kind",
        "scope",
        "dtype",
        "items_per_thread",
        "value_kind",
        "binary_op",
        "broadcast",
    ),
    _CUDAX_CASES,
)
def test_cudax_hierarchy_reduce_compiles_without_external_storage(
    compile_context: _nvrtc.CompileContext,
    _fixed_compiler_target: list[tuple[int, int]],
    group_kind: str,
    scope: SynchronizationScope,
    dtype,
    items_per_thread: int,
    value_kind: str,
    binary_op,
    broadcast: bool,
) -> None:
    group = _resolved_group(group_kind)
    invocable = _group_factory(scope)(
        dtype=dtype,
        group=group,
        binary_op=binary_op,
        items_per_thread=items_per_thread,
        value_kind=value_kind,
        broadcast=broadcast,
        _compile_context=compile_context,
    )
    source = invocable.source

    first_include = source.index("#include")
    enable_macro = source.index("#define _CUDAX_ENABLE_GROUP_FEATURES_IN_LIBCUDACXX")
    disable_macro = source.index("#define _CUDAX_DISABLE_COOPERATIVE_GROUPS_INTEROP")
    assert enable_macro < disable_macro < first_include
    assert invocable.storage_abi is StorageABI.NONE
    assert invocable.execution_scope is scope
    assert invocable.synchronization_scope is SynchronizationScope.NONE
    assert invocable.temp_storage_bytes == 0
    assert invocable.temp_storage_alignment == 1
    assert "TempStorage" not in source
    assert "temp_storage" not in source
    assert "__syncthreads" not in source
    assert "__syncwarp" not in source
    assert "bar.sync" not in source
    assert ("::cuda::experimental::broadcasted" in source) is broadcast
    assert (".value_or(" in source) is (not broadcast)

    if value_kind == "scalar":
        assert invocable.abi_transforms == ("value",)
        assert " item)" in source
        assert "raw_items" not in source
    else:
        assert invocable.abi_transforms == ("ptr",)
        assert isinstance(invocable.parameters[0], _types.Array)
        assert invocable.parameters[0].size == items_per_thread
        assert "void* raw_items" in source
        assert f"(*)[{items_per_thread}]>(raw_items)" in source

    if group_kind == "logical_warp":
        assert "::cuda::experimental::this_warp group_parent" in source
        assert "::cuda::experimental::group_by<8, true>" in source
        assert "::cuda::experimental::lane_synchronizer" in source
    elif group_kind == "mapped_warps":
        assert "::cuda::experimental::this_block group_parent" in source
        assert "::cuda::experimental::group_by<2, true>" in source
        assert "::cuda::experimental::barrier_synchronizer" in source
        # This shared state belongs to construction of the mapped CUDAX group;
        # it is not a provider TempStorage operand or a rewrite-owned barrier.
        assert "group_barriers_storage" in source
    elif group_kind == "cluster":
        assert "::cuda::cluster_dims<2>()" in source
        assert "::cuda::experimental::this_cluster group" in source
    else:
        assert "group_barriers_storage" not in source

    assert len(invocable.files) == 1
    artifact = Path(invocable.files[0])
    assert artifact.suffix == ".ltoir"
    assert artifact.stat().st_size > 0
    ptx = _types._ltoir_to_ptx(
        artifact.read_bytes(),
        name=invocable.symbol,
        cc=_FIXED_CC,
    )
    assert ".version" in ptx
    assert _fixed_compiler_target


def _collect_cub(
    factory,
    compile_context: _nvrtc.CompileContext,
    /,
    **kwargs,
):
    with _types.collect_specializations() as collected:
        result = factory(**kwargs)
    assert len(collected) == 1
    algorithm, threads, block_threads = collected[0]
    assert result is algorithm
    algorithm._compile_context = compile_context
    return algorithm, threads, block_threads


def _cub_source(collected) -> str:
    algorithm, threads, block_threads = collected
    return algorithm._source_code(
        threads=threads,
        block_threads=block_threads,
    )[0]


def _compile_cub_bundle(collected, *, name: str) -> bytes:
    algorithms = [item[0] for item in collected]
    ltoir = _types.prepare_ltoir_bundle(
        algorithms,
        bundle_name=name,
        allow_single=True,
        threads_by_algo={id(item[0]): item[1] for item in collected},
        block_threads_by_algo={id(item[0]): item[2] for item in collected},
    )
    assert isinstance(ltoir, bytes)
    assert ltoir
    return ltoir


def test_deterministic_cub_reduce_variants_compile_with_scoped_storage(
    compile_context: _nvrtc.CompileContext,
) -> None:
    block_cases = (
        (
            _reduce.sum,
            dict(
                dtype=types.int32,
                threads_per_block=_BLOCK_THREADS,
                algorithm="warp_reductions",
            ),
            ".Sum(",
            "::cub::BLOCK_REDUCE_WARP_REDUCTIONS",
        ),
        (
            _reduce.sum,
            dict(
                dtype=types.float64,
                threads_per_block=_BLOCK_THREADS,
                items_per_thread=2,
                value_kind="array",
                algorithm="raking",
            ),
            ".Sum(",
            "::cub::BLOCK_REDUCE_RAKING",
        ),
        (
            _reduce.block_reduce_builtin,
            dict(
                dtype=types.float32,
                threads_per_block=_BLOCK_THREADS,
                binary_op="max",
                algorithm="raking",
            ),
            ".Reduce(",
            "::cub::BLOCK_REDUCE_RAKING",
        ),
        (
            _reduce.block_reduce_builtin,
            dict(
                dtype=types.uint32,
                threads_per_block=_BLOCK_THREADS,
                binary_op="bit_xor",
                algorithm="raking_commutative_only",
            ),
            ".Reduce(",
            "::cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY",
        ),
        (
            _reduce.sum,
            dict(
                dtype=types.uint16,
                threads_per_block=_BLOCK_THREADS,
                num_valid=ArgumentBinding.runtime(),
            ),
            ".Sum(",
            "::cub::BLOCK_REDUCE_WARP_REDUCTIONS",
        ),
        (
            _reduce.sum,
            dict(
                dtype=types.int64,
                threads_per_block=_BLOCK_THREADS,
                num_valid=ArgumentBinding.static(37),
            ),
            ".Sum(",
            "::cub::BLOCK_REDUCE_WARP_REDUCTIONS",
        ),
    )
    warp_cases = (
        (
            _reduce.warp_sum,
            dict(
                dtype=types.int16,
                threads_in_warp=32,
                threads_per_block=_BLOCK_THREADS,
            ),
            ".Sum(",
        ),
        (
            _reduce.warp_reduce_builtin,
            dict(
                dtype=types.float32,
                binary_op="max",
                threads_in_warp=8,
                threads_per_block=_BLOCK_THREADS,
                valid_items=ArgumentBinding.static(5),
            ),
            ".Reduce(",
        ),
        (
            _reduce.warp_reduce_builtin,
            dict(
                dtype=types.uint32,
                binary_op="bit_and",
                threads_in_warp=8,
                threads_per_block=_BLOCK_THREADS,
                valid_items=ArgumentBinding.runtime(),
            ),
            ".Reduce(",
        ),
    )
    collected = [
        _collect_cub(factory, compile_context, **kwargs)
        for factory, kwargs, _method, _algorithm in block_cases
    ]
    collected.extend(
        _collect_cub(factory, compile_context, **kwargs)
        for factory, kwargs, _method in warp_cases
    )

    for item, (_factory, _kwargs, method, algorithm_name) in zip(
        collected[: len(block_cases)],
        block_cases,
    ):
        algorithm = item[0]
        source = _cub_source(item)
        assert method in source
        assert algorithm_name in source
        assert "WARP_REDUCTIONS_NONDETERMINISTIC" not in source
        assert "cub::BlockReduce<" in source
        assert "TempStorage" in source
        assert "__shared__" in source
        assert "__syncthreads();" in source
        assert "__syncwarp" not in source
        assert algorithm.storage_abi is StorageABI.LEADING_POINTER
        assert algorithm.execution_scope is SynchronizationScope.BLOCK
        assert algorithm.synchronization_scope is SynchronizationScope.BLOCK

    for item, (_factory, kwargs, method) in zip(
        collected[len(block_cases) :],
        warp_cases,
    ):
        algorithm = item[0]
        source = _cub_source(item)
        width = kwargs["threads_in_warp"]
        assert method in source
        assert (
            f"cub::WarpReduce<{_types.numba_type_to_cpp(kwargs['dtype'])}, {width}>"
            in source
        )
        assert "TempStorage" in source
        assert "__shared__" in source
        assert "__syncthreads" not in source
        assert "__syncwarp" in source
        assert algorithm.storage_abi is StorageABI.LEADING_POINTER
        assert algorithm.execution_scope is SynchronizationScope.WARP
        assert algorithm.synchronization_scope is SynchronizationScope.WARP

    block_runtime_prefix = _cub_source(collected[4])
    warp_runtime_prefix = _cub_source(collected[-1])
    for source in (block_runtime_prefix, warp_runtime_prefix):
        assert "::cuda::std::int64_t" in source
        assert "static_cast<::cuda::std::int32_t>" in source
        assert 'asm volatile("trap;" : : :);' in source
    assert "if (param_1 < 1 || param_1 > 64)" in block_runtime_prefix
    assert "if (param_2 < 1 || param_2 > 8)" in warp_runtime_prefix
    assert ".Sum(param_0, 37);" in _cub_source(collected[5])
    assert ", 5);" in _cub_source(collected[-2])

    bundle = _compile_cub_bundle(
        collected,
        name="cuda_coop_numba_mlir_deterministic_reduce_variants",
    )
    ptx = _types._ltoir_to_ptx(
        bundle,
        name="cuda_coop_numba_mlir_deterministic_reduce_variants",
        cc=_FIXED_CC,
    )
    assert ".version" in ptx
    assert all(item[0].temp_storage_bytes > 0 for item in collected)
    assert all(item[0].temp_storage_alignment > 0 for item in collected)


def _link_ltoir_files(paths: list[str], *, name: str) -> str:
    from cuda.core import Linker, LinkerOptions, ObjectCode

    objects = [
        ObjectCode.from_ltoir(Path(path).read_bytes(), name=f"{name}_{index}")
        for index, path in enumerate(paths)
    ]
    options = LinkerOptions(
        arch=f"sm_{_FIXED_CC}",
        link_time_optimization=True,
        ptx=True,
    )
    linked = Linker(*objects, options=options).link("ptx")
    return linked.code.decode("utf-8")


def test_stateless_cub_callback_compiles_and_links_both_lto_inputs(
    compile_context: _nvrtc.CompileContext,
) -> None:
    def maximum(lhs, rhs):
        return lhs if lhs > rhs else rhs

    device_maximum = cuda.jit(device=True)(maximum)
    collected = _collect_cub(
        _reduce.reduce,
        compile_context,
        dtype=types.int32,
        threads_per_block=_BLOCK_THREADS,
        binary_op=device_maximum,
        algorithm="warp_reductions",
    )
    algorithm = collected[0]
    source, support_ltoirs, _, declarations = algorithm._source_code(
        threads=collected[1],
        block_threads=collected[2],
    )

    assert len(support_ltoirs) == 1
    assert len(declarations) == 1
    callback_name = next(iter(declarations))
    assert declarations[callback_name] in source
    assert "auto param_1 = []" in source
    assert f"return {callback_name}(wp_0, wp_1);" in source
    assert ".Reduce(param_0, param_1);" in source

    bundle = _compile_cub_bundle(
        [collected],
        name="cuda_coop_numba_mlir_stateless_reduce_callback",
    )
    invocable = _types.make_invocable_from_specialization(
        algorithm,
        threads=collected[1],
        block_threads=collected[2],
    )

    assert len(invocable.files) == 2
    assert all(Path(path).suffix == ".ltoir" for path in invocable.files)
    assert Path(invocable.files[0]).read_bytes() == bundle
    assert Path(invocable.files[1]).read_bytes() == support_ltoirs[0]
    linked_ptx = _link_ltoir_files(
        invocable.files,
        name="cuda_coop_numba_mlir_stateless_reduce_callback",
    )
    assert ".version" in linked_ptx
    assert invocable.temp_storage_bytes > 0
    assert invocable.temp_storage_alignment > 0
