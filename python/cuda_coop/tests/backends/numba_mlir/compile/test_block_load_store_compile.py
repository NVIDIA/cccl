# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""GPU-free, real-toolchain compilation contracts for Block Load/Store."""

from __future__ import annotations

import gc
import os
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import types

from cuda.coop._core import ArgumentBinding, SynchronizationScope
from cuda.coop._core.block import make_block_load_spec, make_block_store_spec
from cuda.coop.numba_mlir import _types
from cuda.coop.numba_mlir._compiler import _caching, _nvrtc
from cuda.coop.numba_mlir._compiler._operations import StorageABI
from cuda.coop.numba_mlir._lowering._core import NumbaMlirCoreAdapter
from cuda.coop.numba_mlir._lowering._load_store import _load_store_value_abis

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]

_FIXED_COMPUTE_CAPABILITY = (9, 0)


@pytest.fixture(autouse=True)
def _fixed_current_device(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Hide runtime discovery while leaving NVRTC and nvJitLink entirely real."""

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
    operation: str = "load",
    block_dim: tuple[int, int, int] = (32, 1, 1),
    dtype=types.int32,
    items_per_thread: int = 2,
    valid_items: ArgumentBinding | None = None,
) -> _types.Algorithm:
    if valid_items is None:
        valid_items = ArgumentBinding.runtime()
    adapter = NumbaMlirCoreAdapter(
        value_abis=_load_store_value_abis(
            dtype=dtype,
            block_dim=block_dim,
            items_per_thread=items_per_thread,
            valid_items=valid_items,
        )
    )
    factory = make_block_load_spec if operation == "load" else make_block_store_spec
    spec = factory(
        dtype=adapter.core_dtype(dtype),
        block_dim=block_dim,
        items_per_thread=items_per_thread,
        algorithm="direct",
        valid_items=valid_items,
    )
    algorithm = adapter.materialize(
        spec.specialization,
        storage_abi=StorageABI.NONE,
        execution_scope=SynchronizationScope.BLOCK,
        synchronization_scope=SynchronizationScope.NONE,
        extra_type_definitions=(_types.numba_type_to_wrapper(dtype),),
    )
    algorithm._compile_context = context
    return algorithm


def _source(algorithm: _types.Algorithm) -> str:
    source, _support_lto_irs, _storage_symbols, _udf_declarations = (
        algorithm._source_code()
    )
    return source


def _storage_abi_variant(source: str, algorithm: _types.Algorithm) -> str:
    """Change only the explicit temporary-storage wrapper ABI."""

    wrapper = f"{algorithm.mangled_name(algorithm.parameters[0])}__abi"
    original = f"{wrapper}(void *__ret, "
    changed = f"{wrapper}(void *__ret, unsigned long long __cuda_coop_storage_abi, "
    assert source.count(original) == 1
    return source.replace(original, changed, 1)


def _cache_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in root.rglob("*") if path.is_file()))


def _stat_identity(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_ino, stat.st_mtime_ns, stat.st_size


def test_direct_load_store_compile_without_temp_storage_or_barriers(
    compile_context: _nvrtc.CompileContext,
    _fixed_current_device: list[tuple[int, int]],
) -> None:
    algorithms = [
        _algorithm(compile_context, operation="load"),
        _algorithm(compile_context, operation="store"),
    ]
    clones = [
        _algorithm(compile_context, operation="load"),
        _algorithm(compile_context, operation="store"),
    ]

    sources = [_source(algorithm) for algorithm in algorithms]
    clone_sources = [_source(algorithm) for algorithm in clones]
    assert sources == clone_sources
    for source in sources:
        assert "::cuda::std::int64_t param_2" in source
        assert "if (param_2 < 0 || param_2 > 64)" in source
        assert 'asm volatile("trap;" : : :);' in source
        assert "static_cast<::cuda::std::int32_t>(param_2)" in source
        assert "checked_param_2);" in source

    for algorithm, source in zip(algorithms, sources):
        explicit_wrappers = tuple(
            f"{algorithm.mangled_name(method)}__abi" for method in algorithm.parameters
        )
        implicit_wrappers = tuple(
            f"{algorithm.mangled_name(method)}_alloc__abi"
            for method in algorithm.parameters
        )
        storage_symbols = algorithm._temp_storage_symbol_names()
        assert all(symbol in source for symbol in explicit_wrappers)
        assert all(symbol not in source for symbol in implicit_wrappers)
        assert all(symbol not in source for symbol in storage_symbols)
        assert "TempStorage" not in source
        assert "temp_storage" not in source
        assert "__syncthreads" not in source
        assert "__syncwarp" not in source
    assert "().Load(" in sources[0]
    assert "().Store(" in sources[1]

    ltoir = _types.prepare_ltoir_bundle(
        algorithms,
        bundle_name="cuda_coop_numba_mlir_block_load_store_compile_test",
    )
    assert isinstance(ltoir, bytes)
    assert ltoir
    assert _fixed_current_device
    assert set(_fixed_current_device) == {_FIXED_COMPUTE_CAPABILITY}

    cc = 10 * _FIXED_COMPUTE_CAPABILITY[0] + _FIXED_COMPUTE_CAPABILITY[1]
    ptx = _types._ltoir_to_ptx(ltoir, name="load_store_metadata", cc=cc)
    assert ".version" in ptx
    assert ".shared" not in ptx
    assert "bar.sync" not in ptx
    assert all(algorithm.temp_storage_bytes == 0 for algorithm in algorithms)
    assert all(algorithm.temp_storage_alignment == 1 for algorithm in algorithms)

    shared_artifact = algorithms[0]._precompiled_ltoir_files[0]
    assert algorithms[1]._precompiled_ltoir_files[0] is shared_artifact
    artifact_path = Path(shared_artifact.name)
    shared_artifact_ref = weakref.ref(shared_artifact)
    assert artifact_path.suffix == ".ltoir"
    assert artifact_path.read_bytes() == ltoir

    invocables = [
        _types.make_invocable_from_specialization(algorithm) for algorithm in algorithms
    ]
    assert all(invocable.files == [str(artifact_path)] for invocable in invocables)
    assert all(
        invocable.temp_storage_bytes == invocable.specialization.temp_storage_bytes
        for invocable in invocables
    )
    assert all(
        invocable.temp_storage_alignment
        == invocable.specialization.temp_storage_alignment
        for invocable in invocables
    )

    del shared_artifact, algorithms
    gc.collect()
    assert shared_artifact_ref() is not None
    assert artifact_path.is_file()
    assert all(invocable.files == [str(artifact_path)] for invocable in invocables)


def test_real_nvrtc_cache_hits_and_invalidates_every_compile_axis(
    compile_context: _nvrtc.CompileContext,
    _fixed_current_device: list[tuple[int, int]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise both cache tiers without replacing either compiler library."""

    monkeypatch.setattr(_caching, "_CACHE_USABLE", True)
    monkeypatch.setattr(_caching, "_CACHE_LOCATION", str(tmp_path))
    _nvrtc.compile_impl.cache_clear()

    base_algorithm = _algorithm(compile_context)
    base_source = _source(base_algorithm)
    base_kwargs = {
        "cpp": base_source,
        "cc": 90,
        "rdc": True,
        "code": "lto",
        "context": compile_context,
    }

    try:
        _version, first = _nvrtc.compile(**base_kwargs)
        first_info = _nvrtc.compile_impl.cache_info()
        assert first_info.hits == 0
        assert first_info.misses == 1
        files = _cache_files(tmp_path)
        assert len(files) == 1
        first_disk_identity = _stat_identity(files[0])

        _version, memory_cached = _nvrtc.compile(**base_kwargs)
        memory_info = _nvrtc.compile_impl.cache_info()
        assert memory_cached == first
        assert memory_info.hits == first_info.hits + 1
        assert memory_info.misses == first_info.misses
        assert _cache_files(tmp_path) == files
        assert _stat_identity(files[0]) == first_disk_identity

        _nvrtc.compile_impl.cache_clear()
        _version, disk_cached = _nvrtc.compile(**base_kwargs)
        disk_info = _nvrtc.compile_impl.cache_info()
        assert disk_cached == first
        assert disk_info.hits == 0
        assert disk_info.misses == 1
        assert _cache_files(tmp_path) == files
        assert _stat_identity(files[0]) == first_disk_identity

        dimension_source = _source(_algorithm(compile_context, block_dim=(8, 4, 1)))
        dtype_source = _source(_algorithm(compile_context, dtype=types.uint16))
        items_source = _source(_algorithm(compile_context, items_per_thread=3))
        static_binding_source = _source(
            _algorithm(
                compile_context,
                valid_items=ArgumentBinding.static(17),
            )
        )
        storage_abi_source = _storage_abi_variant(base_source, base_algorithm)
        changed_context = replace(
            compile_context,
            header_identity=f"{compile_context.header_identity}-changed",
        )
        variants = {
            "source": {**base_kwargs, "cpp": f"{base_source}\nstatic_assert(true);\n"},
            "block-dimension": {**base_kwargs, "cpp": dimension_source},
            "dtype": {**base_kwargs, "cpp": dtype_source},
            "items-per-thread": {**base_kwargs, "cpp": items_source},
            "static-runtime-binding": {
                **base_kwargs,
                "cpp": static_binding_source,
            },
            "storage-abi": {**base_kwargs, "cpp": storage_abi_source},
            "compile-context": {**base_kwargs, "context": changed_context},
            "compute-capability": {**base_kwargs, "cc": 80},
            "relocatable-device-code": {**base_kwargs, "rdc": False},
            "output-kind": {**base_kwargs, "code": "ptx"},
        }

        previous_misses = disk_info.misses
        previous_file_count = len(files)
        for axis, kwargs in variants.items():
            _version, result = _nvrtc.compile(**kwargs)
            info = _nvrtc.compile_impl.cache_info()
            assert info.misses == previous_misses + 1, axis
            assert info.hits == 0, axis
            assert len(_cache_files(tmp_path)) == previous_file_count + 1, axis
            if kwargs["code"] == "lto":
                assert isinstance(result, bytes) and result, axis
            else:
                assert isinstance(result, str) and ".version" in result, axis
            previous_misses = info.misses
            previous_file_count += 1

        original_compiler_options = _nvrtc._compiler_options

        def changed_compiler_options(**kwargs):
            return (
                *original_compiler_options(**kwargs),
                b"-DCUDA_COOP_COMPILE_OPTION_IDENTITY_TEST=1",
            )

        monkeypatch.setattr(_nvrtc, "_compiler_options", changed_compiler_options)
        _version, option_changed = _nvrtc.compile(**base_kwargs)
        option_info = _nvrtc.compile_impl.cache_info()
        assert option_info.misses == previous_misses + 1
        assert len(_cache_files(tmp_path)) == previous_file_count + 1
        assert isinstance(option_changed, bytes) and option_changed

        assert _fixed_current_device
        assert set(_fixed_current_device) == {_FIXED_COMPUTE_CAPABILITY}
    finally:
        _nvrtc.compile_impl.cache_clear()


def test_provider_symbols_and_cached_lto_are_bound_to_the_compilation_target(
    compile_context: _nvrtc.CompileContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_cc = [(9, 0)]

    monkeypatch.setattr(
        _types.cuda,
        "get_current_device",
        lambda: SimpleNamespace(compute_capability=current_cc[0]),
    )
    sm90 = _algorithm(compile_context)
    sm90_source = _source(sm90)
    sm90_symbol = sm90.mangled_name(sm90.parameters[0])

    current_cc[0] = (8, 0)
    sm80 = _algorithm(compile_context)
    sm80_source = _source(sm80)
    sm80_symbol = sm80.mangled_name(sm80.parameters[0])

    assert sm90_source != sm80_source
    assert sm90_symbol != sm80_symbol
    assert sm90._provider_compile_identity[:3] == (90, True, "lto")
    assert sm80._provider_compile_identity[:3] == (80, True, "lto")

    with pytest.raises(RuntimeError, match="different compute capability"):
        sm90.get_lto_ir()
