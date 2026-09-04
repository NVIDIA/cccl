# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ast
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

import cuda.coop as portable_coop
import cuda.coop.numba_mlir as coop
from cuda.coop.numba_mlir import _temp_storage, _thread_data
from cuda.coop.numba_mlir._compiler import _activation, _numba_mlir_compat

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]

_PORTABLE_EXPORTS = [
    "__version__",
    "Hierarchy",
    "TempStorage",
    "TempStorageLike",
    "ThreadData",
    "ThreadDataLike",
    "ThreadGroup",
    "ThreadHierarchy",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
    "exchange",
    "load",
    "reduce",
    "shuffle",
    "store",
    "sum",
]
_QUALIFIED_EXPORTS = [
    *(name for name in _PORTABLE_EXPORTS if name != "__version__"),
    "local",
    "shared",
]
_EXCLUDED_BACKEND_MODULES = (
    "cuda.coop.numba_mlir._dataclass",
    "cuda.coop.numba_mlir._enums",
    "cuda.coop.numba_mlir._group_scan",
    "cuda.coop.numba_mlir._stateful_function",
    "cuda.coop.numba_mlir._compiler._group_scan",
    "cuda.coop.numba_mlir._compiler._rewrite_scan",
    "cuda.coop.numba_mlir._lowering._scan",
    "cuda.coop.numba_mlir._lowering._thread_group",
    "cuda.coop.numba_mlir._lowering._warp",
)


def test_public_exports_are_only_the_supported_group_families():
    assert portable_coop.__all__ == _PORTABLE_EXPORTS
    assert dir(portable_coop) == sorted(_PORTABLE_EXPORTS)
    assert coop.__all__ == _QUALIFIED_EXPORTS
    assert dir(coop) == sorted(_QUALIFIED_EXPORTS)

    excluded_exports = {
        "BlockLoadAlgorithm",
        "BlockReduceAlgorithm",
        "BlockScanAlgorithm",
        "BlockStoreAlgorithm",
        "StatefulFunction",
        "exclusive_scan",
        "gpu_dataclass",
        "inclusive_scan",
        "scan",
        "WarpLoadAlgorithm",
        "WarpStoreAlgorithm",
    }
    assert excluded_exports.isdisjoint(portable_coop.__all__)
    assert excluded_exports.isdisjoint(coop.__all__)

    loaded = set(sys.modules)
    assert "cuda.coop.numba_mlir._group_load_store" in loaded
    assert "cuda.coop.numba_mlir._compiler._rewrite" in loaded
    assert set(_EXCLUDED_BACKEND_MODULES).isdisjoint(loaded)
    assert importlib.import_module("cuda.coop.numba_mlir._lowering").__all__ == ()

    coop_root = Path(portable_coop.__file__).resolve().parent
    assert not (coop_root / "cutlass").exists()


def test_qualified_surface_is_portable_plus_backend_extensions():
    assert set(coop.__all__) - set(portable_coop.__all__) == {"local", "shared"}
    assert set(portable_coop.__all__) - set(coop.__all__) == {"__version__"}

    def call_shape(function):
        return tuple(
            (name, parameter.kind, parameter.default)
            for name, parameter in inspect.signature(function).parameters.items()
        )

    for operation in ("load", "shuffle", "store"):
        assert inspect.signature(getattr(coop, operation)) == inspect.signature(
            getattr(portable_coop, operation)
        )

    portable_exchange = inspect.signature(portable_coop.exchange)
    qualified_exchange = inspect.signature(coop.exchange)
    for name, parameter in portable_exchange.parameters.items():
        assert qualified_exchange.parameters[name] == parameter
    assert qualified_exchange.return_annotation == portable_exchange.return_annotation
    assert tuple(qualified_exchange.parameters)[
        len(portable_exchange.parameters) :
    ] == (
        "ranks",
        "valid_flags",
        "warp_time_slicing",
    )

    assert call_shape(coop.TempStorage) == call_shape(portable_coop.TempStorage)
    for constructor in (
        "this_thread",
        "this_warp",
        "this_block",
        "this_cluster",
        "this_grid",
    ):
        assert call_shape(getattr(coop, constructor)) == call_shape(
            getattr(portable_coop, constructor)
        )

    portable_thread_data = call_shape(portable_coop.ThreadData)
    qualified_thread_data = call_shape(coop.ThreadData)
    assert qualified_thread_data[:-1] == portable_thread_data
    assert qualified_thread_data[-1] == (
        "alignas",
        inspect.Parameter.KEYWORD_ONLY,
        8,
    )
    assert call_shape(coop.ThreadGroup.group_by) == call_shape(
        portable_coop.ThreadGroup.group_by
    )

    assert coop.ThreadDataLike is portable_coop.ThreadDataLike
    assert coop.TempStorageLike is portable_coop.TempStorageLike
    assert coop.ThreadHierarchy is portable_coop.ThreadHierarchy
    assert coop.Hierarchy is portable_coop.Hierarchy

    portable_load_annotations = inspect.get_annotations(
        portable_coop.load,
        eval_str=True,
    )
    qualified_load_annotations = inspect.get_annotations(
        coop.load,
        eval_str=True,
    )
    assert qualified_load_annotations["output"] == portable_load_annotations["output"]
    assert qualified_load_annotations["return"] == portable_load_annotations["return"]

    coop_root = Path(portable_coop.__file__).resolve().parent

    def stub_signatures(path):
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        return [
            (node.name, ast.dump(node.args), ast.dump(node.returns))
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name in {"load", "store"}
        ]

    assert stub_signatures(coop_root / "numba_mlir" / "_group_load_store.pyi") == (
        stub_signatures(coop_root / "_core" / "api" / "load_store.pyi")
    )


def test_group_descriptors_expose_only_canonical_extent_names():
    assert not hasattr(portable_coop.ThreadHierarchy, "thread_count")
    for group_type in (portable_coop.ThreadGroup, coop.ThreadGroup):
        assert not hasattr(group_type, "static_thread_count")
        assert not hasattr(group_type, "thread_count")
        assert hasattr(group_type, "static_size")
        assert hasattr(group_type, "group_thread_count")


def test_stub_only_group_aliases_exist_at_runtime_without_becoming_exports():
    from cuda.coop._core.api import thread_group as portable_groups
    from cuda.coop.numba_mlir import _thread_group as qualified_groups

    for name in ("MemoryGroup", "ReductionGroup", "BlockGroup", "WarpGroup"):
        assert getattr(portable_groups, name) is portable_groups.ThreadGroup
        assert name not in portable_groups.__all__
    for name in ("ReductionGroup", "BlockGroup", "WarpGroup"):
        assert getattr(qualified_groups, name) is qualified_groups.ThreadGroup
        assert name not in qualified_groups.__all__


@pytest.mark.parametrize("module_name", _EXCLUDED_BACKEND_MODULES)
def test_excluded_backend_implementation_modules_remain_absent(module_name):
    assert importlib.util.find_spec(module_name) is None

    coop_root = Path(portable_coop.__file__).resolve().parent
    relative_module = module_name.removeprefix("cuda.coop.")
    module_path = coop_root.joinpath(*relative_module.split("."))
    assert not module_path.with_suffix(".py").exists()
    assert not module_path.is_dir()


def test_python_operator_compilation_is_stateless_only():
    from cuda.coop.numba_mlir import _types

    assert hasattr(_types, "_compile_device_ltoir")
    assert hasattr(_types, "DependentPythonOperator")
    assert hasattr(_types, "StatelessOperator")
    assert not hasattr(_types, "StatefulOperator")
    assert tuple(inspect.signature(_types.numba_type_to_wrapper).parameters) == (
        "numba_type",
    )


@pytest.mark.parametrize(
    "operation", ("exchange", "load", "reduce", "shuffle", "store", "sum")
)
def test_group_markers_use_exact_callable_identity(operation):
    from cuda.coop.numba_mlir._compiler._operations import group_operation_name

    marker = getattr(coop, operation)
    assert group_operation_name(marker) == operation

    def impostor(*args, **kwargs):
        del args, kwargs

    impostor.__module__ = marker.__module__
    impostor.__name__ = marker.__name__
    impostor.__cuda_coop_backend_member__ = operation
    assert group_operation_name(impostor) is None


@pytest.mark.parametrize("operation", ("load", "store"))
def test_lowering_factories_use_exact_callable_identity(operation):
    from cuda.coop._core import SynchronizationScope
    from cuda.coop.numba_mlir import _lowering
    from cuda.coop.numba_mlir._compiler._operations import (
        FactoryOperation,
        StorageABI,
        factory_operation,
    )

    factory = getattr(_lowering, operation)
    assert factory_operation(factory) == FactoryOperation(
        operation=operation,
        namespace="block",
        storage_abi=StorageABI.NONE,
        execution_scope=SynchronizationScope.BLOCK,
        synchronization_scope=SynchronizationScope.NONE,
    )

    def impostor(*args, **kwargs):
        del args, kwargs

    impostor.__module__ = factory.__module__
    impostor.__name__ = factory.__name__
    assert factory_operation(impostor) is None


@pytest.mark.parametrize("operation", ("load", "store"))
def test_physical_warp_factories_use_exact_callable_identity(operation):
    from cuda.coop._core import SynchronizationScope
    from cuda.coop.numba_mlir._compiler._operations import (
        FactoryOperation,
        StorageABI,
        factory_operation,
    )
    from cuda.coop.numba_mlir._lowering import _load_store

    for storage_bearing in (False, True):
        factory = getattr(
            _load_store,
            (
                f"_warp_{operation}_with_storage"
                if storage_bearing
                else f"warp_{operation}"
            ),
        )
        assert factory_operation(factory) == FactoryOperation(
            operation=operation,
            namespace="warp",
            storage_abi=(
                StorageABI.LEADING_POINTER if storage_bearing else StorageABI.NONE
            ),
            execution_scope=SynchronizationScope.WARP,
            synchronization_scope=(
                SynchronizationScope.WARP
                if storage_bearing
                else SynchronizationScope.NONE
            ),
        )

        def impostor(*args, **kwargs):
            del args, kwargs

        impostor.__module__ = factory.__module__
        impostor.__name__ = factory.__name__
        assert factory_operation(impostor) is None


def test_compiler_hooks_are_registered_exactly_once_and_idempotently():
    group_rewrites = importlib.import_module(
        "cuda.coop.numba_mlir._compiler._group_planner"
    )
    storage_rewrites = importlib.import_module(
        "cuda.coop.numba_mlir._compiler._rewrite"
    )
    compat = _numba_mlir_compat._get_numba_mlir_compat()
    snapshot = compat.snapshot_registrations()

    def counts():
        registration_counts = compat.registration_counts(
            snapshot,
            (
                (
                    "CoopGroupHierarchyPlanner",
                    group_rewrites.CoopGroupHierarchyPlanner,
                ),
                (
                    "CoopWholeFunctionPlanner",
                    storage_rewrites.CoopWholeFunctionPlanner,
                ),
            ),
            (
                "CoopSinglePhaseRewrite",
                storage_rewrites.CoopSinglePhaseRewrite,
            ),
        )
        return tuple(registration_counts.values())

    assert counts() == (1, 1, 1)
    _activation._initialize_runtime_hooks()
    _activation._initialize_runtime_hooks()
    assert counts() == (1, 1, 1)


def test_public_runtime_helpers_have_semantic_module_owners():
    assert coop.local is importlib.import_module("numba_cuda_mlir.cuda").local
    assert coop.shared is importlib.import_module("numba_cuda_mlir.cuda").shared
    assert coop.ThreadData is _thread_data.ThreadData
    assert coop.TempStorage is _temp_storage.TempStorage
    assert coop.ThreadDataLike is portable_coop.ThreadDataLike
    assert coop.TempStorageLike is portable_coop.TempStorageLike
    assert coop.ThreadData.__module__ == "cuda.coop.numba_mlir._thread_data"
    assert coop.TempStorage.__module__ == "cuda.coop.numba_mlir._temp_storage"
