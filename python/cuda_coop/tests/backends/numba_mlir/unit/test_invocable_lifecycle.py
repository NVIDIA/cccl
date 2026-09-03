# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import gc
import weakref
from pathlib import Path

import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


def _make_invocable():
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir import _types
    from cuda.coop.numba_mlir._compiler._artifacts import make_binary_tempfile

    algorithm = _types.Algorithm(
        struct_name="BlockLoad",
        method_name="Load",
        c_name="cuda_coop_numba_mlir_test_invocable_lifetime",
        includes=(),
        template_parameters=(),
        parameters=((_types.Pointer(types.uint8), _types.Value(types.int32)),),
    )
    artifact = make_binary_tempfile(b"test-ltoir", ".ltoir")
    return _types.Invocable(
        temp_files=(artifact,),
        owned_temp_files=(artifact,),
        temp_storage_bytes=8,
        temp_storage_alignment=8,
        algorithm=algorithm,
    )


def _registry_sizes():
    from numba_cuda_mlir.descriptor import mlir_target
    from numba_cuda_mlir.extending import typeof_impl, typing_registry

    return (
        len(typeof_impl.registry),
        len(typing_registry.functions),
        len(typing_registry.globals),
        len(typing_registry.attributes),
        len(mlir_target.typing_context._functions),
        len(mlir_target.typing_context._globals),
        len(mlir_target.typing_context._attributes),
    )


def test_invocable_typing_is_local_to_compiler_owners():
    from numba_cuda_mlir.descriptor import mlir_target
    from numba_cuda_mlir.extending import _NumbaCudaMlirOverloadFunctionTemplate

    registry_sizes = _registry_sizes()
    first = _make_invocable()
    second = _make_invocable()
    first_path = Path(first.files[0])
    second_path = Path(second.files[0])
    first_ref = weakref.ref(first)
    second_ref = weakref.ref(second)

    first_type = mlir_target.typing_context.resolve_value_type(first)
    second_type = mlir_target.typing_context.resolve_value_type(second)

    assert len(first_type.templates) == 2
    assert all(
        issubclass(template, _NumbaCudaMlirOverloadFunctionTemplate)
        for template in first_type.templates
    )
    assert first_type is mlir_target.typing_context.resolve_value_type(first)
    assert second_type is mlir_target.typing_context.resolve_value_type(second)
    assert _registry_sizes() == registry_sizes

    # A compiler-owned Function type retains its exact callable and therefore
    # the link input until compilation has finished with it.
    del first
    gc.collect()
    assert first_ref() is not None
    assert first_path.is_file()

    # No process-global typing registry owns the dynamic template. Releasing
    # the compiler's Function type makes the Invocable cycle collectible and
    # runs its artifact finalizer.
    del first_type
    gc.collect()
    assert first_ref() is None
    assert not first_path.exists()

    del second, second_type
    gc.collect()
    assert second_ref() is None
    assert not second_path.exists()
    assert _registry_sizes() == registry_sizes
