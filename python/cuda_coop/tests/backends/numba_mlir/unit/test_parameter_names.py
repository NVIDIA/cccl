# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import replace

import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


def _materialize(specification, *, adapter=None):
    from cuda.coop._core import SynchronizationScope
    from cuda.coop.numba_mlir._compiler._operations import StorageABI
    from cuda.coop.numba_mlir._lowering._core import NumbaMlirCoreAdapter

    return (adapter or NumbaMlirCoreAdapter()).materialize(
        specification,
        storage_abi=StorageABI.NONE,
        execution_scope=SynchronizationScope.NONE,
        synchronization_scope=SynchronizationScope.NONE,
    )


def _source(algorithm):
    return algorithm._source_code(
        compile_identity=(90, True, "lto", (), "test-toolchain")
    )[0]


@pytest.mark.parametrize("operation", ("load", "store"))
def test_data_movement_retains_names_through_dependent_specialization(operation):
    from cuda.coop._core import INT32, ArgumentBinding
    from cuda.coop._core.block import make_block_load_spec, make_block_store_spec

    factory = make_block_load_spec if operation == "load" else make_block_store_spec
    specification = factory(
        dtype=INT32,
        block_dim=(32, 1, 1),
        items_per_thread=2,
        algorithm="direct",
        valid_items=ArgumentBinding.omitted(),
        include_pointer_offset=ArgumentBinding.runtime(),
    ).specialization
    algorithm = _materialize(specification)
    method = algorithm.parameters[0]
    assert [parameter.parameter_name for parameter in method] == (
        ["src", "dst", "offset"] if operation == "load" else ["dst", "src", "offset"]
    )
    source = _source(algorithm)
    abi_signature = next(
        line for line in source.splitlines() if "__abi(void *__ret," in line
    )
    assert "void *src" in abi_signature
    assert "void *dst" in abi_signature
    assert "::cuda::std::int64_t offset" in abi_signature
    memory_name = "src" if operation == "load" else "dst"
    assert f".{operation.title()}(({memory_name} + offset)," in source


def test_input_transform_retains_the_source_parameter_name():
    from cuda.coop._core import INT32, UINT8, Algorithm, Array
    from cuda.coop.numba_mlir._lowering._core import (
        NumbaMlirArrayInputTransform,
        NumbaMlirCoreAdapter,
    )

    specification = Algorithm(
        struct_name="Provider",
        method_name="Run",
        c_name="transformed_provider",
        includes=(),
        template_parameters=(),
        parameters=((Array(INT32, 2, name="flags"),),),
    ).specialize({})
    algorithm = _materialize(
        specification,
        adapter=NumbaMlirCoreAdapter(
            input_transforms={
                "flags": NumbaMlirArrayInputTransform(UINT8, "{value} != 0")
            }
        ),
    )

    assert algorithm.parameters[0][0].parameter_name == "flags"
    source = _source(algorithm)
    assert "::cuda::std::uint8_t *flags" in source
    assert "transformed_flags[0] = flags[0] != 0;" in source
    assert ".Run(transformed_flags);" in source
    assert "void *flags" in source


def test_parameter_renaming_preserves_provider_identity():
    from cuda.coop._core import INT32, Algorithm, Reference, Value
    from cuda.coop.numba_mlir._types import algo_coalesce_key

    definition = Algorithm(
        struct_name="Provider",
        method_name="Run",
        c_name="named_provider",
        includes=(),
        template_parameters=(),
        parameters=((Value(INT32, name="value"), Reference(INT32, name="result")),),
    )
    renamed_definition = replace(
        definition,
        parameters=((Value(INT32, name="input"), Reference(INT32, name="output")),),
    )
    original = _materialize(definition.specialize({}))
    renamed = _materialize(renamed_definition.specialize({}))

    assert algo_coalesce_key(original) == algo_coalesce_key(renamed)
    assert original.mangled_name(original.parameters[0]) == renamed.mangled_name(
        renamed.parameters[0]
    )
    original_source = _source(original)
    renamed_source = _source(renamed)
    assert "__abi(void *__ret, ::cuda::std::int32_t value," in original_source
    assert "__abi(void *__ret, ::cuda::std::int32_t input," in renamed_source
    assert original._private_symbol_key == renamed._private_symbol_key
    assert original._private_symbol_digest == renamed._private_symbol_digest


def test_shared_value_abi_names_do_not_mutate_other_parameters():
    from numba_cuda_mlir import types

    from cuda.coop._core import INT32, Algorithm, Value
    from cuda.coop.numba_mlir import _types
    from cuda.coop.numba_mlir._lowering._core import NumbaMlirCoreAdapter

    shared_abi = _types.Value(types.int32)
    definition = Algorithm(
        struct_name="Provider",
        method_name="Run",
        c_name="shared_abi_provider",
        includes=(),
        template_parameters=(),
        parameters=((Value(INT32, name="first"), Value(INT32, name="second")),),
    )
    algorithm = _materialize(
        definition.specialize({}),
        adapter=NumbaMlirCoreAdapter(
            value_abis={"first": shared_abi, "second": shared_abi}
        ),
    )
    first, second = algorithm.parameters[0]
    assert first is not second
    assert first is not shared_abi
    assert second is not shared_abi
    assert first.parameter_name == "first"
    assert second.parameter_name == "second"
    assert getattr(shared_abi, "parameter_name", None) is None
