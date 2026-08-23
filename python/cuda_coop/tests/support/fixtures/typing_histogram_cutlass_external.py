# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict CUTLASS-installed typing fixture for Histogram compiler values."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING, Any

from typing_extensions import assert_type

if TYPE_CHECKING:
    from cutlass.base_dsl.typing import Float32, Int64
    from cutlass.cute import Tensor, TensorSSA

    import cuda.coop.cutlass as coop
    from cuda import coop as common_coop

    group = coop.this_block()
    assert_type(
        coop.histogram(group, Int64(1), bins=16),
        coop.ThreadData[int],
    )
    assert_type(
        coop.histogram(
            group,
            Int64(1),
            bins=16,
            counter_dtype=Int64,
        ),
        coop.ThreadData[Int64],
    )
    compiler_samples = coop.ThreadData.from_values(Int64(1))
    assert_type(compiler_samples.to_tensor_ssa(dtype=Int64), object)
    assert_type(compiler_samples.to_register_tensor(dtype=Int64), object)

    class PythonIntRegisterTensor:
        @property
        def shape(self) -> object:
            return (2,)

        @property
        def memspace(self) -> object:
            return object()

        def __getitem__(self, index: int, /) -> int:
            return index

    class PythonIntVector:
        @property
        def shape(self) -> object:
            return (2,)

        def __getitem__(self, index: int, /) -> int:
            return index

    class PythonIntProducer:
        def __cuda_coop_thread_data_load__(self) -> PythonIntVector:
            return PythonIntVector()

    # from_values retains the actual values. CUTLASS DSL dtype tokens cast at
    # runtime, but remain external to this stub and therefore fall back to Any.
    assert_type(
        coop.ThreadData.from_values(1, dtype=Int64),
        coop.ThreadData[int],
    )
    assert_type(
        coop.ThreadData.from_fn(2, lambda index: index, dtype=Int64),
        coop.ThreadData[Any],
    )
    assert_type(
        coop.ThreadData.from_register_tensor(
            PythonIntRegisterTensor(),
            dtype=Int64,
        ),
        coop.ThreadData[Any],
    )
    assert_type(
        coop.ThreadData.from_vector(PythonIntVector(), dtype=Int64),
        coop.ThreadData[Any],
    )
    assert_type(
        coop.ThreadData.from_payload(PythonIntVector(), dtype=Int64),
        coop.ThreadData[Any],
    )
    assert_type(
        coop.ThreadData.load(PythonIntProducer(), dtype=Int64),
        coop.ThreadData[Any],
    )
    assert_type(
        common_coop.histogram(
            group,
            compiler_samples,
            bins=16,
            counter_dtype=Int64,
        ),
        common_coop.ThreadDataLike[Int64],
    )

    def accepts_register_tensors(tensor: Tensor, tensor_ssa: TensorSSA) -> None:
        assert_type(
            coop.histogram(group, tensor, bins=16),
            coop.ThreadData[int],
        )
        assert_type(
            coop.histogram(group, tensor_ssa, bins=16),
            coop.ThreadData[int],
        )

    coop.histogram(group, Float32(1), bins=16)  # pyright: ignore[reportArgumentType]
