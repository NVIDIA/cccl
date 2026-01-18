# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TYPE_CHECKING, Callable, Literal, Union

import numba

from .._common import (
    make_binary_tempfile,
    normalize_dim_param,
    normalize_dtype_param,
)
from .._types import (
    Algorithm,
    BasePrimitive,
    Constant,
    Dependency,
    DependentArray,
    DependentPythonOperator,
    Invocable,
    Pointer,
    TemplateParameter,
    numba_type_to_cpp,
    numba_type_to_wrapper,
)

if TYPE_CHECKING:
    import numpy as np

    from ._rewrite import CoopNode


class merge_sort_keys(BasePrimitive):
    is_one_shot = True

    def __init__(
        self,
        dtype: Union[str, type, "np.dtype", "numba.types.Type"],
        threads_per_block: int,
        items_per_thread: int,
        compare_op: Callable,
        methods: Literal["construct", "assign"] = None,
        unique_id: int = None,
        temp_storage=None,
        node: "CoopNode" = None,
    ):
        """Performs a block-wide merge sort over a :ref:`blocked arrangement <flexible-data-arrangement>` of keys.

        Example:
            The code snippet below illustrates a sort of 512 integer keys that
            are partitioned in a :ref:`blocked arrangement <flexible-data-arrangement>` across 128 threads
            where each thread owns 4 consecutive keys. We start by importing necessary modules:

            .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_merge_sort_api.py
                :language: python
                :dedent:
                :start-after: example-begin imports
                :end-before: example-end imports

            Below is the code snippet that demonstrates the usage of the ``merge_sort_keys`` API:

            .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_merge_sort_api.py
                :language: python
                :dedent:
                :start-after: example-begin merge-sort
                :end-before: example-end merge-sort

            Suppose the set of input ``thread_keys`` across the block of threads is
            ``{ [0, 1, 2, 3], [4, 5, 6, 7], ..., [508, 509, 510, 511] }``.
            The corresponding output ``thread_keys`` in those threads will be
            ``{ [511, 510, 509, 508], [507, 506, 505, 504], ..., [3, 2, 1, 0] }``.

        Args:
            dtype: Numba data type of the keys to be sorted

            threads_per_block: The number of threads in a block, either an integer
                or a tuple of 2 or 3 integers

            items_per_thread: The number of items each thread owns

            compare_op: Comparison function object which returns true if the first
                argument is ordered before the second one

            Returns:
                A callable object that can be invoked from a CUDA kernel in single-phase,
                or a two-phase invocable via :func:`BlockMergeSort`.
        """
        if compare_op is None:
            raise ValueError("compare_op must be provided")
        if items_per_thread < 1:
            raise ValueError("items_per_thread must be >= 1")

        self.node = node
        self.temp_storage = temp_storage
        self.dim = normalize_dim_param(threads_per_block)
        self.dtype = normalize_dtype_param(dtype)
        self.items_per_thread = items_per_thread
        self.compare_op = compare_op
        self.unique_id = unique_id

        template_parameters = [
            TemplateParameter("KeyT"),
            TemplateParameter("BLOCK_DIM_X"),
            TemplateParameter("ITEMS_PER_THREAD"),
            TemplateParameter("ValueT"),
            TemplateParameter("BLOCK_DIM_Y"),
            TemplateParameter("BLOCK_DIM_Z"),
        ]

        parameters = [
            [
                DependentArray(
                    Dependency("KeyT"),
                    Dependency("ITEMS_PER_THREAD"),
                    name="keys",
                ),
                DependentPythonOperator(
                    Constant(numba.int8),
                    [Dependency("KeyT"), Dependency("KeyT")],
                    Dependency("Op"),
                    name="compare_op",
                ),
            ]
        ]

        type_definitions = None
        if methods is not None or numba_type_to_cpp(self.dtype) == "storage_t":
            type_definitions = [numba_type_to_wrapper(self.dtype, methods=methods)]

        self.algorithm = Algorithm(
            "BlockMergeSort",
            "Sort",
            "block_merge_sort",
            ["cub/block/block_merge_sort.cuh"],
            template_parameters,
            parameters,
            self,
            type_definitions=type_definitions,
            unique_id=unique_id,
        )

        specialization_kwds = {
            "KeyT": self.dtype,
            "BLOCK_DIM_X": self.dim[0],
            "ITEMS_PER_THREAD": self.items_per_thread,
            "ValueT": "::cub::NullType",
            "BLOCK_DIM_Y": self.dim[1],
            "BLOCK_DIM_Z": self.dim[2],
            "Op": compare_op,
        }

        self.specialization = self.algorithm.specialize(specialization_kwds)

    @classmethod
    def create(
        cls,
        dtype: Union[str, type, "np.dtype", "numba.types.Type"],
        threads_per_block: int,
        items_per_thread: int,
        compare_op: Callable,
        methods: Literal["construct", "assign"] = None,
    ):
        algo = cls(
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            compare_op=compare_op,
            methods=methods,
        )
        specialization = algo.specialization
        return Invocable(
            temp_files=[
                make_binary_tempfile(ltoir, ".ltoir")
                for ltoir in specialization.get_lto_ir()
            ],
            temp_storage_bytes=specialization.temp_storage_bytes,
            temp_storage_alignment=specialization.temp_storage_alignment,
            algorithm=specialization,
        )


def BlockMergeSort(
    dtype: Union[str, type, "np.dtype", "numba.types.Type"],
    threads_per_block: int,
    items_per_thread: int,
    compare_op: Callable,
    methods: Literal["construct", "assign"] = None,
):
    """
    Create a two-phase block merge sort invocable (explicit temp storage).
    """
    dim = normalize_dim_param(threads_per_block)
    dtype = normalize_dtype_param(dtype)

    template = Algorithm(
        "BlockMergeSort",
        "Sort",
        "block_merge_sort",
        ["cub/block/block_merge_sort.cuh"],
        [
            TemplateParameter("KeyT"),
            TemplateParameter("BLOCK_DIM_X"),
            TemplateParameter("ITEMS_PER_THREAD"),
            TemplateParameter("ValueT"),
            TemplateParameter("BLOCK_DIM_Y"),
            TemplateParameter("BLOCK_DIM_Z"),
        ],
        [
            [
                Pointer(numba.uint8),
                DependentArray(Dependency("KeyT"), Dependency("ITEMS_PER_THREAD")),
                DependentPythonOperator(
                    Constant(numba.int8),
                    [Dependency("KeyT"), Dependency("KeyT")],
                    Dependency("Op"),
                ),
            ]
        ],
        type_definitions=[numba_type_to_wrapper(dtype, methods=methods)],
    )
    specialization = template.specialize(
        {
            "KeyT": dtype,
            "BLOCK_DIM_X": dim[0],
            "ITEMS_PER_THREAD": items_per_thread,
            "ValueT": "::cub::NullType",
            "BLOCK_DIM_Y": dim[1],
            "BLOCK_DIM_Z": dim[2],
            "Op": compare_op,
        }
    )
    return Invocable(
        temp_files=[
            make_binary_tempfile(ltoir, ".ltoir")
            for ltoir in specialization.get_lto_ir()
        ],
        temp_storage_bytes=specialization.temp_storage_bytes,
        temp_storage_alignment=specialization.temp_storage_alignment,
        algorithm=specialization,
    )
