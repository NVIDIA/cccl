# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TYPE_CHECKING, Tuple, Union

import numba

from .._common import (
    CUB_BLOCK_SCAN_ALGOS,
    CudaSharedMemConfig,
    dim3,
    make_binary_tempfile,
    normalize_dim_param,
    normalize_dtype_param,
)
from .._types import (
    Algorithm,
    BasePrimitive,
    Dependency,
    DependentArray,
    Invocable,
    Pointer,
    TemplateParameter,
    Value,
)

if TYPE_CHECKING:
    import numpy as np

    from ._rewrite import CoopNode


TEMPLATE_PARAMETERS = [
    TemplateParameter("KeyT"),
    TemplateParameter("BLOCK_DIM_X"),
    TemplateParameter("ITEMS_PER_THREAD"),
    TemplateParameter("ValueT"),
    TemplateParameter("RADIX_BITS"),
    TemplateParameter("MEMOIZE_OUTER_SCAN"),
    TemplateParameter("INNER_SCAN_ALGORITHM"),
    TemplateParameter("SMEM_CONFIG"),
    TemplateParameter("BLOCK_DIM_Y"),
    TemplateParameter("BLOCK_DIM_Z"),
]


METHOD_PARAMETERS_SINGLE_PHASE = [
    [
        DependentArray(Dependency("KeyT"), Dependency("ITEMS_PER_THREAD")),
    ],
    [
        DependentArray(Dependency("KeyT"), Dependency("ITEMS_PER_THREAD")),
        Value(numba.int32, name="begin_bit"),
        Value(numba.int32, name="end_bit"),
    ],
]

METHOD_PARAMETERS_TWO_PHASE = [
    [
        Pointer(numba.uint8),
        DependentArray(Dependency("KeyT"), Dependency("ITEMS_PER_THREAD")),
    ],
    [
        Pointer(numba.uint8),
        DependentArray(Dependency("KeyT"), Dependency("ITEMS_PER_THREAD")),
        Value(numba.int32, name="begin_bit"),
        Value(numba.int32, name="end_bit"),
    ],
]


# N.B. In order to support multi-dimensional block dimensions, we have to
#      defaults for all the template parameters preceding the final Y and
#      Z dimensions.  This will be improved in the future, allowing users
#      to provide overrides for the default values.

TEMPLATE_PARAMETER_DEFAULTS = {
    "ValueT": "::cub::NullType",  # Indicates keys-only sort
    "RADIX_BITS": 4,
    "MEMOIZE_OUTER_SCAN": "true",
    "INNER_SCAN_ALGORITHM": CUB_BLOCK_SCAN_ALGOS["warp_scans"],
    "SMEM_CONFIG": str(CudaSharedMemConfig.BankSizeFourByte),
}


def _get_template_parameter_specializations(
    dtype: numba.types.Type, dim: dim3, items_per_thread: int
) -> dict:
    """
    Returns a dictionary of template parameter specializations for the block
    radix sort algorithm.

    Args:
        dtype: Supplies the Numba data type.

        dim: Supplies the block dimensions.

        items_per_thread: Supplies the number of items each thread owns.

    Returns:
        A dictionary of template parameter specializations.
    """
    specialization = {
        "KeyT": dtype,
        "BLOCK_DIM_X": dim[0],
        "ITEMS_PER_THREAD": items_per_thread,
        "BLOCK_DIM_Y": dim[1],
        "BLOCK_DIM_Z": dim[2],
    }

    specialization.update(TEMPLATE_PARAMETER_DEFAULTS)

    return specialization


class _RadixSortBase(BasePrimitive):
    is_one_shot = True

    def __init__(
        self,
        dtype: Union[str, type, "np.dtype", "numba.types.Type"],
        threads_per_block: Union[int, Tuple[int, int], Tuple[int, int, int], dim3],
        items_per_thread: int,
        descending: bool,
        begin_bit: int = None,
        end_bit: int = None,
        unique_id: int = None,
        temp_storage=None,
        node: "CoopNode" = None,
    ) -> None:
        if items_per_thread < 1:
            raise ValueError("items_per_thread must be >= 1")
        if (begin_bit is None) != (end_bit is None):
            raise ValueError("begin_bit and end_bit must be provided together")

        self.node = node
        self.temp_storage = temp_storage
        self.dim = normalize_dim_param(threads_per_block)
        self.dtype = normalize_dtype_param(dtype)
        self.items_per_thread = items_per_thread
        self.descending = descending
        self.begin_bit = begin_bit
        self.end_bit = end_bit
        self.unique_id = unique_id

        method_name = "SortDescending" if descending else "Sort"
        parameters = (
            [METHOD_PARAMETERS_SINGLE_PHASE[1]]
            if begin_bit is not None
            else [METHOD_PARAMETERS_SINGLE_PHASE[0]]
        )

        self.algorithm = Algorithm(
            "BlockRadixSort",
            method_name,
            "block_radix_sort",
            ["cub/block/block_radix_sort.cuh"],
            TEMPLATE_PARAMETERS,
            parameters,
            self,
            unique_id=unique_id,
        )
        self.specialization = self.algorithm.specialize(
            _get_template_parameter_specializations(
                self.dtype, self.dim, self.items_per_thread
            )
        )

    @classmethod
    def create(
        cls,
        dtype: Union[str, type, "np.dtype", "numba.types.Type"],
        threads_per_block: Union[int, Tuple[int, int], Tuple[int, int, int], dim3],
        items_per_thread: int,
        descending: bool,
    ) -> Invocable:
        dim = normalize_dim_param(threads_per_block)
        dtype = normalize_dtype_param(dtype)

        method_name = "SortDescending" if descending else "Sort"
        template = Algorithm(
            "BlockRadixSort",
            method_name,
            "block_radix_sort",
            ["cub/block/block_radix_sort.cuh"],
            TEMPLATE_PARAMETERS,
            METHOD_PARAMETERS_TWO_PHASE,
        )
        specialization = template.specialize(
            _get_template_parameter_specializations(dtype, dim, items_per_thread)
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


class radix_sort_keys(_RadixSortBase):
    """Performs an ascending block-wide radix sort over a :ref:`blocked arrangement <flexible-data-arrangement>` of keys.

    Example:
        The code snippet below illustrates a sort of 512 integer keys that
        are partitioned in a :ref:`blocked arrangement <flexible-data-arrangement>` across 128 threads
        where each thread owns 4 consecutive keys. We start by importing necessary modules:

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_radix_sort_api.py
            :language: python
            :dedent:
            :start-after: example-begin imports
            :end-before: example-end imports

        Below is the code snippet that demonstrates the usage of the ``radix_sort_keys`` API:

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_radix_sort_api.py
            :language: python
            :dedent:
            :start-after: example-begin radix-sort
            :end-before: example-end radix-sort

        Suppose the set of input ``thread_keys`` across the block of threads is
        ``{ [511, 510, 509, 508], [507, 506, 505, 504], ..., [3, 2, 1, 0] }``.
        The corresponding output ``thread_keys`` in those threads will be
        ``{ [0, 1, 2, 3], [4, 5, 6, 7], ..., [508, 509, 510, 511] }``.

    Args:
        dtype: Data type of the keys to be sorted

        threads_per_block: The number of threads in a block, either an integer
            or a tuple of 2 or 3 integers

        items_per_thread: The number of items each thread owns

    Returns:
        A callable object that can be linked to and invoked from a CUDA kernel
    """

    def __init__(
        self,
        dtype,
        threads_per_block,
        items_per_thread,
        begin_bit: int = None,
        end_bit: int = None,
        unique_id: int = None,
        temp_storage=None,
        node: "CoopNode" = None,
    ):
        super().__init__(
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            descending=False,
            begin_bit=begin_bit,
            end_bit=end_bit,
            unique_id=unique_id,
            temp_storage=temp_storage,
            node=node,
        )


class radix_sort_keys_descending(_RadixSortBase):
    """Performs an descending block-wide radix sort over a :ref:`blocked arrangement <flexible-data-arrangement>` of keys.

    Example:
        The code snippet below illustrates a sort of 512 integer keys that
        are partitioned in a :ref:`blocked arrangement <flexible-data-arrangement>` across 128 threads
        where each thread owns 4 consecutive keys. We start by importing necessary modules:

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_radix_sort_api.py
            :language: python
            :dedent:
            :start-after: example-begin imports
            :end-before: example-end imports

        Below is the code snippet that demonstrates the usage of the ``radix_sort_keys`` API:

        .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_radix_sort_api.py
            :language: python
            :dedent:
            :start-after: example-begin radix-sort-descending
            :end-before: example-end radix-sort-descending

        Suppose the set of input ``thread_keys`` across the block of threads is
        ``{ [0, 1, 2, 3], [4, 5, 6, 7], ..., [508, 509, 510, 511] }``.
        The corresponding output ``thread_keys`` in those threads will be
        ``{ [511, 510, 509, 508], [507, 506, 505, 504], ..., [3, 2, 1, 0] }``.

    Args:
        dtype: Data type of the keys to be sorted

        threads_per_block: The number of threads in a block, either an integer
            or a tuple of 2 or 3 integers

        items_per_thread: The number of items each thread owns

    Returns:
        A callable object that can be linked to and invoked from a CUDA kernel
    """

    def __init__(
        self,
        dtype,
        threads_per_block,
        items_per_thread,
        begin_bit: int = None,
        end_bit: int = None,
        unique_id: int = None,
        temp_storage=None,
        node: "CoopNode" = None,
    ):
        super().__init__(
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            descending=True,
            begin_bit=begin_bit,
            end_bit=end_bit,
            unique_id=unique_id,
            temp_storage=temp_storage,
            node=node,
        )


def BlockRadixSort(
    dtype: Union[str, type, "np.dtype", "numba.types.Type"],
    threads_per_block: Union[int, Tuple[int, int], Tuple[int, int, int], dim3],
    items_per_thread: int,
):
    """
    Create a two-phase ascending block radix sort invocable.
    """
    return _RadixSortBase.create(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        descending=False,
    )


def BlockRadixSortDescending(
    dtype: Union[str, type, "np.dtype", "numba.types.Type"],
    threads_per_block: Union[int, Tuple[int, int], Tuple[int, int, int], dim3],
    items_per_thread: int,
):
    """
    Create a two-phase descending block radix sort invocable.
    """
    return _RadixSortBase.create(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        descending=True,
    )
