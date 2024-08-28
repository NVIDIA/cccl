# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba
from cuda.cooperative.experimental._types import *
from cuda.cooperative.experimental._common import make_binary_tempfile


def radix_sort_keys(dtype, threads_in_block, items_per_thread):
    """Performs an ascending block-wide radix sort over a :ref:`blocked arrangement <flexible-data-arrangement>` of keys.

    Example:
        The code snippet below illustrates a sort of 512 integer keys that
        are partitioned in a :ref:`blocked arrangement <flexible-data-arrangement>` across 128 threads
        where each thread owns 4 consecutive keys. We start by importing necessary modules:

        .. literalinclude:: ../../python/cuda_cooperative/tests/test_block_radix_sort_api.py
            :language: python
            :dedent:
            :start-after: example-begin imports
            :end-before: example-end imports

        Below is the code snippet that demonstrates the usage of the ``radix_sort_keys`` API:

        .. literalinclude:: ../../python/cuda_cooperative/tests/test_block_radix_sort_api.py
            :language: python
            :dedent:
            :start-after: example-begin radix-sort
            :end-before: example-end radix-sort

        Suppose the set of input ``thread_keys`` across the block of threads is
        ``{ [511, 510, 509, 508], [507, 506, 505, 504], ..., [3, 2, 1, 0] }``.
        The corresponding output ``thread_keys`` in those threads will be
        ``{ [0, 1, 2, 3], [4, 5, 6, 7], ..., [508, 509, 510, 511] }``.

    Args:
        dtype: Numba data type of the keys to be sorted
        threads_in_block: The number of threads in a block
        items_per_thread: The number of items each thread owns

    Returns:
        A callable object that can be linked to and invoked from a CUDA kernel
    """
    template = Algorithm('BlockRadixSort',
                         'Sort',
                         'block_radix_sort',
                         ['cub/block/block_radix_sort.cuh'],
                         [TemplateParameter('KeyT'),
                          TemplateParameter('BLOCK_DIM_X'),
                          TemplateParameter('ITEMS_PER_THREAD')],
                         [[Pointer(numba.uint8), DependentArray(Dependency('KeyT'), Dependency('ITEMS_PER_THREAD'))],
                          [Pointer(numba.uint8), DependentArray(Dependency('KeyT'), Dependency(
                              'ITEMS_PER_THREAD')), Value(numba.int32), Value(numba.int32)]
                          ])
    specialization = template.specialize({'KeyT': dtype,
                                          'BLOCK_DIM_X': threads_in_block,
                                          'ITEMS_PER_THREAD': items_per_thread})
    return Invocable(temp_files=[make_binary_tempfile(ltoir, '.ltoir') for ltoir in specialization.get_lto_ir()],
                     temp_storage_bytes=specialization.get_temp_storage_bytes(),
                     algorithm=specialization)


def radix_sort_keys_descending(dtype, threads_in_block, items_per_thread):
    """Performs an descending block-wide radix sort over a :ref:`blocked arrangement <flexible-data-arrangement>` of keys.

    Example:
        The code snippet below illustrates a sort of 512 integer keys that
        are partitioned in a :ref:`blocked arrangement <flexible-data-arrangement>` across 128 threads
        where each thread owns 4 consecutive keys. We start by importing necessary modules:

        .. literalinclude:: ../../python/cuda_cooperative/tests/test_block_radix_sort_api.py
            :language: python
            :dedent:
            :start-after: example-begin imports
            :end-before: example-end imports

        Below is the code snippet that demonstrates the usage of the ``radix_sort_keys`` API:

        .. literalinclude:: ../../python/cuda_cooperative/tests/test_block_radix_sort_api.py
            :language: python
            :dedent:
            :start-after: example-begin radix-sort-descending
            :end-before: example-end radix-sort-descending

        Suppose the set of input ``thread_keys`` across the block of threads is
        ``{ [0, 1, 2, 3], [4, 5, 6, 7], ..., [508, 509, 510, 511] }``.
        The corresponding output ``thread_keys`` in those threads will be
        ``{ [511, 510, 509, 508], [507, 506, 505, 504], ..., [3, 2, 1, 0] }``.

    Args:
        dtype: Numba data type of the keys to be sorted
        threads_in_block: The number of threads in a block
        items_per_thread: The number of items each thread owns

    Returns:
        A callable object that can be linked to and invoked from a CUDA kernel
    """
    template = Algorithm('BlockRadixSort',
                         'SortDescending',
                         'block_radix_sort',
                         ['cub/block/block_radix_sort.cuh'],
                         [TemplateParameter('KeyT'),
                          TemplateParameter('BLOCK_DIM_X'),
                          TemplateParameter('ITEMS_PER_THREAD')],
                         [[Pointer(numba.uint8), DependentArray(Dependency('KeyT'), Dependency('ITEMS_PER_THREAD'))],
                          [Pointer(numba.uint8), DependentArray(Dependency('KeyT'), Dependency(
                              'ITEMS_PER_THREAD')), Value(numba.int32), Value(numba.int32)]
                          ])
    specialization = template.specialize({'KeyT': dtype,
                                          'BLOCK_DIM_X': threads_in_block,
                                          'ITEMS_PER_THREAD': items_per_thread})

    return Invocable(temp_files=[make_binary_tempfile(ltoir, '.ltoir') for ltoir in specialization.get_lto_ir()],
                     temp_storage_bytes=specialization.get_temp_storage_bytes(),
                     algorithm=specialization)
