# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from cuda.cooperative.experimental._types import *
from cuda.cooperative.experimental._common import make_binary_tempfile


def exclusive_sum(dtype, threads_in_block, items_per_thread, prefix_op=None):
    """Computes an exclusive block-wide prefix sum using addition (+) as the scan operator.
    Each thread contributes an array of consecutive input elements.
    The value of 0 is applied as the initial value, and is assigned to first output element in *thread*\ :sub:`0`.

    Example:
        The code snippet below illustrates an exclusive prefix sum of 512 integer items that
        are partitioned in a :ref:`blocked arrangement <flexible-data-arrangement>` across 128 threads
        where each thread owns 4 consecutive items.

        .. literalinclude:: ../../python/cuda_cooperative/tests/test_block_scan_api.py
            :language: python
            :dedent:
            :start-after: example-begin imports
            :end-before: example-end imports

        Below is the code snippet that demonstrates the usage of the ``exclusive_sum`` API:

        .. literalinclude:: ../../python/cuda_cooperative/tests/test_block_scan_api.py
            :language: python
            :dedent:
            :start-after: example-begin exclusive-sum
            :end-before: example-end exclusive-sum

        Suppose the set of input ``thread_data`` across the block of threads is
        ``{ [1, 1, 1, 1], [1, 1, 1, 1], ..., [1, 1, 1, 1] }``.
        The corresponding output ``thread_data`` in those threads will be
        ``{ [0, 1, 2, 3], [4, 5, 6, 7], ..., [508, 509, 510, 511] }``.

    Args:
        dtype: Data type being scanned
        threads_in_block: The number of threads in a block
        items_per_thread: The number of items each thread owns

    Returns:
        A callable object that can be linked to and invoked from a CUDA kernel
    """
    template = Algorithm('BlockScan',
                         'ExclusiveSum',
                         'block_scan',
                         ['cub/block/block_scan.cuh'],
                         [TemplateParameter('T'),
                          TemplateParameter('BLOCK_DIM_X')],
                         [[Pointer(numba.uint8),
                           DependentArray(Dependency(
                               'T'), Dependency('ITEMS_PER_THREAD')),
                           DependentArray(Dependency(
                               'T'), Dependency('ITEMS_PER_THREAD')),
                           DependentOperator(Dependency('T'), [Dependency('T')], Dependency('PrefixOp'))],
                          [Pointer(numba.uint8),
                           DependentArray(Dependency(
                               'T'), Dependency('ITEMS_PER_THREAD')),
                           DependentArray(Dependency('T'), Dependency('ITEMS_PER_THREAD'))]])
    specialization = template.specialize({'T': dtype,
                                          'BLOCK_DIM_X': threads_in_block,
                                          'ITEMS_PER_THREAD': items_per_thread,
                                          'PrefixOp': prefix_op})
    return Invocable(temp_files=[make_binary_tempfile(ltoir, '.ltoir') for ltoir in specialization.get_lto_ir()],
                     temp_storage_bytes=specialization.get_temp_storage_bytes(),
                     algorithm=specialization)
