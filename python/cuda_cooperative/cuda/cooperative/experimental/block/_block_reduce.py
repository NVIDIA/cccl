# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from cuda.cooperative.experimental._types import *
from cuda.cooperative.experimental._common import make_binary_tempfile


def reduce(dtype, threads_in_block, binary_op, methods=None):
    """Computes a block-wide reduction for thread\ :sub:`0` using the specified binary reduction functor.
    Each thread contributes one input element.

    Warning:
        The return value is undefined in threads other than thread\ :sub:`0`.

    Example:
        The code snippet below illustrates a max reduction of 128 integer items that
        are partitioned across 128 threads.

        .. literalinclude:: ../../python/cuda_cooperative/tests/test_block_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin imports
            :end-before: example-end imports

        Below is the code snippet that demonstrates the usage of the ``reduce`` API:

        .. literalinclude:: ../../python/cuda_cooperative/tests/test_block_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin reduce
            :end-before: example-end reduce

        Suppose the set of inputs across the block of threads is
        ``{ 0, 1, 2, 3, ..., 127 }``.
        The corresponding output in the threads thread\ :sub:`0` will be ``{ 127 }``.

    Args:
        dtype: Data type being reduced
        threads_in_block: The number of threads in a block
        binary_op: Binary reduction function

    Returns:
        A callable object that can be linked to and invoked from a CUDA kernel
    """
    template = Algorithm('BlockReduce',
                         'Reduce',
                         'block_reduce',
                         ['cub/block/block_reduce.cuh'],
                         [TemplateParameter('T'),
                          TemplateParameter('BLOCK_DIM_X')],
                         [[Pointer(numba.uint8),
                           DependentReference(Dependency('T')),
                           DependentOperator(Dependency('T'), [Dependency(
                               'T'), Dependency('T')], Dependency('Op')),
                           DependentReference(Dependency('T'), True)]],
                         type_definitions=[numba_type_to_wrapper(dtype, methods=methods)])
    specialization = template.specialize({'T': dtype,
                                          'BLOCK_DIM_X': threads_in_block,
                                          'Op': binary_op})

    return Invocable(temp_files=[make_binary_tempfile(ltoir, '.ltoir') for ltoir in specialization.get_lto_ir()],
                     temp_storage_bytes=specialization.get_temp_storage_bytes(),
                     algorithm=specialization)


def sum(dtype, threads_in_block):
    """Computes a block-wide reduction for thread\ :sub:`0` using addition (+) as the reduction operator.
    Each thread contributes one input element.

    Warning:
        The return value is undefined in threads other than thread\ :sub:`0`.

    Example:
        The code snippet below illustrates a reduction of 128 integer items that
        are partitioned across 128 threads.

        .. literalinclude:: ../../python/cuda_cooperative/tests/test_block_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin imports
            :end-before: example-end imports

        Below is the code snippet that demonstrates the usage of the ``reduce`` API:

        .. literalinclude:: ../../python/cuda_cooperative/tests/test_block_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin sum
            :end-before: example-end sum

        Suppose the set of inputs across the block of threads is
        ``{ 1, 1, 1, 1, ..., 1 }``.
        The corresponding output in the threads thread\ :sub:`0` will be ``{ 128 }``.

    Args:
        dtype: Data type being reduced
        threads_in_block: The number of threads in a block

    Returns:
        A callable object that can be linked to and invoked from a CUDA kernel
    """
    template = Algorithm('BlockReduce',
                         'Sum',
                         'block_reduce',
                         ['cub/block/block_reduce.cuh'],
                         [TemplateParameter('T'),
                          TemplateParameter('BLOCK_DIM_X')],
                         [[Pointer(numba.uint8), DependentReference(Dependency('T')), DependentReference(Dependency('T'), True)],
                          [Pointer(numba.uint8), DependentReference(Dependency('T')), Value(numba.int32), DependentReference(Dependency('T'), True)]])
    specialization = template.specialize({'T': dtype,
                                          'BLOCK_DIM_X': threads_in_block})
    return Invocable(temp_files=[make_binary_tempfile(ltoir, '.ltoir') for ltoir in specialization.get_lto_ir()],
                     temp_storage_bytes=specialization.get_temp_storage_bytes(),
                     algorithm=specialization)
