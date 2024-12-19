# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from cuda.cooperative.experimental._types import *
from cuda.cooperative.experimental._common import make_binary_tempfile, normalize_dim_param


def striped_to_blocked(dtype, threads_in_block, items_per_thread, warp_time_slicing=False):
    template = Algorithm('BlockExchange',
                         'StripedToBlocked',
                         'block_exchange',
                         ['cub/block/block_exchange.cuh'],
                         [TemplateParameter('T'),
                          TemplateParameter('BLOCK_DIM_X'),
                          TemplateParameter('ITEMS_PER_THREAD'),
                          TemplateParameter('WARP_TIME_SLICING'),
                          TemplateParameter('BLOCK_DIM_Y'),
                          TemplateParameter('BLOCK_DIM_Z')],
                         [[Pointer(numba.uint8),
                           DependentArray(Dependency(
                               'T'), Dependency('ITEMS_PER_THREAD'))],
                          [Pointer(numba.uint8),
                           DependentArray(Dependency(
                               'T'), Dependency('ITEMS_PER_THREAD')),
                           DependentArray(Dependency(
                               'T'), Dependency('ITEMS_PER_THREAD'))]])
    dim = normalize_dim_param(threads_in_block)
    specialization = template.specialize({'T': dtype,
                                          'BLOCK_DIM_X': dim[0],
                                          'ITEMS_PER_THREAD': items_per_thread,
                                          'WARP_TIME_SLICING': int(warp_time_slicing),
                                          'BLOCK_DIM_Y': dim[1],
                                          'BLOCK_DIM_Z': dim[2]})
    return Invocable(temp_files=[make_binary_tempfile(ltoir, '.ltoir') for ltoir in specialization.get_lto_ir()],
                     temp_storage_bytes=specialization.get_temp_storage_bytes(),
                     algorithm=specialization)
