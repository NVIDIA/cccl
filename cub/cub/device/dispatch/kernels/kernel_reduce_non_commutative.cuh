// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_reduce_non_commutative.cuh>
#include <cub/detail/uninitialized_copy.cuh>
#include <cub/device/dispatch/kernels/kernel_reduce.cuh>
#include <cub/device/dispatch/tuning/tuning_reduce.cuh>
#include <cub/util_arch.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::reduce_non_commutative
{
//! First pass: each block reduces the shares of its warps and writes one aggregate per block, in block order.
template <typename PolicySelector,
          typename InputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename AccumT,
          typename TransformOpT>
#if _CCCL_HAS_CONCEPTS()
  requires reduce::reduce_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
_CCCL_KERNEL_ATTRIBUTES __launch_bounds__(int(
  current_policy<PolicySelector>()
    .multi_tile.threads_per_block)) void device_reduce_non_commutative_kernel(const InputIteratorT d_in,
                                                                              AccumT* d_block_aggregates,
                                                                              const even_share_t<OffsetT> even_share,
                                                                              ReductionOpT reduction_op,
                                                                              TransformOpT transform_op)
{
  static constexpr ReducePassPolicy policy = current_policy<PolicySelector>().multi_tile;
  using agent =
    agent_t<policy.threads_per_block,
            policy.items_per_thread,
            policy.load_modifier,
            InputIteratorT,
            OffsetT,
            ReductionOpT,
            AccumT,
            TransformOpT>;

  static_assert(
    sizeof(typename agent::TempStorage) <= max_smem_per_block,
    "cub::DeviceReduce::ReduceNonCommutative ran out of CUDA shared memory, which we judged to be extremely "
    "unlikely. Please file an issue at: https://github.com/NVIDIA/cccl/issues");

  __shared__ typename agent::TempStorage temp_storage;

  const AccumT block_aggregate = agent(temp_storage, d_in, reduction_op, transform_op)
                                   .consume_shares(even_share, static_cast<int>(blockIdx.x) * agent::warps_per_block);
  if (threadIdx.x == 0)
  {
    detail::uninitialized_copy_single(d_block_aggregates + blockIdx.x, block_aggregate);
  }
}

//! Reduces the whole input with one block and applies the initial value. Used for small problems, and as the second
//! pass over the block aggregates of the first.
template <typename PolicySelector,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitValueT,
          typename AccumT,
          typename TransformOpT>
#if _CCCL_HAS_CONCEPTS()
  requires reduce::reduce_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
_CCCL_KERNEL_ATTRIBUTES __launch_bounds__(
  int(current_policy<PolicySelector>().single_tile.threads_per_block),
  1) void device_reduce_non_commutative_single_tile_kernel(const InputIteratorT d_in,
                                                           OutputIteratorT d_out,
                                                           const OffsetT num_items,
                                                           ReductionOpT reduction_op,
                                                           const InitValueT init,
                                                           TransformOpT transform_op)
{
  static constexpr ReducePassPolicy policy = current_policy<PolicySelector>().single_tile;
  using agent =
    agent_t<policy.threads_per_block,
            policy.items_per_thread,
            policy.load_modifier,
            InputIteratorT,
            OffsetT,
            ReductionOpT,
            AccumT,
            TransformOpT>;

  static_assert(
    sizeof(typename agent::TempStorage) <= max_smem_per_block,
    "cub::DeviceReduce::ReduceNonCommutative ran out of CUDA shared memory, which we judged to be extremely "
    "unlikely. Please file an issue at: https://github.com/NVIDIA/cccl/issues");

  __shared__ typename agent::TempStorage temp_storage;

  if (num_items == 0)
  {
    if (threadIdx.x == 0)
    {
      reduce::handle_empty_problem(d_out, init);
    }
    return;
  }

  const AccumT block_aggregate = agent(temp_storage, d_in, reduction_op, transform_op).consume_range(num_items);
  if (threadIdx.x == 0)
  {
    reduce::finalize_and_store_aggregate(d_out, reduction_op, init, block_aggregate);
  }
}
} // namespace detail::reduce_non_commutative

CUB_NAMESPACE_END
