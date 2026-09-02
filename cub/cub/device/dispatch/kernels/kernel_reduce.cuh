// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_reduce.cuh>
#include <cub/detail/deferred_parameter.cuh>
#include <cub/device/dispatch/tuning/tuning_reduce.cuh>
#include <cub/grid/grid_even_share.cuh>
#include <cub/util_arch.cuh>

#include <cuda/__device/compute_capability.h>
#include <cuda/atomic>
#include <cuda/std/__type_traits/is_integral.h>

CUB_NAMESPACE_BEGIN

namespace detail::reduce
{
//! All cub::DeviceReduce::* algorithms are using the same implementation. Some of them, however,
//! should use the initial value only for empty problems. If this struct is used as initial value with
//! one of the `DeviceReduce` algorithms, the `init` value wrapped by this struct will only be used
//! for empty problems; it will not be incorporated into the aggregate of non-empty problems.
template <class T>
struct empty_problem_init_t
{
  T init;

  _CCCL_HOST_DEVICE operator T() const
  {
    return init;
  }
};

struct no_init_t
{};

//! If this value is passed as initial value to a `DeviceReduce` algorithm, no initial value will be incorporated into
//! the total aggregate.
inline constexpr auto no_init = no_init_t{};

template <class OutputIteratorT, class InitValueT>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE void handle_empty_problem(OutputIteratorT&& d_out, InitValueT init)
{
  *d_out = init;
}

template <class OutputIteratorT, class InitValueT>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE void
handle_empty_problem(OutputIteratorT&& d_out, empty_problem_init_t<InitValueT> empty_problem_init)
{
  *d_out = empty_problem_init.init;
}

template <class OutputIteratorT>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE void handle_empty_problem(OutputIteratorT&&, no_init_t)
{}

/**
 * @brief Applies initial value to the block aggregate and stores the result to the output iterator.
 *
 * @param d_out Iterator to the output aggregate
 * @param reduction_op Binary reduction functor
 * @param init Initial value
 * @param block_aggregate Aggregate value computed by the block
 */
template <class OutputIteratorT, class ReductionOpT, class InitValueT, class AccumT>
_CCCL_HOST_DEVICE void
finalize_and_store_aggregate(OutputIteratorT d_out, ReductionOpT reduction_op, InitValueT init, AccumT block_aggregate)
{
  *d_out = reduction_op(init, block_aggregate);
}

/**
 * @brief Ignores initial value and stores the block aggregate to the output iterator.
 *
 * @param d_out Iterator to the output aggregate
 * @param block_aggregate Aggregate value computed by the block
 */
template <class OutputIteratorT, class ReductionOpT, class InitValueT, class AccumT>
_CCCL_HOST_DEVICE void finalize_and_store_aggregate(
  OutputIteratorT d_out, ReductionOpT, empty_problem_init_t<InitValueT>, AccumT block_aggregate)
{
  *d_out = block_aggregate;
}

template <class OutputIteratorT, class ReductionOpT, class AccumT>
_CCCL_HOST_DEVICE void
finalize_and_store_aggregate(OutputIteratorT d_out, ReductionOpT, no_init_t, AccumT block_aggregate)
{
  *d_out = block_aggregate;
}

/**
 * @brief Reduce region kernel entry point (multi-block). Computes privatized
 *        reductions, one per thread block.
 *
 * @tparam PolicySelector
 *   The tuning polices
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam KernelNumItemsT
 *   Type of integral problem size or a deferred problem-size descriptor
 *
 * @tparam ReductionOpT
 *   Binary reduction functor type having member
 *   `auto operator()(const T &a, const U &b)`
 *
 * @tparam InitValueT
 *   Initial value type
 *
 * @tparam AccumT
 *   Accumulator type
 *
 * @param[in] d_in
 *   Pointer to the input sequence of data items
 *
 * @param[out] d_out
 *   Pointer to the output aggregate
 *
 * @param[in] kernel_num_items
 *   Immediate problem size or a deferred problem-size descriptor
 *
 * @param[in] even_share
 *   Even-share descriptor for mapping an equal number of tiles onto each
 *   thread block
 *
 * @param[in] reduction_op
 *   Binary reduction functor
 */
template <typename PolicySelector,
          bool StableReductionOrder,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename KernelNumItemsT,
          typename ReductionOpT,
          typename AccumT,
          typename InitValueT,
          typename TransformOpT>
#if _CCCL_HAS_CONCEPTS()
  requires reduce_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
_CCCL_KERNEL_ATTRIBUTES
__launch_bounds__(int(current_policy<PolicySelector>().multi_tile.threads_per_block)) void DeviceReduceKernel(
  const InputIteratorT d_in,
  const OutputIteratorT d_out,
  const KernelNumItemsT kernel_num_items,
  GridEvenShare<OffsetT> even_share,
  ReductionOpT reduction_op,
  [[maybe_unused]] const InitValueT init,
  TransformOpT transform_op)
{
  static constexpr ReducePassPolicy policy = current_policy<PolicySelector>().multi_tile;
  const OffsetT num_items                  = CUB_NS_QUALIFIER::detail::parameter_from_device<OffsetT>(kernel_num_items);

  // Early return from surplus blocks for deferred num_items
  if constexpr (!::cuda::std::is_integral_v<KernelNumItemsT>)
  {
    constexpr int tile_size = policy.threads_per_block * policy.items_per_thread;
    even_share.DispatchInit(num_items, static_cast<int>(gridDim.x), tile_size);

    if constexpr (StableReductionOrder)
    {
      if (static_cast<int>(blockIdx.x) >= even_share.grid_size)
      {
        return;
      }
    }
    else
    {
      // Only block zero handles an empty atomic reduction. For non-empty problems, all surplus blocks return.
      if ((num_items == 0 && blockIdx.x != 0)
          || (num_items != 0 && static_cast<int>(blockIdx.x) >= even_share.grid_size))
      {
        return;
      }
    }
  }

  if constexpr (!StableReductionOrder)
  {
    static_assert(detail::is_cuda_std_plus_v<ReductionOpT>,
                  "Only plus is currently supported in nondeterministic reduce");

    // The atomic code path already finishes in this kernel, so check if we have an empty problem and handle it
    if (num_items == 0)
    {
      if (threadIdx.x == 0)
      {
        reduce::handle_empty_problem(d_out, init);
      }

      return;
    }
  }

  // TODO(bgruber): pass policy directly as template argument to AgentReduce in C++20
  using agent_policy_t = detail::agent_reduce_policy<
    0,
    0,
    AccumT,
    policy.vec_size,
    policy.reduce_algorithm,
    policy.load_modifier,
    NoScaling<policy.threads_per_block, policy.items_per_thread, AccumT>>;

  // Thread block type for reducing input tiles
  using AgentReduceT = AgentReduce<agent_policy_t, InputIteratorT, OffsetT, ReductionOpT, AccumT, TransformOpT>;

  static_assert(sizeof(typename AgentReduceT::TempStorage) <= max_smem_per_block,
                "cub::DeviceReduce ran out of CUDA shared memory, which we judged to be extremely unlikely. Please "
                "file an issue at: https://github.com/NVIDIA/cccl/issues");

  // Shared memory storage
  __shared__ typename AgentReduceT::TempStorage temp_storage;

  // Consume input tiles
  AccumT block_aggregate = AgentReduceT(temp_storage, d_in, reduction_op, transform_op).ConsumeTiles(even_share);

  // Output result, only thread 0 has valid value in block aggregate
  if (threadIdx.x == 0)
  {
    if constexpr (StableReductionOrder)
    {
      detail::uninitialized_copy_single(d_out + blockIdx.x, block_aggregate);
    }
    else
    {
      // TODO: replace this with other atomic operations when specified
      NV_IF_ELSE_TARGET(
        NV_PROVIDES_SM_60,
        ({
          ::cuda::atomic_ref<AccumT, ::cuda::thread_scope_device> atomic_target(d_out[0]);
          atomic_target.fetch_add(blockIdx.x == 0 ? reduction_op(init, block_aggregate) : block_aggregate,
                                  ::cuda::memory_order_relaxed);
        }),
        (atomicAdd(&d_out[0], blockIdx.x == 0 ? reduction_op(init, block_aggregate) : block_aggregate);));
    }
  }
}

/**
 * @brief Reduce a single tile kernel entry point (single-block). Can be used
 *        to aggregate privatized thread block reductions from a previous
 *        multi-block reduction pass.
 *
 * @tparam PolicySelector
 *   The tuning polices
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam OutputIteratorT
 *   Output iterator type for recording the reduced aggregate @iterator
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam ReductionOpT
 *   Binary reduction functor type having member
 *   `T operator()(const T &a, const U &b)`
 *
 * @tparam InitValueT
 *   Initial value type
 *
 * @tparam AccumT
 *   Accumulator type
 *
 * @param[in] d_in
 *   Pointer to the input sequence of data items
 *
 * @param[out] d_out
 *   Pointer to the output aggregate
 *
 * @param[in] num_items
 *   Total number of input data items
 *
 * @param[in] reduction_op
 *   Binary reduction functor
 *
 * @param[in] init
 *   The initial value of the reduction
 */
template <typename PolicySelector,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitValueT,
          typename AccumT,
          typename TransformOpT = ::cuda::std::identity>
#if _CCCL_HAS_CONCEPTS()
  requires reduce_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
_CCCL_KERNEL_ATTRIBUTES __launch_bounds__(
  int(current_policy<PolicySelector>().single_tile.threads_per_block),
  1) void DeviceReduceSingleTileKernel(const InputIteratorT d_in,
                                       OutputIteratorT d_out,
                                       const OffsetT num_items,
                                       ReductionOpT reduction_op,
                                       const InitValueT init,
                                       TransformOpT transform_op)
{
  static constexpr ReducePassPolicy policy = current_policy<PolicySelector>().single_tile;
  // TODO(bgruber): pass policy directly as template argument to AgentReduce in C++20
  using agent_policy_t = detail::agent_reduce_policy<
    /* NominalThreadsPerBlock4B = */ 0,
    /* NominalItemsPerThread4B = */ 0,
    AccumT,
    policy.vec_size,
    policy.reduce_algorithm,
    policy.load_modifier,
    NoScaling<policy.threads_per_block, policy.items_per_thread, AccumT>>;

  // Thread block type for reducing input tiles
  using AgentReduceT = AgentReduce<agent_policy_t, InputIteratorT, OffsetT, ReductionOpT, AccumT, TransformOpT>;

  static_assert(sizeof(typename AgentReduceT::TempStorage) <= max_smem_per_block,
                "cub::DeviceReduce ran out of CUDA shared memory, which we judged to be extremely unlikely. Please "
                "file an issue at: https://github.com/NVIDIA/cccl/issues");

  // Shared memory storage
  __shared__ typename AgentReduceT::TempStorage temp_storage;

  // TODO(NaderAlAwar): This code is intentionally duplicated in DeviceReduceDeferredSingleTileKernel because
  // extracting it into a device function changes the SASS of this kernel. Changes here must also be applied to the
  // copy below.

  // Check if empty problem
  if (num_items == 0)
  {
    if (threadIdx.x == 0)
    {
      detail::reduce::handle_empty_problem(d_out, init);
    }

    return;
  }

  // Consume input tiles
  AccumT block_aggregate =
    AgentReduceT(temp_storage, d_in, reduction_op, transform_op).ConsumeRange(OffsetT(0), num_items);

  // Output result
  if (threadIdx.x == 0)
  {
    detail::reduce::finalize_and_store_aggregate(d_out, reduction_op, init, block_aggregate);
  }
}

//! Single-tile entry point that derives the number of first-pass partials from a deferred problem size.
template <typename PolicySelector,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename KernelNumItemsT,
          typename ReductionOpT,
          typename InitValueT,
          typename AccumT,
          typename TransformOpT = ::cuda::std::identity>
#if _CCCL_HAS_CONCEPTS()
  requires reduce_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
_CCCL_KERNEL_ATTRIBUTES __launch_bounds__(
  int{current_policy<PolicySelector>().single_tile.threads_per_block},
  1) void DeviceReduceDeferredSingleTileKernel(const InputIteratorT d_in,
                                               const OutputIteratorT d_out,
                                               const KernelNumItemsT kernel_num_items,
                                               const int first_pass_grid_size,
                                               ReductionOpT reduction_op,
                                               const InitValueT init,
                                               TransformOpT transform_op)
{
  const OffsetT actual_num_items = CUB_NS_QUALIFIER::detail::parameter_from_device<OffsetT>(kernel_num_items);
  static constexpr ReducePassPolicy first_pass_policy = current_policy<PolicySelector>().multi_tile;
  constexpr int first_pass_tile_size = first_pass_policy.threads_per_block * first_pass_policy.items_per_thread;
  GridEvenShare<OffsetT> even_share;
  even_share.DispatchInit(actual_num_items, first_pass_grid_size, first_pass_tile_size);
  const int num_items = even_share.grid_size;

  static constexpr ReducePassPolicy policy = current_policy<PolicySelector>().single_tile;

  // TODO(bgruber): pass policy directly as template argument to AgentReduce in C++20
  using agent_policy_t = detail::agent_reduce_policy<
    /* NominalThreadsPerBlock4B = */ 0,
    /* NominalItemsPerThread4B = */ 0,
    AccumT,
    policy.vec_size,
    policy.reduce_algorithm,
    policy.load_modifier,
    NoScaling<policy.threads_per_block, policy.items_per_thread, AccumT>>;

  // Thread block type for reducing input tiles
  using AgentReduceT = AgentReduce<agent_policy_t, InputIteratorT, int, ReductionOpT, AccumT, TransformOpT>;

  static_assert(sizeof(typename AgentReduceT::TempStorage) <= max_smem_per_block,
                "cub::DeviceReduce ran out of CUDA shared memory, which we judged to be extremely unlikely. Please "
                "file an issue at: https://github.com/NVIDIA/cccl/issues");

  // Shared memory storage
  __shared__ typename AgentReduceT::TempStorage temp_storage;

  // TODO(NaderAlAwar): This code is intentionally duplicated in DeviceReduceSingleTileKernel because extracting it
  // into a device function changes the SASS of that kernel. Changes here must also be applied to the copy above.

  // Check if empty problem
  if (num_items == 0)
  {
    if (threadIdx.x == 0)
    {
      detail::reduce::handle_empty_problem(d_out, init);
    }

    return;
  }

  // Consume input tiles
  AccumT block_aggregate = AgentReduceT(temp_storage, d_in, reduction_op, transform_op).ConsumeRange(int(0), num_items);

  // Output result
  if (threadIdx.x == 0)
  {
    detail::reduce::finalize_and_store_aggregate(d_out, reduction_op, init, block_aggregate);
  }
}
} // namespace detail::reduce

CUB_NAMESPACE_END
