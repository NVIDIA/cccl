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

#include <cub/detail/deferred_parameter.cuh>
#include <cub/detail/rfa.cuh>
#include <cub/device/dispatch/kernels/kernel_reduce.cuh>
#include <cub/device/dispatch/tuning/tuning_reduce.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_type.cuh>

#include <thrust/type_traits/unwrap_contiguous_iterator.h>

#include <cuda/__device/compute_capability.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::reduce
{
//! Type a normalized problem size resolves to on device: an immediate problem size keeps its integral type, a
//! deferred source resolves to a signed integer wide enough for its element.
template <typename NormalizedNumItemsT, bool = ::cuda::std::is_integral_v<NormalizedNumItemsT>>
struct deterministic_num_items
{
  using type = NormalizedNumItemsT;
};

template <typename NormalizedNumItemsT>
struct deterministic_num_items<NormalizedNumItemsT, false>
{
  using type = ::cuda::std::conditional_t<sizeof(it_value_t<NormalizedNumItemsT>) == 4, int, ::cuda::std::int64_t>;
};

template <typename NormalizedNumItemsT>
using deterministic_num_items_t = typename deterministic_num_items<NormalizedNumItemsT>::type;

/**
 * @brief Deterministically Reduce region kernel entry point (multi-block). Computes privatized
 *        reductions, one per thread block in deterministic fashion
 *
 * @tparam PolicySelector
 *   Tuning policy selector
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam NormalizedNumItemsT
 *   Integral problem size or a deferred problem-size descriptor
 *
 * @tparam ReductionOpT
 *   Binary reduction functor type having member
 *   `auto operator()(const T &a, const U &b)`
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
 * @param[in] normalized_num_items
 *   Immediate problem size or a deferred problem-size descriptor
 *
 * @param[in] reduction_op
 *   Binary reduction functor
 */
template <typename PolicySelector,
          typename InputIteratorT,
          typename NormalizedNumItemsT,
          typename ReductionOpT,
          typename AccumT,
          typename TransformOpT>
_CCCL_KERNEL_ATTRIBUTES
__launch_bounds__(int(current_policy<PolicySelector>().multi_tile.threads_per_block)) void DeterministicDeviceReduceKernel(
  InputIteratorT d_in,
  AccumT* d_out,
  const NormalizedNumItemsT normalized_num_items,
  ReductionOpT reduction_op,
  TransformOpT transform_op,
  const int reduce_grid_size)
{
  constexpr ReducePassPolicy policy = current_policy<PolicySelector>().multi_tile;
  constexpr int items_per_thread    = policy.items_per_thread;
  constexpr int threads_per_block   = policy.threads_per_block;

  // A 64-bit deferred problem size is consumed in a single launch instead of host-side chunks, so the index
  // computations must be 64-bit as well.
  using num_items_t = deterministic_num_items_t<NormalizedNumItemsT>;
  using index_t     = ::cuda::std::make_unsigned_t<num_items_t>;

  const num_items_t num_items = CUB_NS_QUALIFIER::detail::resolve_parameter<num_items_t>(normalized_num_items);

  using block_reduce_t = BlockReduce<AccumT, threads_per_block, policy.reduce_algorithm>;

  // Shared memory storage
  __shared__ typename block_reduce_t::TempStorage temp_storage;

  using ftype              = typename AccumT::ftype;
  constexpr int bin_length = AccumT::max_index + AccumT::max_fold;
  const int tid            = threads_per_block * blockIdx.x + threadIdx.x;

  ftype* shared_bins = detail::rfa::get_shared_bin_array<ftype, bin_length>();

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int index = static_cast<int>(threadIdx.x); index < bin_length; index += threads_per_block)
  {
    shared_bins[index] = AccumT::initialize_bin(index);
  }

  __syncthreads();

  AccumT thread_aggregate{};
  int count = 0;

  int n_threads = reduce_grid_size * threads_per_block;

  _CCCL_PRAGMA_UNROLL_FULL()
  for (index_t i = tid; i < static_cast<index_t>(num_items); i += (n_threads * items_per_thread))
  {
    ftype items[items_per_thread] = {};
    for (int j = 0; j < items_per_thread; j++)
    {
      const index_t idx = i + j * n_threads;
      if (idx < static_cast<index_t>(num_items))
      {
        items[j] = transform_op(d_in[idx]);
      }
    }

    ftype abs_max_val = ::cuda::std::fabs(items[0]);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 1; j < items_per_thread; j++)
    {
      abs_max_val = ::cuda::std::fmax(::cuda::std::fabs(items[j]), abs_max_val);
    }

    thread_aggregate.set_max_val(abs_max_val);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < items_per_thread; j++)
    {
      thread_aggregate.unsafe_add(items[j]);
      count++;
      if (count >= thread_aggregate.endurance())
      {
        thread_aggregate.renorm();
        count = 0;
      }
    }
  }

  AccumT block_aggregate = block_reduce_t(temp_storage).Reduce(thread_aggregate, [](AccumT lhs, AccumT rhs) -> AccumT {
    AccumT rtn = lhs;
    rtn += rhs;
    return rtn;
  });

  // Output result
  if (threadIdx.x == 0)
  {
    detail::uninitialized_copy_single(d_out + blockIdx.x, block_aggregate);
  }
}

/**
 * @brief Deterministically Reduce a single tile kernel entry point (single-block). Can be used
 *        to aggregate privatized thread block reductions from a previous
 *        multi-block reduction pass.
 *
 * @tparam PolicySelector
 *   Tuning policy selector
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam OutputIteratorT
 *   Output iterator type for recording the reduced aggregate @iterator
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
          typename ReductionOpT,
          typename InitValueT,
          typename AccumT,
          typename TransformOpT = ::cuda::std::identity>
_CCCL_KERNEL_ATTRIBUTES __launch_bounds__(
  int(current_policy<PolicySelector>().single_tile.threads_per_block),
  1) void DeterministicDeviceReduceSingleTileKernel(InputIteratorT d_in,
                                                    OutputIteratorT d_out,
                                                    int num_items,
                                                    ReductionOpT reduction_op,
                                                    InitValueT init,
                                                    TransformOpT transform_op)
{
  constexpr ReducePassPolicy policy = current_policy<PolicySelector>().single_tile;
  constexpr int threads_per_block   = policy.threads_per_block;

  using block_reduce_t = BlockReduce<AccumT, threads_per_block, policy.reduce_algorithm>;

  // Shared memory storage
  __shared__ typename block_reduce_t::TempStorage temp_storage;

  // TODO(NaderAlAwar): This code is intentionally duplicated in DeterministicDeviceReduceDeferredSingleTileKernel
  // because extracting it into a device function changes the SASS of this kernel. Changes here must also be applied to
  // the copy below.

  // Check if empty problem
  if (num_items == 0)
  {
    if (threadIdx.x == 0)
    {
      *d_out = init;
    }
    return;
  }

  using float_type         = typename AccumT::ftype;
  constexpr int bin_length = AccumT::max_index + AccumT::max_fold;

  float_type* shared_bins = detail::rfa::get_shared_bin_array<float_type, bin_length>();

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int index = static_cast<int>(threadIdx.x); index < bin_length; index += threads_per_block)
  {
    shared_bins[index] = AccumT::initialize_bin(index);
  }

  __syncthreads();

  AccumT thread_aggregate{};

  // Consume block aggregates of previous kernel
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = static_cast<int>(threadIdx.x); i < num_items; i += threads_per_block)
  {
    thread_aggregate += transform_op(d_in[i]);
  }

  AccumT block_aggregate = block_reduce_t(temp_storage).Reduce(thread_aggregate, reduction_op, num_items);

  // Output result
  if (threadIdx.x == 0)
  {
    detail::reduce::finalize_and_store_aggregate(d_out, reduction_op, init, block_aggregate.conv_to_fp());
  }
}

//! Single-tile entry point that consumes all first-pass partials of a reduction with a deferred problem size. The
//! first pass launches a worst-case grid whose surplus blocks write empty accumulators, which are exact identities
//! under binned accumulation, so every partial can be consumed. The deferred problem size is read only to detect an
//! empty problem, which must store `init` unmodified.
template <typename PolicySelector,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename NormalizedNumItemsT,
          typename ReductionOpT,
          typename InitValueT,
          typename AccumT,
          typename TransformOpT = ::cuda::std::identity>
_CCCL_KERNEL_ATTRIBUTES __launch_bounds__(
  int(current_policy<PolicySelector>().single_tile.threads_per_block),
  1) void DeterministicDeviceReduceDeferredSingleTileKernel(InputIteratorT d_in,
                                                            OutputIteratorT d_out,
                                                            const NormalizedNumItemsT normalized_num_items,
                                                            const int first_pass_grid_size,
                                                            ReductionOpT reduction_op,
                                                            InitValueT init,
                                                            TransformOpT transform_op)
{
  using actual_num_items_t = deterministic_num_items_t<NormalizedNumItemsT>;
  const actual_num_items_t actual_num_items =
    CUB_NS_QUALIFIER::detail::resolve_parameter<actual_num_items_t>(normalized_num_items);
  const int num_items = first_pass_grid_size;

  constexpr ReducePassPolicy policy = current_policy<PolicySelector>().single_tile;
  constexpr int threads_per_block   = policy.threads_per_block;

  using block_reduce_t = BlockReduce<AccumT, threads_per_block, policy.reduce_algorithm>;

  // Shared memory storage
  __shared__ typename block_reduce_t::TempStorage temp_storage;

  // TODO(NaderAlAwar): This code is intentionally duplicated in DeterministicDeviceReduceSingleTileKernel because
  // extracting it into a device function changes the SASS of that kernel. Changes here must also be applied to the
  // copy above.

  // Check if empty problem
  if (actual_num_items == 0)
  {
    if (threadIdx.x == 0)
    {
      *d_out = init;
    }
    return;
  }

  using float_type         = typename AccumT::ftype;
  constexpr int bin_length = AccumT::max_index + AccumT::max_fold;

  float_type* shared_bins = detail::rfa::get_shared_bin_array<float_type, bin_length>();

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int index = static_cast<int>(threadIdx.x); index < bin_length; index += threads_per_block)
  {
    shared_bins[index] = AccumT::initialize_bin(index);
  }

  __syncthreads();

  AccumT thread_aggregate{};

  // Consume block aggregates of previous kernel
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = static_cast<int>(threadIdx.x); i < num_items; i += threads_per_block)
  {
    thread_aggregate += transform_op(d_in[i]);
  }

  AccumT block_aggregate = block_reduce_t(temp_storage).Reduce(thread_aggregate, reduction_op, num_items);

  // Output result
  if (threadIdx.x == 0)
  {
    detail::reduce::finalize_and_store_aggregate(d_out, reduction_op, init, block_aggregate.conv_to_fp());
  }
}
} // namespace detail::reduce

CUB_NAMESPACE_END
