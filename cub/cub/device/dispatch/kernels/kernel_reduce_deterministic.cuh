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

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__device/compute_capability.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::reduce
{
// Type a kernel problem-size argument resolves to on device: an immediate problem size keeps its integral type, a
// deferred source resolves to an integer wide enough for its element. Only a signed 32-bit element is guaranteed
// to fit a single 32-bit chunk; every other element type is widened to 64 bits and consumed in chunks, with an
// unsigned element resolving to an unsigned type so that its full value range is representable.
template <typename KernelNumItemsT, bool = ::cuda::std::is_integral_v<KernelNumItemsT>>
struct deterministic_num_items
{
  using type = KernelNumItemsT;
};

template <typename KernelNumItemsT>
struct deterministic_num_items<KernelNumItemsT, false>
{
  using element_t = it_value_t<KernelNumItemsT>;
  using type      = ::cuda::std::conditional_t<
         ::cuda::std::is_signed_v<element_t> && sizeof(element_t) == 4,
         int,
         ::cuda::std::conditional_t<::cuda::std::is_signed_v<element_t>, ::cuda::std::int64_t, ::cuda::std::uint64_t>>;
};

template <typename KernelNumItemsT>
using deterministic_num_items_t = typename deterministic_num_items<KernelNumItemsT>::type;

// Number of first-pass blocks that produce a partial for a deferred problem size: the worst-case launch is trimmed
// to the grid the host would have computed had it been able to read the problem size. Must be used consistently by
// both reduction passes.
template <typename PolicySelector, typename NumItemsT>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE int
deferred_reduce_grid_size(NumItemsT num_items, int launched_grid_size) noexcept
{
  constexpr ReducePassPolicy policy = detail::current_policy<PolicySelector>().multi_tile;
  constexpr int tile_size           = policy.threads_per_block * policy.items_per_thread;
  const NumItemsT num_tiles         = ::cuda::ceil_div(num_items, NumItemsT{tile_size});
  return static_cast<int>(::cuda::std::min(static_cast<NumItemsT>(launched_grid_size), num_tiles));
}

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
 * @tparam KernelNumItemsT
 *   Type of integral problem size or a deferred problem-size descriptor
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
 * @param[in] kernel_num_items
 *   Immediate problem size or a deferred problem-size descriptor
 *
 * @param[in] reduction_op
 *   Binary reduction functor
 */
template <typename PolicySelector,
          typename InputIteratorT,
          typename KernelNumItemsT,
          typename ReductionOpT,
          typename AccumT,
          typename TransformOpT>
_CCCL_KERNEL_ATTRIBUTES
__launch_bounds__(int(current_policy<PolicySelector>().multi_tile.threads_per_block)) void DeterministicDeviceReduceKernel(
  InputIteratorT d_in,
  AccumT* d_out,
  const KernelNumItemsT kernel_num_items,
  ReductionOpT reduction_op,
  TransformOpT transform_op,
  const int reduce_grid_size)
{
  constexpr ReducePassPolicy policy = current_policy<PolicySelector>().multi_tile;
  constexpr int items_per_thread    = policy.items_per_thread;
  constexpr int threads_per_block   = policy.threads_per_block;

  // A 64-bit deferred problem size is consumed in a single launch that loops over 32-bit chunks in the kernel.
  using num_items_t = deterministic_num_items_t<KernelNumItemsT>;

  const num_items_t num_items = detail::parameter_from_device<num_items_t>(kernel_num_items);

  // The worst-case grid of a deferred problem size is trimmed to the blocks that receive at least one tile. Both the
  // early exit and the loop stride must use the trimmed grid so that the remaining blocks cover the whole input.
  const int active_grid_size = [&] {
    if constexpr (::cuda::std::is_integral_v<KernelNumItemsT>)
    {
      return reduce_grid_size;
    }
    else
    {
      return detail::reduce::deferred_reduce_grid_size<PolicySelector>(num_items, reduce_grid_size);
    }
  }();

  if constexpr (!::cuda::std::is_integral_v<KernelNumItemsT>)
  {
    if (static_cast<int>(blockIdx.x) >= active_grid_size)
    {
      return;
    }
  }

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

  int n_threads = active_grid_size * threads_per_block;

  if constexpr (sizeof(num_items_t) == 8)
  {
    // Loop over 32-bit chunks so that the hot loop keeps the index arithmetic and register footprint of the 32-bit
    // path below; only the count of remaining items is 64-bit and the iterator advances once per chunk. Binned
    // accumulation is exact, so accumulating across chunk boundaries produces the same bits as the host-side chunking
    // of an immediate problem size.
    // TODO(NaderAlAwar): The chunk body is intentionally duplicated from the 32-bit loop below because extracting it
    // into a device function changes the SASS of the immediate instantiations. Changes here must also be applied to
    // the copy below.
    constexpr auto num_items_per_chunk = num_items_t{::cuda::std::numeric_limits<int>::max()};
    for (num_items_t remaining = num_items; remaining != 0;)
    {
      const auto chunk_num_items = static_cast<int>(::cuda::std::min(remaining, num_items_per_chunk));

      _CCCL_PRAGMA_UNROLL_FULL()
      for (unsigned i = tid; i < static_cast<unsigned>(chunk_num_items); i += (n_threads * items_per_thread))
      {
        ftype items[items_per_thread] = {};
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int j = 0; j < items_per_thread; j++)
        {
          const unsigned idx = i + j * n_threads;
          if (idx < static_cast<unsigned>(chunk_num_items))
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

      d_in += chunk_num_items;
      // chunk_num_items == min(remaining, num_items_per_chunk) <= remaining, so the countdown lands exactly on zero
      // and cannot wrap when num_items_t is unsigned.
      remaining -= chunk_num_items;
    }
  }
  else
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (unsigned i = tid; i < static_cast<unsigned>(num_items); i += (n_threads * items_per_thread))
    {
      ftype items[items_per_thread] = {};
      for (int j = 0; j < items_per_thread; j++)
      {
        const unsigned idx = i + j * n_threads;
        if (idx < static_cast<unsigned>(num_items))
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

//! Single-tile entry point for a reduction with a deferred problem size. Derives on device how many blocks of the
//! worst-case first-pass grid produced a partial (the surplus blocks exited without writing one) and consumes exactly
//! those partials. An empty problem stores `init` unmodified.
template <typename PolicySelector,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename KernelNumItemsT,
          typename ReductionOpT,
          typename InitValueT,
          typename AccumT,
          typename TransformOpT = ::cuda::std::identity>
_CCCL_KERNEL_ATTRIBUTES __launch_bounds__(
  int(current_policy<PolicySelector>().single_tile.threads_per_block),
  1) void DeterministicDeviceReduceDeferredSingleTileKernel(const InputIteratorT d_in,
                                                            const OutputIteratorT d_out,
                                                            const KernelNumItemsT kernel_num_items,
                                                            const int first_pass_grid_size,
                                                            ReductionOpT reduction_op,
                                                            const InitValueT init,
                                                            TransformOpT transform_op)
{
  using actual_num_items_t                  = deterministic_num_items_t<KernelNumItemsT>;
  const actual_num_items_t actual_num_items = detail::parameter_from_device<actual_num_items_t>(kernel_num_items);
  const int num_items =
    detail::reduce::deferred_reduce_grid_size<PolicySelector>(actual_num_items, first_pass_grid_size);

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
