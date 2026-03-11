// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include <cub/detail/rfa.cuh>
#include <cub/device/dispatch/kernels/kernel_reduce.cuh>
#include <cub/device/dispatch/tuning/tuning_reduce_deterministic.cuh>

#include <thrust/type_traits/unwrap_contiguous_iterator.h>

#include <cuda/__device/arch_id.h>

CUB_NAMESPACE_BEGIN

namespace detail::reduce
{
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
 * @param[in] num_items
 *   Total number of input data items
 *
 * @param[in] reduction_op
 *   Binary reduction functor
 */
template <typename PolicySelector, typename InputIteratorT, typename ReductionOpT, typename AccumT, typename TransformOpT>
CUB_DETAIL_KERNEL_ATTRIBUTES __launch_bounds__(int(
  PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10})
    .reduce.block_threads)) void DeterministicDeviceReduceKernel(InputIteratorT d_in,
                                                                 AccumT* d_out,
                                                                 int num_items,
                                                                 ReductionOpT reduction_op,
                                                                 TransformOpT transform_op,
                                                                 const int reduce_grid_size)
{
  constexpr rfa::reduce_policy policy = PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10}).reduce;
  constexpr int items_per_thread      = policy.items_per_thread;
  constexpr int block_threads         = policy.block_threads;

  using block_reduce_t = BlockReduce<AccumT, block_threads, policy.block_algorithm>;

  // Shared memory storage
  __shared__ typename block_reduce_t::TempStorage temp_storage;

  using ftype              = typename AccumT::ftype;
  constexpr int bin_length = AccumT::max_index + AccumT::max_fold;
  const int tid            = block_threads * blockIdx.x + threadIdx.x;

  ftype* shared_bins = detail::rfa::get_shared_bin_array<ftype, bin_length>();

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int index = threadIdx.x; index < bin_length; index += block_threads)
  {
    shared_bins[index] = AccumT::initialize_bin(index);
  }

  __syncthreads();

  AccumT thread_aggregate{};
  int count = 0;

  int n_threads = reduce_grid_size * block_threads;

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
 * @tparam InitT
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
          typename InitT,
          typename AccumT,
          typename TransformOpT = ::cuda::std::identity>
CUB_DETAIL_KERNEL_ATTRIBUTES __launch_bounds__(
  int(PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10}).single_tile.block_threads),
  1) void DeterministicDeviceReduceSingleTileKernel(InputIteratorT d_in,
                                                    OutputIteratorT d_out,
                                                    int num_items,
                                                    ReductionOpT reduction_op,
                                                    InitT init,
                                                    TransformOpT transform_op)
{
  constexpr rfa::single_tile_policy policy = PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10}).single_tile;
  constexpr int block_threads              = policy.block_threads;

  using block_reduce_t = BlockReduce<AccumT, block_threads, policy.block_algorithm>;

  // Shared memory storage
  __shared__ typename block_reduce_t::TempStorage temp_storage;

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
  for (int index = threadIdx.x; index < bin_length; index += block_threads)
  {
    shared_bins[index] = AccumT::initialize_bin(index);
  }

  __syncthreads();

  AccumT thread_aggregate{};

  // Consume block aggregates of previous kernel
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = threadIdx.x; i < num_items; i += block_threads)
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
