/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_nondeterministic_reduce.cuh>
#include <cub/grid/grid_even_share.cuh>

CUB_NAMESPACE_BEGIN

namespace detail
{
namespace nondeterministic_reduce
{

/**
 * All cub::DeviceReduce::* algorithms are using the same implementation. Some of them, however,
 * should use initial value only for empty problems. If this struct is used as initial value with
 * one of the `DeviceReduce` algorithms, the `init` value wrapped by this struct will only be used
 * for empty problems; it will not be incorporated into the aggregate of non-empty problems.
 */
template <class T>
struct empty_problem_init_t
{
  T init;

  _CCCL_HOST_DEVICE operator T() const
  {
    return init;
  }
};

template <class InitT>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE InitT unwrap_empty_problem_init(InitT init)
{
  return init;
}

template <class InitT>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE InitT unwrap_empty_problem_init(empty_problem_init_t<InitT> empty_problem_init)
{
  return empty_problem_init.init;
}

/**
 * @brief Applies initial value to the block aggregate and stores the result to the output iterator.
 *
 * @param d_out Iterator to the output aggregate
 * @param reduction_op Binary reduction functor
 * @param init Initial value
 * @param block_aggregate Aggregate value computed by the block
 */
template <class OutputIteratorT, class ReductionOpT, class InitT, class AccumT>
_CCCL_HOST_DEVICE void
finalize_and_store_aggregate(OutputIteratorT d_out, ReductionOpT reduction_op, InitT init, AccumT block_aggregate)
{
  *d_out = reduction_op(init, block_aggregate);
}

/**
 * @brief Ignores initial value and stores the block aggregate to the output iterator.
 *
 * @param d_out Iterator to the output aggregate
 * @param block_aggregate Aggregate value computed by the block
 */
template <class OutputIteratorT, class ReductionOpT, class InitT, class AccumT>
_CCCL_HOST_DEVICE void
finalize_and_store_aggregate(OutputIteratorT d_out, ReductionOpT, empty_problem_init_t<InitT>, AccumT block_aggregate)
{
  *d_out = block_aggregate;
}

/**
 * @brief Reduce region kernel entry point (multi-block). Computes privatized
 *        reductions, one per thread block.
 *
 * @tparam ChainedPolicyT
 *   Chained tuning policy
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam ReductionOpT
 *   Binary reduction functor type having member
 *   `auto operator()(const T &a, const U &b)`
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
 * @param[in] even_share
 *   Even-share descriptor for mapping an equal number of tiles onto each
 *   thread block
 *
 * @param[in] reduction_op
 *   Binary reduction functor
 */
template <typename ChainedPolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitT,
          typename AccumT,
          typename TransformOpT,
          typename CounterT>
CUB_DETAIL_KERNEL_ATTRIBUTES
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::ReduceLastBlockPolicy::BLOCK_THREADS)) void DeviceReduceLastBlockKernel(
  InputIteratorT d_in,
  OutputIteratorT d_out,
  AccumT* d_block_reductions,
  GridEvenShare<OffsetT> even_share,
  ReductionOpT reduction_op,
  InitT init,
  TransformOpT transform_op,
  CounterT* counter)
{
  // Thread block type for reducing input tiles
  using AgentReduceT = detail::nondeterministic_reduce::AgentReduce<
    typename ChainedPolicyT::ActivePolicy::ReduceLastBlockPolicy,
    InputIteratorT,
    AccumT*,
    OffsetT,
    ReductionOpT,
    AccumT,
    TransformOpT>;

  // Thread block type for reducing input tiles
  using FinalAgentReduceT = detail::nondeterministic_reduce::AgentReduce<
    typename ChainedPolicyT::ActivePolicy::ReduceLastBlockPolicy,
    AccumT*,
    OutputIteratorT,
    OffsetT,
    ReductionOpT,
    AccumT>;

  // Shared memory storage
  union temp_storage_t
  {
    typename AgentReduceT::TempStorage partial_reduce;
    typename FinalAgentReduceT::TempStorage final_reduce;
  };

  __shared__ temp_storage_t temp_storage;

  // Consume input tiles
  AccumT block_aggregate =
    AgentReduceT(temp_storage.partial_reduce, d_in, reduction_op, transform_op).ConsumeTiles(even_share);

  // Output result
  if (threadIdx.x == 0)
  {
    // ony thread 0 has valid value in block aggregate
    detail::uninitialized_copy_single(d_block_reductions + blockIdx.x, block_aggregate);
  }

  __threadfence();

  int perform_final_reduce = false;
  if (threadIdx.x == 0)
  {
    CounterT old_counter = atomicAdd(counter, static_cast<CounterT>(1));
    perform_final_reduce = old_counter == gridDim.x - 1;
  }

  if (__syncthreads_or(perform_final_reduce))
  {
    // Consume input tiles
    AccumT block_aggregate =
      FinalAgentReduceT(temp_storage.final_reduce, d_block_reductions, reduction_op).ConsumeRange(OffsetT(0), gridDim.x);

    // Output result
    if (threadIdx.x == 0)
    {
      detail::nondeterministic_reduce::finalize_and_store_aggregate(d_out, reduction_op, init, block_aggregate);
    }
  }
}

template <typename ChainedPolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitT,
          typename AccumT,
          typename TransformOpT>
CUB_DETAIL_KERNEL_ATTRIBUTES
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::ReduceAtomicPolicy::BLOCK_THREADS)) void DeviceReduceAtomicKernel(
  InputIteratorT d_in,
  OutputIteratorT d_out,
  OffsetT num_items,
#if TUNE_USE_GRID_EVEN_SHARE
  GridEvenShare<OffsetT> even_share,
#endif
  ReductionOpT reduction_op,
  InitT init,
  TransformOpT transform_op)
{
  // Thread block type for reducing input tiles
  using AgentReduceT = detail::nondeterministic_reduce::AgentReduce<
    typename ChainedPolicyT::ActivePolicy::ReduceAtomicPolicy,
    InputIteratorT,
    AccumT*,
    OffsetT,
    ReductionOpT,
    AccumT,
    TransformOpT>;

  // Shared memory storage
  __shared__ typename AgentReduceT::TempStorage temp_storage;

#if TUNE_USE_GRID_EVEN_SHARE
  // Consume input tiles
  AccumT block_aggregate = AgentReduceT(temp_storage, d_in, reduction_op, transform_op).ConsumeTiles(even_share);
#else
  AccumT block_aggregate =
    AgentReduceT(temp_storage, d_in, reduction_op, transform_op)
      .ConsumeRange(blockIdx.x * AgentReduceT::TILE_ITEMS,
                    _CUDA_VSTD::min(static_cast<OffsetT>((blockIdx.x + 1) * AgentReduceT::TILE_ITEMS), num_items));
#endif

  // Output result
  if (threadIdx.x == 0)
  {
    // ony thread 0 has valid value in block aggregate
    // detail::uninitialized_copy_single(d_block_reductions + blockIdx.x, block_aggregate);
    if (blockIdx.x == 0)
    {
      atomicAdd(d_out, init);
    }

    atomicAdd(d_out, block_aggregate);
  }
}

} // namespace nondeterministic_reduce
} // namespace detail

CUB_NAMESPACE_END
