// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! @rst
//! The ``cub::WarpReduceBatched`` class provides :ref:`collective <collective-primitives>` methods for
//! performing batched parallel reductions of multiple arrays partitioned across a CUDA thread warp.
//! @endrst

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/type_traits.cuh>
#include <cub/util_type.cuh>
#include <cub/warp/specializations/warp_reduce_batched_wspro.cuh>

#include <cuda/cmath>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/warp>

CUB_NAMESPACE_BEGIN

//! @rst
//! The ``WarpReduceBatched`` class provides :ref:`collective <collective-primitives>` methods for performing
//! batched parallel reductions of multiple arrays partitioned across a CUDA thread warp using the WSPRO algorithm.
//!
//! Overview
//! ++++++++
//!
//! - Performs batched reductions of Batches arrays, each containing LogicalWarpThreads elements
//! - Completes in Batches-1 stages using the WSPRO (Warp Shuffle Parallel Reduction Optimization) algorithm
//! - Standard approach (Batches sequential calls to WarpReduce) requires ``Batches * log2(LogicalWarpThreads)`` stages
//! - Best performance when ``Batches == LogicalWarpThreads`` (both powers of 2)
//! - **Output semantics:** Thread i's ``outputs`` array contains:
//!   - ``outputs[0]`` = reduction of array i
//!   - ``outputs[1]`` = reduction of array (i + LogicalWarpThreads), if it exists
//!   - ``outputs[k]`` = reduction of array (i + k * LogicalWarpThreads)
//!
//! Performance Characteristics
//! +++++++++++++++++++++++++++
//!
//! - **Stage count:** Batches-1 stages vs Batches * log2(LogicalWarpThreads) for sequential WarpReduce calls
//! - **Example (Batches=8, LogicalWarpThreads=8):** 7 stages (batched) vs 24 stages (sequential) = 3.4x reduction
//! - **Example (Batches=16, LogicalWarpThreads=16):** 15 stages (batched) vs 64 stages (sequential) = 4.3x reduction
//! - Uses warp ``SHFL`` instructions for communication
//! - No shared memory required
//!
//! When to Use
//! +++++++++++
//!
//! - When you need to reduce multiple independent batches within a warp
//! - When ``Batches`` and ``LogicalWarpThreads`` are powers of 2 (required for ``LogicalWarpThreads``, recommended for
//! ``Batches``)
//! - When computing many small reductions (e.g., per-pixel reductions)
//!
//! Simple Examples
//! +++++++++++++++
//!
//! @warpcollective{WarpReduceBatched}
//!
//! The code snippet below illustrates reduction of 8 batches of 8 elements each:
//!
//! .. code-block:: c++
//!
//!    #include <cub/cub.cuh>
//!
//!    __global__ void ExampleKernel(...)
//!    {
//!        // Specialize WarpReduceBatched for 8 batches of 8 int elements
//!        using WarpReduceBatched = cub::WarpReduceBatched<int, 8, 8>;
//!
//!        // Allocate shared memory (none needed for shuffle-based implementation)
//!        __shared__ typename WarpReduceBatched::TempStorage temp_storage;
//!
//!        // Each thread provides 8 input values (one element from each of 8 arrays)
//!        int thread_data[8];
//!        for (int i = 0; i < 8; i++)
//!        {
//!            thread_data[i] = ...; // Load element i from thread's position
//!        }
//!
//!        // Perform batched reduction (7 stages vs 24 for sequential)
//!        int results[8];
//!        WarpReduceBatched(temp_storage).Reduce(thread_data, results, ::cuda::std::plus<>{});
//!
//!        // Thread i now has the sum of array i in results[i]
//!    }
//!
//! @endrst
//!
//! @tparam T
//!   The reduction input/output element type
//!
//! @tparam Batches
//!   The number of arrays to reduce in batch. Best performance when Batches = LogicalWarpThreads.
//!
//! @tparam LogicalWarpThreads
//!   The number of threads per logical warp / elements per array. Must be a power-of-two in range [2, 32].
//!   Default is the warp size of the targeted CUDA compute-capability (e.g., 32).
//!
template <typename T, int Batches, int LogicalWarpThreads = detail::warp_threads>
class WarpReduceBatched
{
  static_assert(::cuda::is_power_of_two(LogicalWarpThreads), "LogicalWarpThreads must be a power of two");
  // TODO: Should we allow LogicalWarpThreads = 1? (in which case everything is just no-op/copy)
  static_assert(LogicalWarpThreads > 1 && LogicalWarpThreads <= detail::warp_threads,
                "LogicalWarpThreads must be in the range [2, 32]");
  // TODO: Should we restrict to Batches > 1?
  static_assert(Batches >= 1, "Batches must be >= 1");

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

public:
  /// Internal specialization.
  using InternalWarpReduceBatched = detail::WarpReduceBatchedWspro<T, Batches, LogicalWarpThreads>;

#endif // _CCCL_DOXYGEN_INVOKED

private:
  //---------------------------------------------------------------------
  // Constants and type definitions
  //---------------------------------------------------------------------

  static constexpr auto max_out_per_thread = ::cuda::ceil_div(Batches, LogicalWarpThreads);

  /// Shared memory storage layout type for WarpReduceBatched
  using _TempStorage = typename InternalWarpReduceBatched::TempStorage;

  /// Shared storage reference
  _TempStorage& temp_storage;

public:
  /// \smemstorage{WarpReduceBatched}
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //! @name Collective constructors
  //! @{

  //! @rst
  //! Collective constructor using the specified memory allocation as temporary storage.
  //! Logical warp and lane identifiers are constructed from ``threadIdx.x``.
  //! @endrst
  //!
  //! @param[in] temp_storage Reference to memory allocation having layout type TempStorage
  _CCCL_DEVICE _CCCL_FORCEINLINE WarpReduceBatched(TempStorage& temp_storage)
      : temp_storage{temp_storage.Alias()}
  {}

  //! @}
  //! @name Batched reductions
  //! @{

  //! @rst
  //! Performs batched reduction of ``Batches`` arrays using the specified binary reduction operator.
  //!
  //! Each thread provides ``Batches`` input values (one element from each batch).
  //! The warp collectively reduces each of the ``Batches`` batches (each containing ``LogicalWarpThreads`` elements).
  //! Thread *i* stores results sequentially in its ``outputs`` array:
  //! ``outputs[0]`` = result of batch *i*, ``outputs[1]`` = result of batch *(i + LogicalWarpThreads)*, etc.
  //!
  //! **Algorithm Performance:**
  //!
  //! - Completes in ``Batches - 1 + log2(LogicalWarpThreads / Batches)`` stages
  //! - vs ``Batches * log2(LogicalWarpThreads)`` stages for ``Batches`` sequential ``WarpReduce`` calls
  //! - Example: ``Batches=8``, ``LogicalWarpThreads=8`` -> 7 stages (batched) vs 24 stages (sequential)
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates batched reduction of 8 batches:
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!
  //!    __global__ void ExampleKernel(...)
  //!    {
  //!        using WarpReduceBatched = cub::WarpReduceBatched<int, 8, 8>;
  //!
  //!        cuda::std::array<int, 8> inputs = {...};  // Each thread provides 8 values
  //!        cuda::std::array<int, 1> output;
  //!
  //!        WarpReduceBatched.Reduce(
  //!            inputs, output, cuda::std::plus<>{});
  //!
  //!        // Logical warp lane i now has sum of batch i in output[0]
  //!    }
  //!
  //! @endrst
  //!
  //! @tparam InputT
  //!   **[inferred]** Input array-like type (C-array, cuda::std::array, cuda::std::span, etc.)
  //!
  //! @tparam OutputT
  //!   **[inferred]** Output array-like type (C-array, cuda::std::array, cuda::std::span, etc.)
  //!
  //! @tparam ReductionOp
  //!   **[inferred]** Binary reduction operator type having member
  //!   `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] inputs
  //!   Statically-sized array-like container of Batches input values from calling thread
  //!
  //! @param[out] outputs
  //!   Statically-sized array-like container where thread i stores reductions sequentially:
  //!   ``outputs[0]`` = result of batch i, ``outputs[1]`` = result of batch (i + LogicalWarpThreads), etc.
  //!
  //! @param[in] reduction_op
  //!   Binary reduction operator
  //!
  //! @param[in] lane_mask
  //!   Lane mask to restrict the reduction to a subset of the logical warps present in the physical warp.
  //!   Default is all logical warps.
  template <typename InputT, typename OutputT, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Reduce(const InputT& inputs,
         OutputT& outputs,
         ReductionOp reduction_op,
         ::cuda::std::uint32_t lane_mask = ::cuda::device::lane_mask::all().value())
  {
    static_assert(::cub::detail::is_fixed_size_random_access_range_v<InputT>,
                  "InputT must support the subscript operator[] and have a compile-time size");
    static_assert(::cub::detail::is_fixed_size_random_access_range_v<OutputT>,
                  "OutputT must support the subscript operator[] and have a compile-time size");
    static_assert(::cub::detail::static_size_v<InputT> == Batches, "Input size must match Batches");
    static_assert(::cub::detail::static_size_v<OutputT> == max_out_per_thread,
                  "Output size must match ceil_div(Batches, LogicalWarpThreads)");
    // These restrictions could be relaxed to allow type-conversions
    static_assert(::cuda::std::is_same_v<::cuda::std::iter_value_t<InputT>, T>, "Input element type must match T");
    static_assert(::cuda::std::is_same_v<::cuda::std::iter_value_t<OutputT>, T>, "Output element type must match T");

    InternalWarpReduceBatched{temp_storage}.Reduce(inputs, outputs, reduction_op, lane_mask);
  }

  // TODO: Public for benchmarking purposes only.
  template <typename InputT, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ReduceInplace(
    InputT& inputs, ReductionOp reduction_op, ::cuda::std::uint32_t lane_mask = ::cuda::device::lane_mask::all().value())
  {
    static_assert(detail::is_fixed_size_random_access_range_v<InputT>,
                  "InputT must support the subscript operator[] and have a compile-time size");
    static_assert(detail::static_size_v<InputT> == Batches, "Input size must match Batches");
    static_assert(::cuda::std::is_same_v<::cuda::std::iter_value_t<InputT>, T>, "Input element type must match T");

    InternalWarpReduceBatched{temp_storage}.ReduceInplace(inputs, reduction_op, lane_mask);
  }

  //! @rst
  //! Performs batched sum reduction of Batches arrays.
  //!
  //! Convenience wrapper for ``Reduce`` with ``::cuda::std::plus<>`` operator.
  //!
  //! @smemwarpreuse
  //!
  //! @endrst
  //!
  //! @tparam InputT
  //!   **[inferred]** Input array-like type (C-array, cuda::std::array, cuda::std::span, etc.)
  //!
  //! @tparam OutputT
  //!   **[inferred]** Output array-like type (C-array, cuda::std::array, cuda::std::span, etc.)
  //!
  //! @param[in] inputs
  //!   Statically-sized array-like container of Batches input values from calling thread
  //!
  //! @param[out] outputs
  //!   Statically-sized array-like container where thread i stores sums sequentially:
  //!   ``outputs[0]`` = sum of array i, ``outputs[1]`` = sum of array (i + LogicalWarpThreads), etc.
  //!
  //! @param[in] lane_mask
  //!   Lane mask to restrict the reduction to a subset of the logical warps present in the physical warp.
  //!   Default is all logical warps.
  template <typename InputT, typename OutputT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Sum(
    const InputT& inputs, OutputT& outputs, ::cuda::std::uint32_t lane_mask = ::cuda::device::lane_mask::all().value())
  {
    Reduce(inputs, outputs, ::cuda::std::plus<>{}, lane_mask);
  }

  //! @}
};

CUB_NAMESPACE_END
