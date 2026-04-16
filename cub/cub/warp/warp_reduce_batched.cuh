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

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__cmath/pow2.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__iterator/iterator_traits.h>

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
//! The code snippet below illustrates reduction of 3 batches across 32 threads in each of 2 (logical) warps:
//!
//! .. literalinclude:: ../../../cub/test/catch2_test_warp_reduce_batched_api.cu
//!     :language: c++
//!     :dedent:
//!     :start-after: example-begin warp-reduce-batched-overview
//!     :end-before: example-end warp-reduce-batched-overview
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
template <typename T, int Batches, int LogicalWarpThreads = detail::warp_threads, bool SyncPhysicalWarp = false>
class WarpReduceBatched
{
  static_assert(::cuda::is_power_of_two(LogicalWarpThreads), "LogicalWarpThreads must be a power of two");
  // TODO: Should we allow LogicalWarpThreads = 1? (in which case everything is just no-op/copy)
  static_assert(LogicalWarpThreads > 1 && LogicalWarpThreads <= detail::warp_threads,
                "LogicalWarpThreads must be in the range [2, 32]");
  static_assert(Batches >= 1, "Batches must be >= 1");

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

public:
  //! Internal specialization.
  using InternalWarpReduceBatched = detail::warp_reduce_batched_wspro<T, Batches, LogicalWarpThreads, SyncPhysicalWarp>;

#endif // _CCCL_DOXYGEN_INVOKED

private:
  static constexpr auto max_out_per_thread = ::cuda::ceil_div(Batches, LogicalWarpThreads);

  //! Shared memory storage layout type for WarpReduceBatched
  using _TempStorage = typename InternalWarpReduceBatched::TempStorage;

  //! Shared storage reference
  _TempStorage& temp_storage;

public:
  //! \smemstorage{WarpReduceBatched}
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
  //! Thread ``i`` returns the result for batch ``i``. For ``Batches > LogicalWarpThreads``, use ``ReduceToStriped()``
  //! or ``ReduceToBlocked()`` instead.
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates reduction of 3 batches across 16 threads in the branched-off first logical warp
  //! (using ``cuda::std::array`` inputs and outputs):
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_warp_reduce_batched_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin warp-reduce-batched-reduce
  //!     :end-before: example-end warp-reduce-batched-reduce
  //!
  //! @endrst
  //!
  //! @tparam InputT
  //!   **[inferred]** Input array-like type (C-array, cuda::std::array, cuda::std::span, etc.)
  //!
  //! @tparam ReductionOp
  //!   **[inferred]** Binary reduction operator type having member
  //!   `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] inputs
  //!   Statically-sized array-like container of Batches input values from calling thread
  //!
  //! @param[in] reduction_op
  //!   Binary reduction operator
  //!
  //! @return
  //!   The reduction of the input values of the batch corresponding to the logical lane.
  template <typename InputT, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(const InputT& inputs, ReductionOp reduction_op)
  {
    static_assert(detail::is_fixed_size_random_access_range_v<InputT>,
                  "InputT must support the subscript operator[] and have a compile-time size");
    static_assert(detail::static_size_v<InputT> == Batches, "Input size must match Batches");
    // These restrictions could be relaxed to allow type-conversions
    static_assert(::cuda::std::is_same_v<::cuda::std::iter_value_t<InputT>, T>, "Input element type must match T");
    static_assert(max_out_per_thread == 1,
                  "For Batches > LogicalWarpThreads, use ReduceToStriped() or ReduceToBlocked()");

    ::cuda::std::array<T, 1> output;
    // ReduceToBlocked() and ReduceToStriped() do the same for max_out_per_thread == 1.
    ReduceToStriped(inputs, output, reduction_op);
    return output[0];
  }

  //! @rst
  //! Performs batched reduction of ``Batches`` arrays using the specified binary reduction operator.
  //!
  //! Each thread provides ``Batches`` input values (one element from each batch).
  //! The warp collectively reduces each of the ``Batches`` batches (each containing ``LogicalWarpThreads`` elements).
  //! Thread ``i`` stores results in its ``outputs`` array in a striped manner:
  //! ``outputs[0]`` = result of batch ``i``, ``outputs[1]`` = result of batch ``i + LogicalWarpThreads``, etc.
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates reduction of 3 batches across 16 threads in the branched-off first logical warp
  //! (using ``cuda::std::array`` inputs and outputs):
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_warp_reduce_batched_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin warp-reduce-batched-reduce-to-striped
  //!     :end-before: example-end warp-reduce-batched-reduce-to-striped
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
  template <typename InputT, typename OutputT, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ReduceToStriped(const InputT& inputs, OutputT& outputs, ReductionOp reduction_op)
  {
    static_assert(detail::is_fixed_size_random_access_range_v<InputT>,
                  "InputT must support the subscript operator[] and have a compile-time size");
    static_assert(detail::is_fixed_size_random_access_range_v<OutputT>,
                  "OutputT must support the subscript operator[] and have a compile-time size");
    static_assert(detail::static_size_v<InputT> == Batches, "Input size must match Batches");
    static_assert(detail::static_size_v<OutputT> == max_out_per_thread,
                  "Output size must match ceil_div(Batches, LogicalWarpThreads)");
    // These restrictions could be relaxed to allow type-conversions
    static_assert(::cuda::std::is_same_v<::cuda::std::iter_value_t<InputT>, T>, "Input element type must match T");
    static_assert(::cuda::std::is_same_v<::cuda::std::iter_value_t<OutputT>, T>, "Output element type must match T");

    InternalWarpReduceBatched{temp_storage}.Reduce</* ToBlocked = */ false>(inputs, outputs, reduction_op);
  }

  //! @rst
  //! Performs batched reduction of ``Batches`` arrays using the specified binary reduction operator.
  //!
  //! Each thread provides ``Batches`` input values (one element from each batch).
  //! The warp collectively reduces each of the ``Batches`` batches (each containing ``LogicalWarpThreads`` elements).
  //! Thread *i* stores results in its ``outputs`` array in a blocked manner:
  //! ``outputs[0]`` = result of batch ``i * ceil_div(Batches, LogicalWarpThreads)``, ``outputs[1]`` = result of batch
  //! ``i * ceil_div(Batches, LogicalWarpThreads) + 1``, etc.
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates reduction of 3 batches across 16 threads in the branched-off first logical warp
  //! (using ``cuda::std::array`` inputs and outputs):
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_warp_reduce_batched_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin warp-reduce-batched-reduce-to-blocked
  //!     :end-before: example-end warp-reduce-batched-reduce-to-blocked
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
  template <typename InputT, typename OutputT, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ReduceToBlocked(const InputT& inputs, OutputT& outputs, ReductionOp reduction_op)
  {
    static_assert(detail::is_fixed_size_random_access_range_v<InputT>,
                  "InputT must support the subscript operator[] and have a compile-time size");
    static_assert(detail::is_fixed_size_random_access_range_v<OutputT>,
                  "OutputT must support the subscript operator[] and have a compile-time size");
    static_assert(detail::static_size_v<InputT> == Batches, "Input size must match Batches");
    static_assert(detail::static_size_v<OutputT> == max_out_per_thread,
                  "Output size must match ceil_div(Batches, LogicalWarpThreads)");
    // These restrictions could be relaxed to allow type-conversions
    static_assert(::cuda::std::is_same_v<::cuda::std::iter_value_t<InputT>, T>, "Input element type must match T");
    static_assert(::cuda::std::is_same_v<::cuda::std::iter_value_t<OutputT>, T>, "Output element type must match T");

    InternalWarpReduceBatched{temp_storage}.Reduce</* ToBlocked = */ true>(inputs, outputs, reduction_op);
  }

  //! @rst
  //! Performs batched sum reduction of Batches arrays.
  //!
  //! Convenience wrapper for ``Reduce`` with ``::cuda::std::plus<>`` operator.
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates reduction of 3 batches across 2 threads in each of 4 logical warps
  //! meaning more than one output per thread (using ``cuda::std::span`` inputs and outputs):
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_warp_reduce_batched_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin warp-reduce-batched-sum
  //!     :end-before: example-end warp-reduce-batched-sum
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
  //! @return
  //!   The sum of the input values of the batch corresponding to the logical lane.
  template <typename InputT>
  _CCCL_DEVICE _CCCL_FORCEINLINE T Sum(const InputT& inputs)
  {
    return Reduce(inputs, ::cuda::std::plus<>{});
  }

  //! @rst
  //! Performs batched sum reduction of Batches arrays.
  //!
  //! Convenience wrapper for ``ReduceToStriped`` with ``::cuda::std::plus<>`` operator.
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates reduction of 3 batches across 2 threads in each of 4 logical warps
  //! meaning more than one output per thread (using ``cuda::std::span`` inputs and outputs):
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_warp_reduce_batched_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin warp-reduce-batched-sum-to-striped
  //!     :end-before: example-end warp-reduce-batched-sum-to-striped
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
  template <typename InputT, typename OutputT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void SumToStriped(const InputT& inputs, OutputT& outputs)
  {
    ReduceToStriped(inputs, outputs, ::cuda::std::plus<>{});
  }

  //! @rst
  //! Performs batched sum reduction of Batches arrays.
  //!
  //! Convenience wrapper for ``ReduceToBlocked`` with ``::cuda::std::plus<>`` operator.
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates reduction of 3 batches across 2 threads in each of 4 logical warps
  //! meaning more than one output per thread (using ``cuda::std::span`` inputs and outputs):
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_warp_reduce_batched_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin warp-reduce-batched-sum-to-blocked
  //!     :end-before: example-end warp-reduce-batched-sum-to-blocked
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
  template <typename InputT, typename OutputT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void SumToBlocked(const InputT& inputs, OutputT& outputs)
  {
    ReduceToBlocked(inputs, outputs, ::cuda::std::plus<>{});
  }

  //! @}
};

CUB_NAMESPACE_END
