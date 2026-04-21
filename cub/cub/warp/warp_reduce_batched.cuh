// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! @rst
//! The ``cub::WarpReduceBatched`` class provides :ref:`collective <collective-primitives>` methods for
//! performing batched parallel reductions of multiple arrays partitioned across a CUDA warp.
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
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__type_traits/is_same.h>

CUB_NAMESPACE_BEGIN

//! @rst
//! The ``WarpReduceBatched`` class provides :ref:`collective <collective-primitives>` methods for performing
//! batched parallel reductions of multiple batches of items partitioned across a CUDA warp.
//!
//! Overview
//! ++++++++
//!
//! - A `reduction <http://en.wikipedia.org/wiki/Reduce_(higher-order_function)>`__ (or *fold*) uses a binary combining
//!   operator to compute a single aggregate from a list of input elements. Parallel reductions are in general only
//!   deterministic when the reduction operator is both commutative and associative.
//! - Supports "logical" warps smaller than the physical warp size (e.g., logical warps of 8 threads).
//! - This primitive performs batched reductions taking one item per batch per thread.
//! - Results are distributed among the threads. When there are more batches than logical warp threads, results can be
//!   distributed among threads in either striped or blocked manner.
//! - The number of batches must be non-negative. Compile-time and register pressure increase with the number of
//!   batches.
//!
//! Performance Characteristics
//! +++++++++++++++++++++++++++
//!
//! - Uses special instructions when applicable (e.g., warp ``SHFL`` instructions).
//! - Uses synchronization-free communication between warp lanes when applicable.
//! - Should generally give much better performance than sequential ``WarpReduce`` calls independent of blocked or
//! striped output arrangements.
//! - Achieves peak efficiency when the number of batches is a multiple of the number of logical warp threads.
//! - For smaller than physical warp size logical warps, using ``SyncPhysicalWarp = true`` should in general give better
//! performance than ``SyncPhysicalWarp = false``.
//!   Note that it can cause a deadlock if not all non-exited logical warps from the same physical warp participate in
//!   the reduction (due to branches).
//! - Any uneven number of batches is less efficient than the next higher even number.
//! - For more batches than logical warp threads, the striped output can be slightly more performant than blocked output
//!   depending on the number of batches and the number of logical warp threads.
//! - Blocked output should generally give much better performance than converting striped output to blocked using e.g.
//! ``WarpExchange::StripedToBlocked()``.
//! - For types of less than 4 bytes future optimization might let blocked output outperform striped output.
//!
//! Simple Example
//! +++++++++++++++
//!
//! @warpcollective{WarpReduceBatched}
//!
//! The code snippet below illustrates reduction of 3 batches across 32 threads in each of 2 warps:
//!
//! .. literalinclude:: ../../../cub/test/warp/catch2_test_warp_reduce_batched_api.cu
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
//!   The number of batches to reduce. Also corresponds to the size of the range of inputs for each thread.
//!
//! @tparam LogicalWarpThreads
//!   **[optional]** The number of threads per "logical" warp (may be less than the number of
//!   hardware warp threads but has to be a power of two).  Default is the warp size of the targeted CUDA
//!   compute-capability (e.g., 32 threads for SM80).
//!
template <typename T, int Batches, int LogicalWarpThreads = detail::warp_threads, bool SyncPhysicalWarp = false>
class WarpReduceBatched
{
  static_assert(::cuda::is_power_of_two(LogicalWarpThreads), "LogicalWarpThreads must be a power of two");
  static_assert(LogicalWarpThreads > 0 && LogicalWarpThreads <= detail::warp_threads,
                "LogicalWarpThreads must be in the range [1, 32]");
  static_assert(Batches >= 0, "Batches must be >= 0");

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

public:
  //! Internal specialization.
  using InternalWarpReduceBatched = detail::warp_reduce_batched_wspro<T, Batches, LogicalWarpThreads, SyncPhysicalWarp>;

#endif // _CCCL_DOXYGEN_INVOKED

private:
  static constexpr auto max_out_per_thread = ::cuda::ceil_div(Batches, LogicalWarpThreads);

  template <class InputT, class OutputT, class ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE void check_constraints()
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
    static_assert(detail::has_binary_call_operator<ReductionOp, T>::value,
                  "ReductionOp must have the binary call operator: operator(T, T)");
  };

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
  _CCCL_DEVICE_API _CCCL_FORCEINLINE WarpReduceBatched(TempStorage& temp_storage)
      : temp_storage{temp_storage.Alias()}
  {}

  //! @}
  //! @name Generic reductions
  //! @{

  //! @rst
  //! Computes a warp-wide reduction for each batch in the calling warp using the specified binary reduction
  //! functor.
  //! Thread ``i`` returns the result for batch ``i``.
  //! Returned items that have no corresponding input batch are invalid.
  //! For more batches than logical warp threads or generic code that could result in zero batches, use
  //! ``ReduceToStriped()`` or ``ReduceToBlocked()`` instead.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates reduction of 3 batches across 16 threads in the branched-off first logical warp
  //! (using ``cuda::std::array`` inputs and outputs):
  //!
  //! .. literalinclude:: ../../../cub/test/warp/catch2_test_warp_reduce_batched_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin warp-reduce-batched-reduce
  //!     :end-before: example-end warp-reduce-batched-reduce
  //!
  //! @endrst
  //!
  //! @tparam InputT
  //!   **[inferred]** The data type to be reduced having member
  //!   ``operator[](int i)`` and must be statically-sized (``size()`` method or static array)
  //!
  //! @tparam ReductionOp
  //!   **[inferred]** Binary reduction operator type having member
  //!   ``T operator()(const T &a, const T &b)``
  //!
  //! @param[in] inputs
  //!   Statically-sized input range holding ``Batches`` items
  //!
  //! @param[in] reduction_op
  //!   Binary reduction operator
  //!
  //! @return
  //!   The reduction of the input values of the batch corresponding to the logical lane.
  template <typename InputT, typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Reduce(const InputT& inputs, ReductionOp reduction_op)
  {
    static_assert(max_out_per_thread == 1,
                  "For Batches > LogicalWarpThreads (or Batches == 0), use ReduceToStriped() or ReduceToBlocked()");

    ::cuda::std::array<T, 1> output;
    // ReduceToBlocked() and ReduceToStriped() do the same for max_out_per_thread == 1.
    ReduceToStriped(inputs, output, reduction_op);
    return output[0];
  }

  //! @rst
  //! Computes a warp-wide reduction for each batch in the calling warp using the specified binary reduction
  //! functor. The user must provide an output range of ``max_out_per_thread = ceil_div(Batches, LogicalWarpThreads)``
  //! items. Logical lane ``i`` stores results in its output range in a striped manner:
  //! ``outputs[0]`` = result of batch ``i``, ``outputs[1]`` = result of batch ``i + LogicalWarpThreads``, etc.
  //! Items in the output range that have no corresponding input batch are invalid.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates reduction of 3 batches across 2 threads in every second logical warp
  //! (using ``cuda::std::array`` inputs and outputs):
  //!
  //! .. literalinclude:: ../../../cub/test/warp/catch2_test_warp_reduce_batched_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin warp-reduce-batched-reduce-to-striped
  //!     :end-before: example-end warp-reduce-batched-reduce-to-striped
  //!
  //! @endrst
  //!
  //! @tparam InputT
  //!   **[inferred]** The data type to be reduced having member
  //!   ``operator[](int i)`` and must be statically-sized (``size()`` method or static array)
  //!
  //! @tparam OutputT
  //!   **[inferred]** The data type to hold results having member
  //!   ``operator[](int i)`` and must be statically-sized (``size()`` method or static array)
  //!
  //! @tparam ReductionOp
  //!   **[inferred]** Binary reduction operator type having member
  //!   ``T operator()(const T &a, const T &b)``
  //!
  //! @param[in] inputs
  //!   Statically-sized input range holding ``Batches`` items
  //!
  //! @param[out] outputs
  //!   Statically-sized output range holding ``ceil_div(Batches, LogicalWarpThreads)`` items
  //!
  //! @param[in] reduction_op
  //!   Binary reduction operator
  template <typename InputT, typename OutputT, typename ReductionOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  ReduceToStriped(const InputT& inputs, OutputT& outputs, ReductionOp reduction_op)
  {
    check_constraints<InputT, OutputT, ReductionOp>();
    InternalWarpReduceBatched{temp_storage}.template Reduce</* ToBlocked = */ false>(inputs, outputs, reduction_op);
  }

  //! @rst
  //! Computes a warp-wide reduction for each batch in the calling warp using the specified binary reduction
  //! functor. The user must provide an output range of ``max_out_per_thread = ceil_div(Batches, LogicalWarpThreads)``
  //! items. Logical lane *i* stores results in its output range in a blocked manner:
  //! ``outputs[0]`` = result of batch ``i * max_out_per_thread``, ``outputs[1]`` = result of batch
  //! ``i * max_out_per_thread + 1``, etc.
  //! Items in the output range that have no corresponding input batch are invalid.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates reduction of 3 batches across 2 threads in every second logical warp
  //! (using ``cuda::std::array`` inputs and outputs):
  //!
  //! .. literalinclude:: ../../../cub/test/warp/catch2_test_warp_reduce_batched_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin warp-reduce-batched-reduce-to-blocked
  //!     :end-before: example-end warp-reduce-batched-reduce-to-blocked
  //!
  //! @endrst
  //!
  //! @tparam InputT
  //!   **[inferred]** The data type to be reduced having member
  //!   ``operator[](int i)`` and must be statically-sized (``size()`` method or static array)
  //!
  //! @tparam OutputT
  //!   **[inferred]** The data type to hold results having member
  //!   ``operator[](int i)`` and must be statically-sized (``size()`` method or static array)
  //!
  //! @tparam ReductionOp
  //!   **[inferred]** Binary reduction operator type having member
  //!   ``T operator()(const T &a, const T &b)``
  //!
  //! @param[in] inputs
  //!   Statically-sized input range holding ``Batches`` items
  //!
  //! @param[out] outputs
  //!   Statically-sized output range holding ``ceil_div(Batches, LogicalWarpThreads)`` items
  //!
  //! @param[in] reduction_op
  //!   Binary reduction operator
  template <typename InputT, typename OutputT, typename ReductionOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  ReduceToBlocked(const InputT& inputs, OutputT& outputs, ReductionOp reduction_op)
  {
    check_constraints<InputT, OutputT, ReductionOp>();
    InternalWarpReduceBatched{temp_storage}.template Reduce</* ToBlocked = */ true>(inputs, outputs, reduction_op);
  }

  //! @}
  //! @name Sum reductions
  //! @{
  //! @rst
  //! Computes a warp-wide sum for each batch in the calling warp.
  //! Thread ``i`` returns the result for batch ``i``.
  //! Returned items that have no corresponding input batch are invalid.
  //! For more batches than logical warp threads or generic code that could result in zero batches, use
  //! ``SumToStriped()`` or ``SumToBlocked()`` instead.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates reduction of 3 batches across 4 threads in each of 2 logical warps
  //!
  //! .. literalinclude:: ../../../cub/test/warp/catch2_test_warp_reduce_batched_api.cu
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
  //!   **[inferred]** The data type to be reduced having member
  //!   ``operator[](int i)`` and must be statically-sized (``size()`` method or static array)
  //!
  //! @param[in] inputs
  //!   Statically-sized input range holding ``Batches`` items
  //!
  //! @return
  //!   The sum of the input values of the batch corresponding to the logical lane.
  template <typename InputT>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Sum(const InputT& inputs)
  {
    return Reduce(inputs, ::cuda::std::plus<>{});
  }

  //! @rst
  //! Computes a warp-wide sum for each batch in the calling warp.
  //! The user must provide an output range of ``max_out_per_thread = ceil_div(Batches, LogicalWarpThreads)`` items.
  //! Logical lane ``i`` stores results in its output range in a striped manner:
  //! ``outputs[0]`` = result of batch ``i``, ``outputs[1]`` = result of batch ``i + LogicalWarpThreads``, etc.
  //! Items in the output range that have no corresponding input batch are invalid.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates reduction of 5 batches across 2 threads in each of 4 logical warps
  //! meaning more than one output per thread (using ``cuda::std::span`` inputs and outputs):
  //!
  //! .. literalinclude:: ../../../cub/test/warp/catch2_test_warp_reduce_batched_api.cu
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
  //!   **[inferred]** The data type to be reduced having member
  //!   ``operator[](int i)`` and must be statically-sized (``size()`` method or static array)
  //!
  //! @tparam OutputT
  //!   **[inferred]** The data type to hold results having member
  //!   ``operator[](int i)`` and must be statically-sized (``size()`` method or static array)
  //!
  //! @param[in] inputs
  //!   Statically-sized input range holding ``Batches`` items
  //!
  //! @param[out] outputs
  //!   Statically-sized output range holding ``ceil_div(Batches, LogicalWarpThreads)`` items
  template <typename InputT, typename OutputT>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void SumToStriped(const InputT& inputs, OutputT& outputs)
  {
    ReduceToStriped(inputs, outputs, ::cuda::std::plus<>{});
  }

  //! @rst
  //! Computes a warp-wide sum for each batch in the calling warp.
  //! The user must provide an output range of ``max_out_per_thread = ceil_div(Batches, LogicalWarpThreads)`` items.
  //! Logical lane *i* stores results in its output range in a blocked manner:
  //! ``outputs[0]`` = result of batch ``i * max_out_per_thread``, ``outputs[1]`` = result of batch
  //! ``i * max_out_per_thread + 1``, etc.
  //! Items in the output range that have no corresponding input batch are invalid.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! @smemwarpreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates reduction of 5 batches across 2 threads in each of 4 logical warps
  //! meaning more than one output per thread (using ``cuda::std::span`` inputs and outputs):
  //!
  //! .. literalinclude:: ../../../cub/test/warp/catch2_test_warp_reduce_batched_api.cu
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
  //!   **[inferred]** The data type to be reduced having member
  //!   ``operator[](int i)`` and must be statically-sized (``size()`` method or static array)
  //!
  //! @tparam OutputT
  //!   **[inferred]** The data type to hold results having member
  //!   ``operator[](int i)`` and must be statically-sized (``size()`` method or static array)
  //!
  //! @param[in] inputs
  //!   Statically-sized input range holding ``Batches`` items
  //!
  //! @param[out] outputs
  //!   Statically-sized output range holding ``ceil_div(Batches, LogicalWarpThreads)`` items
  template <typename InputT, typename OutputT>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void SumToBlocked(const InputT& inputs, OutputT& outputs)
  {
    ReduceToBlocked(inputs, outputs, ::cuda::std::plus<>{});
  }

  //! @}
};

CUB_NAMESPACE_END
