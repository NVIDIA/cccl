// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

//! @file
//! cub::DeviceSegmentedReduce provides device-wide, parallel operations for computing a batched reduction across
//! multiple sequences of data items residing within device-accessible memory.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/choose_offset.cuh>
#include <cub/detail/device_memory_resource.cuh>
#include <cub/detail/temporary_storage.cuh>
#include <cub/device/dispatch/dispatch_fixed_size_segmented_reduce.cuh>
#include <cub/device/dispatch/dispatch_segmented_reduce.cuh>
#include <cub/iterator/arg_index_input_iterator.cuh>
#include <cub/util_type.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>
#include <cuda/__functional/maximum.h>
#include <cuda/__functional/minimum.h>
#include <cuda/__memory_resource/get_memory_resource.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/pair.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

CUB_NAMESPACE_BEGIN

namespace detail
{
namespace segmented_reduce
{
struct get_tuning_query_t
{};

template <class Derived>
struct tuning
{
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto query(const get_tuning_query_t&) const noexcept -> Derived
  {
    return static_cast<const Derived&>(*this);
  }
};

struct default_tuning : tuning<default_tuning>
{
  template <class AccumT, class Offset, class OpT>
  using fn = detail::reduce::policy_hub<AccumT, Offset, OpT>;
};
} // namespace segmented_reduce
} // namespace detail

//! @rst
//! DeviceSegmentedReduce provides device-wide, parallel operations for
//! computing a reduction across multiple sequences of data items
//! residing within device-accessible memory.
//!
//! Overview
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! A `reduction <http://en.wikipedia.org/wiki/Reduce_(higher-order_function)>`_
//! (or *fold*) uses a binary combining operator to compute a single aggregate
//! from a sequence of input elements.
//!
//! Usage Considerations
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @cdp_class{DeviceSegmentedReduce}
//!
//! @endrst
struct DeviceSegmentedReduce
{
  //! @rst
  //! Computes a device-wide segmented reduction using the specified
  //! binary ``reduction_op`` functor.
  //!
  //! - Does not support binary reduction operators that are non-commutative.
  //! - Provides "run-to-run" determinism for pseudo-associative reduction
  //!   (e.g., addition of floating point types) on the same GPU device.
  //!   However, results for pseudo-associative reduction may be inconsistent
  //!   from one device to a another device of a different compute-capability
  //!   because CUB can employ different tile-sizing for different architectures.
  //! - When input a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - Let ``s`` be in ``[0, num_segments)``. The range
  //!   ``[d_out + d_begin_offsets[s], d_out + d_end_offsets[s])`` shall not
  //!   overlap ``[d_in + d_begin_offsets[s], d_in + d_end_offsets[s])``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates a custom min-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-reduce
  //!     :end-before: example-end segmented-reduce-reduce
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets @iterator
  //!
  //! @tparam ReductionOpT
  //!   **[inferred]** Binary reduction functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam T
  //!   **[inferred]** Data element type that is convertible to the `value` type of `InputIteratorT`
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_in``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the *i*\ :sup:`th` is considered empty.
  //!   @endrst
  //!
  //! @param[in] reduction_op
  //!   Binary reduction functor
  //!
  //! @param[in] initial_value
  //!   Initial value of the reduction for each segment
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename ReductionOpT,
            typename T>
  CUB_RUNTIME_FUNCTION static cudaError_t Reduce(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    ReductionOpT reduction_op,
    T initial_value,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::Reduce");

    using OffsetT = detail::common_iterator_value_t<BeginOffsetIteratorT, EndOffsetIteratorT>;
    static_assert(::cuda::std::is_integral_v<OffsetT>, "Offset iterator value type should be integral.");
    if constexpr (::cuda::std::is_integral_v<OffsetT>)
    {
      return DispatchSegmentedReduce<
        InputIteratorT,
        OutputIteratorT,
        BeginOffsetIteratorT,
        EndOffsetIteratorT,
        OffsetT,
        ReductionOpT,
        T>::Dispatch(d_temp_storage,
                     temp_storage_bytes,
                     d_in,
                     d_out,
                     num_segments,
                     d_begin_offsets,
                     d_end_offsets,
                     reduction_op,
                     initial_value, // zero-initialize
                     stream);
    }
    _CCCL_UNREACHABLE();
  }

  //! @rst
  //! Computes a device-wide segmented reduction using the specified
  //! binary ``reduction_op`` functor and a fixed segment size.
  //!
  //! - Does not support binary reduction operators that are non-commutative.
  //! - Provides "run-to-run" determinism for pseudo-associative reduction
  //!   (e.g., addition of floating point types) on the same GPU device.
  //!   However, results for pseudo-associative reduction may be inconsistent
  //!   from one device to a another device of a different compute-capability
  //!   because CUB can employ different tile-sizing for different architectures.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates a custom min-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin fixed-size-segmented-reduce-reduce
  //!     :end-before: example-end fixed-size-segmented-reduce-reduce
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam ReductionOpT
  //!   **[inferred]** Binary reduction functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam T
  //!   **[inferred]** Data element type that is convertible to the `value` type of `InputIteratorT`
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregates
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] segment_size
  //!   The fixed segment size of each segment
  //!
  //! @param[in] reduction_op
  //!   Binary reduction functor
  //!
  //! @param[in] initial_value
  //!   Initial value of the reduction for each segment
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename ReductionOpT, typename T>
  CUB_RUNTIME_FUNCTION static cudaError_t Reduce(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    int segment_size,
    ReductionOpT reduction_op,
    T initial_value,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::Reduce");

    // `offset_t` a.k.a `SegmentSizeT` is fixed to `int` type now, but later can be changed to accept
    // integral constant or larger integral types
    using offset_t = int;

    return detail::reduce::
      DispatchFixedSizeSegmentedReduce<InputIteratorT, OutputIteratorT, offset_t, ReductionOpT, T>::Dispatch(
        d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, segment_size, reduction_op, initial_value, stream);
  }

  //! @rst
  //! Computes a device-wide segmented sum using the addition (``+``) operator.
  //!
  //! - Uses ``0`` as the initial value of the reduction for each segment.
  //! - When input a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - Does not support ``+`` operators that are non-commutative.
  //! - Let ``s`` be in ``[0, num_segments)``. The range
  //!   ``[d_out + d_begin_offsets[s], d_out + d_end_offsets[s])`` shall not
  //!   overlap ``[d_in + d_begin_offsets[s], d_in + d_end_offsets[s])``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the sum reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-sum
  //!     :end-before: example-end segmented-reduce-sum
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments`, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_in``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the *i*\ :sup:`th` is considered empty.
  //!   @endrst
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename = ::cuda::std::void_t<typename ::cuda::std::iterator_traits<BeginOffsetIteratorT>::value_type,
                                           typename ::cuda::std::iterator_traits<EndOffsetIteratorT>::value_type>>
  CUB_RUNTIME_FUNCTION static cudaError_t
  Sum(void* d_temp_storage,
      size_t& temp_storage_bytes,
      InputIteratorT d_in,
      OutputIteratorT d_out,
      ::cuda::std::int64_t num_segments,
      BeginOffsetIteratorT d_begin_offsets,
      EndOffsetIteratorT d_end_offsets,
      cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::Sum");

    using OffsetT = detail::common_iterator_value_t<BeginOffsetIteratorT, EndOffsetIteratorT>;
    using OutputT = detail::non_void_value_t<OutputIteratorT, detail::it_value_t<InputIteratorT>>;
    using init_t  = OutputT;
    static_assert(::cuda::std::is_integral_v<OffsetT>, "Offset iterator value type should be integral.");
    if constexpr (::cuda::std::is_integral_v<OffsetT>)
    {
      return DispatchSegmentedReduce<
        InputIteratorT,
        OutputIteratorT,
        BeginOffsetIteratorT,
        EndOffsetIteratorT,
        OffsetT,
        ::cuda::std::plus<>,
        init_t>::Dispatch(d_temp_storage,
                          temp_storage_bytes,
                          d_in,
                          d_out,
                          num_segments,
                          d_begin_offsets,
                          d_end_offsets,
                          ::cuda::std::plus<>{},
                          init_t{}, // zero-initialize
                          stream);
    }
    _CCCL_UNREACHABLE();
  }

  //! @rst
  //! Computes a device-wide segmented sum using the addition (``+``) operator.
  //!
  //! - Uses ``0`` as the initial value of the reduction for each segment.
  //! - When input a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - Does not support ``+`` operators that are non-commutative.
  //! - Let ``s`` be in ``[0, num_segments)``. The range
  //!   ``[d_out + d_begin_offsets[s], d_out + d_end_offsets[s])`` shall not
  //!   overlap ``[d_in + d_begin_offsets[s], d_in + d_end_offsets[s])``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)``.
  //! - Can use a specific stream or cuda memory resource through the `env` parameter
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the sum reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-sum-env
  //!     :end-before: example-end segmented-reduce-sum-env
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets @iterator
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is `cuda::std::execution::env<>`.
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments`, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_in``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the *i*\ :sup:`th` is considered empty.
  //!   @endrst
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename      = ::cuda::std::void_t<typename ::cuda::std::iterator_traits<BeginOffsetIteratorT>::value_type,
                                                typename ::cuda::std::iterator_traits<EndOffsetIteratorT>::value_type>,
            typename EnvT = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t
  Sum(InputIteratorT d_in,
      OutputIteratorT d_out,
      ::cuda::std::int64_t num_segments,
      BeginOffsetIteratorT d_begin_offsets,
      EndOffsetIteratorT d_end_offsets,
      EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSegmentedReduce::Sum");

    using OffsetT = detail::common_iterator_value_t<BeginOffsetIteratorT, EndOffsetIteratorT>;
    using OutputT = detail::non_void_value_t<OutputIteratorT, detail::it_value_t<InputIteratorT>>;
    using init_t  = OutputT;
    using AccumT  = ::cuda::std::__accumulator_t<::cuda::std::plus<>, cub::detail::it_value_t<InputIteratorT>, init_t>;

    using segmented_reduce_tuning_t = ::cuda::std::execution::
      __query_result_or_t<EnvT, detail::segmented_reduce::get_tuning_query_t, detail::segmented_reduce::default_tuning>;

    using policy_t = typename segmented_reduce_tuning_t::template fn<AccumT, OffsetT, ::cuda::std::plus<>>;

    using requirements_t = ::cuda::std::execution::
      __query_result_or_t<EnvT, ::cuda::execution::__get_requirements_t, ::cuda::std::execution::env<>>;

    using requested_determinism_t =
      ::cuda::std::execution::__query_result_or_t<requirements_t, //
                                                  ::cuda::execution::determinism::__get_determinism_t,
                                                  ::cuda::execution::determinism::run_to_run_t>;

    using dispatch_t = DispatchSegmentedReduce<
      InputIteratorT,
      OutputIteratorT,
      BeginOffsetIteratorT,
      EndOffsetIteratorT,
      OffsetT,
      ::cuda::std::plus<>,
      init_t,
      AccumT,
      policy_t>;

    // Static assert to reject gpu_to_gpu determinism since it's not properly implemented atm
    static_assert(!::cuda::std::is_same_v<requested_determinism_t, ::cuda::execution::determinism::gpu_to_gpu_t>,
                  "gpu_to_gpu determinism is not supported for device segmented reductions ");

    static_assert(::cuda::std::is_integral_v<OffsetT>, "Offset iterator value type should be integral.");
    if constexpr (::cuda::std::is_integral_v<OffsetT>)
    {
      auto stream = ::cuda::std::execution::__query_or(env, ::cuda::get_stream, ::cuda::stream_ref{cudaStream_t{}});
      auto mr =
        ::cuda::std::execution::__query_or(env, ::cuda::mr::__get_memory_resource, detail::device_memory_resource{});

      void* d_temp_storage      = nullptr;
      size_t temp_storage_bytes = 0;

      // Query the required temporary storage size
      cudaError_t error = dispatch_t::Dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        num_segments,
        d_begin_offsets,
        d_end_offsets,
        ::cuda::std::plus<>{},
        init_t{}, // zero-initialize
        stream.get());
      if (error != cudaSuccess)
      {
        return error;
      }

      // TODO(gevtushenko): use uninitialized buffer whenit's available
      error = CubDebug(detail::temporary_storage::allocate(stream, d_temp_storage, temp_storage_bytes, mr));
      if (error != cudaSuccess)
      {
        return error;
      }

      // Run the algorithm
      error = dispatch_t::Dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        num_segments,
        d_begin_offsets,
        d_end_offsets,
        ::cuda::std::plus<>{},
        init_t{}, // zero-initialize
        stream.get());

      // Try to deallocate regardless of the error to avoid memory leaks
      cudaError_t deallocate_error =
        CubDebug(detail::temporary_storage::deallocate(stream, d_temp_storage, temp_storage_bytes, mr));

      if (error != cudaSuccess)
      {
        // Reduction error takes precedence over deallocation error since it happens first
        return error;
      }
      return deallocate_error;
    }
    _CCCL_UNREACHABLE();
  }

  //! @rst
  //! Computes a device-wide segmented sum using the addition (``+``) operator.
  //!
  //! - Uses ``0`` as the initial value of the reduction for each segment.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the sum reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin fixed-size-segmented-reduce-sum
  //!     :end-before: example-end fixed-size-segmented-reduce-sum
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] segment_size
  //!   The fixed segment size of each segment
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  Sum(void* d_temp_storage,
      size_t& temp_storage_bytes,
      InputIteratorT d_in,
      OutputIteratorT d_out,
      ::cuda::std::int64_t num_segments,
      int segment_size,
      cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::Sum");

    // `offset_t` a.k.a `SegmentSizeT` is fixed to `int` type now, but later can be changed to accept
    // integral constant or larger integral types
    using offset_t = int;
    using output_t = detail::non_void_value_t<OutputIteratorT, detail::it_value_t<InputIteratorT>>;

    return detail::reduce::
      DispatchFixedSizeSegmentedReduce<InputIteratorT, OutputIteratorT, offset_t, ::cuda::std::plus<>, output_t>::Dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        num_segments,
        segment_size,
        ::cuda::std::plus{},
        output_t{},
        stream);
  }

  //! @rst
  //! Computes a device-wide segmented minimum using the less-than (``<``) operator.
  //!
  //! - Uses ``::cuda::std::numeric_limits<T>::max()`` as the initial value of the reduction for each segment.
  //! - When input a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased for both
  //!   the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where the latter is
  //!   specified as ``segment_offsets + 1``).
  //! - Does not support ``<`` operators that are non-commutative.
  //! - Let ``s`` be in ``[0, num_segments)``. The range
  //!   ``[d_out + d_begin_offsets[s], d_out + d_end_offsets[s])`` shall not
  //!   overlap ``[d_in + d_begin_offsets[s], d_in + d_end_offsets[s])``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the min-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-custommin
  //!     :end-before: example-end segmented-reduce-custommin
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-min
  //!     :end-before: example-end segmented-reduce-min
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_in``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the *i*\ :sup:`th` is considered empty.
  //!   @endrst
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename = ::cuda::std::void_t<typename ::cuda::std::iterator_traits<BeginOffsetIteratorT>::value_type,
                                           typename ::cuda::std::iterator_traits<EndOffsetIteratorT>::value_type>>
  CUB_RUNTIME_FUNCTION static cudaError_t
  Min(void* d_temp_storage,
      size_t& temp_storage_bytes,
      InputIteratorT d_in,
      OutputIteratorT d_out,
      ::cuda::std::int64_t num_segments,
      BeginOffsetIteratorT d_begin_offsets,
      EndOffsetIteratorT d_end_offsets,
      cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::Min");

    using OffsetT = detail::common_iterator_value_t<BeginOffsetIteratorT, EndOffsetIteratorT>;
    using InputT  = detail::it_value_t<InputIteratorT>;
    using init_t  = InputT;
    static_assert(::cuda::std::is_integral_v<OffsetT>, "Offset iterator value type should be integral.");
    if constexpr (::cuda::std::is_integral_v<OffsetT>)
    {
      return DispatchSegmentedReduce<
        InputIteratorT,
        OutputIteratorT,
        BeginOffsetIteratorT,
        EndOffsetIteratorT,
        OffsetT,
        ::cuda::minimum<>,
        init_t>::Dispatch(d_temp_storage,
                          temp_storage_bytes,
                          d_in,
                          d_out,
                          num_segments,
                          d_begin_offsets,
                          d_end_offsets,
                          ::cuda::minimum<>{},
                          ::cuda::std::numeric_limits<init_t>::max(),
                          stream);
    }
    _CCCL_UNREACHABLE();
  }

  //! @rst
  //! Computes a device-wide segmented minimum using the less-than (``<``) operator.
  //!
  //! - Uses ``::cuda::std::numeric_limits<T>::max()`` as the initial value of the reduction for each segment.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the min-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-custommin
  //!     :end-before: example-end segmented-reduce-custommin
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin fixed-size-segmented-reduce-min
  //!     :end-before: example-end fixed-size-segmented-reduce-min
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] segment_size
  //!   The fixed segment size of each segment
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  Min(void* d_temp_storage,
      size_t& temp_storage_bytes,
      InputIteratorT d_in,
      OutputIteratorT d_out,
      ::cuda::std::int64_t num_segments,
      int segment_size,
      cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::Min");

    // `offset_t` a.k.a `SegmentSizeT` is fixed to `int` type now, but later can be changed to accept
    // integral constant or larger integral types
    using offset_t = int;
    using input_t  = detail::it_value_t<InputIteratorT>;

    return detail::reduce::
      DispatchFixedSizeSegmentedReduce<InputIteratorT, OutputIteratorT, offset_t, ::cuda::minimum<>, input_t>::Dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        num_segments,
        segment_size,
        ::cuda::minimum<>{},
        ::cuda::std::numeric_limits<input_t>::max(),
        stream);
  }

  //! @rst
  //! Finds the first device-wide minimum in each segment using the
  //! less-than (``<``) operator, also returning the in-segment index of that item.
  //!
  //! - The output value type of ``d_out`` is ``cub::KeyValuePair<int, T>``
  //!   (assuming the value type of ``d_in`` is ``T``)
  //!
  //!   - The minimum of the *i*\ :sup:`th` segment is written to
  //!     ``d_out[i].value`` and its offset in that segment is written to ``d_out[i].key``.
  //!   - The ``{1, ::cuda::std::numeric_limits<T>::max()}`` tuple is produced for zero-length inputs
  //!
  //! - When input a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased for both
  //!   the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where the latter
  //!   is specified as ``segment_offsets + 1``).
  //! - Does not support ``<`` operators that are non-commutative.
  //! - Let ``s`` be in ``[0, num_segments)``. The range
  //!   ``[d_out + d_begin_offsets[s], d_out + d_end_offsets[s])`` shall not
  //!   overlap ``[d_in + d_begin_offsets[s], d_in + d_end_offsets[s])``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the argmin-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-argmin
  //!     :end-before: example-end segmented-reduce-argmin
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items (of some type `T`) @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate
  //!   (having value type `KeyValuePair<int, T>`) @iterator
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   ending offsets @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_in``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the *i*\ :sup:`th` is considered empty.
  //!   @endrst
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename = ::cuda::std::void_t<typename ::cuda::std::iterator_traits<BeginOffsetIteratorT>::value_type,
                                           typename ::cuda::std::iterator_traits<EndOffsetIteratorT>::value_type>>
  CUB_RUNTIME_FUNCTION static cudaError_t ArgMin(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::ArgMin");

    // Using common iterator value type is a breaking change, see:
    // https://github.com/NVIDIA/cccl/pull/414#discussion_r1330632615
    using OffsetT = int; // detail::common_iterator_value_t<BeginOffsetIteratorT, EndOffsetIteratorT>;

    using InputValueT  = detail::it_value_t<InputIteratorT>;
    using OutputTupleT = detail::non_void_value_t<OutputIteratorT, KeyValuePair<OffsetT, InputValueT>>;
    using OutputKeyT   = typename OutputTupleT::Key;
    using OutputValueT = typename OutputTupleT::Value;
    using AccumT       = OutputTupleT;
    using InitT        = detail::reduce::empty_problem_init_t<AccumT>;

    static_assert(::cuda::std::is_same_v<int, OutputKeyT>, "Output key type must be int.");

    // Wrapped input iterator to produce index-value <OffsetT, InputT> tuples
    using ArgIndexInputIteratorT = ArgIndexInputIterator<InputIteratorT, OffsetT, OutputValueT>;
    ArgIndexInputIteratorT d_indexed_in(d_in);

    InitT initial_value{AccumT(1, ::cuda::std::numeric_limits<InputValueT>::max())};

    static_assert(::cuda::std::is_integral_v<OffsetT>, "Offset iterator value type should be integral.");
    if constexpr (::cuda::std::is_integral_v<OffsetT>)
    {
      return DispatchSegmentedReduce<
        ArgIndexInputIteratorT,
        OutputIteratorT,
        BeginOffsetIteratorT,
        EndOffsetIteratorT,
        OffsetT,
        cub::ArgMin,
        InitT,
        AccumT>::Dispatch(d_temp_storage,
                          temp_storage_bytes,
                          d_indexed_in,
                          d_out,
                          num_segments,
                          d_begin_offsets,
                          d_end_offsets,
                          cub::ArgMin{},
                          initial_value,
                          stream);
    }
    _CCCL_UNREACHABLE();
  }

  //! @rst
  //! Finds the first device-wide minimum in each segment using the
  //! less-than (``<``) operator, also returning the in-segment index of that item.
  //!
  //! - The output value type of ``d_out`` is ``::cuda::std::pair<int, T>``
  //!   (assuming the value type of ``d_in`` is ``T``)
  //!
  //!   - The minimum of the *i*\ :sup:`th` segment is written to
  //!     ``d_out[i].second`` and its offset in that segment is written to ``d_out[i].first``.
  //!   - The ``{1, ::cuda::std::numeric_limits<T>::max()}`` tuple is produced for zero-length inputs
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the argmin-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin fixed-size-segmented-reduce-argmin
  //!     :end-before: example-end fixed-size-segmented-reduce-argmin
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items (of some type `T`) @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate
  //!   (having value type `cuda::std::pair<int, T>`) @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] segment_size
  //!   The fixed segment size of each segment
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t ArgMin(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    int segment_size,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::ArgMin");

    // `offset_t` a.k.a `SegmentSizeT` is fixed to `int` type now, but later can be changed to accept
    // integral constant or larger integral types
    using offset_t = int;

    // The input type
    using input_value_t = cub::detail::it_value_t<InputIteratorT>;

    // The output tuple type
    using output_tuple_t = cub::detail::non_void_value_t<OutputIteratorT, ::cuda::std::pair<offset_t, input_value_t>>;

    using accum_t = output_tuple_t;

    using init_t = detail::reduce::empty_problem_init_t<accum_t>;

    using output_key_t   = typename output_tuple_t::first_type;
    using output_value_t = typename output_tuple_t::second_type;

    static_assert(::cuda::std::is_same_v<int, output_key_t>, "Output key type must be int.");

    // Wrapped input iterator to produce index-value <offset_t, InputT> tuples
    auto d_indexed_in = THRUST_NS_QUALIFIER::make_transform_iterator(
      THRUST_NS_QUALIFIER::counting_iterator<::cuda::std::int64_t>{0},
      detail::reduce::generate_idx_value<InputIteratorT, output_value_t>(d_in, segment_size));

    using arg_index_input_iterator_t = decltype(d_indexed_in);

    // Initial value
    init_t initial_value{accum_t(1, ::cuda::std::numeric_limits<input_value_t>::max())};

    return detail::reduce::DispatchFixedSizeSegmentedReduce<
      arg_index_input_iterator_t,
      OutputIteratorT,
      offset_t,
      cub::detail::arg_min,
      init_t,
      accum_t>::Dispatch(d_temp_storage,
                         temp_storage_bytes,
                         d_indexed_in,
                         d_out,
                         num_segments,
                         segment_size,
                         cub::detail::arg_min(),
                         initial_value,
                         stream);
  }

  //! @rst
  //! Computes a device-wide segmented maximum using the greater-than (``>``) operator.
  //!
  //! - Uses ``::cuda::std::numeric_limits<T>::lowest()`` as the initial value of the reduction.
  //! - When input a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - Does not support ``>`` operators that are non-commutative.
  //! - Let ``s`` be in ``[0, num_segments)``. The range
  //!   ``[d_out + d_begin_offsets[s], d_out + d_end_offsets[s])`` shall not
  //!   overlap ``[d_in + d_begin_offsets[s], d_in + d_end_offsets[s])``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the max-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-max
  //!     :end-before: example-end segmented-reduce-max
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_in``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the *i*\ :sup:`th` is considered empty.
  //!   @endrst
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename = ::cuda::std::void_t<typename ::cuda::std::iterator_traits<BeginOffsetIteratorT>::value_type,
                                           typename ::cuda::std::iterator_traits<EndOffsetIteratorT>::value_type>>
  CUB_RUNTIME_FUNCTION static cudaError_t
  Max(void* d_temp_storage,
      size_t& temp_storage_bytes,
      InputIteratorT d_in,
      OutputIteratorT d_out,
      ::cuda::std::int64_t num_segments,
      BeginOffsetIteratorT d_begin_offsets,
      EndOffsetIteratorT d_end_offsets,
      cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::Max");

    using OffsetT = detail::common_iterator_value_t<BeginOffsetIteratorT, EndOffsetIteratorT>;
    using InputT  = cub::detail::it_value_t<InputIteratorT>;
    using init_t  = InputT;

    static_assert(::cuda::std::is_integral_v<OffsetT>, "Offset iterator value type should be integral.");
    if constexpr (::cuda::std::is_integral_v<OffsetT>)
    {
      return DispatchSegmentedReduce<
        InputIteratorT,
        OutputIteratorT,
        BeginOffsetIteratorT,
        EndOffsetIteratorT,
        OffsetT,
        ::cuda::maximum<>,
        init_t>::Dispatch(d_temp_storage,
                          temp_storage_bytes,
                          d_in,
                          d_out,
                          num_segments,
                          d_begin_offsets,
                          d_end_offsets,
                          ::cuda::maximum<>{},
                          ::cuda::std::numeric_limits<init_t>::lowest(),
                          stream);
    }
    _CCCL_UNREACHABLE();
  }

  //! @rst
  //! Computes a device-wide segmented maximum using the greater-than (``>``) operator.
  //!
  //! - Uses ``::cuda::std::numeric_limits<T>::lowest()`` as the initial value of the reduction.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the max-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin fixed-size-segmented-reduce-max
  //!     :end-before: example-end fixed-size-segmented-reduce-max
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] segment_size
  //!   The fixed segment size of each segment
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  Max(void* d_temp_storage,
      size_t& temp_storage_bytes,
      InputIteratorT d_in,
      OutputIteratorT d_out,
      ::cuda::std::int64_t num_segments,
      int segment_size,
      cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::Max");

    // `offset_t` a.k.a `SegmentSizeT` is fixed to `int` type now, but later can be changed to accept
    // integral constant or larger integral types
    using offset_t = int;
    using input_t  = detail::it_value_t<InputIteratorT>;

    return detail::reduce::
      DispatchFixedSizeSegmentedReduce<InputIteratorT, OutputIteratorT, offset_t, ::cuda::maximum<>, input_t>::Dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        num_segments,
        segment_size,
        ::cuda::maximum<>{},
        ::cuda::std::numeric_limits<input_t>::lowest(),
        stream);
  }

  //! @rst
  //! Finds the first device-wide maximum in each segment using the
  //! greater-than (``>``) operator, also returning the in-segment index of that item
  //!
  //! - The output value type of ``d_out`` is ``cub::KeyValuePair<int, T>``
  //!   (assuming the value type of ``d_in`` is ``T``)
  //!
  //!   - The maximum of the *i*\ :sup:`th` segment is written to
  //!     ``d_out[i].value`` and its offset in that segment is written to ``d_out[i].key``.
  //!   - The ``{1, ::cuda::std::numeric_limits<T>::lowest()}`` tuple is produced for zero-length inputs
  //!
  //! - When input a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - Does not support ``>`` operators that are non-commutative.
  //! - Let ``s`` be in ``[0, num_segments)``. The range
  //!   ``[d_out + d_begin_offsets[s], d_out + d_end_offsets[s])`` shall not
  //!   overlap ``[d_in + d_begin_offsets[s], d_in + d_end_offsets[s])``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the argmax-reduction of a device vector
  //! of `int` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-argmax
  //!     :end-before: example-end segmented-reduce-argmax
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items
  //!   (of some type `T`) @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate
  //!   (having value type `KeyValuePair<int, T>`) @iterator
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   ending offsets @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work
  //!   is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length `num_segments`, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_in``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the *i*\ :sup:`th` is considered empty.
  //!   @endrst
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename = ::cuda::std::void_t<typename ::cuda::std::iterator_traits<BeginOffsetIteratorT>::value_type,
                                           typename ::cuda::std::iterator_traits<EndOffsetIteratorT>::value_type>>
  CUB_RUNTIME_FUNCTION static cudaError_t ArgMax(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::ArgMax");

    // Using common iterator value type is a breaking change, see:
    // https://github.com/NVIDIA/cccl/pull/414#discussion_r1330632615
    using OffsetT = int; // detail::common_iterator_value_t<BeginOffsetIteratorT, EndOffsetIteratorT>;

    using InputValueT  = cub::detail::it_value_t<InputIteratorT>;
    using OutputTupleT = cub::detail::non_void_value_t<OutputIteratorT, KeyValuePair<OffsetT, InputValueT>>;
    using AccumT       = OutputTupleT;
    using InitT        = detail::reduce::empty_problem_init_t<AccumT>;
    using OutputKeyT   = typename OutputTupleT::Key;
    using OutputValueT = typename OutputTupleT::Value;

    static_assert(::cuda::std::is_same_v<int, OutputKeyT>, "Output key type must be int.");

    // Wrapped input iterator to produce index-value <OffsetT, InputT> tuples
    using ArgIndexInputIteratorT = ArgIndexInputIterator<InputIteratorT, OffsetT, OutputValueT>;
    ArgIndexInputIteratorT d_indexed_in(d_in);

    InitT initial_value{AccumT(1, ::cuda::std::numeric_limits<InputValueT>::lowest())};

    static_assert(::cuda::std::is_integral_v<OffsetT>, "Offset iterator value type should be integral.");
    if constexpr (::cuda::std::is_integral_v<OffsetT>)
    {
      return DispatchSegmentedReduce<
        ArgIndexInputIteratorT,
        OutputIteratorT,
        BeginOffsetIteratorT,
        EndOffsetIteratorT,
        OffsetT,
        cub::ArgMax,
        InitT,
        AccumT>::Dispatch(d_temp_storage,
                          temp_storage_bytes,
                          d_indexed_in,
                          d_out,
                          num_segments,
                          d_begin_offsets,
                          d_end_offsets,
                          cub::ArgMax{},
                          initial_value,
                          stream);
    }
    _CCCL_UNREACHABLE();
  }

  //! @rst
  //! Finds the first device-wide maximum in each segment using the
  //! greater-than (``>``) operator, also returning the in-segment index of that item
  //!
  //! - The output value type of ``d_out`` is ``::cuda::std::pair<int, T>``
  //!   (assuming the value type of ``d_in`` is ``T``)
  //!
  //!   - The maximum of the *i*\ :sup:`th` segment is written to
  //!     ``d_out[i].second`` and its offset in that segment is written to ``d_out[i].first``.
  //!   - The ``{1, ::cuda::std::numeric_limits<T>::lowest()}`` tuple is produced for zero-length inputs
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the argmax-reduction of a device vector
  //! of `int` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin fixed-size-segmented-reduce-argmax
  //!     :end-before: example-end fixed-size-segmented-reduce-argmax
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items
  //!   (of some type `T`) @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate
  //!   (having value type `cuda::std::pair<int, T>`) @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work
  //!   is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] segment_size
  //!   The fixed segment size of each segment
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t ArgMax(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    int segment_size,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::ArgMax");

    // `offset_t` a.k.a `SegmentSizeT` is fixed to `int` type now, but later can be changed to accept
    // integral constant or larger integral types
    using input_t = int;

    using input_value_t  = detail::it_value_t<InputIteratorT>;
    using output_tuple_t = detail::non_void_value_t<OutputIteratorT, ::cuda::std::pair<input_t, input_value_t>>;
    using accum_t        = output_tuple_t;
    using init_t         = detail::reduce::empty_problem_init_t<accum_t>;
    using output_key_t   = typename output_tuple_t::first_type;
    using output_value_t = typename output_tuple_t::second_type;

    static_assert(::cuda::std::is_same_v<int, output_key_t>, "Output key type must be int.");

    // Wrapped input iterator to produce index-value <input_t, InputT> tuples
    auto d_indexed_in = THRUST_NS_QUALIFIER::make_transform_iterator(
      THRUST_NS_QUALIFIER::counting_iterator<::cuda::std::int64_t>{0},
      detail::reduce::generate_idx_value<InputIteratorT, output_value_t>(d_in, segment_size));
    using arg_index_input_iterator_t = decltype(d_indexed_in);

    init_t initial_value{accum_t(1, ::cuda::std::numeric_limits<input_value_t>::lowest())};

    return detail::reduce::DispatchFixedSizeSegmentedReduce<
      arg_index_input_iterator_t,
      OutputIteratorT,
      input_t,
      detail::arg_max,
      init_t,
      accum_t>::Dispatch(d_temp_storage,
                         temp_storage_bytes,
                         d_indexed_in,
                         d_out,
                         num_segments,
                         segment_size,
                         detail::arg_max(),
                         initial_value,
                         stream);
  }
};

CUB_NAMESPACE_END
