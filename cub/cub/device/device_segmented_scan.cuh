// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! cub::DeviceSegmentedScan provides device-wide, parallel operations for computing a batched prefix
//! scan across multiple sequences of data items residing within device-accessible memory.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/dispatch/dispatch_segmented_scan.cuh>

#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

//! @rst
//! DeviceSegmentedScan provides device-wide, parallel operations for computing a
//! batched prefix scan across multiple sequences of data items residing within
//! device-accessible memory.
//!
//! Overview
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! Given a sequence of input elements and a binary reduction operator, a
//! `prefix scan <http://en.wikipedia.org/wiki/Prefix_sum>`_ produces an output
//! sequence where each element is computed to be the reduction of the elements
//! occurring earlier in the input sequence. *Prefix sum* connotes a prefix scan
//! with the addition operator. The term *inclusive* indicates that the
//! \ *i*\ :sup:`th` output reduction incorporates the \ *i*\ :sup:`th` input.
//! The term *exclusive* indicates the *i*\ :sup:`th` input is not
//! incorporated into the \ *i*\ :sup:`th` output reduction. When the input and
//! output sequences are the same, the scan is performed in-place.
//!
//! In order to provide an efficient parallel implementation, the binary reduction operator must be associative. That
//! is, ``op(op(a, b), c)`` must be equivalent to ``op(a, op(b, c))`` for any input values ``a``, ``b``, and ``c``.
//!
//! Usage Considerations
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @cdp_class{DeviceSegmentedScan}
//!
//! @endrst
struct DeviceSegmentedScan
{
  //! @rst
  //! Computes a device-wide segmented exclusive prefix sum.
  //!
  //! - Results are not deterministic for computation of prefix sum on floating-point types
  //!   and may vary from run to run.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The input and output sequences
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! Preconditions
  //! +++++++++++++
  //!
  //! - When ``d_in`` and ``d_out`` are equal, the segmented scan is performed in-place.
  //!   The range ``[d_in, d_in + num_items_in)`` and ``[d_out, d_out + num_items_out)``
  //!   shall not overlap in any other way.
  //! - ``d_in`` and ``d_out`` must not be null pointers
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive segmented prefix sum of an ``int``
  //! device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or, equivalently
  //!    // #include <cub/device/device_segmented_scan.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int  num_segments;   // e.g., 3
  //!    int  *d_in;          // e.g., [8, 6, 7, 5, 3, -2, 9]
  //!    int  *d_offsets;     // e.g., [0, 2, 5, 7]
  //!    int  *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::ExclusiveSegmentedSum(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, d_offsets, d_offsets + 1, num_segments);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run exclusive prefix sum
  //!    cub::DeviceScan::ExclusiveSegmentedSum(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, d_offsets, d_offsets + 1, num_segments);
  //!
  //!    // d_out <-- [0, 8, 0, 7, 12, 0, -2]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segmented scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing segmented scan outputs @iterator
  //!
  //! @tparam BeginOffsetIteratorInputT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets in the input data
  //!   sequence @iterator
  //!
  //! @tparam EndOffsetIteratorInputT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets in the input data sequence
  //!   @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] d_in_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_in_begin_offsets[i]`` is the first
  //!   element of the \ *i*\ :sup:`th` data segment in ``d_in`` and in ``d_out``.
  //!   @endrst
  //!
  //! @param[in] d_in_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_in_end_offsets[i] - 1`` is the last element of
  //!   the \ *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_in_end_offsets[i] - 1 <= d_in_begin_offsets[i]``, the \ *i*\ :sup:`th`
  //!   is considered empty.
  //!   @endrst
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented prefix scan data.
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorInputT,
            typename EndOffsetIteratorInputT>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveSegmentedSum(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT d_in_begin_offsets,
    EndOffsetIteratorInputT d_in_end_offsets,
    ::cuda::std::int64_t num_segments,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedScan::ExclusiveSegmentedSum");

    using offset_t              = detail::common_iterator_value_t<BeginOffsetIteratorInputT, EndOffsetIteratorInputT>;
    using integral_offset_check = ::cuda::std::is_integral<offset_t>;

    static_assert(integral_offset_check::value, "Offset iterator value type should be integral.");

    using scan_op_t = ::cuda::std::plus<>;
    scan_op_t scan_op{};

    using init_value_t = cub::detail::it_value_t<InputIteratorT>;
    init_value_t init_value{};

    return cub::detail::segmented_scan::dispatch_segmented_scan<
      InputIteratorT,
      OutputIteratorT,
      BeginOffsetIteratorInputT,
      EndOffsetIteratorInputT,
      BeginOffsetIteratorInputT,
      scan_op_t,
      detail::InputValue<init_value_t>>::
      dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        num_segments,
        d_in_begin_offsets,
        d_in_end_offsets,
        d_in_begin_offsets,
        scan_op,
        detail::InputValue<init_value_t>(init_value),
        stream);
  }

  //! @rst
  //! Computes a device-wide segmented exclusive prefix sum.
  //!
  //! - Results are not deterministic for computation of prefix sum on floating-point types
  //!   and may vary from run to run.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The input and output sequences
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive segmented prefix sum of an ``int``
  //! device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_segmented_scan_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin exclusive-segmented-sum-three-offsets
  //!     :end-before: example-end exclusive-segmented-sum-three-offsets
  //!
  //! @endrst
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segmented scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing segmented scan outputs @iterator
  //!
  //! @tparam BeginOffsetIteratorInputT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets in the input data
  //!   sequence @iterator
  //!
  //! @tparam EndOffsetIteratorInputT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets in the input data sequence
  //!   @iterator
  //!
  //! @tparam BeginOffsetIteratorOutputT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets in the output sequence
  //!   @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] d_in_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_in_begin_offsets[i]`` is the first
  //!   element of the \ *i*\ :sup:`th` data segment in ``d_in``
  //!   @endrst
  //!
  //! @param[in] d_in_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_in_end_offsets[i] - 1`` is the last element of
  //!   the \ *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_in_end_offsets[i] - 1 <= d_in_begin_offsets[i]``, the \ *i*\ :sup:`th`
  //!   is considered empty.
  //!   @endrst
  //!
  //! @param[in] d_out_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_out_begin_offsets[i]`` is the first
  //!   element of the \ *i*\ :sup:`th` data segment in ``d_out``
  //!   @endrst
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented prefix scan data.
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorInputT,
            typename EndOffsetIteratorInputT,
            typename BeginOffsetIteratorOutputT>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveSegmentedSum(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT d_in_begin_offsets,
    EndOffsetIteratorInputT d_in_end_offsets,
    BeginOffsetIteratorOutputT d_out_begin_offsets,
    ::cuda::std::int64_t num_segments,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedScan::ExclusiveSegmentedSum");

    using offset_t =
      detail::common_iterator_value_t<BeginOffsetIteratorInputT, EndOffsetIteratorInputT, BeginOffsetIteratorOutputT>;
    using integral_offset_check = ::cuda::std::is_integral<offset_t>;

    static_assert(integral_offset_check::value, "Offset iterator value type should be integral.");

    using scan_op_t = ::cuda::std::plus<>;
    scan_op_t scan_op{};

    using init_value_t = cub::detail::it_value_t<InputIteratorT>;
    init_value_t init_value{};

    return cub::detail::segmented_scan::dispatch_segmented_scan<
      InputIteratorT,
      OutputIteratorT,
      BeginOffsetIteratorInputT,
      EndOffsetIteratorInputT,
      BeginOffsetIteratorOutputT,
      scan_op_t,
      detail::InputValue<init_value_t>>::
      dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        num_segments,
        d_in_begin_offsets,
        d_in_end_offsets,
        d_out_begin_offsets,
        scan_op,
        detail::InputValue<init_value_t>(init_value),
        stream);
  }

  //! @rst
  //! Computes a device-wide segmented exclusive prefix scan using the specified
  //! binary associative ``scan_op`` functor. The ``init_value`` value is applied as
  //! the initial value, and is assigned to the first element in each output segment.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The input and output sequences
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive segmented prefix scan of an ``int``
  //! device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_segmented_scan_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin exclusive-segmented-scan-two-offsets
  //!     :end-before: example-end exclusive-segmented-scan-two-offsets
  //!
  //! @endrst
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segmented scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing segmented scan outputs @iterator
  //!
  //! @tparam BeginOffsetIteratorInputT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets in the input data
  //!   sequence @iterator
  //!
  //! @tparam EndOffsetIteratorInputT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets in the input data sequence
  //!   @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary associative scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam InitValueT
  //!  **[inferred]** Type of the `init_value`
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] d_in_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_in_begin_offsets[i]`` is the first
  //!   element of the \ *i*\ :sup:`th` data segment in ``d_in`` and in ``d_out``
  //!   @endrst
  //!
  //! @param[in] d_in_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_in_end_offsets[i] - 1`` is the last element of
  //!   the \ *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_in_end_offsets[i] - 1 <= d_in_begin_offsets[i]``, the \ *i*\ :sup:`th`
  //!   is considered empty.
  //!   @endrst
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented prefix scan data.
  //!
  //! @param[in] scan_op
  //!   Binary associative scan functor
  //!
  //! @param[in] init_value
  //!   Initial value to seed the exclusive scan for each segment in the output sequence
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorInputT,
            typename EndOffsetIteratorInputT,
            typename ScanOpT,
            typename InitValueT>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveSegmentedScan(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT d_in_begin_offsets,
    EndOffsetIteratorInputT d_in_end_offsets,
    ::cuda::std::int64_t num_segments,
    ScanOpT scan_op,
    InitValueT init_value,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedScan::ExclusiveSegmentedScan");

    using offset_t              = detail::common_iterator_value_t<BeginOffsetIteratorInputT, EndOffsetIteratorInputT>;
    using integral_offset_check = ::cuda::std::is_integral<offset_t>;

    static_assert(integral_offset_check::value, "Offset iterator value type should be integral.");

    return cub::detail::segmented_scan::dispatch_segmented_scan<
      InputIteratorT,
      OutputIteratorT,
      BeginOffsetIteratorInputT,
      EndOffsetIteratorInputT,
      BeginOffsetIteratorInputT,
      ScanOpT,
      detail::InputValue<InitValueT>>::
      dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        num_segments,
        d_in_begin_offsets,
        d_in_end_offsets,
        d_in_begin_offsets,
        scan_op,
        detail::InputValue<InitValueT>(init_value),
        stream);
  }

  //! @rst
  //! Computes a device-wide segmented exclusive prefix scan using the specified
  //! binary associative ``scan_op`` functor. The ``init_value`` value is applied as
  //! the initial value, and is assigned to the first element in each output segment.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The input and output sequences
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! @endrst
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segmented scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing segmented scan outputs @iterator
  //!
  //! @tparam BeginOffsetIteratorInputT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets in the input data
  //!   sequence @iterator
  //!
  //! @tparam EndOffsetIteratorInputT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets in the input data sequence
  //!   @iterator
  //!
  //! @tparam BeginOffsetIteratorOutputT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets in the output sequence
  //!   @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary associative scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam InitValueT
  //!  **[inferred]** Type of the `init_value`
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] d_in_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_in_begin_offsets[i]`` is the first
  //!   element of the \ *i*\ :sup:`th` data segment in ``d_in``
  //!   @endrst
  //!
  //! @param[in] d_in_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_in_end_offsets[i] - 1`` is the last element of
  //!   the \ *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_in_end_offsets[i] - 1 <= d_in_begin_offsets[i]``, the \ *i*\ :sup:`th`
  //!   is considered empty.
  //!   @endrst
  //!
  //! @param[in] d_out_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_out_begin_offsets[i]`` is the first
  //!   element of the \ *i*\ :sup:`th` data segment in ``d_out``
  //!   @endrst
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented prefix scan data.
  //!
  //! @param[in] scan_op
  //!   Binary associative scan functor
  //!
  //! @param[in] init_value
  //!   Initial value to seed the exclusive scan for each segment in the output sequence
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorInputT,
            typename EndOffsetIteratorInputT,
            typename BeginOffsetIteratorOutputT,
            typename ScanOpT,
            typename InitValueT>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveSegmentedScan(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT d_in_begin_offsets,
    EndOffsetIteratorInputT d_in_end_offsets,
    BeginOffsetIteratorOutputT d_out_begin_offsets,
    ::cuda::std::int64_t num_segments,
    ScanOpT scan_op,
    InitValueT init_value,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedScan::ExclusiveSegmentedScan");

    using offset_t =
      detail::common_iterator_value_t<BeginOffsetIteratorInputT, EndOffsetIteratorInputT, BeginOffsetIteratorOutputT>;
    using integral_offset_check = ::cuda::std::is_integral<offset_t>;

    static_assert(integral_offset_check::value, "Offset iterator value type should be integral.");

    return cub::detail::segmented_scan::dispatch_segmented_scan<
      InputIteratorT,
      OutputIteratorT,
      BeginOffsetIteratorInputT,
      EndOffsetIteratorInputT,
      BeginOffsetIteratorOutputT,
      ScanOpT,
      detail::InputValue<InitValueT>>::
      dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        num_segments,
        d_in_begin_offsets,
        d_in_end_offsets,
        d_out_begin_offsets,
        scan_op,
        detail::InputValue<InitValueT>(init_value),
        stream);
  }

  //! @rst
  //! Computes a device-wide segmented inclusive prefix sum.
  //!
  //! - Results are not deterministic for computation of prefix sum on floating-point types
  //!   and may vary from run to run.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The input and output sequences
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the inclusive segmented prefix sum of an ``int``
  //! device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_segmented_scan_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin inclusive-segmented-sum-two-offsets
  //!     :end-before: example-end inclusive-segmented-sum-two-offsets
  //!
  //! @endrst
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segmented scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing segmented scan outputs @iterator
  //!
  //! @tparam BeginOffsetIteratorInputT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets in the input data
  //!   sequence @iterator
  //!
  //! @tparam EndOffsetIteratorInputT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets in the input data sequence
  //!   @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary associative scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] d_in_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_in_begin_offsets[i]`` is the first
  //!   element of the \ *i*\ :sup:`th` data segment in ``d_in`` and in ``d_out``
  //!   @endrst
  //!
  //! @param[in] d_in_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_in_end_offsets[i] - 1`` is the last element of
  //!   the \ *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_in_end_offsets[i] - 1 <= d_in_begin_offsets[i]``, the \ *i*\ :sup:`th`
  //!   is considered empty.
  //!   @endrst
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented prefix scan data.
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorInputT,
            typename EndOffsetIteratorInputT>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveSegmentedSum(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT d_in_begin_offsets,
    EndOffsetIteratorInputT d_in_end_offsets,
    ::cuda::std::int64_t num_segments,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedScan::InclusiveSegmentedSum");

    using offset_t              = detail::common_iterator_value_t<BeginOffsetIteratorInputT, EndOffsetIteratorInputT>;
    using integral_offset_check = ::cuda::std::is_integral<offset_t>;

    static_assert(integral_offset_check::value, "Offset iterator value type should be integral.");

    using scan_op_t = ::cuda::std::plus<>;
    scan_op_t scan_op{};

    return cub::detail::segmented_scan::dispatch_segmented_scan<
      InputIteratorT,
      OutputIteratorT,
      BeginOffsetIteratorInputT,
      EndOffsetIteratorInputT,
      BeginOffsetIteratorInputT,
      scan_op_t,
      NullType>::dispatch(d_temp_storage,
                          temp_storage_bytes,
                          d_in,
                          d_out,
                          num_segments,
                          d_in_begin_offsets,
                          d_in_end_offsets,
                          d_in_begin_offsets,
                          scan_op,
                          NullType(),
                          stream);
  }

  //! @rst
  //! Computes a device-wide segmented inclusive prefix sum.
  //!
  //! - Results are not deterministic for computation of prefix sum on floating-point types
  //!   and may vary from run to run.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The input and output sequences
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the inclusive segmented prefix sum of an ``int``
  //! device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_segmented_scan_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin inclusive-segmented-sum-three-offsets
  //!     :end-before: example-end inclusive-segmented-sum-three-offsets
  //!
  //! @endrst
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segmented scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing segmented scan outputs @iterator
  //!
  //! @tparam BeginOffsetIteratorInputT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets in the input data
  //!   sequence @iterator
  //!
  //! @tparam EndOffsetIteratorInputT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets in the input data sequence
  //!   @iterator
  //!
  //! @tparam BeginOffsetIteratorOutputT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets in the output sequence
  //!   @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary associative scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] d_in_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_in_begin_offsets[i]`` is the first
  //!   element of the \ *i*\ :sup:`th` data segment in ``d_in``
  //!   @endrst
  //!
  //! @param[in] d_in_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_in_end_offsets[i] - 1`` is the last element of
  //!   the \ *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_in_end_offsets[i] - 1 <= d_in_begin_offsets[i]``, the \ *i*\ :sup:`th`
  //!   is considered empty.
  //!   @endrst
  //!
  //! @param[in] d_out_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_out_begin_offsets[i]`` is the first
  //!   element of the \ *i*\ :sup:`th` data segment in ``d_out``
  //!   @endrst
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented prefix scan data.
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorInputT,
            typename EndOffsetIteratorInputT,
            typename BeginOffsetIteratorOutputT>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveSegmentedSum(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT d_in_begin_offsets,
    EndOffsetIteratorInputT d_in_end_offsets,
    BeginOffsetIteratorOutputT d_out_begin_offsets,
    ::cuda::std::int64_t num_segments,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedScan::InclusiveSegmentedSum");

    using offset_t =
      detail::common_iterator_value_t<BeginOffsetIteratorInputT, EndOffsetIteratorInputT, BeginOffsetIteratorOutputT>;
    using integral_offset_check = ::cuda::std::is_integral<offset_t>;

    static_assert(integral_offset_check::value, "Offset iterator value type should be integral.");

    using scan_op_t = ::cuda::std::plus<>;
    scan_op_t scan_op{};

    return cub::detail::segmented_scan::dispatch_segmented_scan<
      InputIteratorT,
      OutputIteratorT,
      BeginOffsetIteratorInputT,
      EndOffsetIteratorInputT,
      BeginOffsetIteratorOutputT,
      scan_op_t,
      NullType>::dispatch(d_temp_storage,
                          temp_storage_bytes,
                          d_in,
                          d_out,
                          num_segments,
                          d_in_begin_offsets,
                          d_in_end_offsets,
                          d_out_begin_offsets,
                          scan_op,
                          NullType(),
                          stream);
  }

  //! @rst
  //! Computes a device-wide segmented inclusive prefix scan using the specified binary associative ``scan_op`` functor.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The input and output sequences
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! @endrst
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segmented scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing segmented scan outputs @iterator
  //!
  //! @tparam BeginOffsetIteratorInputT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets in the input data
  //!   sequence @iterator
  //!
  //! @tparam EndOffsetIteratorInputT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets in the input data sequence
  //!   @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary associative scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] d_in_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_in_begin_offsets[i]`` is the first
  //!   element of the \ *i*\ :sup:`th` data segment in ``d_in`` and in ``d_out``
  //!   @endrst
  //!
  //! @param[in] d_in_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_in_end_offsets[i] - 1`` is the last element of
  //!   the \ *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_in_end_offsets[i] - 1 <= d_in_begin_offsets[i]``, the \ *i*\ :sup:`th`
  //!   is considered empty.
  //!   @endrst
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented prefix scan data.
  //!
  //! @param[in] scan_op
  //!   Binary associative scan functor
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorInputT,
            typename EndOffsetIteratorInputT,
            typename ScanOpT>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveSegmentedScan(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT d_in_begin_offsets,
    EndOffsetIteratorInputT d_in_end_offsets,
    ::cuda::std::int64_t num_segments,
    ScanOpT scan_op,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedScan::InclusiveSegmentedScan");

    using offset_t              = detail::common_iterator_value_t<BeginOffsetIteratorInputT, EndOffsetIteratorInputT>;
    using integral_offset_check = ::cuda::std::is_integral<offset_t>;

    static_assert(integral_offset_check::value, "Offset iterator value type should be integral.");

    return cub::detail::segmented_scan::dispatch_segmented_scan<
      InputIteratorT,
      OutputIteratorT,
      BeginOffsetIteratorInputT,
      EndOffsetIteratorInputT,
      BeginOffsetIteratorInputT,
      ScanOpT,
      NullType>::dispatch(d_temp_storage,
                          temp_storage_bytes,
                          d_in,
                          d_out,
                          num_segments,
                          d_in_begin_offsets,
                          d_in_end_offsets,
                          d_in_begin_offsets,
                          scan_op,
                          NullType(),
                          stream);
  }

  //! @rst
  //! Computes a device-wide segmented inclusive prefix scan using the specified binary associative ``scan_op`` functor.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The input and output sequences
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive segmented prefix sum of an ``int``
  //! device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_segmented_scan_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin inclusive-segmented-scan-three-offsets
  //!     :end-before: example-end inclusive-segmented-scan-three-offsets
  //!
  //! @endrst
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segmented scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing segmented scan outputs @iterator
  //!
  //! @tparam BeginOffsetIteratorInputT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets in the input data
  //!   sequence @iterator
  //!
  //! @tparam EndOffsetIteratorInputT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets in the input data sequence
  //!   @iterator
  //!
  //! @tparam BeginOffsetIteratorOutputT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets in the output sequence
  //!   @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary associative scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] d_in_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_in_begin_offsets[i]`` is the first
  //!   element of the \ *i*\ :sup:`th` data segment in ``d_in``
  //!   @endrst
  //!
  //! @param[in] d_in_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_in_end_offsets[i] - 1`` is the last element of
  //!   the \ *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_in_end_offsets[i] - 1 <= d_in_begin_offsets[i]``, the \ *i*\ :sup:`th`
  //!   is considered empty.
  //!   @endrst
  //!
  //! @param[in] d_out_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_out_begin_offsets[i]`` is the first
  //!   element of the \ *i*\ :sup:`th` data segment in ``d_out``
  //!   @endrst
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented prefix scan data.
  //!
  //! @param[in] scan_op
  //!   Binary associative scan functor
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorInputT,
            typename EndOffsetIteratorInputT,
            typename BeginOffsetIteratorOutputT,
            typename ScanOpT>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveSegmentedScan(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT d_in_begin_offsets,
    EndOffsetIteratorInputT d_in_end_offsets,
    BeginOffsetIteratorOutputT d_out_begin_offsets,
    ::cuda::std::int64_t num_segments,
    ScanOpT scan_op,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedScan::InclusiveSegmentedScan");

    using offset_t =
      detail::common_iterator_value_t<BeginOffsetIteratorInputT, EndOffsetIteratorInputT, BeginOffsetIteratorOutputT>;
    using integral_offset_check = ::cuda::std::is_integral<offset_t>;

    static_assert(integral_offset_check::value, "Offset iterator value type should be integral.");

    return cub::detail::segmented_scan::dispatch_segmented_scan<
      InputIteratorT,
      OutputIteratorT,
      BeginOffsetIteratorInputT,
      EndOffsetIteratorInputT,
      BeginOffsetIteratorOutputT,
      ScanOpT,
      NullType>::dispatch(d_temp_storage,
                          temp_storage_bytes,
                          d_in,
                          d_out,
                          num_segments,
                          d_in_begin_offsets,
                          d_in_end_offsets,
                          d_out_begin_offsets,
                          scan_op,
                          NullType(),
                          stream);
  }

  //! @rst
  //! Computes a device-wide segmented inclusive prefix scan using the specified binary associative ``scan_op`` functor.
  //! The result of applying the ``scan_op`` binary operator to ``init_value`` value and the first value in each input
  //! segment is assigned to the first value of the corresponding output segment.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The input and output sequences
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive segmented prefix scan of an ``int``
  //! device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_segmented_scan_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin inclusive-segmented-scan-init-two-offsets
  //!     :end-before: example-end inclusive-segmented-scan-init-two-offsets
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segmented scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing segmented scan outputs @iterator
  //!
  //! @tparam BeginOffsetIteratorInputT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets in the input data
  //!   sequence @iterator
  //!
  //! @tparam EndOffsetIteratorInputT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets in the input data sequence
  //!   @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary associative scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam InitValueT
  //!  **[inferred]** Type of the `init_value`
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] d_in_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_in_begin_offsets[i]`` is the first
  //!   element of the \ *i*\ :sup:`th` data segment in ``d_in`` and in ``d_out``
  //!   @endrst
  //!
  //! @param[in] d_in_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_in_end_offsets[i] - 1`` is the last element of
  //!   the \ *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_in_end_offsets[i] - 1 <= d_in_begin_offsets[i]``, the \ *i*\ :sup:`th`
  //!   is considered empty.
  //!   @endrst
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented prefix scan data.
  //!
  //! @param[in] scan_op
  //!   Binary associative scan functor
  //!
  //! @param[in] init_value
  //!   Initial value to seed the exclusive scan for each segment in the output sequence
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorInputT,
            typename EndOffsetIteratorInputT,
            typename ScanOpT,
            typename InitValueT>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveSegmentedScanInit(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT d_in_begin_offsets,
    EndOffsetIteratorInputT d_in_end_offsets,
    ::cuda::std::int64_t num_segments,
    ScanOpT scan_op,
    InitValueT init_value,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedScan::InclusiveSegmentedScanInit");

    using offset_t              = detail::common_iterator_value_t<BeginOffsetIteratorInputT, EndOffsetIteratorInputT>;
    using integral_offset_check = ::cuda::std::is_integral<offset_t>;

    static_assert(integral_offset_check::value, "Offset iterator value type should be integral.");
    static_assert(!::cuda::std::is_same_v<InitValueT, NullType>);

    using accum_t = ::cuda::std::__accumulator_t<ScanOpT, cub::detail::it_value_t<InputIteratorT>, InitValueT>;

    return cub::detail::segmented_scan::dispatch_segmented_scan<
      InputIteratorT,
      OutputIteratorT,
      BeginOffsetIteratorInputT,
      EndOffsetIteratorInputT,
      BeginOffsetIteratorInputT,
      ScanOpT,
      detail::InputValue<InitValueT>,
      accum_t,
      ForceInclusive::Yes>::dispatch(d_temp_storage,
                                     temp_storage_bytes,
                                     d_in,
                                     d_out,
                                     num_segments,
                                     d_in_begin_offsets,
                                     d_in_end_offsets,
                                     d_in_begin_offsets,
                                     scan_op,
                                     detail::InputValue<InitValueT>(init_value),
                                     stream);
  }

  //! @rst
  //! Computes a device-wide segmented inclusive prefix scan using the specified binary associative ``scan_op`` functor.
  //! The result of applying the ``scan_op`` binary operator to ``init_value`` value and the first value in each input
  //! segment is assigned to the first value of the corresponding output segment.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The input and output sequences
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segmented scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing segmented scan outputs @iterator
  //!
  //! @tparam BeginOffsetIteratorInputT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets in the input data
  //!   sequence @iterator
  //!
  //! @tparam EndOffsetIteratorInputT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets in the input data sequence
  //!   @iterator
  //!
  //! @tparam BeginOffsetIteratorOutputT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets in the output sequence
  //!   @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary associative scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam InitValueT
  //!  **[inferred]** Type of the `init_value`
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] d_in_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_in_begin_offsets[i]`` is the first
  //!   element of the \ *i*\ :sup:`th` data segment in ``d_in``
  //!   @endrst
  //!
  //! @param[in] d_in_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_in_end_offsets[i] - 1`` is the last element of
  //!   the \ *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_in_end_offsets[i] - 1 <= d_in_begin_offsets[i]``, the \ *i*\ :sup:`th`
  //!   is considered empty.
  //!   @endrst
  //!
  //! @param[in] d_out_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_out_begin_offsets[i]`` is the first
  //!   element of the \ *i*\ :sup:`th` data segment in ``d_out``
  //!   @endrst
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented prefix scan data.
  //!
  //! @param[in] scan_op
  //!   Binary associative scan functor
  //!
  //! @param[in] init_value
  //!   Initial value to seed the exclusive scan for each segment in the output sequence
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorInputT,
            typename EndOffsetIteratorInputT,
            typename BeginOffsetIteratorOutputT,
            typename ScanOpT,
            typename InitValueT>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveSegmentedScanInit(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT d_in_begin_offsets,
    EndOffsetIteratorInputT d_in_end_offsets,
    BeginOffsetIteratorOutputT d_out_begin_offsets,
    ::cuda::std::int64_t num_segments,
    ScanOpT scan_op,
    InitValueT init_value,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedScan::InclusiveSegmentedScanInit");

    using offset_t =
      detail::common_iterator_value_t<BeginOffsetIteratorInputT, EndOffsetIteratorInputT, BeginOffsetIteratorOutputT>;
    using integral_offset_check = ::cuda::std::is_integral<offset_t>;

    static_assert(integral_offset_check::value, "Offset iterator value type should be integral.");
    static_assert(!::cuda::std::is_same_v<InitValueT, NullType>);

    using accum_t = ::cuda::std::__accumulator_t<ScanOpT, cub::detail::it_value_t<InputIteratorT>, InitValueT>;

    return cub::detail::segmented_scan::dispatch_segmented_scan<
      InputIteratorT,
      OutputIteratorT,
      BeginOffsetIteratorInputT,
      EndOffsetIteratorInputT,
      BeginOffsetIteratorOutputT,
      ScanOpT,
      detail::InputValue<InitValueT>,
      accum_t,
      ForceInclusive::Yes>::dispatch(d_temp_storage,
                                     temp_storage_bytes,
                                     d_in,
                                     d_out,
                                     num_segments,
                                     d_in_begin_offsets,
                                     d_in_end_offsets,
                                     d_out_begin_offsets,
                                     scan_op,
                                     detail::InputValue<InitValueT>(init_value),
                                     stream);
  }
};

CUB_NAMESPACE_END
