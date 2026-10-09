// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! cub::DeviceSegmentedScan provides device-wide, parallel operations for computing a batched prefix
//! scan across multiple sequences of data items residing within device-accessible memory.

#pragma once

#include <cub/config.cuh>

#ifndef CCCL_DISABLE_NVRTC_COMPATIBILITY_CHECK
#  if _CCCL_COMPILER(NVRTC)
#    error \
      "Including <cub/device/device_segmented_scan.cuh> is not supported when compiling with NVRTC. Include block-, warp-, or thread-level primitives instead (e.g. <cub/block/block_reduce.cuh>). You can define CCCL_DISABLE_NVRTC_COMPATIBILITY_CHECK to disable this warning."
#  endif // _CCCL_COMPILER(NVRTC)
#endif // CCCL_DISABLE_NVRTC_COMPATIBILITY_CHECK

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/env_dispatch.cuh>
#include <cub/detail/type_traits.cuh>
#include <cub/device/dispatch/dispatch_segmented_scan.cuh>

#include <cuda/__execution/tune.h>
#include <cuda/__functional/call_or.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/cstdint>
#include <cuda/std/optional>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_scan
{
struct get_segmented_scan_num_items_t
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class EnvT)
  _CCCL_REQUIRES(::cuda::std::execution::__queryable_with<EnvT, get_segmented_scan_num_items_t>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const EnvT& env) const noexcept
  {
    static_assert(noexcept(env.query(*this)));
    return env.query(*this);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr bool query(::cuda::std::execution::forwarding_query_t) noexcept
  {
    return true;
  }
};

struct get_segmented_scan_load_balancing_t
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class EnvT)
  _CCCL_REQUIRES(::cuda::std::execution::__queryable_with<EnvT, get_segmented_scan_load_balancing_t>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const EnvT& env) const noexcept
  {
    static_assert(noexcept(env.query(*this)));
    return env.query(*this);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr bool query(::cuda::std::execution::forwarding_query_t) noexcept
  {
    return true;
  }
};

struct load_balancing_t
{};

template <ForceInclusive EnforceInclusive,
          bool ReuseInputBegin,
          typename EnvT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename BeginOffsetIteratorInputT,
          typename EndOffsetIteratorInputT,
          typename BeginOffsetIteratorOutputT,
          typename ScanOpT,
          typename InitValueT,
          typename PolicySelector>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch_from_env(
  const EnvT& env,
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIteratorT d_in,
  OutputIteratorT d_out,
  ::cuda::std::int64_t num_segments,
  BeginOffsetIteratorInputT input_begin_offsets,
  EndOffsetIteratorInputT input_end_offsets,
  BeginOffsetIteratorOutputT output_begin_offsets,
  ScanOpT scan_op,
  InitValueT init_value,
  cudaStream_t stream,
  PolicySelector policy_selector)
{
  if constexpr (::cuda::std::execution::__queryable_with<EnvT, get_segmented_scan_load_balancing_t>)
  {
    ::cuda::std::optional<::cuda::std::int64_t> num_items;
    if constexpr (::cuda::std::execution::__queryable_with<EnvT, get_segmented_scan_num_items_t>)
    {
      num_items = get_segmented_scan_num_items_t{}(env);
    }

    using accum_t = deduced_accum_t<ScanOpT, InitValueT, it_value_t<InputIteratorT>>;
    using tuning_env_t =
      ::cuda::__call_result_or_t<::cuda::execution::__get_tuning_t, ::cuda::std::execution::env<>, EnvT>;
    using default_load_balanced_policy_selector_t = load_balanced_policy_selector_from_types<accum_t>;
    using load_balanced_policy_selector_t         = ::cuda::std::execution::
      __query_result_or_t<tuning_env_t, SegmentedScanLoadBalancedPolicy, default_load_balanced_policy_selector_t>;

    if constexpr (ReuseInputBegin)
    {
      return dispatch_load_balanced<EnforceInclusive>(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        num_segments,
        general_segments<BeginOffsetIteratorInputT, EndOffsetIteratorInputT>{input_begin_offsets, input_end_offsets, {}},
        scan_op,
        init_value,
        num_items,
        stream,
        policy_selector,
        load_balanced_policy_selector_t{});
    }
    else
    {
      return dispatch_load_balanced<EnforceInclusive>(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        num_segments,
        general_segments<BeginOffsetIteratorInputT, EndOffsetIteratorInputT, BeginOffsetIteratorOutputT>{
          input_begin_offsets, input_end_offsets, output_begin_offsets},
        scan_op,
        init_value,
        num_items,
        stream,
        policy_selector,
        load_balanced_policy_selector_t{});
    }
  }
  else if constexpr (::cuda::std::execution::__queryable_with<EnvT, get_segmented_scan_num_items_t>)
  {
    return dispatch_with_num_items<EnforceInclusive>(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_segments,
      input_begin_offsets,
      input_end_offsets,
      output_begin_offsets,
      scan_op,
      init_value,
      get_segmented_scan_num_items_t{}(env),
      stream,
      policy_selector);
  }
  else
  {
    return dispatch<EnforceInclusive>(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_segments,
      input_begin_offsets,
      input_end_offsets,
      output_begin_offsets,
      scan_op,
      init_value,
      1,
      worker::block,
      stream,
      policy_selector);
  }
}
} // namespace detail::segmented_scan

//! @rst
//! Creates an environment property that supplies the total number of items in all segments. The total is
//! ``sum(max(end[i] - begin[i], 0))``. When the mean segment length is at most half of a block's tile, supplying the
//! total lets one block scan ``floor(2 * tile size / mean segment length)`` consecutive segments, capped by the
//! active :cpp:struct:`cub::SegmentedScanPolicy`; the tile size is ``block.threads_per_block`` times
//! ``block.items_per_thread``. Supplying it is optional, but a supplied value must equal the total; otherwise the
//! behavior is undefined. A negative value returns ``cudaErrorInvalidValue``.
//!
//! .. versionadded:: 3.6.0
//! @endrst
[[nodiscard]] CUB_RUNTIME_FUNCTION inline auto segmented_scan_num_items(::cuda::std::int64_t num_items)
{
  return ::cuda::std::execution::prop{detail::segmented_scan::get_segmented_scan_num_items_t{}, num_items};
}

//! @rst
//! Environment property that opts into load balancing for skewed segment lengths, where a few segments contain most of
//! the items. It adds a pass over the segment offsets, kernel launches, and temporary storage, so it pays off for
//! moderate skew on large inputs, but on small inputs only under heavy skew.
//!
//! .. versionadded:: 3.6.0
//! @endrst
_CCCL_GLOBAL_CONSTANT auto segmented_scan_load_balancing = ::cuda::std::execution::prop{
  detail::segmented_scan::get_segmented_scan_load_balancing_t{}, detail::segmented_scan::load_balancing_t{}};

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
//! Choosing how segments are scheduled
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! By default, each segment is scanned by one thread block. This is the best choice when segment lengths are similar
//! or the input is small.
//!
//! - If the total number of items in all segments is known, pass it in the environment with
//!   ``cub::segmented_scan_num_items(n)``. When the mean segment length is at most half of a block's tile, one block
//!   scans ``floor(2 * tile size / mean segment length)`` consecutive segments, capped by the active policy. The
//!   tile size is ``block.threads_per_block`` times ``block.items_per_thread``. The value must equal the total;
//!   otherwise the behavior is undefined. A negative value returns ``cudaErrorInvalidValue``.
//! - If a few segments contain most of the items, opt into load balancing with
//!   ``cub::segmented_scan_load_balancing``. Items are divided evenly among thread blocks regardless of segment
//!   boundaries, so one long segment does not hold up the scan. This path costs an extra pass over the segment
//!   offsets, one or two extra kernel launches, and temporary storage of about ``num_segments`` offsets. Because of
//!   this fixed cost, it pays off for moderate skew on large inputs, but on small inputs only under heavy skew.
//!   ``num_items`` is optional on this path; supplying it trims the launch and temporary storage, and a supplied
//!   value must equal the total.
//! - With load balancing, a segment's floating-point sum may be grouped differently depending on the GPU and input
//!   size.
//! - Output segments must not overlap. Input segments may.
//!
//!  .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_scan_env_api.cu
//!      :language: c++
//!      :dedent:
//!      :start-after: example-begin segmented-scan-num-items
//!      :end-before: example-end segmented-scan-num-items
//!
//!  .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_scan_env_api.cu
//!      :language: c++
//!      :dedent:
//!      :start-after: example-begin segmented-scan-load-balancing
//!      :end-before: example-end segmented-scan-load-balancing
//!
//! Tuning
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! All algorithms in DeviceSegmentedScan that accept an environment can be tuned by passing a custom
//! :ref:`policy selector <cub-policy-selectors>` that returns a :cpp:struct:`cub::SegmentedScanPolicy`, as shown in
//! the example below:
//!
//!  .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_scan_env_api.cu
//!      :language: c++
//!      :dedent:
//!      :start-after: example-begin segmented-scan-policy-selector
//!      :end-before: example-end segmented-scan-policy-selector
//!
//!  .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_scan_env_api.cu
//!      :language: c++
//!      :dedent:
//!      :start-after: example-begin segmented-scan-tuning
//!      :end-before: example-end segmented-scan-tuning
//!
//! With ``cub::segmented_scan_load_balancing`` in the environment, the load-balanced algorithm is tuned by a policy
//! selector that returns a :cpp:struct:`cub::SegmentedScanLoadBalancedPolicy`. Pass it to ``cuda::execution::tune``,
//! alone or together with a selector returning a :cpp:struct:`cub::SegmentedScanPolicy`:
//!
//!  .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_scan_env_api.cu
//!      :language: c++
//!      :dedent:
//!      :start-after: example-begin segmented-scan-load-balanced-policy-selector
//!      :end-before: example-end segmented-scan-load-balanced-policy-selector
//!
//!  .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_scan_env_api.cu
//!      :language: c++
//!      :dedent:
//!      :start-after: example-begin segmented-scan-load-balanced-tuning
//!      :end-before: example-end segmented-scan-load-balanced-tuning
//!
//! @endrst
struct DeviceSegmentedScan
{
private:
  template <typename... It>
  CUB_RUNTIME_FUNCTION static void check_common_iterator_value_is_integral()
  {
    using offset_t = detail::common_iterator_value_t<It...>;
    static_assert(::cuda::std::is_integral_v<offset_t>, "Offset iterator value type should be integral.");
  }

public:
  //! @rst
  //! Computes a device-wide segmented exclusive prefix sum.
  //!
  //! - Results are not deterministic for computation of prefix sum on floating-point types
  //!   and may vary from run to run.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The input and output sequences
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! .. versionadded:: 3.3.0
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
  //!   @devicestorage
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
    cudaStream_t stream = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedScan::ExclusiveSegmentedSum");

    check_common_iterator_value_is_integral<BeginOffsetIteratorInputT, EndOffsetIteratorInputT>();

    using scan_op_t = ::cuda::std::plus<>;
    const scan_op_t scan_op{};

    using init_value_t = detail::it_value_t<InputIteratorT>;
    const init_value_t init_value{};

    return detail::segmented_scan::dispatch(
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
      1,
      detail::segmented_scan::worker::block,
      stream);
  }

  //! @rst
  //! Computes a device-wide segmented exclusive prefix sum.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - Results are not deterministic for computation of prefix sum on floating-point types
  //!   and may vary from run to run.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The input and output sequences
  //!   shall not overlap in any other way.
  //! - Can use a specific stream or cuda memory resource through the ``env`` parameter.
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
  //! The code snippet below illustrates an exclusive segmented prefix sum using a stream environment.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_scan_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin exclusive-segmented-sum-env
  //!     :end-before: example-end exclusive-segmented-sum-env
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
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
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
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   Accepts ``cub::segmented_scan_num_items`` and ``cub::segmented_scan_load_balancing``, described under
  //!   "Choosing how segments are scheduled" in the class documentation.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorInputT,
            typename EndOffsetIteratorInputT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveSegmentedSum(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT d_in_begin_offsets,
    EndOffsetIteratorInputT d_in_end_offsets,
    ::cuda::std::int64_t num_segments,
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSegmentedScan::ExclusiveSegmentedSum");

    check_common_iterator_value_is_integral<BeginOffsetIteratorInputT, EndOffsetIteratorInputT>();

    using scan_op_t = ::cuda::std::plus<>;
    scan_op_t scan_op{};

    using init_value_t = cub::detail::it_value_t<InputIteratorT>;
    init_value_t init_value{};

    using accum_t = detail::segmented_scan::
      deduced_accum_t<scan_op_t, detail::InputValue<init_value_t>, detail::it_value_t<InputIteratorT>>;

    using default_policy_selector = detail::segmented_scan::policy_selector_from_types<accum_t>;

    return detail::dispatch_with_env_and_tuning<default_policy_selector>(
      env, [&](auto policy_selector, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t stream) {
        return cub::detail::segmented_scan::dispatch_from_env<ForceInclusive::No, true>(
          env,
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
          stream,
          policy_selector);
      });
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
  //! .. versionadded:: 3.3.0
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive segmented prefix sum of an ``int``
  //! device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_scan_api.cu
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
  //!   @devicestorage
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
    cudaStream_t stream = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedScan::ExclusiveSegmentedSum");

    check_common_iterator_value_is_integral<BeginOffsetIteratorInputT,
                                            EndOffsetIteratorInputT,
                                            BeginOffsetIteratorOutputT>();

    using scan_op_t = ::cuda::std::plus<>;
    const scan_op_t scan_op{};

    using init_value_t = cub::detail::it_value_t<InputIteratorT>;
    const init_value_t init_value{};

    return cub::detail::segmented_scan::dispatch(
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
      1,
      detail::segmented_scan::worker::block,
      stream);
  }

  //! @rst
  //! Computes a device-wide segmented exclusive prefix sum with separate input and output offsets.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - Results are not deterministic for computation of prefix sum on floating-point types
  //!   and may vary from run to run.
  //! - Can use a specific stream or cuda memory resource through the ``env`` parameter.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates an exclusive segmented prefix sum with separate input
  //! and output offsets using a stream environment.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_scan_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin exclusive-segmented-sum-separate-env
  //!     :end-before: example-end exclusive-segmented-sum-separate-env
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
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
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
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   Accepts ``cub::segmented_scan_num_items`` and ``cub::segmented_scan_load_balancing``, described under
  //!   "Choosing how segments are scheduled" in the class documentation.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorInputT,
            typename EndOffsetIteratorInputT,
            typename BeginOffsetIteratorOutputT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveSegmentedSum(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT d_in_begin_offsets,
    EndOffsetIteratorInputT d_in_end_offsets,
    BeginOffsetIteratorOutputT d_out_begin_offsets,
    ::cuda::std::int64_t num_segments,
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSegmentedScan::ExclusiveSegmentedSum");

    check_common_iterator_value_is_integral<BeginOffsetIteratorInputT,
                                            EndOffsetIteratorInputT,
                                            BeginOffsetIteratorOutputT>();

    using scan_op_t = ::cuda::std::plus<>;
    scan_op_t scan_op{};

    using init_value_t = cub::detail::it_value_t<InputIteratorT>;
    init_value_t init_value{};

    using accum_t = detail::segmented_scan::
      deduced_accum_t<scan_op_t, detail::InputValue<init_value_t>, detail::it_value_t<InputIteratorT>>;

    using default_policy_selector = detail::segmented_scan::policy_selector_from_types<accum_t>;

    return detail::dispatch_with_env_and_tuning<default_policy_selector>(
      env, [&](auto policy_selector, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t stream) {
        return cub::detail::segmented_scan::dispatch_from_env<ForceInclusive::No, false>(
          env,
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
          stream,
          policy_selector);
      });
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
  //! .. versionadded:: 3.3.0
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive segmented prefix scan of an ``int``
  //! device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_scan_api.cu
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
  //!   @devicestorage
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
    cudaStream_t stream = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedScan::ExclusiveSegmentedScan");

    check_common_iterator_value_is_integral<BeginOffsetIteratorInputT, EndOffsetIteratorInputT>();

    return cub::detail::segmented_scan::dispatch(
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
      1,
      detail::segmented_scan::worker::block,
      stream);
  }

  //! @rst
  //! Computes a device-wide segmented exclusive prefix scan using the specified
  //! binary associative ``scan_op`` functor. The ``init_value`` value is applied as
  //! the initial value, and is assigned to the first element in each output segment.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The input and output sequences
  //!   shall not overlap in any other way.
  //! - Can use a specific stream or cuda memory resource through the ``env`` parameter.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates an exclusive segmented prefix scan with a
  //! user-supplied initial value using a stream environment.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_scan_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin exclusive-segmented-scan-env
  //!     :end-before: example-end exclusive-segmented-scan-env
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
  //!   **[inferred]** Type of the ``init_value``
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
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
  //! @param[in] scan_op
  //!   Binary associative scan functor
  //!
  //! @param[in] init_value
  //!   Initial value to seed the exclusive scan for each segment in the output sequence
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   Accepts ``cub::segmented_scan_num_items`` and ``cub::segmented_scan_load_balancing``, described under
  //!   "Choosing how segments are scheduled" in the class documentation.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorInputT,
            typename EndOffsetIteratorInputT,
            typename ScanOpT,
            typename InitValueT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveSegmentedScan(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT d_in_begin_offsets,
    EndOffsetIteratorInputT d_in_end_offsets,
    ::cuda::std::int64_t num_segments,
    ScanOpT scan_op,
    InitValueT init_value,
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSegmentedScan::ExclusiveSegmentedScan");

    check_common_iterator_value_is_integral<BeginOffsetIteratorInputT, EndOffsetIteratorInputT>();

    using accum_t = detail::segmented_scan::
      deduced_accum_t<ScanOpT, detail::InputValue<InitValueT>, detail::it_value_t<InputIteratorT>>;

    using default_policy_selector = detail::segmented_scan::policy_selector_from_types<accum_t>;

    return detail::dispatch_with_env_and_tuning<default_policy_selector>(
      env, [&](auto policy_selector, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t stream) {
        return cub::detail::segmented_scan::dispatch_from_env<ForceInclusive::No, true>(
          env,
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
          stream,
          policy_selector);
      });
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
  //! .. versionadded:: 3.3.0
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
  //!   @devicestorage
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
    cudaStream_t stream = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedScan::ExclusiveSegmentedScan");

    check_common_iterator_value_is_integral<BeginOffsetIteratorInputT,
                                            EndOffsetIteratorInputT,
                                            BeginOffsetIteratorOutputT>();

    return cub::detail::segmented_scan::dispatch(
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
      1,
      detail::segmented_scan::worker::block,
      stream);
  }

  //! @rst
  //! Computes a device-wide segmented exclusive prefix scan with separate input and output offsets
  //! using the specified binary associative ``scan_op`` functor. The ``init_value`` value is applied as
  //! the initial value, and is assigned to the first element in each output segment.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The input and output sequences
  //!   shall not overlap in any other way.
  //! - Can use a specific stream or cuda memory resource through the ``env`` parameter.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates an exclusive segmented prefix scan with separate input
  //! and output offsets and a user-supplied initial value using a stream environment.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_scan_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin exclusive-segmented-scan-separate-env
  //!     :end-before: example-end exclusive-segmented-scan-separate-env
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
  //!   **[inferred]** Type of the ``init_value``
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
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
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   Accepts ``cub::segmented_scan_num_items`` and ``cub::segmented_scan_load_balancing``, described under
  //!   "Choosing how segments are scheduled" in the class documentation.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorInputT,
            typename EndOffsetIteratorInputT,
            typename BeginOffsetIteratorOutputT,
            typename ScanOpT,
            typename InitValueT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveSegmentedScan(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT d_in_begin_offsets,
    EndOffsetIteratorInputT d_in_end_offsets,
    BeginOffsetIteratorOutputT d_out_begin_offsets,
    ::cuda::std::int64_t num_segments,
    ScanOpT scan_op,
    InitValueT init_value,
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSegmentedScan::ExclusiveSegmentedScan");

    check_common_iterator_value_is_integral<BeginOffsetIteratorInputT,
                                            EndOffsetIteratorInputT,
                                            BeginOffsetIteratorOutputT>();

    using accum_t = detail::segmented_scan::
      deduced_accum_t<ScanOpT, detail::InputValue<InitValueT>, detail::it_value_t<InputIteratorT>>;

    using default_policy_selector = detail::segmented_scan::policy_selector_from_types<accum_t>;

    return detail::dispatch_with_env_and_tuning<default_policy_selector>(
      env, [&](auto policy_selector, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t stream) {
        return cub::detail::segmented_scan::dispatch_from_env<ForceInclusive::No, false>(
          env,
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
          stream,
          policy_selector);
      });
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
  //! .. versionadded:: 3.3.0
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the inclusive segmented prefix sum of an ``int``
  //! device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_scan_api.cu
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
  //!   @devicestorage
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
    cudaStream_t stream = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedScan::InclusiveSegmentedSum");

    check_common_iterator_value_is_integral<BeginOffsetIteratorInputT, EndOffsetIteratorInputT>();

    using scan_op_t = ::cuda::std::plus<>;
    const scan_op_t scan_op{};

    return cub::detail::segmented_scan::dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_segments,
      d_in_begin_offsets,
      d_in_end_offsets,
      d_in_begin_offsets,
      scan_op,
      NullType(),
      1,
      detail::segmented_scan::worker::block,
      stream);
  }

  //! @rst
  //! Computes a device-wide segmented inclusive prefix sum.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - Results are not deterministic for computation of prefix sum on floating-point types
  //!   and may vary from run to run.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The input and output sequences
  //!   shall not overlap in any other way.
  //! - Can use a specific stream or cuda memory resource through the ``env`` parameter.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates an inclusive segmented prefix sum using a stream environment.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_scan_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin inclusive-segmented-sum-env
  //!     :end-before: example-end inclusive-segmented-sum-env
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
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
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
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   Accepts ``cub::segmented_scan_num_items`` and ``cub::segmented_scan_load_balancing``, described under
  //!   "Choosing how segments are scheduled" in the class documentation.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorInputT,
            typename EndOffsetIteratorInputT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t InclusiveSegmentedSum(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT d_in_begin_offsets,
    EndOffsetIteratorInputT d_in_end_offsets,
    ::cuda::std::int64_t num_segments,
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSegmentedScan::InclusiveSegmentedSum");

    check_common_iterator_value_is_integral<BeginOffsetIteratorInputT, EndOffsetIteratorInputT>();

    using scan_op_t = ::cuda::std::plus<>;
    scan_op_t scan_op{};

    using accum_t = detail::segmented_scan::deduced_accum_t<scan_op_t, NullType, detail::it_value_t<InputIteratorT>>;

    using default_policy_selector = detail::segmented_scan::policy_selector_from_types<accum_t>;

    return detail::dispatch_with_env_and_tuning<default_policy_selector>(
      env, [&](auto policy_selector, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t stream) {
        return cub::detail::segmented_scan::dispatch_from_env<ForceInclusive::No, true>(
          env,
          d_temp_storage,
          temp_storage_bytes,
          d_in,
          d_out,
          num_segments,
          d_in_begin_offsets,
          d_in_end_offsets,
          d_in_begin_offsets,
          scan_op,
          NullType(),
          stream,
          policy_selector);
      });
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
  //! .. versionadded:: 3.3.0
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the inclusive segmented prefix sum of an ``int``
  //! device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_scan_api.cu
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
  //!   @devicestorage
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
    cudaStream_t stream = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedScan::InclusiveSegmentedSum");

    check_common_iterator_value_is_integral<BeginOffsetIteratorInputT,
                                            EndOffsetIteratorInputT,
                                            BeginOffsetIteratorOutputT>();

    using scan_op_t = ::cuda::std::plus<>;
    const scan_op_t scan_op{};

    return cub::detail::segmented_scan::dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_segments,
      d_in_begin_offsets,
      d_in_end_offsets,
      d_out_begin_offsets,
      scan_op,
      NullType(),
      1,
      detail::segmented_scan::worker::block,
      stream);
  }

  //! @rst
  //! Computes a device-wide segmented inclusive prefix sum with separate input and output offsets.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - Results are not deterministic for computation of prefix sum on floating-point types
  //!   and may vary from run to run.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The input and output sequences
  //!   shall not overlap in any other way.
  //! - Can use a specific stream or cuda memory resource through the ``env`` parameter.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates an inclusive segmented prefix sum with separate input
  //! and output offsets using a stream environment.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_scan_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin inclusive-segmented-sum-separate-env
  //!     :end-before: example-end inclusive-segmented-sum-separate-env
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
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
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
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   Accepts ``cub::segmented_scan_num_items`` and ``cub::segmented_scan_load_balancing``, described under
  //!   "Choosing how segments are scheduled" in the class documentation.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorInputT,
            typename EndOffsetIteratorInputT,
            typename BeginOffsetIteratorOutputT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t InclusiveSegmentedSum(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT d_in_begin_offsets,
    EndOffsetIteratorInputT d_in_end_offsets,
    BeginOffsetIteratorOutputT d_out_begin_offsets,
    ::cuda::std::int64_t num_segments,
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSegmentedScan::InclusiveSegmentedSum");

    check_common_iterator_value_is_integral<BeginOffsetIteratorInputT,
                                            EndOffsetIteratorInputT,
                                            BeginOffsetIteratorOutputT>();

    using scan_op_t = ::cuda::std::plus<>;
    scan_op_t scan_op{};

    using accum_t = detail::segmented_scan::deduced_accum_t<scan_op_t, NullType, detail::it_value_t<InputIteratorT>>;

    using default_policy_selector = detail::segmented_scan::policy_selector_from_types<accum_t>;

    return detail::dispatch_with_env_and_tuning<default_policy_selector>(
      env, [&](auto policy_selector, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t stream) {
        return cub::detail::segmented_scan::dispatch_from_env<ForceInclusive::No, false>(
          env,
          d_temp_storage,
          temp_storage_bytes,
          d_in,
          d_out,
          num_segments,
          d_in_begin_offsets,
          d_in_end_offsets,
          d_out_begin_offsets,
          scan_op,
          NullType(),
          stream,
          policy_selector);
      });
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
  //! .. versionadded:: 3.3.0
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
  //!   @devicestorage
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
    cudaStream_t stream = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedScan::InclusiveSegmentedScan");

    check_common_iterator_value_is_integral<BeginOffsetIteratorInputT, EndOffsetIteratorInputT>();

    return cub::detail::segmented_scan::dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_segments,
      d_in_begin_offsets,
      d_in_end_offsets,
      d_in_begin_offsets,
      scan_op,
      NullType(),
      1,
      detail::segmented_scan::worker::block,
      stream);
  }

  //! @rst
  //! Computes a device-wide segmented inclusive prefix scan using the specified binary associative ``scan_op`` functor.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The input and output sequences
  //!   shall not overlap in any other way.
  //! - Can use a specific stream or cuda memory resource through the ``env`` parameter.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates an inclusive segmented prefix scan with a
  //! user-supplied scan operator using a stream environment.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_scan_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin inclusive-segmented-scan-env
  //!     :end-before: example-end inclusive-segmented-scan-env
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
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
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
  //! @param[in] scan_op
  //!   Binary associative scan functor
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   Accepts ``cub::segmented_scan_num_items`` and ``cub::segmented_scan_load_balancing``, described under
  //!   "Choosing how segments are scheduled" in the class documentation.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorInputT,
            typename EndOffsetIteratorInputT,
            typename ScanOpT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t InclusiveSegmentedScan(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT d_in_begin_offsets,
    EndOffsetIteratorInputT d_in_end_offsets,
    ::cuda::std::int64_t num_segments,
    ScanOpT scan_op,
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSegmentedScan::InclusiveSegmentedScan");

    check_common_iterator_value_is_integral<BeginOffsetIteratorInputT, EndOffsetIteratorInputT>();

    using accum_t = detail::segmented_scan::deduced_accum_t<ScanOpT, NullType, detail::it_value_t<InputIteratorT>>;

    using default_policy_selector = detail::segmented_scan::policy_selector_from_types<accum_t>;

    return detail::dispatch_with_env_and_tuning<default_policy_selector>(
      env, [&](auto policy_selector, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t stream) {
        return cub::detail::segmented_scan::dispatch_from_env<ForceInclusive::No, true>(
          env,
          d_temp_storage,
          temp_storage_bytes,
          d_in,
          d_out,
          num_segments,
          d_in_begin_offsets,
          d_in_end_offsets,
          d_in_begin_offsets,
          scan_op,
          NullType(),
          stream,
          policy_selector);
      });
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
  //! .. versionadded:: 3.3.0
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive segmented prefix sum of an ``int``
  //! device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_scan_api.cu
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
  //!   @devicestorage
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
    cudaStream_t stream = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedScan::InclusiveSegmentedScan");

    check_common_iterator_value_is_integral<BeginOffsetIteratorInputT,
                                            EndOffsetIteratorInputT,
                                            BeginOffsetIteratorOutputT>();

    return cub::detail::segmented_scan::dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_segments,
      d_in_begin_offsets,
      d_in_end_offsets,
      d_out_begin_offsets,
      scan_op,
      NullType(),
      1,
      detail::segmented_scan::worker::block,
      stream);
  }

  //! @rst
  //! Computes a device-wide segmented inclusive prefix scan with separate input and output offsets
  //! using the specified binary associative ``scan_op`` functor.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The input and output sequences
  //!   shall not overlap in any other way.
  //! - Can use a specific stream or cuda memory resource through the ``env`` parameter.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates an inclusive segmented prefix scan with separate input
  //! and output offsets and a user-supplied scan operator using a stream environment.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_scan_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin inclusive-segmented-scan-separate-env
  //!     :end-before: example-end inclusive-segmented-scan-separate-env
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
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
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
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   Accepts ``cub::segmented_scan_num_items`` and ``cub::segmented_scan_load_balancing``, described under
  //!   "Choosing how segments are scheduled" in the class documentation.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorInputT,
            typename EndOffsetIteratorInputT,
            typename BeginOffsetIteratorOutputT,
            typename ScanOpT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t InclusiveSegmentedScan(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT d_in_begin_offsets,
    EndOffsetIteratorInputT d_in_end_offsets,
    BeginOffsetIteratorOutputT d_out_begin_offsets,
    ::cuda::std::int64_t num_segments,
    ScanOpT scan_op,
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSegmentedScan::InclusiveSegmentedScan");

    check_common_iterator_value_is_integral<BeginOffsetIteratorInputT,
                                            EndOffsetIteratorInputT,
                                            BeginOffsetIteratorOutputT>();

    using accum_t = detail::segmented_scan::deduced_accum_t<ScanOpT, NullType, detail::it_value_t<InputIteratorT>>;

    using default_policy_selector = detail::segmented_scan::policy_selector_from_types<accum_t>;

    return detail::dispatch_with_env_and_tuning<default_policy_selector>(
      env, [&](auto policy_selector, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t stream) {
        return cub::detail::segmented_scan::dispatch_from_env<ForceInclusive::No, false>(
          env,
          d_temp_storage,
          temp_storage_bytes,
          d_in,
          d_out,
          num_segments,
          d_in_begin_offsets,
          d_in_end_offsets,
          d_out_begin_offsets,
          scan_op,
          NullType(),
          stream,
          policy_selector);
      });
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
  //! .. versionadded:: 3.3.0
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive segmented prefix scan of an ``int``
  //! device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_scan_api.cu
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
  //!   @devicestorage
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
    cudaStream_t stream = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedScan::InclusiveSegmentedScanInit");

    check_common_iterator_value_is_integral<BeginOffsetIteratorInputT, EndOffsetIteratorInputT>();
    static_assert(!::cuda::std::is_same_v<InitValueT, NullType>);

    return cub::detail::segmented_scan::dispatch<ForceInclusive::Yes>(
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
      1,
      detail::segmented_scan::worker::block,
      stream);
  }

  //! @rst
  //! Computes a device-wide segmented inclusive prefix scan using the specified binary associative ``scan_op`` functor.
  //! The result of applying the ``scan_op`` binary operator to ``init_value`` value and the first value in each input
  //! segment is assigned to the first value of the corresponding output segment.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The input and output sequences
  //!   shall not overlap in any other way.
  //! - Can use a specific stream or cuda memory resource through the ``env`` parameter.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates an inclusive segmented prefix scan with a
  //! user-supplied initial value using a stream environment.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_scan_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin inclusive-segmented-scan-init-env
  //!     :end-before: example-end inclusive-segmented-scan-init-env
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
  //!   **[inferred]** Type of the ``init_value``
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
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
  //!   Initial value to seed the inclusive scan for each segment
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   Accepts ``cub::segmented_scan_num_items`` and ``cub::segmented_scan_load_balancing``, described under
  //!   "Choosing how segments are scheduled" in the class documentation.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorInputT,
            typename EndOffsetIteratorInputT,
            typename ScanOpT,
            typename InitValueT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t InclusiveSegmentedScanInit(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT d_in_begin_offsets,
    EndOffsetIteratorInputT d_in_end_offsets,
    ::cuda::std::int64_t num_segments,
    ScanOpT scan_op,
    InitValueT init_value,
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSegmentedScan::InclusiveSegmentedScanInit");

    check_common_iterator_value_is_integral<BeginOffsetIteratorInputT, EndOffsetIteratorInputT>();
    static_assert(!::cuda::std::is_same_v<InitValueT, NullType>);

    using accum_t = detail::segmented_scan::
      deduced_accum_t<ScanOpT, detail::InputValue<InitValueT>, detail::it_value_t<InputIteratorT>>;

    using default_policy_selector = detail::segmented_scan::policy_selector_from_types<accum_t>;

    return detail::dispatch_with_env_and_tuning<default_policy_selector>(
      env, [&](auto policy_selector, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t stream) {
        return cub::detail::segmented_scan::dispatch_from_env<ForceInclusive::Yes, true>(
          env,
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
          stream,
          policy_selector);
      });
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
  //! .. versionadded:: 3.3.0
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
  //!   @devicestorage
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
    cudaStream_t stream = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedScan::InclusiveSegmentedScanInit");

    check_common_iterator_value_is_integral<BeginOffsetIteratorInputT,
                                            EndOffsetIteratorInputT,
                                            BeginOffsetIteratorOutputT>();
    static_assert(!::cuda::std::is_same_v<InitValueT, NullType>);

    return cub::detail::segmented_scan::dispatch<ForceInclusive::Yes>(
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
      1,
      detail::segmented_scan::worker::block,
      stream);
  }

  //! @rst
  //! Computes a device-wide segmented inclusive prefix scan with separate input and output offsets
  //! using the specified binary associative ``scan_op`` functor. The result of applying the ``scan_op``
  //! binary operator to ``init_value`` value and the first value in each input segment is assigned to
  //! the first value of the corresponding output segment.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The input and output sequences
  //!   shall not overlap in any other way.
  //! - Can use a specific stream or cuda memory resource through the ``env`` parameter.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates an inclusive segmented prefix scan with separate input
  //! and output offsets and a user-supplied initial value using a stream environment.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_scan_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin inclusive-segmented-scan-init-separate-env
  //!     :end-before: example-end inclusive-segmented-scan-init-separate-env
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
  //!   **[inferred]** Type of the ``init_value``
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
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
  //!   Initial value to seed the inclusive scan for each segment
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   Accepts ``cub::segmented_scan_num_items`` and ``cub::segmented_scan_load_balancing``, described under
  //!   "Choosing how segments are scheduled" in the class documentation.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorInputT,
            typename EndOffsetIteratorInputT,
            typename BeginOffsetIteratorOutputT,
            typename ScanOpT,
            typename InitValueT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t InclusiveSegmentedScanInit(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    BeginOffsetIteratorInputT d_in_begin_offsets,
    EndOffsetIteratorInputT d_in_end_offsets,
    BeginOffsetIteratorOutputT d_out_begin_offsets,
    ::cuda::std::int64_t num_segments,
    ScanOpT scan_op,
    InitValueT init_value,
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSegmentedScan::InclusiveSegmentedScanInit");

    check_common_iterator_value_is_integral<BeginOffsetIteratorInputT,
                                            EndOffsetIteratorInputT,
                                            BeginOffsetIteratorOutputT>();
    static_assert(!::cuda::std::is_same_v<InitValueT, NullType>);

    using accum_t = detail::segmented_scan::
      deduced_accum_t<ScanOpT, detail::InputValue<InitValueT>, detail::it_value_t<InputIteratorT>>;

    using default_policy_selector = detail::segmented_scan::policy_selector_from_types<accum_t>;

    return detail::dispatch_with_env_and_tuning<default_policy_selector>(
      env, [&](auto policy_selector, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t stream) {
        return cub::detail::segmented_scan::dispatch_from_env<ForceInclusive::Yes, false>(
          env,
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
          stream,
          policy_selector);
      });
  }
};

CUB_NAMESPACE_END
