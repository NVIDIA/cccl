// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! cub::detail::rld provides the dispatch layer of cub::DeviceRunLengthDecode. Both algorithms are expressed as a
//! batched copy (cub::detail::batch_memcpy::dispatch) that copies run i from an iterator repeating the i-th run value
//! to the output position of the i-th run. The run-length based variant first computes these output positions with an
//! exclusive scan (cub::detail::scan::dispatch) over the run lengths.

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
#include <cub/detail/temporary_storage.cuh>
#include <cub/device/dispatch/dispatch_batch_memcpy.cuh>
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/device/dispatch/tuning/tuning_batch_memcpy.cuh>
#include <cub/device/dispatch/tuning/tuning_scan.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_type.cuh>

#include <cuda/__iterator/constant_iterator.h>
#include <cuda/__iterator/counting_iterator.h>
#include <cuda/__iterator/transform_iterator.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::rld
{
//! Type of the output offsets computed from the run lengths. It is 64-bit independently of the run length type, since
//! the sum of many 32-bit run lengths can exceed 2^32.
using decoded_offset_t = ::cuda::std::int64_t;

//! Type of the number of runs handed to the scan dispatch, which requires an unsigned offset type.
using scan_offset_t = choose_offset_t<::cuda::std::int64_t>;

//! Integer type large enough to hold any offset in [0, num_thread_blocks_launched) of the batched copy, see
//! cub::DeviceCopy::Batched.
using copy_block_offset_t = ::cuda::std::uint32_t;

//! Maps a run value to the source range of the run, which repeats the run value.
template <typename ValueT>
struct run_value_to_source_op
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API ::cuda::constant_iterator<ValueT> operator()(ValueT run_value) const
  {
    return ::cuda::constant_iterator<ValueT>{run_value};
  }
};

//! Maps the output offset of a run to the output iterator positioned at the first decoded item of that run.
template <typename OutputIteratorT, typename RunOffsetT>
struct run_offset_to_destination_op
{
  OutputIteratorT d_out;

  [[nodiscard]] _CCCL_HOST_DEVICE_API OutputIteratorT operator()(RunOffsetT run_offset) const
  {
    return d_out + run_offset;
  }
};

//! Computes the length of run i from the run offsets as d_run_offsets[i + 1] - d_run_offsets[i].
template <typename RunOffsetsIteratorT, typename RunSizeT>
struct run_offsets_to_length_op
{
  RunOffsetsIteratorT d_run_offsets;

  [[nodiscard]] _CCCL_HOST_DEVICE_API RunSizeT operator()(::cuda::std::int64_t run) const
  {
    using run_offset_t       = it_value_t<RunOffsetsIteratorT>;
    const run_offset_t begin = d_run_offsets[run];
    const run_offset_t end   = d_run_offsets[run + 1];
    return static_cast<RunSizeT>(end - begin);
  }
};

//! Copies the i-th run value to the output range given by the i-th destination iterator and the i-th run length.
//! @p TuningEnvT may carry a policy selector returning a cub::BatchedCopyPolicy.
template <typename RunValuesIteratorT, typename DestinationsIteratorT, typename RunLengthsIteratorT, typename TuningEnvT>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t copy_runs(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  RunValuesIteratorT d_run_values,
  DestinationsIteratorT d_destinations,
  RunLengthsIteratorT d_run_lengths,
  ::cuda::std::int64_t num_runs,
  cudaStream_t stream,
  const TuningEnvT&)
{
  using value_t = it_value_t<RunValuesIteratorT>;
  using policy_selector_t =
    ::cuda::std::execution::__query_result_or_t<TuningEnvT, BatchedCopyPolicy, batch_memcpy::policy_selector>;
  return batch_memcpy::dispatch<CopyAlg::Copy, copy_block_offset_t>(
    d_temp_storage,
    temp_storage_bytes,
    ::cuda::make_transform_iterator(d_run_values, run_value_to_source_op<value_t>{}),
    d_destinations,
    d_run_lengths,
    num_runs,
    stream,
    policy_selector_t{});
}

//! Run-length decode from run values and run lengths. The output offsets of the runs are computed into temporary
//! storage by an exclusive scan over the run lengths before the runs are copied.
//!
//! @p TuningEnvT may carry a policy selector returning a cub::ScanPolicy, applied to the scan, and one returning a
//! cub::BatchedCopyPolicy, applied to the copy.
template <typename RunValuesIteratorT,
          typename RunLengthsIteratorT,
          typename OutputIteratorT,
          typename TuningEnvT = ::cuda::std::execution::env<>>
CUB_RUNTIME_FUNCTION cudaError_t dispatch(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  RunValuesIteratorT d_run_values,
  RunLengthsIteratorT d_run_lengths,
  OutputIteratorT d_out,
  ::cuda::std::int64_t num_runs,
  cudaStream_t stream,
  const TuningEnvT& tuning_env = {})
{
  // Type of the run lengths as seen by the batched copy
  using run_size_t = choose_offset_t<it_value_t<RunLengthsIteratorT>>;

  using destination_op_t = run_offset_to_destination_op<OutputIteratorT, decoded_offset_t>;

  // The offsets are accumulated in decoded_offset_t, since their sum can exceed the range of the run length type
  using scan_op_t   = ::cuda::std::plus<decoded_offset_t>;
  using scan_init_t = InputValue<decoded_offset_t>;
  using default_scan_policy_selector_t =
    scan::policy_selector_from_types<RunLengthsIteratorT, decoded_offset_t*, decoded_offset_t, scan_offset_t, scan_op_t>;
  using scan_policy_selector_t =
    ::cuda::std::execution::__query_result_or_t<TuningEnvT, ScanPolicy, default_scan_policy_selector_t>;

  const auto d_run_sizes = ::cuda::make_transform_iterator(d_run_lengths, CastOp<run_size_t>{});

  // The run offsets are live during both the scan and the copy. The scan's and the copy's own temporary storage are
  // only used one after another in stream order and can therefore alias each other.
  temporary_storage::layout<2> storage_layout;
  const auto run_offsets_alloc =
    storage_layout.get_slot(0)->create_alias<decoded_offset_t>(static_cast<size_t>(num_runs));
  auto scan_storage_alloc = storage_layout.get_slot(1)->create_alias<::cuda::std::uint8_t>();
  auto copy_storage_alloc = storage_layout.get_slot(1)->create_alias<::cuda::std::uint8_t>();

  // The scan reads d_run_lengths unwrapped: wrapping it in a cuda::transform_iterator requires the user iterator to
  // provide operator++ (checked by the scan's contiguous-iterator detection) and hides contiguous inputs from the scan.
  size_t scan_storage_bytes = 0;
  if (const auto error = CubDebug(scan::dispatch(
        nullptr,
        scan_storage_bytes,
        d_run_lengths,
        static_cast<decoded_offset_t*>(nullptr),
        scan_op_t{},
        scan_init_t{decoded_offset_t{0}},
        static_cast<scan_offset_t>(num_runs),
        stream,
        scan_policy_selector_t{})))
  {
    return error;
  }

  size_t copy_storage_bytes = 0;
  if (const auto error = CubDebug(copy_runs(
        nullptr,
        copy_storage_bytes,
        d_run_values,
        ::cuda::make_transform_iterator(static_cast<decoded_offset_t*>(nullptr), destination_op_t{d_out}),
        d_run_sizes,
        num_runs,
        stream,
        tuning_env)))
  {
    return error;
  }

  scan_storage_alloc.grow(scan_storage_bytes);
  copy_storage_alloc.grow(copy_storage_bytes);

  if (d_temp_storage == nullptr)
  {
    temp_storage_bytes = storage_layout.get_size();
    return cudaSuccess;
  }

  if (num_runs == 0)
  {
    return cudaSuccess;
  }

  if (const auto error = CubDebug(storage_layout.map_to_buffer(d_temp_storage, temp_storage_bytes)))
  {
    return error;
  }

  decoded_offset_t* const d_run_offsets = run_offsets_alloc.get();

  if (const auto error = CubDebug(scan::dispatch(
        scan_storage_alloc.get(),
        scan_storage_bytes,
        d_run_lengths,
        d_run_offsets,
        scan_op_t{},
        scan_init_t{decoded_offset_t{0}},
        static_cast<scan_offset_t>(num_runs),
        stream,
        scan_policy_selector_t{})))
  {
    return error;
  }

  return CubDebug(copy_runs(
    copy_storage_alloc.get(),
    copy_storage_bytes,
    d_run_values,
    ::cuda::make_transform_iterator(d_run_offsets, destination_op_t{d_out}),
    d_run_sizes,
    num_runs,
    stream,
    tuning_env));
}

//! Run-length decode from run values and run offsets. Run i covers [d_run_offsets[i], d_run_offsets[i + 1]) of the
//! output, so the destinations and lengths of the runs are computed on the fly and no temporary storage beyond the
//! copy's own is needed.
//!
//! @p TuningEnvT may carry a policy selector returning a cub::BatchedCopyPolicy, applied to the copy.
template <typename RunValuesIteratorT,
          typename RunOffsetsIteratorT,
          typename OutputIteratorT,
          typename TuningEnvT = ::cuda::std::execution::env<>>
CUB_RUNTIME_FUNCTION cudaError_t dispatch_from_offsets(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  RunValuesIteratorT d_run_values,
  RunOffsetsIteratorT d_run_offsets,
  OutputIteratorT d_out,
  ::cuda::std::int64_t num_runs,
  cudaStream_t stream,
  const TuningEnvT& tuning_env = {})
{
  using run_offset_t = it_value_t<RunOffsetsIteratorT>;
  // Type of the run lengths as seen by the batched copy
  using run_size_t       = choose_offset_t<run_offset_t>;
  using destination_op_t = run_offset_to_destination_op<OutputIteratorT, run_offset_t>;

  return copy_runs(
    d_temp_storage,
    temp_storage_bytes,
    d_run_values,
    ::cuda::make_transform_iterator(d_run_offsets, destination_op_t{d_out}),
    ::cuda::make_transform_iterator(::cuda::counting_iterator<::cuda::std::int64_t>{0},
                                    run_offsets_to_length_op<RunOffsetsIteratorT, run_size_t>{d_run_offsets}),
    num_runs,
    stream,
    tuning_env);
}
} // namespace detail::rld

CUB_NAMESPACE_END
