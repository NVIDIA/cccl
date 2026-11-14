// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! cub::DeviceRunLengthDecode provides device-wide, parallel operations for expanding a run-length encoded sequence
//! residing within device-accessible memory.

#pragma once

#include <cub/config.cuh>

#ifndef CCCL_DISABLE_NVRTC_COMPATIBILITY_CHECK
#  if _CCCL_COMPILER(NVRTC)
#    error \
      "Including <cub/device/device_run_length_decode.cuh> is not supported when compiling with NVRTC. Include block-, warp-, or thread-level primitives instead (e.g. <cub/block/block_run_length_decode.cuh>). You can define CCCL_DISABLE_NVRTC_COMPATIBILITY_CHECK to disable this warning."
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
#include <cub/device/dispatch/dispatch_rld.cuh>

#include <cuda/std/__execution/env.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

//! @rst
//! DeviceRunLengthDecode provides device-wide, parallel operations for expanding a run-length encoded sequence
//! residing within device-accessible memory.
//!
//! Overview
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! A run-length encoded sequence consists of runs, where each run is described by a value and either its length or its
//! offset in the decoded sequence. Run-length decoding writes each run value as many times as the length of its run,
//! with the runs in order. It inverts :cpp:func:`cub::DeviceRunLengthEncode::Encode`, whose unique values and run
//! lengths can be passed to ``Decode`` directly.
//!
//! ``Decode`` takes the length of each run, ``DecodeFromOffsets`` takes the offset of each run in the decoded
//! sequence. Runs of length zero are allowed at any position and do not produce any output.
//!
//! Usage Considerations
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @cdp_class{DeviceRunLengthDecode}
//!
//! Performance
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! Both algorithms copy the runs using the same implementation as :cpp:func:`cub::DeviceCopy::Batched`, which
//! balances the work across runs of very different lengths. Only the number of runs has to be known on the host, and
//! the required temporary storage grows linearly with it. ``Decode`` additionally computes the output offsets of the
//! runs with a prefix sum over the run lengths and keeps one 64-bit offset per run in temporary storage.
//!
//! Tuning
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! The algorithms that accept an environment can be tuned by passing a custom :ref:`policy selector
//! <cub-policy-selectors>` that returns a :cpp:struct:`cub::BatchedCopyPolicy`, which is applied to copying the runs.
//! For ``Decode``, a policy selector that returns a :cpp:struct:`cub::ScanPolicy` is applied to the prefix sum over the
//! run lengths. Both can be passed together, as shown in the example below:
//!
//!  .. literalinclude:: ../../../cub/test/catch2_test_device_run_length_decode_env_api.cu
//!      :language: c++
//!      :dedent:
//!      :start-after: example-begin decode-policy-selectors
//!      :end-before: example-end decode-policy-selectors
//!
//!  .. literalinclude:: ../../../cub/test/catch2_test_device_run_length_decode_env_api.cu
//!      :language: c++
//!      :dedent:
//!      :start-after: example-begin decode-tuning
//!      :end-before: example-end decode-tuning
//!
//! @endrst
struct DeviceRunLengthDecode
{
  //! @rst
  //! Expands a run-length encoded sequence given by the value and the length of each run.
  //!
  //! .. versionadded:: 3.6.0
  //!
  //! - The value of the *i*\ :sup:`th` run, ``d_run_values[i]``, is written ``d_run_lengths[i]`` times to ``d_out``,
  //!   starting at ``d_out[d_run_lengths[0] + ... + d_run_lengths[i - 1]]``.
  //! - ``d_out`` must provide room for ``d_run_lengths[0] + ... + d_run_lengths[num_runs - 1]`` items. This total is
  //!   not returned. For runs produced by :cpp:func:`cub::DeviceRunLengthEncode::Encode`, it is the ``num_items``
  //!   passed to ``Encode``. Otherwise, it can be computed with :cpp:func:`cub::DeviceReduce::Sum`.
  //! - Run lengths must be non-negative. Runs of length zero are allowed at any position.
  //! - The total may exceed 2\ :sup:`32` items, independently of the run length type.
  //! - In-place operations are not supported. The output range must not overlap
  //!   ``[d_run_values, d_run_values + num_runs)`` or ``[d_run_lengths, d_run_lengths + num_runs)``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the run-length decoding of a sequence of ``int`` values.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_run_length_decode_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin decode-run-lengths
  //!     :end-before: example-end decode-run-lengths
  //!
  //! @endrst
  //!
  //! @tparam RunValuesIteratorT
  //!   **[inferred]** Random-access input iterator type for reading the run values @iterator
  //!
  //! @tparam RunLengthsIteratorT
  //!   **[inferred]** Random-access input iterator type for reading the run lengths @iterator.
  //!   Its value type must be an integral type.
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing the decoded items @iterator
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!   Supports customization of the stream via ``cuda::get_stream`` and of the tuning via
  //!   ``cuda::execution::tune``.
  //!
  //! @param[in] d_temp_storage
  //!   @devicestorage
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_run_values
  //!   Iterator to the value of each run
  //!
  //! @param[in] d_run_lengths
  //!   Iterator to the length of each run
  //!
  //! @param[out] d_out
  //!   Iterator to the beginning of the decoded sequence
  //!
  //! @param[in] num_runs
  //!   Number of runs (i.e., the length of ``d_run_values`` and ``d_run_lengths``). Must be non-negative.
  //!
  //! @param[in] env
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  template <typename RunValuesIteratorT,
            typename RunLengthsIteratorT,
            typename OutputIteratorT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t Decode(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    RunValuesIteratorT d_run_values,
    RunLengthsIteratorT d_run_lengths,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_runs,
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceRunLengthDecode::Decode");
    return detail::dispatch_with_env(
      d_temp_storage, temp_storage_bytes, env, [&](auto tuning, void* storage, size_t& bytes, cudaStream_t stream) {
        return detail::rld::dispatch(storage, bytes, d_run_values, d_run_lengths, d_out, num_runs, stream, tuning);
      });
  }

  //! @rst
  //! Expands a run-length encoded sequence given by the value and the length of each run.
  //!
  //! .. versionadded:: 3.6.0
  //!
  //! This is an environment-based API that allows customization of:
  //!
  //! - Stream: Query via ``cuda::get_stream``
  //! - Memory resource: Query via ``cuda::mr::get_memory_resource``
  //! - Tuning: Query via ``cuda::execution::tune``
  //!
  //! - The value of the *i*\ :sup:`th` run, ``d_run_values[i]``, is written ``d_run_lengths[i]`` times to ``d_out``,
  //!   starting at ``d_out[d_run_lengths[0] + ... + d_run_lengths[i - 1]]``.
  //! - ``d_out`` must provide room for ``d_run_lengths[0] + ... + d_run_lengths[num_runs - 1]`` items. This total is
  //!   not returned. For runs produced by :cpp:func:`cub::DeviceRunLengthEncode::Encode`, it is the ``num_items``
  //!   passed to ``Encode``. Otherwise, it can be computed with :cpp:func:`cub::DeviceReduce::Sum`.
  //! - Run lengths must be non-negative. Runs of length zero are allowed at any position.
  //! - The total may exceed 2\ :sup:`32` items, independently of the run length type.
  //! - In-place operations are not supported. The output range must not overlap
  //!   ``[d_run_values, d_run_values + num_runs)`` or ``[d_run_lengths, d_run_lengths + num_runs)``.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the run-length decoding of a sequence of ``int`` values on a stream.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_run_length_decode_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin decode-run-lengths-env
  //!     :end-before: example-end decode-run-lengths-env
  //!
  //! @endrst
  //!
  //! @tparam RunValuesIteratorT
  //!   **[inferred]** Random-access input iterator type for reading the run values @iterator
  //!
  //! @tparam RunLengthsIteratorT
  //!   **[inferred]** Random-access input iterator type for reading the run lengths @iterator.
  //!   Its value type must be an integral type.
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing the decoded items @iterator
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_run_values
  //!   Iterator to the value of each run
  //!
  //! @param[in] d_run_lengths
  //!   Iterator to the length of each run
  //!
  //! @param[out] d_out
  //!   Iterator to the beginning of the decoded sequence
  //!
  //! @param[in] num_runs
  //!   Number of runs (i.e., the length of ``d_run_values`` and ``d_run_lengths``). Must be non-negative.
  //!
  //! @param[in] env
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  template <typename RunValuesIteratorT,
            typename RunLengthsIteratorT,
            typename OutputIteratorT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t
  Decode(RunValuesIteratorT d_run_values,
         RunLengthsIteratorT d_run_lengths,
         OutputIteratorT d_out,
         ::cuda::std::int64_t num_runs,
         const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceRunLengthDecode::Decode");
    return detail::dispatch_with_env(env, [&](auto tuning, void* storage, size_t& bytes, cudaStream_t stream) {
      return detail::rld::dispatch(storage, bytes, d_run_values, d_run_lengths, d_out, num_runs, stream, tuning);
    });
  }

  //! @rst
  //! Expands a run-length encoded sequence given by the value of each run and the offset of each run in the decoded
  //! sequence.
  //!
  //! .. versionadded:: 3.6.0
  //!
  //! - ``d_run_offsets`` provides ``num_runs + 1`` offsets. The *i*\ :sup:`th` run occupies the positions
  //!   ``[d_run_offsets[i], d_run_offsets[i + 1])`` of ``d_out``, i.e., its value ``d_run_values[i]`` is written to
  //!   ``d_out[d_run_offsets[i]]`` through ``d_out[d_run_offsets[i + 1] - 1]``.
  //! - The offsets must be non-decreasing. Runs of length zero (equal consecutive offsets) are allowed at any position.
  //! - Only the positions ``[d_run_offsets[0], d_run_offsets[num_runs])`` of ``d_out`` are written.
  //! - In-place operations are not supported. The output range must not overlap
  //!   ``[d_run_values, d_run_values + num_runs)`` or ``[d_run_offsets, d_run_offsets + num_runs + 1)``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the run-length decoding of a sequence of ``int`` values.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_run_length_decode_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin decode-run-offsets
  //!     :end-before: example-end decode-run-offsets
  //!
  //! @endrst
  //!
  //! @tparam RunValuesIteratorT
  //!   **[inferred]** Random-access input iterator type for reading the run values @iterator
  //!
  //! @tparam RunOffsetsIteratorT
  //!   **[inferred]** Random-access input iterator type for reading the run offsets @iterator.
  //!   Its value type must be an integral type.
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing the decoded items @iterator
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!   Supports customization of the stream via ``cuda::get_stream`` and of the tuning via
  //!   ``cuda::execution::tune``.
  //!
  //! @param[in] d_temp_storage
  //!   @devicestorage
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_run_values
  //!   Iterator to the value of each run
  //!
  //! @param[in] d_run_offsets
  //!   Iterator to the ``num_runs + 1`` offsets delimiting the runs in the decoded sequence
  //!
  //! @param[out] d_out
  //!   Iterator to the beginning of the decoded sequence
  //!
  //! @param[in] num_runs
  //!   Number of runs (i.e., the length of ``d_run_values``). Must be non-negative.
  //!
  //! @param[in] env
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  template <typename RunValuesIteratorT,
            typename RunOffsetsIteratorT,
            typename OutputIteratorT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t DecodeFromOffsets(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    RunValuesIteratorT d_run_values,
    RunOffsetsIteratorT d_run_offsets,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_runs,
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceRunLengthDecode::DecodeFromOffsets");
    return detail::dispatch_with_env(
      d_temp_storage, temp_storage_bytes, env, [&](auto tuning, void* storage, size_t& bytes, cudaStream_t stream) {
        return detail::rld::dispatch_from_offsets(
          storage, bytes, d_run_values, d_run_offsets, d_out, num_runs, stream, tuning);
      });
  }

  //! @rst
  //! Expands a run-length encoded sequence given by the value of each run and the offset of each run in the decoded
  //! sequence.
  //!
  //! .. versionadded:: 3.6.0
  //!
  //! This is an environment-based API that allows customization of:
  //!
  //! - Stream: Query via ``cuda::get_stream``
  //! - Memory resource: Query via ``cuda::mr::get_memory_resource``
  //! - Tuning: Query via ``cuda::execution::tune``
  //!
  //! - ``d_run_offsets`` provides ``num_runs + 1`` offsets. The *i*\ :sup:`th` run occupies the positions
  //!   ``[d_run_offsets[i], d_run_offsets[i + 1])`` of ``d_out``, i.e., its value ``d_run_values[i]`` is written to
  //!   ``d_out[d_run_offsets[i]]`` through ``d_out[d_run_offsets[i + 1] - 1]``.
  //! - The offsets must be non-decreasing. Runs of length zero (equal consecutive offsets) are allowed at any position.
  //! - Only the positions ``[d_run_offsets[0], d_run_offsets[num_runs])`` of ``d_out`` are written.
  //! - In-place operations are not supported. The output range must not overlap
  //!   ``[d_run_values, d_run_values + num_runs)`` or ``[d_run_offsets, d_run_offsets + num_runs + 1)``.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the run-length decoding of a sequence of ``int`` values on a stream.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_run_length_decode_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin decode-run-offsets-env
  //!     :end-before: example-end decode-run-offsets-env
  //!
  //! @endrst
  //!
  //! @tparam RunValuesIteratorT
  //!   **[inferred]** Random-access input iterator type for reading the run values @iterator
  //!
  //! @tparam RunOffsetsIteratorT
  //!   **[inferred]** Random-access input iterator type for reading the run offsets @iterator.
  //!   Its value type must be an integral type.
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing the decoded items @iterator
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_run_values
  //!   Iterator to the value of each run
  //!
  //! @param[in] d_run_offsets
  //!   Iterator to the ``num_runs + 1`` offsets delimiting the runs in the decoded sequence
  //!
  //! @param[out] d_out
  //!   Iterator to the beginning of the decoded sequence
  //!
  //! @param[in] num_runs
  //!   Number of runs (i.e., the length of ``d_run_values``). Must be non-negative.
  //!
  //! @param[in] env
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  template <typename RunValuesIteratorT,
            typename RunOffsetsIteratorT,
            typename OutputIteratorT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t DecodeFromOffsets(
    RunValuesIteratorT d_run_values,
    RunOffsetsIteratorT d_run_offsets,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_runs,
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceRunLengthDecode::DecodeFromOffsets");
    return detail::dispatch_with_env(env, [&](auto tuning, void* storage, size_t& bytes, cudaStream_t stream) {
      return detail::rld::dispatch_from_offsets(
        storage, bytes, d_run_values, d_run_offsets, d_out, num_runs, stream, tuning);
    });
  }
};

CUB_NAMESPACE_END
