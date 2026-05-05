// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/env_dispatch.cuh>
#include <cub/device/dispatch/dispatch_merge.cuh>
#include <cub/util_namespace.cuh>

#include <cuda/std/__functional/operations.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

//! DeviceMerge provides device-wide, parallel operations for merging two sorted sequences of values (called keys) or
//! key-value pairs in device-accessible memory. The sorting order is determined by a comparison functor (default:
//! less-than), which has to establish a [strict weak ordering].
//!
//! [strict weak ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
struct DeviceMerge
{
  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //! Merges two sorted sequences of values (called keys) into a sorted output sequence. Merging is unstable,
  //! which means any two equivalent values (neither value is ordered before the other) may be written to the output
  //! sequence in any order.
  //!
  //! .. versionadded:: 2.7.0
  //!    First appears in CUDA Toolkit 12.8.
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //! The code snippet below illustrates the merging of two device vectors of `int` keys.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_merge_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin merge-keys
  //!     :end-before: example-end merge-keys
  //!
  //! @endrst
  //!
  //! @tparam KeyIteratorIn1 **[deduced]** Random access iterator to the first sorted input sequence. Must have the same
  //! value type as KeyIteratorIn2.
  //! @tparam KeyIteratorIn2 **[deduced]** Random access iterator to the second sorted input sequence. Must have the
  //! same value type as KeyIteratorIn1.
  //! @tparam KeyIteratorOut **[deduced]** Random access iterator to the output sequence.
  //! @tparam CompareOp **[deduced]** Binary predicate to compare the input iterator's value types. Must have a
  //! signature equivalent to `bool operator()(Key lhs, Key rhs)` and establish a [strict weak ordering].
  //!
  //! @param[in] d_temp_storage Device-accessible allocation of temporary storage. When `nullptr`, the required
  //! allocation size is written to `temp_storage_bytes` and no work is done.
  //! @param[in,out] temp_storage_bytes Reference to size in bytes of `d_temp_storage` allocation.
  //! @param[in] keys_in1 Iterator to the beginning of the first sorted input sequence.
  //! @param[in] num_keys1 Number of keys in the first input sequence.
  //! @param[in] keys_in2 Iterator to the beginning of the second sorted input sequence.
  //! @param[in] num_keys2 Number of keys in the second input sequence.
  //! @param[out] keys_out Iterator to the beginning of the output sequence.
  //! @param[in] compare_op Comparison function object, returning true if the first argument is ordered before the
  //! second. Must establish a [strict weak ordering].
  //! @param[in] stream **[optional]** CUDA stream to launch kernels into. Default is stream<sub>0</sub>.
  //!
  //! [strict weak ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
  template <typename KeyIteratorIn1,
            typename KeyIteratorIn2,
            typename KeyIteratorOut,
            typename CompareOp = ::cuda::std::less<>>
  CUB_RUNTIME_FUNCTION static cudaError_t MergeKeys(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyIteratorIn1 keys_in1,
    ::cuda::std::int64_t num_keys1,
    KeyIteratorIn2 keys_in2,
    ::cuda::std::int64_t num_keys2,
    KeyIteratorOut keys_out,
    CompareOp compare_op = {},
    cudaStream_t stream  = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceMerge::MergeKeys");
    // offset type is just int64_t
    return detail::merge::dispatch(
      d_temp_storage,
      temp_storage_bytes,
      keys_in1,
      static_cast<NullType*>(nullptr),
      num_keys1,
      keys_in2,
      static_cast<NullType*>(nullptr),
      num_keys2,
      keys_out,
      static_cast<NullType*>(nullptr),
      compare_op,
      stream);
  }

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //! Merges two sorted sequences of values (called keys) into a sorted output sequence. Merging is unstable,
  //! which means any two equivalent values (neither value is ordered before the other) may be written to the output
  //! sequence in any order.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! This is an environment-based API that allows customization of:
  //!
  //! - Stream: Query via ``cuda::get_stream``
  //! - Memory resource: Query via ``cuda::mr::get_memory_resource``
  //!
  //! Snippet
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_merge_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin merge-keys-env
  //!     :end-before: example-end merge-keys-env
  //!
  //! @endrst
  //!
  //! @tparam KeyIteratorIn1
  //!   **[deduced]** Random access iterator to the first sorted input sequence. Must have the same
  //!   value type as KeyIteratorIn2.
  //!
  //! @tparam KeyIteratorIn2
  //!   **[deduced]** Random access iterator to the second sorted input sequence. Must have the
  //!   same value type as KeyIteratorIn1.
  //!
  //! @tparam KeyIteratorOut
  //!   **[deduced]** Random access iterator to the output sequence.
  //!
  //! @tparam CompareOp
  //!   **[deduced]** Binary predicate to compare the input iterator's value types. Must have a
  //!   signature equivalent to `bool operator()(Key lhs, Key rhs)` and establish a [strict weak ordering].
  //!
  //! @tparam EnvT
  //!   **[deduced]** Environment type (e.g., `cuda::std::execution::env<...>`)
  //!
  //! @param[in] keys_in1
  //!   Iterator to the beginning of the first sorted input sequence.
  //!
  //! @param[in] num_keys1
  //!   Number of keys in the first input sequence.
  //!
  //! @param[in] keys_in2
  //!   Iterator to the beginning of the second sorted input sequence.
  //!
  //! @param[in] num_keys2
  //!   Number of keys in the second input sequence.
  //!
  //! @param[out] keys_out
  //!   Iterator to the beginning of the output sequence.
  //!
  //! @param[in] compare_op
  //!   Comparison function object, returning true if the first argument is ordered before the
  //!   second. Must establish a [strict weak ordering].
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  //! [strict weak ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
  template <
    typename KeyIteratorIn1,
    typename KeyIteratorIn2,
    typename KeyIteratorOut,
    typename CompareOp = ::cuda::std::less<>,
    typename EnvT      = ::cuda::std::execution::env<>,
    ::cuda::std::enable_if_t<
      !::cuda::std::is_same_v<KeyIteratorIn1, void*> && !::cuda::std::is_same_v<KeyIteratorIn1, ::cuda::std::nullptr_t>,
      int> = 0,
    ::cuda::std::enable_if_t<::cuda::std::indirect_binary_predicate<CompareOp, KeyIteratorIn1, KeyIteratorIn2>, int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t MergeKeys(
    KeyIteratorIn1 keys_in1,
    ::cuda::std::int64_t num_keys1,
    KeyIteratorIn2 keys_in2,
    ::cuda::std::int64_t num_keys2,
    KeyIteratorOut keys_out,
    CompareOp compare_op = {},
    EnvT env             = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceMerge::MergeKeys");

    using default_policy_selector =
      detail::merge::policy_selector_from_types<detail::it_value_t<KeyIteratorIn1>, NullType, int64_t>;
    return detail::dispatch_with_env_and_tuning<default_policy_selector>(
      env, [&](auto policy_selector, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t stream) {
        return detail::merge::dispatch(
          d_temp_storage,
          temp_storage_bytes,
          keys_in1,
          static_cast<NullType*>(nullptr),
          num_keys1,
          keys_in2,
          static_cast<NullType*>(nullptr),
          num_keys2,
          keys_out,
          static_cast<NullType*>(nullptr),
          compare_op,
          stream,
          policy_selector);
      });
  }

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //! Merges two sorted sequences of key-value pairs into a sorted output sequence. Merging is unstable,
  //! which means any two equivalent values (neither value is ordered before the other) may be written to the output
  //! sequence in any order.
  //!
  //! .. versionadded:: 2.7.0
  //!    First appears in CUDA Toolkit 12.8.
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //! The code snippet below illustrates the merging of two device vectors of `int` keys.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_merge_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin merge-pairs
  //!     :end-before: example-end merge-pairs
  //!
  //! @endrst
  //!
  //! @tparam KeyIteratorIn1 **[deduced]** Random access iterator to the keys of the first sorted input sequence. Must
  //! have the same value type as KeyIteratorIn2.
  //! @tparam ValueIteratorIn1 **[deduced]** Random access iterator to the values of the first sorted input sequence.
  //! Must have the same value type as ValueIteratorIn2.
  //! @tparam KeyIteratorIn2 **[deduced]** Random access iterator to the second sorted input sequence. Must have the
  //! same value type as KeyIteratorIn1.
  //! @tparam ValueIteratorIn2 **[deduced]** Random access iterator to the values of the second sorted input sequence.
  //! Must have the same value type as ValueIteratorIn1.
  //! @tparam KeyIteratorOut **[deduced]** Random access iterator to the keys of the output sequence.
  //! @tparam ValueIteratorOut **[deduced]** Random access iterator to the values of the output sequence.
  //! @tparam CompareOp **[deduced]** Binary predicate to compare the key input iterator's value types. Must have a
  //! signature equivalent to `bool operator()(Key lhs, Key rhs)` and establish a [strict weak ordering].
  //!
  //! @param[in] d_temp_storage Device-accessible allocation of temporary storage. When `nullptr`, the required
  //! allocation size is written to `temp_storage_bytes` and no work is done.
  //! @param[in,out] temp_storage_bytes Reference to size in bytes of `d_temp_storage` allocation.
  //! @param[in] keys_in1 Iterator to the beginning of the keys of the first sorted input sequence.
  //! @param[in] values_in1 Iterator to the beginning of the values of the first sorted input sequence.
  //! @param[in] num_pairs1 Number of key-value pairs in the first input sequence.
  //! @param[in] keys_in2 Iterator to the beginning of the keys of the second sorted input sequence.
  //! @param[in] values_in2 Iterator to the beginning of the values of the second sorted input sequence.
  //! @param[in] num_pairs2 Number of key-value pairs in the second input sequence.
  //! @param[out] keys_out Iterator to the beginning of the keys of the output sequence.
  //! @param[out] values_out Iterator to the beginning of the values of the output sequence.
  //! @param[in] compare_op Comparison function object, returning true if the first argument is ordered before the
  //! second. Must establish a [strict weak ordering].
  //! @param[in] stream **[optional]** CUDA stream to launch kernels into. Default is stream<sub>0</sub>.
  //!
  //! [strict weak ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
  template <typename KeyIteratorIn1,
            typename ValueIteratorIn1,
            typename KeyIteratorIn2,
            typename ValueIteratorIn2,
            typename KeyIteratorOut,
            typename ValueIteratorOut,
            typename CompareOp = ::cuda::std::less<>>
  CUB_RUNTIME_FUNCTION static cudaError_t MergePairs(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyIteratorIn1 keys_in1,
    ValueIteratorIn1 values_in1,
    ::cuda::std::int64_t num_pairs1,
    KeyIteratorIn2 keys_in2,
    ValueIteratorIn2 values_in2,
    ::cuda::std::int64_t num_pairs2,
    KeyIteratorOut keys_out,
    ValueIteratorOut values_out,
    CompareOp compare_op = {},
    cudaStream_t stream  = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceMerge::MergePairs");
    // offset type is just int64_t
    return detail::merge::dispatch(
      d_temp_storage,
      temp_storage_bytes,
      keys_in1,
      values_in1,
      num_pairs1,
      keys_in2,
      values_in2,
      num_pairs2,
      keys_out,
      values_out,
      compare_op,
      stream);
  }

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //! Merges two sorted sequences of key-value pairs into a sorted output sequence. Merging is unstable,
  //! which means any two equivalent values (neither value is ordered before the other) may be written to the output
  //! sequence in any order.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! This is an environment-based API that allows customization of:
  //!
  //! - Stream: Query via ``cuda::get_stream``
  //! - Memory resource: Query via ``cuda::mr::get_memory_resource``
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_merge_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin merge-pairs-env
  //!     :end-before: example-end merge-pairs-env
  //!
  //! @endrst
  //!
  //! @tparam KeyIteratorIn1
  //!   **[deduced]** Random access iterator to the keys of the first sorted input sequence. Must
  //!   have the same value type as KeyIteratorIn2.
  //!
  //! @tparam ValueIteratorIn1
  //!   **[deduced]** Random access iterator to the values of the first sorted input sequence.
  //!   Must have the same value type as ValueIteratorIn2.
  //!
  //! @tparam KeyIteratorIn2
  //!   **[deduced]** Random access iterator to the second sorted input sequence. Must have the
  //!   same value type as KeyIteratorIn1.
  //!
  //! @tparam ValueIteratorIn2
  //!   **[deduced]** Random access iterator to the values of the second sorted input sequence.
  //!   Must have the same value type as ValueIteratorIn1.
  //!
  //! @tparam KeyIteratorOut
  //!   **[deduced]** Random access iterator to the keys of the output sequence.
  //!
  //! @tparam ValueIteratorOut
  //!   **[deduced]** Random access iterator to the values of the output sequence.
  //!
  //! @tparam CompareOp
  //!   **[deduced]** Binary predicate to compare the key input iterator's value types. Must have a
  //!   signature equivalent to `bool operator()(Key lhs, Key rhs)` and establish a [strict weak ordering].
  //!
  //! @tparam EnvT
  //!   **[deduced]** Environment type (e.g., `cuda::std::execution::env<...>`)
  //!
  //! @param[in] keys_in1
  //!   Iterator to the beginning of the keys of the first sorted input sequence.
  //!
  //! @param[in] values_in1
  //!   Iterator to the beginning of the values of the first sorted input sequence.
  //!
  //! @param[in] num_pairs1
  //!   Number of key-value pairs in the first input sequence.
  //!
  //! @param[in] keys_in2
  //!   Iterator to the beginning of the keys of the second sorted input sequence.
  //!
  //! @param[in] values_in2
  //!   Iterator to the beginning of the values of the second sorted input sequence.
  //!
  //! @param[in] num_pairs2
  //!   Number of key-value pairs in the second input sequence.
  //!
  //! @param[out] keys_out
  //!   Iterator to the beginning of the keys of the output sequence.
  //!
  //! @param[out] values_out
  //!   Iterator to the beginning of the values of the output sequence.
  //!
  //! @param[in] compare_op
  //!   Comparison function object, returning true if the first argument is ordered before the
  //!   second. Must establish a [strict weak ordering].
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  //! [strict weak ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
  template <
    typename KeyIteratorIn1,
    typename ValueIteratorIn1,
    typename KeyIteratorIn2,
    typename ValueIteratorIn2,
    typename KeyIteratorOut,
    typename ValueIteratorOut,
    typename CompareOp = ::cuda::std::less<>,
    typename EnvT      = ::cuda::std::execution::env<>,
    ::cuda::std::enable_if_t<
      !::cuda::std::is_same_v<KeyIteratorIn1, void*> && !::cuda::std::is_same_v<KeyIteratorIn1, ::cuda::std::nullptr_t>,
      int> = 0,
    ::cuda::std::enable_if_t<::cuda::std::indirect_binary_predicate<CompareOp, KeyIteratorIn1, KeyIteratorIn2>, int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t MergePairs(
    KeyIteratorIn1 keys_in1,
    ValueIteratorIn1 values_in1,
    ::cuda::std::int64_t num_pairs1,
    KeyIteratorIn2 keys_in2,
    ValueIteratorIn2 values_in2,
    ::cuda::std::int64_t num_pairs2,
    KeyIteratorOut keys_out,
    ValueIteratorOut values_out,
    CompareOp compare_op = {},
    EnvT env             = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceMerge::MergePairs");
    using default_policy_selector = detail::merge::
      policy_selector_from_types<detail::it_value_t<KeyIteratorIn1>, detail::it_value_t<ValueIteratorIn1>, int64_t>;
    return detail::dispatch_with_env_and_tuning<default_policy_selector>(
      env, [&](auto policy_selector, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t stream) {
        return detail::merge::dispatch(
          d_temp_storage,
          temp_storage_bytes,
          keys_in1,
          values_in1,
          num_pairs1,
          keys_in2,
          values_in2,
          num_pairs2,
          keys_out,
          values_out,
          compare_op,
          stream,
          policy_selector);
      });
  }
};

CUB_NAMESPACE_END
