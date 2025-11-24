// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! cub::DeviceTopK provides device-wide, parallel operations for finding the K largest (or smallest) items from
//! sequences of data

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
#include <cub/device/dispatch/dispatch_topk.cuh>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/std/__execution/env.h>

CUB_NAMESPACE_BEGIN

namespace detail
{
template <topk::select SelectDirection,
          typename KeyInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename NumItemsT,
          typename NumOutItemsT,
          typename EnvT>
CUB_RUNTIME_FUNCTION static cudaError_t dispatch_topk_hub(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  KeyInputIteratorT d_keys_in,
  KeyOutputIteratorT d_keys_out,
  ValueInputIteratorT d_values_in,
  ValueOutputIteratorT d_values_out,
  NumItemsT num_items,
  NumOutItemsT k,
  EnvT env)
{
  // Offset type selection
  using offset_t     = choose_offset_t<NumItemsT>;
  using out_offset_t = ::cuda::std::
    conditional_t<sizeof(offset_t) < sizeof(choose_offset_t<NumOutItemsT>), offset_t, choose_offset_t<NumOutItemsT>>;

  // Query environment properties to determine if the user-requested configuration is supported
  static_assert(!::cuda::std::execution::__queryable_with<EnvT, ::cuda::execution::determinism::__get_determinism_t>,
                "Determinism should be used inside requires to have an effect.");
  using requirements_t = ::cuda::std::execution::
    __query_result_or_t<EnvT, ::cuda::execution::__get_requirements_t, ::cuda::std::execution::env<>>;
  using requested_determinism_t =
    ::cuda::std::execution::__query_result_or_t<requirements_t,
                                                ::cuda::execution::determinism::__get_determinism_t,
                                                ::cuda::execution::determinism::run_to_run_t>;
  using requested_order_t =
    ::cuda::std::execution::__query_result_or_t<requirements_t,
                                                ::cuda::execution::output_ordering::__get_output_ordering_t,
                                                ::cuda::execution::output_ordering::sorted_t>;
  constexpr auto is_determinism_not_guaranteed =
    ::cuda::std::is_same_v<requested_determinism_t, ::cuda::execution::determinism::not_guaranteed_t>;
  constexpr auto is_output_order_unsorted =
    ::cuda::std::is_same_v<requested_order_t, ::cuda::execution::output_ordering::unsorted_t>;

  // We only support the case where determinism is not guaranteed and output order is unsorted
  static_assert(is_determinism_not_guaranteed && is_output_order_unsorted,
                "cub::DeviceTopK only supports the case where determinism is not guaranteed and output order is "
                "unsorted.");

  // Query relevant properties from the environment
  auto stream = ::cuda::std::execution::__query_or(env, ::cuda::get_stream, ::cuda::stream_ref{cudaStream_t{}});

  return topk::DispatchTopK<
    KeyInputIteratorT,
    KeyOutputIteratorT,
    ValueInputIteratorT,
    ValueOutputIteratorT,
    offset_t,
    out_offset_t,
    SelectDirection>::Dispatch(d_temp_storage,
                               temp_storage_bytes,
                               d_keys_in,
                               d_keys_out,
                               d_values_in,
                               d_values_out,
                               static_cast<offset_t>(num_items),
                               static_cast<out_offset_t>(k),
                               stream.get());
}
} // namespace detail

//! @rst
//! DeviceTopK provides device-wide, parallel operations for finding the largest (or smallest) K items from sequences of
//! unordered data items residing within device-accessible memory.
//!
//! Overview
//! ++++++++++++++++++++++++++
//!
//! The TopK algorithm tries to find the largest (or smallest) K items in an unordered list. A related problem is called
//! `K selection problem <https://en.wikipedia.org/wiki/Selection_algorithm>`_, which finds the Kth largest
//! (or smallest) values in a list.
//! DeviceTopK will return K items in an unspecified order as results. It is based on an algorithm called
//! `AIR TopK <https://dl.acm.org/doi/10.1145/3581784.3607062>`_.
//!
//! Supported Types
//! ++++++++++++++++++++++++++
//!
//! DeviceTopK can process all of the built-in C++ numeric primitive types (`unsigned char`, `int`, `double`, etc.) as
//! well as CUDA's `__half`  and `__nv_bfloat16` 16-bit floating-point types.
//!
//! Determinism
//! ++++++++++++++++++++++++++
//!
//! DeviceTopK currently only supports unordered output, which may be non-deterministic for certain inputs.
//! That is, if there are multiple items across the k-th position that compare equal, the subset of tied elements that
//! ends up in the returned top‑k is not uniquely defined and may vary between runs. This behavior has to be explicitly
//! acknowledged by the user by passing `cuda::execution::determinism::not_guaranteed`.
//!
//! Usage Considerations
//! ++++++++++++++++++++++++++
//!
//! @cdp_class{DeviceTopK}
//!
//! Performance
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @linear_performance{top-k}
//!
//! @endrst
struct DeviceTopK
{
  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! Finds the largest K keys and their corresponding values from an unordered input sequence of key-value pairs.
  //!
  //! - @devicestorage
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The following code snippet demonstrates how to use the `cub::DeviceTopK::MaxPairs` function to find the largest K
  //! items:
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_topk_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin topk-max-pairs-non-deterministic-unsorted
  //!     :end-before: example-end topk-max-pairs-non-deterministic-unsorted
  //!
  //! @endrst
  //!
  //! @tparam KeyInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input keys @iterator
  //!
  //! @tparam KeyOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing output keys @iterator
  //!
  //! @tparam ValueInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input values @iterator
  //!
  //! @tparam ValueOutputIteratorT
  //!   **[inferred]** Random-access input iterator type for writing output values @iterator
  //!
  //! @tparam NumItemsT
  //!  The integral type of variable num_items
  //!
  //! @tparam NumOutItemsT
  //!  The integral type of variable k
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the required allocation size is written to
  //!   `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_keys_in
  //!   Random-access iterator to the input sequence containing the keys
  //!
  //! @param[out] d_keys_out
  //!   Random-access iterator to the output sequence of keys, where K values will be written to
  //!
  //! @param[in] d_values_in
  //!   Random-access iterator to the input sequence containing the values associated to each key
  //!
  //! @param[out] d_values_out
  //!   Random-access iterator to the output sequence of values, corresponding to the top k keys, where k values will be
  //!   written to
  //!
  //! @param[in] num_items
  //!   Number of items to be read and processed from `d_keys_in` and `d_values_in` each
  //!
  //! @param[in] k
  //!   The value of K, which is the number of largest pairs to find from `num_items` pairs. Capped to a maximum of
  //!   `num_items`.
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is `cuda::std::execution::env{}`.
  //!   @endrst
  template <typename KeyInputIteratorT,
            typename KeyOutputIteratorT,
            typename ValueInputIteratorT,
            typename ValueOutputIteratorT,
            typename NumItemsT,
            typename NumOutItemsT,
            typename EnvT = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t MaxPairs(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    NumItemsT num_items,
    NumOutItemsT k,
    EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceTopK::MaxPairs");

    return detail::dispatch_topk_hub<detail::topk::select::max>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      d_values_in,
      d_values_out,
      num_items,
      k,
      ::cuda::std::move(env));
  }

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! Finds the lowest K keys and their corresponding values from an unordered input sequence of key-value pairs.
  //!
  //! - @devicestorage
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The following code snippet demonstrates how to use the `cub::DeviceTopK::MinPairs` function to find the lowest K
  //! items:
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_topk_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin topk-min-pairs-non-deterministic-unsorted
  //!     :end-before: example-end topk-min-pairs-non-deterministic-unsorted
  //!
  //! @endrst
  //!
  //! @tparam KeyInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input keys @iterator
  //!
  //! @tparam KeyOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing output keys @iterator
  //!
  //! @tparam ValueInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input values @iterator
  //!
  //! @tparam ValueOutputIteratorT
  //!   **[inferred]** Random-access input iterator type for writing output values @iterator
  //!
  //! @tparam NumItemsT
  //!  The integral type of variable num_items
  //!
  //! @tparam NumOutItemsT
  //!  The integral type of variable k
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_keys_in
  //!   Random-access iterator to the input sequence containing the keys
  //!
  //! @param[out] d_keys_out
  //!   Random-access iterator to the output sequence of keys, where K values will be written to
  //!
  //! @param[in] d_values_in
  //!   Random-access iterator to the input sequence containing the values associated to each key
  //!
  //! @param[out] d_values_out
  //!   Random-access iterator to the output sequence of values, corresponding to the top k keys, where k values will be
  //!   written to
  //!
  //! @param[in] num_items
  //!   Number of items to be read and processed from `d_keys_in` and `d_values_in` each
  //!
  //! @param[in] k
  //!   The value of K, which is the number of lowest pairs to find from `num_items` pairs. Capped to a maximum of
  //!   `num_items`.
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is `cuda::std::execution::env{}`.
  //!   @endrst
  template <typename KeyInputIteratorT,
            typename KeyOutputIteratorT,
            typename ValueInputIteratorT,
            typename ValueOutputIteratorT,
            typename NumItemsT,
            typename NumOutItemsT,
            typename EnvT = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t MinPairs(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    NumItemsT num_items,
    NumOutItemsT k,
    EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceTopK::MinPairs");

    return detail::dispatch_topk_hub<detail::topk::select::min>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      d_values_in,
      d_values_out,
      num_items,
      k,
      ::cuda::std::move(env));
  }

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! Finds the largest K keys from an unordered input sequence of keys.
  //!
  //! - @devicestorage
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The following code snippet demonstrates how to use the `cub::DeviceTopK::MinKeys` function to find the largest K
  //! items:
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_topk_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin topk-max-keys-non-deterministic-unsorted
  //!     :end-before: example-end topk-max-keys-non-deterministic-unsorted
  //!
  //! @endrst
  //!
  //! @tparam KeyInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input keys @iterator
  //!
  //! @tparam KeyOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing output keys @iterator
  //!
  //! @tparam NumItemsT
  //!  The integral type of variable num_items
  //!
  //! @tparam NumOutItemsT
  //!  The integral type of variable k
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_keys_in
  //!   Random-access iterator to the input sequence containing the keys
  //!
  //! @param[out] d_keys_out
  //!   Random-access iterator to the output sequence of keys, where K values will be written to
  //!
  //! @param[in] num_items
  //!   Number of items to be read and processed from `d_keys_in`
  //!
  //! @param[in] k
  //!   The value of K, which is the number of largest pairs to find from `num_items` pairs. Capped to a maximum of
  //!   `num_items`.
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is `cuda::std::execution::env{}`.
  //!   @endrst
  template <typename KeyInputIteratorT,
            typename KeyOutputIteratorT,
            typename NumItemsT,
            typename NumOutItemsT,
            typename EnvT = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t MaxKeys(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    NumItemsT num_items,
    NumOutItemsT k,
    EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceTopK::MaxKeys");

    return detail::dispatch_topk_hub<detail::topk::select::max>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      static_cast<NullType*>(nullptr),
      static_cast<NullType*>(nullptr),
      num_items,
      k,
      ::cuda::std::move(env));
  }

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! Finds the lowest K keys from an unordered input sequence of keys.
  //!
  //! - @devicestorage
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The following code snippet demonstrates how to use the `cub::DeviceTopK::MinKeys` function to find the lowest K
  //! items:
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_topk_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin topk-min-keys-non-deterministic-unsorted
  //!     :end-before: example-end topk-min-keys-non-deterministic-unsorted
  //!
  //! @endrst
  //!
  //! @tparam KeyInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input keys @iterator
  //!
  //! @tparam KeyOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing output keys @iterator
  //!
  //! @tparam NumItemsT
  //!  The integral type of variable num_items
  //!
  //! @tparam NumOutItemsT
  //!  The integral type of variable k
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_keys_in
  //!   Random-access iterator to the input sequence containing the keys
  //!
  //! @param[out] d_keys_out
  //!   Random-access iterator to the output sequence of keys, where K values will be written to
  //!
  //! @param[in] num_items
  //!   Number of items to be read and processed from `d_keys_in`
  //!
  //! @param[in] k
  //!   The value of K, which is the number of largest pairs to find from `num_items` pairs. Capped to a maximum of
  //!   `num_items`.
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is `cuda::std::execution::env{}`.
  //!   @endrst
  template <typename KeyInputIteratorT,
            typename KeyOutputIteratorT,
            typename NumItemsT,
            typename NumOutItemsT,
            typename EnvT = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t MinKeys(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    NumItemsT num_items,
    NumOutItemsT k,
    EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceTopK::MinKeys");

    return detail::dispatch_topk_hub<detail::topk::select::min>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      static_cast<NullType*>(nullptr),
      static_cast<NullType*>(nullptr),
      num_items,
      k,
      ::cuda::std::move(env));
  }
};

CUB_NAMESPACE_END
