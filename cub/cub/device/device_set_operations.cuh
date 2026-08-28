// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#ifndef CCCL_DISABLE_NVRTC_COMPATIBILITY_CHECK
#  if _CCCL_COMPILER(NVRTC)
#    error \
      "Including <cub/device/device_set_operations.cuh> is not supported when compiling with NVRTC. Include block-, warp-, or thread-level primitives instead (e.g. <cub/block/block_reduce.cuh>). You can define CCCL_DISABLE_NVRTC_COMPATIBILITY_CHECK to disable this warning."
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
#include <cub/device/dispatch/dispatch_set_operations.cuh>
#include <cub/util_namespace.cuh>
#include <cub/util_type.cuh>

#include <cuda/__functional/call_or.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

//! @rst
//! DeviceSetOps provides device-wide, parallel operations for computing set operations (difference, intersection,
//! symmetric difference, union) over two *sorted* input sequences of keys (and optionally associated values). The
//! ordering is determined by a comparison functor (default: less-than) that must establish a `strict weak ordering
//! <https://en.cppreference.com/w/cpp/concepts/strict_weak_order>`_.
//!
//! The result is written to an output sequence and its length -- which is data dependent -- is written to
//! ``d_num_selected_out`` (following the same convention as :cpp:struct:`cub::DeviceSelect`). The semantics match the
//! C++ standard library's ``std::set_*`` algorithms, including their handling of duplicate elements.
//! @endrst
struct DeviceSetOps
{
private:
  template <typename SetOp,
            typename KeyIteratorIn1,
            typename KeyIteratorIn2,
            typename KeyIteratorOut,
            typename NumSelectedIteratorT,
            typename OffsetT,
            typename CompareOp,
            typename EnvT>
  CUB_RUNTIME_FUNCTION static cudaError_t set_op_keys(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyIteratorIn1 d_keys_in1,
    OffsetT num_keys1,
    KeyIteratorIn2 d_keys_in2,
    OffsetT num_keys2,
    KeyIteratorOut d_keys_out,
    NumSelectedIteratorT d_num_selected_out,
    CompareOp compare_op,
    const EnvT& env)
  {
    const auto stream = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStream_t{}}, env).get();
    return detail::set_ops::dispatch<
      KeyIteratorIn1,
      KeyIteratorIn2,
      NullType*,
      NullType*,
      KeyIteratorOut,
      NullType*,
      OffsetT,
      CompareOp,
      SetOp,
      NumSelectedIteratorT>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in1,
      d_keys_in2,
      static_cast<NullType*>(nullptr),
      static_cast<NullType*>(nullptr),
      num_keys1,
      num_keys2,
      d_keys_out,
      static_cast<NullType*>(nullptr),
      compare_op,
      SetOp{},
      d_num_selected_out,
      stream);
  }

  template <typename SetOp,
            typename KeyIteratorIn1,
            typename ValueIteratorIn1,
            typename KeyIteratorIn2,
            typename ValueIteratorIn2,
            typename KeyIteratorOut,
            typename ValueIteratorOut,
            typename NumSelectedIteratorT,
            typename OffsetT,
            typename CompareOp,
            typename EnvT>
  CUB_RUNTIME_FUNCTION static cudaError_t set_op_pairs(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyIteratorIn1 d_keys_in1,
    ValueIteratorIn1 d_values_in1,
    OffsetT num_pairs1,
    KeyIteratorIn2 d_keys_in2,
    ValueIteratorIn2 d_values_in2,
    OffsetT num_pairs2,
    KeyIteratorOut d_keys_out,
    ValueIteratorOut d_values_out,
    NumSelectedIteratorT d_num_selected_out,
    CompareOp compare_op,
    const EnvT& env)
  {
    const auto stream = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStream_t{}}, env).get();
    return detail::set_ops::dispatch<
      KeyIteratorIn1,
      KeyIteratorIn2,
      ValueIteratorIn1,
      ValueIteratorIn2,
      KeyIteratorOut,
      ValueIteratorOut,
      OffsetT,
      CompareOp,
      SetOp,
      NumSelectedIteratorT>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in1,
      d_keys_in2,
      d_values_in1,
      d_values_in2,
      num_pairs1,
      num_pairs2,
      d_keys_out,
      d_values_out,
      compare_op,
      SetOp{},
      d_num_selected_out,
      stream);
  }

  template <typename SetOp,
            typename KeyIteratorIn1,
            typename KeyIteratorIn2,
            typename KeyIteratorOut,
            typename NumSelectedIteratorT,
            typename OffsetT,
            typename CompareOp,
            typename EnvT>
  CUB_RUNTIME_FUNCTION static cudaError_t set_op_keys_env(
    KeyIteratorIn1 d_keys_in1,
    OffsetT num_keys1,
    KeyIteratorIn2 d_keys_in2,
    OffsetT num_keys2,
    KeyIteratorOut d_keys_out,
    NumSelectedIteratorT d_num_selected_out,
    CompareOp compare_op,
    const EnvT& env)
  {
    using default_policy_selector =
      detail::set_ops::policy_selector_from_types<KeyIteratorIn1, NullType*, KeyIteratorIn2, NullType*, OffsetT>;
    return detail::dispatch_with_env_and_tuning<default_policy_selector>(
      env, [&](auto policy_selector, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t stream) {
        return detail::set_ops::dispatch<
          KeyIteratorIn1,
          KeyIteratorIn2,
          NullType*,
          NullType*,
          KeyIteratorOut,
          NullType*,
          OffsetT,
          CompareOp,
          SetOp,
          NumSelectedIteratorT,
          decltype(policy_selector)>(
          d_temp_storage,
          temp_storage_bytes,
          d_keys_in1,
          d_keys_in2,
          static_cast<NullType*>(nullptr),
          static_cast<NullType*>(nullptr),
          num_keys1,
          num_keys2,
          d_keys_out,
          static_cast<NullType*>(nullptr),
          compare_op,
          SetOp{},
          d_num_selected_out,
          stream,
          policy_selector);
      });
  }

  template <typename SetOp,
            typename KeyIteratorIn1,
            typename ValueIteratorIn1,
            typename KeyIteratorIn2,
            typename ValueIteratorIn2,
            typename KeyIteratorOut,
            typename ValueIteratorOut,
            typename NumSelectedIteratorT,
            typename OffsetT,
            typename CompareOp,
            typename EnvT>
  CUB_RUNTIME_FUNCTION static cudaError_t set_op_pairs_env(
    KeyIteratorIn1 d_keys_in1,
    ValueIteratorIn1 d_values_in1,
    OffsetT num_pairs1,
    KeyIteratorIn2 d_keys_in2,
    ValueIteratorIn2 d_values_in2,
    OffsetT num_pairs2,
    KeyIteratorOut d_keys_out,
    ValueIteratorOut d_values_out,
    NumSelectedIteratorT d_num_selected_out,
    CompareOp compare_op,
    const EnvT& env)
  {
    using default_policy_selector = detail::set_ops::
      policy_selector_from_types<KeyIteratorIn1, ValueIteratorIn1, KeyIteratorIn2, ValueIteratorIn2, OffsetT>;
    return detail::dispatch_with_env_and_tuning<default_policy_selector>(
      env, [&](auto policy_selector, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t stream) {
        return detail::set_ops::dispatch<
          KeyIteratorIn1,
          KeyIteratorIn2,
          ValueIteratorIn1,
          ValueIteratorIn2,
          KeyIteratorOut,
          ValueIteratorOut,
          OffsetT,
          CompareOp,
          SetOp,
          NumSelectedIteratorT,
          decltype(policy_selector)>(
          d_temp_storage,
          temp_storage_bytes,
          d_keys_in1,
          d_keys_in2,
          d_values_in1,
          d_values_in2,
          num_pairs1,
          num_pairs2,
          d_keys_out,
          d_values_out,
          compare_op,
          SetOp{},
          d_num_selected_out,
          stream,
          policy_selector);
      });
  }

  // SFINAE guard shared by the environment-based overloads: it disambiguates them from the explicit-temp-storage
  // overloads (whose first parameter is a `void*`) and requires a usable comparison predicate.
  template <typename KeyIteratorIn1, typename KeyIteratorIn2, typename CompareOp>
  static constexpr bool enable_env_overload =
    !::cuda::std::is_same_v<KeyIteratorIn1, void*> && !::cuda::std::is_same_v<KeyIteratorIn1, ::cuda::std::nullptr_t>
    && ::cuda::std::indirect_binary_predicate<CompareOp, KeyIteratorIn1, KeyIteratorIn2>;

public:
  //! @rst
  //! Computes the set difference ``keys1 \ keys2`` of two sorted key sequences, writing the number of emitted keys to
  //! ``d_num_selected_out``.
  //! @endrst
  template <typename KeyIteratorIn1,
            typename KeyIteratorIn2,
            typename KeyIteratorOut,
            typename NumSelectedIteratorT,
            typename OffsetT,
            typename CompareOp = ::cuda::std::less<>,
            typename EnvT      = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t SetDifference(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyIteratorIn1 d_keys_in1,
    OffsetT num_keys1,
    KeyIteratorIn2 d_keys_in2,
    OffsetT num_keys2,
    KeyIteratorOut d_keys_out,
    NumSelectedIteratorT d_num_selected_out,
    CompareOp compare_op = {},
    const EnvT& env      = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSetOps::SetDifference");
    return set_op_keys<detail::set_ops::serial_set_difference>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in1,
      num_keys1,
      d_keys_in2,
      num_keys2,
      d_keys_out,
      d_num_selected_out,
      compare_op,
      env);
  }

  //! @rst
  //! Environment-based overload of @ref SetDifference that allocates the temporary storage from the memory resource
  //! provided by ``env`` (default: ``cuda::mr::device_memory_resource``). The stream and tuning are also queried from
  //! ``env``.
  //! @endrst
  template <typename KeyIteratorIn1,
            typename KeyIteratorIn2,
            typename KeyIteratorOut,
            typename NumSelectedIteratorT,
            typename OffsetT,
            typename CompareOp = ::cuda::std::less<>,
            typename EnvT      = ::cuda::std::execution::env<>,
            ::cuda::std::enable_if_t<enable_env_overload<KeyIteratorIn1, KeyIteratorIn2, CompareOp>, int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t SetDifference(
    KeyIteratorIn1 d_keys_in1,
    OffsetT num_keys1,
    KeyIteratorIn2 d_keys_in2,
    OffsetT num_keys2,
    KeyIteratorOut d_keys_out,
    NumSelectedIteratorT d_num_selected_out,
    CompareOp compare_op = {},
    const EnvT& env      = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSetOps::SetDifference");
    return set_op_keys_env<detail::set_ops::serial_set_difference>(
      d_keys_in1, num_keys1, d_keys_in2, num_keys2, d_keys_out, d_num_selected_out, compare_op, env);
  }

  //! @rst
  //! Computes the set intersection ``keys1 ∩ keys2`` of two sorted key sequences, writing the number of emitted keys to
  //! ``d_num_selected_out``.
  //! @endrst
  template <typename KeyIteratorIn1,
            typename KeyIteratorIn2,
            typename KeyIteratorOut,
            typename NumSelectedIteratorT,
            typename OffsetT,
            typename CompareOp = ::cuda::std::less<>,
            typename EnvT      = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t SetIntersection(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyIteratorIn1 d_keys_in1,
    OffsetT num_keys1,
    KeyIteratorIn2 d_keys_in2,
    OffsetT num_keys2,
    KeyIteratorOut d_keys_out,
    NumSelectedIteratorT d_num_selected_out,
    CompareOp compare_op = {},
    const EnvT& env      = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSetOps::SetIntersection");
    return set_op_keys<detail::set_ops::serial_set_intersection>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in1,
      num_keys1,
      d_keys_in2,
      num_keys2,
      d_keys_out,
      d_num_selected_out,
      compare_op,
      env);
  }

  //! @rst
  //! Environment-based overload of @ref SetIntersection that allocates the temporary storage from the memory resource
  //! provided by ``env`` (default: ``cuda::mr::device_memory_resource``). The stream and tuning are also queried from
  //! ``env``.
  //! @endrst
  template <typename KeyIteratorIn1,
            typename KeyIteratorIn2,
            typename KeyIteratorOut,
            typename NumSelectedIteratorT,
            typename OffsetT,
            typename CompareOp = ::cuda::std::less<>,
            typename EnvT      = ::cuda::std::execution::env<>,
            ::cuda::std::enable_if_t<enable_env_overload<KeyIteratorIn1, KeyIteratorIn2, CompareOp>, int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t SetIntersection(
    KeyIteratorIn1 d_keys_in1,
    OffsetT num_keys1,
    KeyIteratorIn2 d_keys_in2,
    OffsetT num_keys2,
    KeyIteratorOut d_keys_out,
    NumSelectedIteratorT d_num_selected_out,
    CompareOp compare_op = {},
    const EnvT& env      = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSetOps::SetIntersection");
    return set_op_keys_env<detail::set_ops::serial_set_intersection>(
      d_keys_in1, num_keys1, d_keys_in2, num_keys2, d_keys_out, d_num_selected_out, compare_op, env);
  }

  //! @rst
  //! Computes the set symmetric difference ``keys1 △ keys2`` of two sorted key sequences, writing the number of emitted
  //! keys to ``d_num_selected_out``.
  //! @endrst
  template <typename KeyIteratorIn1,
            typename KeyIteratorIn2,
            typename KeyIteratorOut,
            typename NumSelectedIteratorT,
            typename OffsetT,
            typename CompareOp = ::cuda::std::less<>,
            typename EnvT      = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t SetSymmetricDifference(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyIteratorIn1 d_keys_in1,
    OffsetT num_keys1,
    KeyIteratorIn2 d_keys_in2,
    OffsetT num_keys2,
    KeyIteratorOut d_keys_out,
    NumSelectedIteratorT d_num_selected_out,
    CompareOp compare_op = {},
    const EnvT& env      = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSetOps::SetSymmetricDifference");
    return set_op_keys<detail::set_ops::serial_set_symmetric_difference>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in1,
      num_keys1,
      d_keys_in2,
      num_keys2,
      d_keys_out,
      d_num_selected_out,
      compare_op,
      env);
  }

  //! @rst
  //! Environment-based overload of @ref SetSymmetricDifference that allocates the temporary storage from the memory
  //! resource provided by ``env`` (default: ``cuda::mr::device_memory_resource``). The stream and tuning are also
  //! queried from ``env``.
  //! @endrst
  template <typename KeyIteratorIn1,
            typename KeyIteratorIn2,
            typename KeyIteratorOut,
            typename NumSelectedIteratorT,
            typename OffsetT,
            typename CompareOp = ::cuda::std::less<>,
            typename EnvT      = ::cuda::std::execution::env<>,
            ::cuda::std::enable_if_t<enable_env_overload<KeyIteratorIn1, KeyIteratorIn2, CompareOp>, int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t SetSymmetricDifference(
    KeyIteratorIn1 d_keys_in1,
    OffsetT num_keys1,
    KeyIteratorIn2 d_keys_in2,
    OffsetT num_keys2,
    KeyIteratorOut d_keys_out,
    NumSelectedIteratorT d_num_selected_out,
    CompareOp compare_op = {},
    const EnvT& env      = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSetOps::SetSymmetricDifference");
    return set_op_keys_env<detail::set_ops::serial_set_symmetric_difference>(
      d_keys_in1, num_keys1, d_keys_in2, num_keys2, d_keys_out, d_num_selected_out, compare_op, env);
  }

  //! @rst
  //! Computes the set union ``keys1 ∪ keys2`` of two sorted key sequences, writing the number of emitted keys to
  //! ``d_num_selected_out``.
  //! @endrst
  template <typename KeyIteratorIn1,
            typename KeyIteratorIn2,
            typename KeyIteratorOut,
            typename NumSelectedIteratorT,
            typename OffsetT,
            typename CompareOp = ::cuda::std::less<>,
            typename EnvT      = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t SetUnion(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyIteratorIn1 d_keys_in1,
    OffsetT num_keys1,
    KeyIteratorIn2 d_keys_in2,
    OffsetT num_keys2,
    KeyIteratorOut d_keys_out,
    NumSelectedIteratorT d_num_selected_out,
    CompareOp compare_op = {},
    const EnvT& env      = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSetOps::SetUnion");
    return set_op_keys<detail::set_ops::serial_set_union>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in1,
      num_keys1,
      d_keys_in2,
      num_keys2,
      d_keys_out,
      d_num_selected_out,
      compare_op,
      env);
  }

  //! @rst
  //! Environment-based overload of @ref SetUnion that allocates the temporary storage from the memory resource provided
  //! by ``env`` (default: ``cuda::mr::device_memory_resource``). The stream and tuning are also queried from ``env``.
  //! @endrst
  template <typename KeyIteratorIn1,
            typename KeyIteratorIn2,
            typename KeyIteratorOut,
            typename NumSelectedIteratorT,
            typename OffsetT,
            typename CompareOp = ::cuda::std::less<>,
            typename EnvT      = ::cuda::std::execution::env<>,
            ::cuda::std::enable_if_t<enable_env_overload<KeyIteratorIn1, KeyIteratorIn2, CompareOp>, int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t SetUnion(
    KeyIteratorIn1 d_keys_in1,
    OffsetT num_keys1,
    KeyIteratorIn2 d_keys_in2,
    OffsetT num_keys2,
    KeyIteratorOut d_keys_out,
    NumSelectedIteratorT d_num_selected_out,
    CompareOp compare_op = {},
    const EnvT& env      = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSetOps::SetUnion");
    return set_op_keys_env<detail::set_ops::serial_set_union>(
      d_keys_in1, num_keys1, d_keys_in2, num_keys2, d_keys_out, d_num_selected_out, compare_op, env);
  }

  //! @rst
  //! Key-value variant of @ref SetDifference. Keys present in the output are accompanied by the value from the first
  //! input sequence.
  //! @endrst
  template <typename KeyIteratorIn1,
            typename ValueIteratorIn1,
            typename KeyIteratorIn2,
            typename ValueIteratorIn2,
            typename KeyIteratorOut,
            typename ValueIteratorOut,
            typename NumSelectedIteratorT,
            typename OffsetT,
            typename CompareOp = ::cuda::std::less<>,
            typename EnvT      = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t SetDifferencePairs(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyIteratorIn1 d_keys_in1,
    ValueIteratorIn1 d_values_in1,
    OffsetT num_pairs1,
    KeyIteratorIn2 d_keys_in2,
    ValueIteratorIn2 d_values_in2,
    OffsetT num_pairs2,
    KeyIteratorOut d_keys_out,
    ValueIteratorOut d_values_out,
    NumSelectedIteratorT d_num_selected_out,
    CompareOp compare_op = {},
    const EnvT& env      = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSetOps::SetDifferencePairs");
    return set_op_pairs<detail::set_ops::serial_set_difference>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in1,
      d_values_in1,
      num_pairs1,
      d_keys_in2,
      d_values_in2,
      num_pairs2,
      d_keys_out,
      d_values_out,
      d_num_selected_out,
      compare_op,
      env);
  }

  //! @rst
  //! Environment-based overload of @ref SetDifferencePairs that allocates the temporary storage from the memory
  //! resource provided by ``env`` (default: ``cuda::mr::device_memory_resource``). The stream and tuning are also
  //! queried from
  //! ``env``.
  //! @endrst
  template <typename KeyIteratorIn1,
            typename ValueIteratorIn1,
            typename KeyIteratorIn2,
            typename ValueIteratorIn2,
            typename KeyIteratorOut,
            typename ValueIteratorOut,
            typename NumSelectedIteratorT,
            typename OffsetT,
            typename CompareOp = ::cuda::std::less<>,
            typename EnvT      = ::cuda::std::execution::env<>,
            ::cuda::std::enable_if_t<enable_env_overload<KeyIteratorIn1, KeyIteratorIn2, CompareOp>, int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t SetDifferencePairs(
    KeyIteratorIn1 d_keys_in1,
    ValueIteratorIn1 d_values_in1,
    OffsetT num_pairs1,
    KeyIteratorIn2 d_keys_in2,
    ValueIteratorIn2 d_values_in2,
    OffsetT num_pairs2,
    KeyIteratorOut d_keys_out,
    ValueIteratorOut d_values_out,
    NumSelectedIteratorT d_num_selected_out,
    CompareOp compare_op = {},
    const EnvT& env      = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSetOps::SetDifferencePairs");
    return set_op_pairs_env<detail::set_ops::serial_set_difference>(
      d_keys_in1,
      d_values_in1,
      num_pairs1,
      d_keys_in2,
      d_values_in2,
      num_pairs2,
      d_keys_out,
      d_values_out,
      d_num_selected_out,
      compare_op,
      env);
  }

  //! @rst
  //! Key-value variant of @ref SetIntersection. Keys present in the output are accompanied by the value from the first
  //! input sequence.
  //! @endrst
  template <typename KeyIteratorIn1,
            typename ValueIteratorIn1,
            typename KeyIteratorIn2,
            typename ValueIteratorIn2,
            typename KeyIteratorOut,
            typename ValueIteratorOut,
            typename NumSelectedIteratorT,
            typename OffsetT,
            typename CompareOp = ::cuda::std::less<>,
            typename EnvT      = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t SetIntersectionPairs(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyIteratorIn1 d_keys_in1,
    ValueIteratorIn1 d_values_in1,
    OffsetT num_pairs1,
    KeyIteratorIn2 d_keys_in2,
    ValueIteratorIn2 d_values_in2,
    OffsetT num_pairs2,
    KeyIteratorOut d_keys_out,
    ValueIteratorOut d_values_out,
    NumSelectedIteratorT d_num_selected_out,
    CompareOp compare_op = {},
    const EnvT& env      = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSetOps::SetIntersectionPairs");
    return set_op_pairs<detail::set_ops::serial_set_intersection>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in1,
      d_values_in1,
      num_pairs1,
      d_keys_in2,
      d_values_in2,
      num_pairs2,
      d_keys_out,
      d_values_out,
      d_num_selected_out,
      compare_op,
      env);
  }

  //! @rst
  //! Environment-based overload of @ref SetIntersectionPairs that allocates the temporary storage from the memory
  //! resource provided by ``env`` (default: ``cuda::mr::device_memory_resource``). The stream and tuning are also
  //! queried from ``env``.
  //! @endrst
  template <typename KeyIteratorIn1,
            typename ValueIteratorIn1,
            typename KeyIteratorIn2,
            typename ValueIteratorIn2,
            typename KeyIteratorOut,
            typename ValueIteratorOut,
            typename NumSelectedIteratorT,
            typename OffsetT,
            typename CompareOp = ::cuda::std::less<>,
            typename EnvT      = ::cuda::std::execution::env<>,
            ::cuda::std::enable_if_t<enable_env_overload<KeyIteratorIn1, KeyIteratorIn2, CompareOp>, int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t SetIntersectionPairs(
    KeyIteratorIn1 d_keys_in1,
    ValueIteratorIn1 d_values_in1,
    OffsetT num_pairs1,
    KeyIteratorIn2 d_keys_in2,
    ValueIteratorIn2 d_values_in2,
    OffsetT num_pairs2,
    KeyIteratorOut d_keys_out,
    ValueIteratorOut d_values_out,
    NumSelectedIteratorT d_num_selected_out,
    CompareOp compare_op = {},
    const EnvT& env      = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSetOps::SetIntersectionPairs");
    return set_op_pairs_env<detail::set_ops::serial_set_intersection>(
      d_keys_in1,
      d_values_in1,
      num_pairs1,
      d_keys_in2,
      d_values_in2,
      num_pairs2,
      d_keys_out,
      d_values_out,
      d_num_selected_out,
      compare_op,
      env);
  }

  //! @rst
  //! Key-value variant of @ref SetSymmetricDifference. Keys taken from the first input carry the value from the first
  //! input; keys taken from the second input carry the value from the second input.
  //! @endrst
  template <typename KeyIteratorIn1,
            typename ValueIteratorIn1,
            typename KeyIteratorIn2,
            typename ValueIteratorIn2,
            typename KeyIteratorOut,
            typename ValueIteratorOut,
            typename NumSelectedIteratorT,
            typename OffsetT,
            typename CompareOp = ::cuda::std::less<>,
            typename EnvT      = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t SetSymmetricDifferencePairs(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyIteratorIn1 d_keys_in1,
    ValueIteratorIn1 d_values_in1,
    OffsetT num_pairs1,
    KeyIteratorIn2 d_keys_in2,
    ValueIteratorIn2 d_values_in2,
    OffsetT num_pairs2,
    KeyIteratorOut d_keys_out,
    ValueIteratorOut d_values_out,
    NumSelectedIteratorT d_num_selected_out,
    CompareOp compare_op = {},
    const EnvT& env      = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSetOps::SetSymmetricDifferencePairs");
    return set_op_pairs<detail::set_ops::serial_set_symmetric_difference>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in1,
      d_values_in1,
      num_pairs1,
      d_keys_in2,
      d_values_in2,
      num_pairs2,
      d_keys_out,
      d_values_out,
      d_num_selected_out,
      compare_op,
      env);
  }

  //! @rst
  //! Environment-based overload of @ref SetSymmetricDifferencePairs that allocates the temporary storage from the
  //! memory resource provided by ``env`` (default: ``cuda::mr::device_memory_resource``). The stream and tuning are
  //! also queried from ``env``.
  //! @endrst
  template <typename KeyIteratorIn1,
            typename ValueIteratorIn1,
            typename KeyIteratorIn2,
            typename ValueIteratorIn2,
            typename KeyIteratorOut,
            typename ValueIteratorOut,
            typename NumSelectedIteratorT,
            typename OffsetT,
            typename CompareOp = ::cuda::std::less<>,
            typename EnvT      = ::cuda::std::execution::env<>,
            ::cuda::std::enable_if_t<enable_env_overload<KeyIteratorIn1, KeyIteratorIn2, CompareOp>, int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t SetSymmetricDifferencePairs(
    KeyIteratorIn1 d_keys_in1,
    ValueIteratorIn1 d_values_in1,
    OffsetT num_pairs1,
    KeyIteratorIn2 d_keys_in2,
    ValueIteratorIn2 d_values_in2,
    OffsetT num_pairs2,
    KeyIteratorOut d_keys_out,
    ValueIteratorOut d_values_out,
    NumSelectedIteratorT d_num_selected_out,
    CompareOp compare_op = {},
    const EnvT& env      = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSetOps::SetSymmetricDifferencePairs");
    return set_op_pairs_env<detail::set_ops::serial_set_symmetric_difference>(
      d_keys_in1,
      d_values_in1,
      num_pairs1,
      d_keys_in2,
      d_values_in2,
      num_pairs2,
      d_keys_out,
      d_values_out,
      d_num_selected_out,
      compare_op,
      env);
  }

  //! @rst
  //! Key-value variant of @ref SetUnion. In case of a tie, the key and value are taken from the first input sequence.
  //! @endrst
  template <typename KeyIteratorIn1,
            typename ValueIteratorIn1,
            typename KeyIteratorIn2,
            typename ValueIteratorIn2,
            typename KeyIteratorOut,
            typename ValueIteratorOut,
            typename NumSelectedIteratorT,
            typename OffsetT,
            typename CompareOp = ::cuda::std::less<>,
            typename EnvT      = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t SetUnionPairs(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyIteratorIn1 d_keys_in1,
    ValueIteratorIn1 d_values_in1,
    OffsetT num_pairs1,
    KeyIteratorIn2 d_keys_in2,
    ValueIteratorIn2 d_values_in2,
    OffsetT num_pairs2,
    KeyIteratorOut d_keys_out,
    ValueIteratorOut d_values_out,
    NumSelectedIteratorT d_num_selected_out,
    CompareOp compare_op = {},
    const EnvT& env      = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSetOps::SetUnionPairs");
    return set_op_pairs<detail::set_ops::serial_set_union>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in1,
      d_values_in1,
      num_pairs1,
      d_keys_in2,
      d_values_in2,
      num_pairs2,
      d_keys_out,
      d_values_out,
      d_num_selected_out,
      compare_op,
      env);
  }

  //! @rst
  //! Environment-based overload of @ref SetUnionPairs that allocates the temporary storage from the memory resource
  //! provided by ``env`` (default: ``cuda::mr::device_memory_resource``). The stream and tuning are also queried from
  //! ``env``.
  //! @endrst
  template <typename KeyIteratorIn1,
            typename ValueIteratorIn1,
            typename KeyIteratorIn2,
            typename ValueIteratorIn2,
            typename KeyIteratorOut,
            typename ValueIteratorOut,
            typename NumSelectedIteratorT,
            typename OffsetT,
            typename CompareOp = ::cuda::std::less<>,
            typename EnvT      = ::cuda::std::execution::env<>,
            ::cuda::std::enable_if_t<enable_env_overload<KeyIteratorIn1, KeyIteratorIn2, CompareOp>, int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t SetUnionPairs(
    KeyIteratorIn1 d_keys_in1,
    ValueIteratorIn1 d_values_in1,
    OffsetT num_pairs1,
    KeyIteratorIn2 d_keys_in2,
    ValueIteratorIn2 d_values_in2,
    OffsetT num_pairs2,
    KeyIteratorOut d_keys_out,
    ValueIteratorOut d_values_out,
    NumSelectedIteratorT d_num_selected_out,
    CompareOp compare_op = {},
    const EnvT& env      = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSetOps::SetUnionPairs");
    return set_op_pairs_env<detail::set_ops::serial_set_union>(
      d_keys_in1,
      d_values_in1,
      num_pairs1,
      d_keys_in2,
      d_values_in2,
      num_pairs2,
      d_keys_out,
      d_values_out,
      d_num_selected_out,
      compare_op,
      env);
  }
};

CUB_NAMESPACE_END
