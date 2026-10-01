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

#include <cub/detail/choose_offset.cuh>
#include <cub/detail/env_dispatch.cuh>
#include <cub/device/dispatch/dispatch_set_operations.cuh>
#include <cub/util_namespace.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/__execution/env.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_same.h>

CUB_NAMESPACE_BEGIN

// TODO(bgruber): expose DeviceSetOps publicly (move it out of the detail namespace). It is kept internal for now
// because the offset-type handling is not robust enough yet: choose_offset_t deduces the offset from a single size
// type, which requires that type to be able to represent the sum of both input lengths -- a dangerous precondition to
// place on users. Before exposing it, either hard-code a 64-bit offset type or derive the offset type more carefully
// from the two input size types.
namespace detail
{
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
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t set_op_keys(
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
    return detail::dispatch_with_env(
      d_temp_storage,
      temp_storage_bytes,
      env,
      [&](auto tuning_env, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t stream) {
        return detail::set_ops::dispatch(
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
          tuning_env);
      });
  }

  template <typename SetOp,
            typename KeyIteratorIn1,
            typename KeyIteratorIn2,
            typename KeyIteratorOut,
            typename NumSelectedIteratorT,
            typename OffsetT,
            typename CompareOp,
            typename EnvT>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t set_op_keys_env(
    KeyIteratorIn1 d_keys_in1,
    OffsetT num_keys1,
    KeyIteratorIn2 d_keys_in2,
    OffsetT num_keys2,
    KeyIteratorOut d_keys_out,
    NumSelectedIteratorT d_num_selected_out,
    CompareOp compare_op,
    const EnvT& env)
  {
    return detail::dispatch_with_env(
      env, [&](auto tuning_env, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t stream) {
        return detail::set_ops::dispatch(
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
          tuning_env);
      });
  }

public:
  //! Computes the set difference `keys1 \ keys2` of two sorted key sequences, writing the number of emitted keys to
  //! `d_num_selected_out`.
  //!
  //! @tparam NumKeysT
  //!   Type of `num_keys1` and `num_keys2`. Their sum (the combined input size) must be representable by this type.
  template <typename KeyIteratorIn1,
            typename KeyIteratorIn2,
            typename KeyIteratorOut,
            typename NumSelectedIteratorT,
            typename NumKeysT,
            typename CompareOp = ::cuda::std::less<>,
            typename EnvT      = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t SetDifference(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyIteratorIn1 d_keys_in1,
    NumKeysT num_keys1,
    KeyIteratorIn2 d_keys_in2,
    NumKeysT num_keys2,
    KeyIteratorOut d_keys_out,
    NumSelectedIteratorT d_num_selected_out,
    CompareOp compare_op = {},
    const EnvT& env      = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::detail::DeviceSetOps::SetDifference");
    using offset_t = detail::choose_offset_t<NumKeysT>;
    return set_op_keys<detail::set_ops::serial_set_difference>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in1,
      static_cast<offset_t>(num_keys1),
      d_keys_in2,
      static_cast<offset_t>(num_keys2),
      d_keys_out,
      d_num_selected_out,
      compare_op,
      env);
  }

  //! @rst
  //! Environment-based overload of @ref SetDifference that allocates the temporary storage from the memory resource
  //! provided by ``env`` (default: ``cuda::mr::device_memory_resource``). The stream and tuning are also queried from
  //! ``env``.
  //!
  //! Snippet
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_set_operations_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin set-difference-env
  //!     :end-before: example-end set-difference-env
  //!
  //! @endrst
  //!
  //! @tparam NumKeysT
  //!   Type of `num_keys1` and `num_keys2`. Their sum (the combined input size) must be representable by this type.
  template <
    typename KeyIteratorIn1,
    typename KeyIteratorIn2,
    typename KeyIteratorOut,
    typename NumSelectedIteratorT,
    typename NumKeysT,
    typename CompareOp                                                            = ::cuda::std::less<>,
    typename EnvT                                                                 = ::cuda::std::execution::env<>,
    ::cuda::std::enable_if_t<!::cuda::std::is_same_v<KeyIteratorIn1, void*>, int> = 0,
    ::cuda::std::enable_if_t<!::cuda::std::is_same_v<KeyIteratorIn1, ::cuda::std::nullptr_t>, int> = 0,
    ::cuda::std::enable_if_t<::cuda::std::indirect_binary_predicate<CompareOp, KeyIteratorIn1, KeyIteratorIn2>, int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t SetDifference(
    KeyIteratorIn1 d_keys_in1,
    NumKeysT num_keys1,
    KeyIteratorIn2 d_keys_in2,
    NumKeysT num_keys2,
    KeyIteratorOut d_keys_out,
    NumSelectedIteratorT d_num_selected_out,
    CompareOp compare_op = {},
    const EnvT& env      = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::detail::DeviceSetOps::SetDifference");
    using offset_t = detail::choose_offset_t<NumKeysT>;
    return set_op_keys_env<detail::set_ops::serial_set_difference>(
      d_keys_in1,
      static_cast<offset_t>(num_keys1),
      d_keys_in2,
      static_cast<offset_t>(num_keys2),
      d_keys_out,
      d_num_selected_out,
      compare_op,
      env);
  }
};
} // namespace detail

CUB_NAMESPACE_END
