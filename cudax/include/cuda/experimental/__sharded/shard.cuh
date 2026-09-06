//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief `shard<T>`: one contiguous, placed piece of a sharded array.
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__places/places.cuh>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
using ::cuda::experimental::places::data_place;
using ::cuda::experimental::places::exec_place;

/**
 * @brief A single contiguous piece of a sharded array.
 *
 * A shard couples a span of elements with its placement: the `data_place`
 * where the memory lives, the `exec_place` to activate when operating on it,
 * and a reference stream for stream-ordered operations. `global_offset` is
 * the shard's starting index in the logical (whole-array) index space.
 */
template <typename _Tp>
struct shard
{
  _Tp* data            = nullptr; //!< pointer to the shard's elements
  size_t size          = 0; //!< number of valid elements (logical size)
  size_t capacity      = 0; //!< allocated capacity (>= size)
  size_t global_offset = 0; //!< starting index in the logical array
  data_place place; //!< where the memory lives
  exec_place exec; //!< execution place to activate for this shard
  cudaStream_t stream = nullptr; //!< reference stream for stream-ordered operations

  // Iterators over valid elements
  _Tp* begin()
  {
    return data;
  }
  _Tp* end()
  {
    return data + size;
  }
  const _Tp* begin() const
  {
    return data;
  }
  const _Tp* end() const
  {
    return data + size;
  }

  /// @brief Logical size in bytes.
  [[nodiscard]] _CCCL_HOST_DEVICE_API size_t size_bytes() const noexcept
  {
    return size * sizeof(_Tp);
  }

  /// @brief Allocated capacity in bytes.
  [[nodiscard]] _CCCL_HOST_DEVICE_API size_t capacity_bytes() const noexcept
  {
    return capacity * sizeof(_Tp);
  }

  /// @brief Reset the logical size to the full capacity (buffer reuse).
  void reset_to_capacity()
  {
    size = capacity;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool empty() const noexcept
  {
    return size == 0 || data == nullptr;
  }

  /// @brief First global index covered by this shard.
  [[nodiscard]] _CCCL_HOST_DEVICE_API size_t global_begin() const noexcept
  {
    return global_offset;
  }

  /// @brief One-past-the-last global index covered by this shard.
  [[nodiscard]] _CCCL_HOST_DEVICE_API size_t global_end() const noexcept
  {
    return global_offset + size;
  }

  /// @brief Whether a global index falls within this shard.
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool contains(size_t global_idx) const noexcept
  {
    return global_idx >= global_offset && global_idx < global_offset + size;
  }

  /// @brief Convert a global index to a shard-local index.
  [[nodiscard]] _CCCL_HOST_DEVICE_API size_t to_local(size_t global_idx) const noexcept
  {
    _CCCL_ASSERT(contains(global_idx), "shard::to_local: global index outside this shard");
    return global_idx - global_offset;
  }

  /// @brief Convert a shard-local index to a global index.
  [[nodiscard]] _CCCL_HOST_DEVICE_API size_t to_global(size_t local_idx) const noexcept
  {
    _CCCL_ASSERT(local_idx < size, "shard::to_global: local index out of range");
    return global_offset + local_idx;
  }
};
} // namespace cuda::experimental::sharded
