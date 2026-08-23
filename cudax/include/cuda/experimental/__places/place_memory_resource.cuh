//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief `place_memory_resource`: a `data_place` adapter modeling the
 *        `cuda::mr` resource concepts.
 *
 * Independent of `place_group`: any `data_place` (device, host, managed,
 * green-context, locality-domain) becomes usable wherever CCCL expects a
 * memory resource. `place_group` builds its per-place conveniences
 * (`memory_resource(i)`, `env(i)`) on top of this adapter.
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

#include <cuda/memory_resource>
#include <cuda/stream>

#include <cuda/experimental/__places/places.cuh>

#include <cstddef>
#include <cstdio>
#include <exception>
#include <stdexcept>

#include <cuda_runtime.h>

namespace cuda::experimental::places
{
// ============================================================================
// place_memory_resource: per-place memory resource over data_place
// ============================================================================

/**
 * @brief A memory resource that allocates from a `data_place`, modeling the
 * `cuda::mr` resource concepts.
 *
 * This makes any place (device, host, managed, green-context, locality
 * domain) usable wherever CCCL expects a memory resource — in particular in
 * the environments accepted by CUB's single-call device algorithms, so that
 * algorithm temporaries land on the same place as the data they serve.
 *
 * Stream-ordered places allocate/deallocate on the provided stream; other
 * places fall back to their immediate allocation path.
 */
class place_memory_resource
{
public:
  /// @brief Construct a memory resource allocating from @p place.
  explicit place_memory_resource(data_place place)
      : place_(mv(place))
      , is_stream_ordered_(place_.allocation_is_stream_ordered())
  {}

  /// @brief The underlying data place.
  const data_place& place() const
  {
    return place_;
  }

  /// @brief Whether allocations are stream-ordered on this place.
  bool is_stream_ordered() const
  {
    return is_stream_ordered_;
  }

  /// @brief Stream-ordered allocation (models the `cuda::mr` resource concept).
  [[nodiscard]] void*
  allocate(::cuda::stream_ref stream, ::std::size_t bytes, ::std::size_t /*alignment*/ = alignof(::std::max_align_t))
  {
    if (bytes == 0)
    {
      return nullptr;
    }
    cudaStream_t cuda_stream = is_stream_ordered_ ? stream.get() : nullptr;
    return place_.allocate(static_cast<::std::ptrdiff_t>(bytes), cuda_stream);
  }

  /// @brief Stream-ordered deallocation (models the `cuda::mr` resource concept).
  void deallocate(::cuda::stream_ref stream,
                  void* ptr,
                  ::std::size_t bytes,
                  ::std::size_t /*alignment*/ = alignof(::std::max_align_t)) noexcept
  {
    if (ptr == nullptr)
    {
      return;
    }
    cudaStream_t cuda_stream = is_stream_ordered_ ? stream.get() : nullptr;
    // Deallocation is noexcept by the convention of the cuda::mr resources
    // (their bodies never throw); a deallocation failure is not recoverable,
    // so report it rather than terminate through the noexcept boundary.
    try
    {
      place_.deallocate(ptr, bytes, cuda_stream);
    }
    catch (const ::std::exception& e)
    {
      ::fprintf(stderr, "place_memory_resource::deallocate failed: %s\n", e.what());
    }
  }

  /// @brief Synchronous allocation (models the `cuda::mr` synchronous resource concept).
  [[nodiscard]] void* allocate_sync(::std::size_t bytes, ::std::size_t /*alignment*/ = alignof(::std::max_align_t))
  {
    if (bytes == 0)
    {
      return nullptr;
    }
    return place_.allocate(static_cast<::std::ptrdiff_t>(bytes), nullptr);
  }

  /// @brief Synchronous deallocation (models the `cuda::mr` synchronous resource concept).
  void
  deallocate_sync(void* ptr, ::std::size_t bytes, ::std::size_t /*alignment*/ = alignof(::std::max_align_t)) noexcept
  {
    if (ptr == nullptr)
    {
      return;
    }
    try
    {
      place_.deallocate(ptr, bytes, nullptr);
    }
    catch (const ::std::exception& e)
    {
      ::fprintf(stderr, "place_memory_resource::deallocate_sync failed: %s\n", e.what());
    }
  }

  /// @brief Two resources are equal when they allocate from the same place.
  friend bool operator==(const place_memory_resource& lhs, const place_memory_resource& rhs) noexcept
  {
    return lhs.place_ == rhs.place_;
  }

  friend bool operator!=(const place_memory_resource& lhs, const place_memory_resource& rhs) noexcept
  {
    return !(lhs == rhs);
  }

private:
  data_place place_;
  bool is_stream_ordered_;
};
} // namespace cuda::experimental::places
