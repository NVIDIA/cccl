// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * \file
 * Utilities for device-accessible temporary storages.
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/util_debug.cuh>
#include <cub/util_namespace.cuh>

#include <cuda/__cmath/round_up.h>
#include <cuda/__memory/align_up.h>

CUB_NAMESPACE_BEGIN

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

namespace detail
{
//! @brief Alias temporaries to externally-allocated device storage (or simply return the amount of storage needed).
//!
//! @param[in] d_temp_storage
//!   Device-accessible allocation of temporary storage.
//!   When nullptr, the required allocation size is written to @p temp_storage_bytes and no work is done.
//!
//! @param[in,out] temp_storage_bytes
//!   Size in bytes of @p d_temp_storage allocation
//!
//! @param[in,out] allocations
//!   Pointers to device allocations needed
//!
//! @param[in] allocation_sizes
//!   Sizes in bytes of device allocations needed
template <int NumAllocations>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE cudaError_t alias_temporaries(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  void* (&allocations)[NumAllocations],
  const size_t (&allocation_sizes)[NumAllocations])
{
  constexpr size_t align_bytes = 256;

  // Compute exclusive prefix sum over allocation requests
  size_t allocation_offsets[NumAllocations];
  size_t bytes_needed = 0;
  for (int i = 0; i < NumAllocations; ++i)
  {
    allocation_offsets[i] = bytes_needed;
    bytes_needed += ::cuda::round_up(allocation_sizes[i], +align_bytes);
  }
  bytes_needed += align_bytes - 1;

  // Check if the caller is simply requesting the size of the storage allocation
  if (!d_temp_storage)
  {
    temp_storage_bytes = bytes_needed;
    return cudaSuccess;
  }

  // Check if enough storage provided
  if (temp_storage_bytes < bytes_needed)
  {
    return CubDebug(cudaErrorInvalidValue);
  }

  // Alias
  d_temp_storage = ::cuda::align_up(d_temp_storage, align_bytes);
  for (int i = 0; i < NumAllocations; ++i)
  {
    allocations[i] = static_cast<char*>(d_temp_storage) + allocation_offsets[i];
  }

  return cudaSuccess;
}
} // namespace detail

#endif // _CCCL_DOXYGEN_INVOKED

CUB_NAMESPACE_END
