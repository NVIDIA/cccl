//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/** @file
 * @brief Widely used artifacts used by most of the library (subset that is compatible with nvrtc)
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

#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>
#include <cuda/experimental/__stf/utility/hash.cuh>

namespace cuda::experimental::stf
{
namespace reserved
{

template <typename Kernel>
::std::pair<int /*min_grid_size*/, int /*block_size*/>
compute_occupancy(Kernel&& f, size_t dynamicSMemSize = 0, int blockSizeLimit = 0)
{
  using key_t = ::std::pair<size_t /*dynamicSMemSize*/, int /*blockSizeLimit*/>;
  static ::std::
    unordered_map<key_t, ::std::pair<int /*min_grid_size*/, int /*block_size*/>, ::cuda::experimental::stf::hash<key_t>>
      occupancy_cache;
  const auto key = ::std::make_pair(dynamicSMemSize, blockSizeLimit);

  if (auto i = occupancy_cache.find(key); i != occupancy_cache.end())
  {
    // Cache hit
    return i->second;
  }
  // Miss
  auto& result = occupancy_cache[key];
  if constexpr (::std::is_same_v<::std::decay_t<Kernel>, CUfunction>)
  {
    cuda_safe_call(
      cuOccupancyMaxPotentialBlockSize(&result.first, &result.second, f, nullptr, dynamicSMemSize, blockSizeLimit));
  }
  else
  {
    cuda_safe_call(
      cudaOccupancyMaxPotentialBlockSize(&result.first, &result.second, f, dynamicSMemSize, blockSizeLimit));
  }
  return result;
}

/**
 * This method computes the block and grid sizes to optimize thread occupancy.
 *
 * If cooperative kernels are needed, the grid size is capped to the number of
 * blocks
 *
 * - min_grid_size and max_block_size are the grid and block sizes to
 *   _optimize_ occupancy
 * - block_size_limit is the absolute maximum of threads in a block due to
 *   resource constraints
 */
template <typename Fun>
void compute_kernel_limits(
  const Fun&& f,
  int& min_grid_size,
  int& max_block_size,
  size_t shared_mem_bytes,
  bool cooperative,
  int& block_size_limit)
{
  static_assert(::std::is_function<typename ::std::remove_pointer<Fun>::type>::value,
                "Template parameter Fun must be a pointer to a function type.");

  ::std::tie(min_grid_size, max_block_size) = compute_occupancy(f, shared_mem_bytes);

  if (cooperative)
  {
    // For cooperative kernels, the number of blocks is limited. We compute the number of SM on device 0 and assume
    // we have a homogeneous machine.
    static const int sm_count = cuda_try<cudaDeviceGetAttribute>(cudaDevAttrMultiProcessorCount, 0);

    // TODO there could be more than 1 block per SM, but we do not know the actual block sizes for now ...
    min_grid_size = ::std::min(min_grid_size, sm_count);
  }

  /* Compute the maximum block size (not the optimal size) */
  static const auto maxThreadsPerBlock = [&] {
    cudaFuncAttributes result;
    cuda_safe_call(cudaFuncGetAttributes(&result, f));
    return result.maxThreadsPerBlock;
  }();
  block_size_limit = maxThreadsPerBlock;
}

} // end namespace reserved
} // namespace cuda::experimental::stf
