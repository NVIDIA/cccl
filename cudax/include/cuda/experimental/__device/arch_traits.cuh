//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__DEVICE_ARCH_TRAITS
#define _CUDAX__DEVICE_ARCH_TRAITS

#include <cuda/__cccl_config>
#include <cuda/std/__exception/cuda_error.h>
#include <cuda/std/limits>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

namespace cuda::experimental
{

namespace detail
{
struct arch_common_traits
{
  // TODO: Should these be in block_level, grid_level?
  // Maximum number of threads per block
  static constexpr int max_threads_per_block = 1024;

  // Maximum x-dimension of a block
  static constexpr int max_block_dim_x = 1024;

  // Maximum y-dimension of a block
  static constexpr int max_block_dim_y = 1024;

  // Maximum z-dimension of a block
  static constexpr int max_block_dim_z = 64;

  // Maximum x-dimension of a grid
  static constexpr int max_grid_dim_x = cuda::std::numeric_limits<int>::max();

  // Maximum y-dimension of a grid
  static constexpr int max_grid_dim_y = 64 * 1024 - 1;

  // Maximum z-dimension of a grid
  static constexpr int max_grid_dim_z = 64 * 1024 - 1;

  // Maximum amount of shared memory available to a thread block in bytes
  static constexpr int max_shared_memory_per_block = 48 * 1024;

  // Memory available on device for __constant__ variables in a CUDA C kernel in bytes
  static constexpr int total_constant_memory = 64 * 1024;

  // Warp size in threads
  static constexpr int warp_size = 32;

  // Maximum number of concurrent grids on the device
  static constexpr int max_resident_grids = 128;

  // TODO: Not sure if we need these:
  // true if the device can concurrently copy memory between host and device
  // while executing a kernel, or false if not
  static constexpr bool gpu_overlap = true;

  // true if the device can map host memory into CUDA address space
  static constexpr bool can_map_host_memory = true;

  // true if the device supports executing multiple kernels within the same
  // context simultaneously, or false if not. It is not guaranteed that multiple
  // kernels will be resident on the device concurrently so this feature should
  // not be relied upon for correctness.
  static constexpr bool concurrent_kernels = true;

  // true if the device supports stream priorities, or false if not
  static constexpr bool stream_priorities_supported = true;

  // true if device supports caching globals in L1 cache, false if not
  static constexpr bool global_l1_cache_supported = true;

  // true if device supports caching locals in L1 cache, false if not
  static constexpr bool local_l1_cache_supported = true;

  // TODO: We might want to have these per-arch
  // Maximum number of 32-bit registers available to a thread block
  static constexpr int max_registers_per_block = 64 * 1024;

  // Maximum number of 32-bit registers available to a multiprocessor; this
  // number is shared by all thread blocks simultaneously resident on a
  // multiprocessor
  static constexpr int max_registers_per_multiprocessor = 64 * 1024;

  // Maximum number of 32-bit registers available to a thread
  static constexpr int max_registers_per_thread = 255;
};
} // namespace detail

struct arch_traits_t : public detail::arch_common_traits
{
  // Major compute capability version number
  int compute_capability_major;

  // Minor compute capability version number
  int compute_capability_minor;

  // Compute capability version number in 10 * major + minor format
  int compute_capability;

  // Maximum amount of shared memory available to a multiprocessor in bytes;
  // this amount is shared by all thread blocks simultaneously resident on a
  // multiprocessor
  int max_shared_memory_per_multiprocessor;

  // Maximum number of thread blocks that can reside on a multiprocessor
  int max_blocks_per_multiprocessor;

  // Maximum resident warps per multiprocessor
  int max_warps_per_multiprocessor;

  // Maximum resident threads per multiprocessor
  int max_threads_per_multiprocessor;

  // Shared memory reserved by CUDA driver per block in bytes
  int reserved_shared_memory_per_block;

  // Maximum per block shared memory size on the device. This value can be opted
  // into when using dynamic_shared_memory with NonPortableSize set to true
  int max_shared_memory_per_block_optin;

  // TODO: Do we want these?:
  // true if architecture supports clusters
  bool cluster_supported;

  // true if architecture supports redux intrinsic instructions
  bool redux_intrinisic;

  // true if architecture supports elect intrinsic instructions
  bool elect_intrinsic;

  // true if architecture supports asynchronous copy instructions
  bool cp_async_supported;

  // true if architecture supports tensor memory access instructions
  bool tma_supported;
};

namespace detail
{

inline constexpr arch_traits_t sm_700_traits = []() constexpr {
  arch_traits_t __traits{};
  __traits.compute_capability_major             = 7;
  __traits.compute_capability_minor             = 0;
  __traits.compute_capability                   = 70;
  __traits.max_shared_memory_per_multiprocessor = 96 * 1024;
  __traits.max_blocks_per_multiprocessor        = 32;
  __traits.max_warps_per_multiprocessor         = 64;
  __traits.max_threads_per_multiprocessor =
    __traits.max_warps_per_multiprocessor * detail::arch_common_traits::warp_size;
  __traits.reserved_shared_memory_per_block = 0;
  __traits.max_shared_memory_per_block_optin =
    __traits.max_shared_memory_per_multiprocessor - __traits.reserved_shared_memory_per_block;

  __traits.cluster_supported  = false;
  __traits.redux_intrinisic   = false;
  __traits.elect_intrinsic    = false;
  __traits.cp_async_supported = false;
  __traits.tma_supported      = false;

  return __traits;
}();

inline constexpr arch_traits_t sm_750_traits = []() constexpr {
  arch_traits_t __traits{};
  __traits.compute_capability_major             = 7;
  __traits.compute_capability_minor             = 5;
  __traits.compute_capability                   = 75;
  __traits.max_shared_memory_per_multiprocessor = 64 * 1024;
  __traits.max_blocks_per_multiprocessor        = 16;
  __traits.max_warps_per_multiprocessor         = 32;
  __traits.max_threads_per_multiprocessor =
    __traits.max_warps_per_multiprocessor * detail::arch_common_traits::warp_size;
  __traits.reserved_shared_memory_per_block = 0;
  __traits.max_shared_memory_per_block_optin =
    __traits.max_shared_memory_per_multiprocessor - __traits.reserved_shared_memory_per_block;

  __traits.cluster_supported  = false;
  __traits.redux_intrinisic   = false;
  __traits.elect_intrinsic    = false;
  __traits.cp_async_supported = false;
  __traits.tma_supported      = false;

  return __traits;
}();

inline constexpr arch_traits_t sm_860_traits = []() constexpr {
  arch_traits_t __traits{};
  __traits.compute_capability_major             = 8;
  __traits.compute_capability_minor             = 6;
  __traits.compute_capability                   = 86;
  __traits.max_shared_memory_per_multiprocessor = 100 * 1024;
  __traits.max_blocks_per_multiprocessor        = 16;
  __traits.max_warps_per_multiprocessor         = 48;
  __traits.max_threads_per_multiprocessor =
    __traits.max_warps_per_multiprocessor * detail::arch_common_traits::warp_size;
  __traits.reserved_shared_memory_per_block = 1024;
  __traits.max_shared_memory_per_block_optin =
    __traits.max_shared_memory_per_multiprocessor - __traits.reserved_shared_memory_per_block;

  __traits.cluster_supported  = false;
  __traits.redux_intrinisic   = true;
  __traits.elect_intrinsic    = false;
  __traits.cp_async_supported = true;
  __traits.tma_supported      = false;

  return __traits;
}();

// TODO Should this be provided outside detail? Just using arch template seems better
template <unsigned int SmVersion>
_CCCL_HOST_DEVICE inline constexpr arch_traits_t arch_traits()
{
  if constexpr (SmVersion == 60)
  {
    return detail::sm_700_traits;
  }
  else if constexpr (SmVersion == 70)
  {
    return detail::sm_700_traits;
  }
  else if constexpr (SmVersion == 75)
  {
    return detail::sm_750_traits;
  }
  else if constexpr (SmVersion == 80)
  {
    return detail::sm_860_traits;
  }
  else
  {
    static_assert(SmVersion == 86, "Unknown architecture");
    return detail::sm_860_traits;
  }
}

} // namespace detail

//! @brief Retrieve architecture traits of the specified architecture
//!
//! @param __sm_version Compute capability in 10 * major + minor format for which the architecture traits are requested
//!
//! @throws cuda_error if the requested architecture is unknown
_CCCL_HOST_DEVICE inline constexpr arch_traits_t arch_traits(unsigned int __sm_version)
{
  switch (__sm_version)
  {
    case 70:
      return detail::sm_700_traits;
    case 75:
      return detail::sm_750_traits;
    case 86:
      return detail::sm_860_traits;
    default:
      __throw_cuda_error(cudaErrorInvalidValue, "Traits requested for an unknown architecture");
      break;
  }
}

//! @brief Type representing a CUDA device architecture. It provides traits from arch_traits_t in form of static members
template <unsigned int __SmVersion>
struct arch : public detail::arch_common_traits
{
private:
  static constexpr arch_traits_t __traits = detail::arch_traits<__SmVersion>();

public:
  static constexpr int compute_capability_major             = __traits.compute_capability_major;
  static constexpr int compute_capability_minor             = __traits.compute_capability_minor;
  static constexpr int compute_capability                   = __traits.compute_capability;
  static constexpr int max_shared_memory_per_multiprocessor = __traits.max_shared_memory_per_multiprocessor;
  static constexpr int max_warps_per_multiprocessor         = __traits.max_warps_per_multiprocessor;
  static constexpr int max_blocks_per_multiprocessor        = __traits.max_blocks_per_multiprocessor;
  static constexpr int max_threads_per_multiprocessor       = __traits.max_threads_per_multiprocessor;
  static constexpr int reserved_shared_memory_per_block     = __traits.reserved_shared_memory_per_block;
  static constexpr int max_shared_memory_per_block_optin    = __traits.max_shared_memory_per_block_optin;

  static constexpr bool cluster_supported  = __traits.cluster_supported;
  static constexpr bool redux_intrinisic   = __traits.redux_intrinisic;
  static constexpr bool elect_intrinsic    = __traits.elect_intrinsic;
  static constexpr bool cp_async_supported = __traits.cp_async_supported;
  static constexpr bool tma_supported      = __traits.tma_supported;

  constexpr operator arch_traits_t() const
  {
    return __traits;
  }
};

// TODO: possible this should be __device__ constexpr variable?
//! @brief Provides architecture traits of the architecture matching __CUDA_ARCH__ macro
_CCCL_DEVICE constexpr inline arch_traits_t current_arch()
{
#ifdef __CUDA_ARCH__
  return detail::arch_traits<__CUDA_ARCH__ / 10>();
#else
  // Should be unreachable in __device__ function
  return arch_traits_t{};
#endif
}

// TODO might need to remove this since sm_80 and sm_86 is both called ampere :(
using arch_volta  = arch<70>;
using arch_turing = arch<75>;

} // namespace cuda::experimental

#endif // _CUDAX__DEVICE_ARCH_TRAITS
