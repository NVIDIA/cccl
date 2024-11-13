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

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__exception/cuda_error.h>
#include <cuda/std/limits>

#include <cuda/experimental/__device/attributes.cuh>

namespace cuda::experimental
{

namespace detail
{
struct arch_common_traits
{
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

  // Compute capability version number in 100 * major + 10 * minor format
  int compute_capability;

  // Maximum amount of shared memory available to a multiprocessor in bytes;
  // this amount is shared by all thread blocks simultaneously resident on a
  // multiprocessor
  int max_shared_memory_per_multiprocessor;

  // Maximum number of thread blocks that can reside on a multiprocessor
  int max_blocks_per_multiprocessor;

  // Maximum resident threads per multiprocessor
  int max_threads_per_multiprocessor;

  // Maximum resident warps per multiprocessor
  int max_warps_per_multiprocessor;

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

inline constexpr arch_traits_t sm_600_traits = []() constexpr {
  arch_traits_t __traits{};
  __traits.compute_capability_major             = 6;
  __traits.compute_capability_minor             = 0;
  __traits.compute_capability                   = 600;
  __traits.max_shared_memory_per_multiprocessor = 64 * 1024;
  __traits.max_blocks_per_multiprocessor        = 32;
  __traits.max_threads_per_multiprocessor       = 2048;
  __traits.max_warps_per_multiprocessor =
    __traits.max_threads_per_multiprocessor / detail::arch_common_traits::warp_size;
  __traits.reserved_shared_memory_per_block  = 0;
  __traits.max_shared_memory_per_block_optin = 48 * 1024;

  __traits.cluster_supported  = false;
  __traits.redux_intrinisic   = false;
  __traits.elect_intrinsic    = false;
  __traits.cp_async_supported = false;
  __traits.tma_supported      = false;

  return __traits;
}();

inline constexpr arch_traits_t sm_700_traits = []() constexpr {
  arch_traits_t __traits{};
  __traits.compute_capability_major             = 7;
  __traits.compute_capability_minor             = 0;
  __traits.compute_capability                   = 700;
  __traits.max_shared_memory_per_multiprocessor = 96 * 1024;
  __traits.max_blocks_per_multiprocessor        = 32;
  __traits.max_threads_per_multiprocessor       = 2048;
  __traits.max_warps_per_multiprocessor =
    __traits.max_threads_per_multiprocessor / detail::arch_common_traits::warp_size;
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
  __traits.compute_capability                   = 750;
  __traits.max_shared_memory_per_multiprocessor = 64 * 1024;
  __traits.max_blocks_per_multiprocessor        = 16;
  __traits.max_threads_per_multiprocessor       = 1024;
  __traits.max_warps_per_multiprocessor =
    __traits.max_threads_per_multiprocessor / detail::arch_common_traits::warp_size;
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

inline constexpr arch_traits_t sm_800_traits = []() constexpr {
  arch_traits_t __traits{};
  __traits.compute_capability_major             = 8;
  __traits.compute_capability_minor             = 0;
  __traits.compute_capability                   = 800;
  __traits.max_shared_memory_per_multiprocessor = 164 * 1024;
  __traits.max_blocks_per_multiprocessor        = 32;
  __traits.max_threads_per_multiprocessor       = 2048;
  __traits.max_warps_per_multiprocessor =
    __traits.max_threads_per_multiprocessor / detail::arch_common_traits::warp_size;
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

inline constexpr arch_traits_t sm_860_traits = []() constexpr {
  arch_traits_t __traits{};
  __traits.compute_capability_major             = 8;
  __traits.compute_capability_minor             = 6;
  __traits.compute_capability                   = 860;
  __traits.max_shared_memory_per_multiprocessor = 100 * 1024;
  __traits.max_blocks_per_multiprocessor        = 16;
  __traits.max_threads_per_multiprocessor       = 1536;
  __traits.max_warps_per_multiprocessor =
    __traits.max_threads_per_multiprocessor / detail::arch_common_traits::warp_size;
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

inline constexpr arch_traits_t sm_890_traits = []() constexpr {
  arch_traits_t __traits{};
  __traits.compute_capability_major             = 8;
  __traits.compute_capability_minor             = 9;
  __traits.compute_capability                   = 890;
  __traits.max_shared_memory_per_multiprocessor = 100 * 1024;
  __traits.max_blocks_per_multiprocessor        = 24;
  __traits.max_threads_per_multiprocessor       = 1536;
  __traits.max_warps_per_multiprocessor =
    __traits.max_threads_per_multiprocessor / detail::arch_common_traits::warp_size;
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

inline constexpr arch_traits_t sm_900_traits = []() constexpr {
  arch_traits_t __traits{};
  __traits.compute_capability_major             = 9;
  __traits.compute_capability_minor             = 0;
  __traits.compute_capability                   = 900;
  __traits.max_shared_memory_per_multiprocessor = 228 * 1024;
  __traits.max_blocks_per_multiprocessor        = 32;
  __traits.max_threads_per_multiprocessor       = 2048;
  __traits.max_warps_per_multiprocessor =
    __traits.max_threads_per_multiprocessor / detail::arch_common_traits::warp_size;
  __traits.reserved_shared_memory_per_block = 1024;
  __traits.max_shared_memory_per_block_optin =
    __traits.max_shared_memory_per_multiprocessor - __traits.reserved_shared_memory_per_block;

  __traits.cluster_supported  = true;
  __traits.redux_intrinisic   = true;
  __traits.elect_intrinsic    = true;
  __traits.cp_async_supported = true;
  __traits.tma_supported      = true;

  return __traits;
}();

inline constexpr unsigned int __highest_known_arch = 900;

} // namespace detail

//! @brief Retrieve architecture traits of the specified architecture
//!
//! @param __sm_version Compute capability in 100 * major + 10 * minor format for which the architecture traits are
//! requested
//!
//! @throws cuda_error if the requested architecture is unknown
_CCCL_HOST_DEVICE inline constexpr arch_traits_t arch_traits(unsigned int __sm_version)
{
  switch (__sm_version)
  {
    case 600:
      return detail::sm_600_traits;
    case 700:
      return detail::sm_700_traits;
    case 750:
      return detail::sm_750_traits;
    case 800:
      return detail::sm_800_traits;
    case 860:
      return detail::sm_860_traits;
    case 890:
      return detail::sm_890_traits;
    case 900:
      return detail::sm_900_traits;
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
  static constexpr arch_traits_t __traits = arch_traits(__SmVersion);

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

//! @brief Provides architecture traits of the architecture matching __CUDA_ARCH__ macro
_CCCL_DEVICE constexpr inline arch_traits_t current_arch()
{
#ifdef __CUDA_ARCH__
  return arch_traits(__CUDA_ARCH__);
#else
  // Should be unreachable in __device__ function
  return arch_traits_t{};
#endif
}

namespace detail
{
_CCCL_NODISCARD inline constexpr arch_traits_t __arch_traits_might_be_unknown(int __device, unsigned int __arch)
{
  if (__arch <= __highest_known_arch)
  {
    return arch_traits(__arch);
  }
  else
  {
    // If the architecture is unknown, we need to craft the arch_traits from attributes
    arch_traits_t __traits{};
    __traits.compute_capability_major = __arch / 100;
    __traits.compute_capability_minor = (__arch / 10) % 10;
    __traits.compute_capability       = __arch;
    __traits.max_shared_memory_per_multiprocessor =
      detail::__device_attrs::max_shared_memory_per_multiprocessor(__device);
    __traits.max_blocks_per_multiprocessor  = detail::__device_attrs::max_blocks_per_multiprocessor(__device);
    __traits.max_threads_per_multiprocessor = detail::__device_attrs::max_threads_per_multiprocessor(__device);
    __traits.max_warps_per_multiprocessor =
      __traits.max_threads_per_multiprocessor / detail::arch_common_traits::warp_size;
    __traits.reserved_shared_memory_per_block = detail::__device_attrs::reserved_shared_memory_per_block(__device);
    __traits.max_shared_memory_per_block_optin =
      __traits.max_shared_memory_per_multiprocessor - __traits.reserved_shared_memory_per_block;

    __traits.cluster_supported  = __arch >= 900;
    __traits.redux_intrinisic   = __arch >= 800;
    __traits.elect_intrinsic    = __arch >= 900;
    __traits.cp_async_supported = __arch >= 800;
    __traits.tma_supported      = __arch >= 900;
    return __traits;
  }
}
} // namespace detail

} // namespace cuda::experimental

#endif // _CUDAX__DEVICE_ARCH_TRAITS
