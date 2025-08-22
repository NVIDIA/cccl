//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___DEVICE_ARCH_TRAITS_H
#define _CUDA___DEVICE_ARCH_TRAITS_H

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)
#  include <cuda/__device/attributes.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/limits>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA
namespace arch
{

inline constexpr int __arch_specific_id_multiplier = 100000;

// @brief Architecture identifier
// This type identifies an architecture. It has more possible entries than just numeric values of the compute
// capability. For example, sm_90 and sm_90a have the same compute capability, but the identifier is different.
enum class id : int
{
  sm_60   = 60,
  sm_61   = 61,
  sm_70   = 70,
  sm_75   = 75,
  sm_80   = 80,
  sm_86   = 86,
  sm_89   = 89,
  sm_90   = 90,
  sm_100  = 100,
  sm_103  = 103,
  sm_110  = 110,
  sm_120  = 120,
  sm_90a  = 90 * __arch_specific_id_multiplier,
  sm_100a = 100 * __arch_specific_id_multiplier,
  sm_103a = 103 * __arch_specific_id_multiplier,
  sm_110a = 110 * __arch_specific_id_multiplier,
  sm_120a = 120 * __arch_specific_id_multiplier,
};

// @brief Architecture traits
// This type contains information about an architecture that is constant across devices of that architecture.
struct traits_t
{
  // Maximum number of threads per block
  const int max_threads_per_block = 1024;

  // Maximum x-dimension of a block
  const int max_block_dim_x = 1024;

  // Maximum y-dimension of a block
  const int max_block_dim_y = 1024;

  // Maximum z-dimension of a block
  const int max_block_dim_z = 64;

  // Maximum x-dimension of a grid
  const int max_grid_dim_x = ::cuda::std::numeric_limits<int32_t>::max();

  // Maximum y-dimension of a grid
  const int max_grid_dim_y = 64 * 1024 - 1;

  // Maximum z-dimension of a grid
  const int max_grid_dim_z = 64 * 1024 - 1;

  // Maximum amount of shared memory available to a thread block in bytes
  const int max_shared_memory_per_block = 48 * 1024;

  // Memory available on device for __constant__ variables in a CUDA C kernel in bytes
  const int total_constant_memory = 64 * 1024;

  // Warp size in threads
  const int warp_size = 32;

  // Maximum number of concurrent grids on the device
  const int max_resident_grids = 128;

  // true if the device can concurrently copy memory between host and device
  // while executing a kernel, or false if not
  const bool gpu_overlap = true;

  // true if the device can map host memory into CUDA address space
  const bool can_map_host_memory = true;

  // true if the device supports executing multiple kernels within the same
  // context simultaneously, or false if not. It is not guaranteed that multiple
  // kernels will be resident on the device concurrently so this feature should
  // not be relied upon for correctness.
  const bool concurrent_kernels = true;

  // true if the device supports stream priorities, or false if not
  const bool stream_priorities_supported = true;

  // true if device supports caching globals in L1 cache, false if not
  const bool global_l1_cache_supported = true;

  // true if device supports caching locals in L1 cache, false if not
  const bool local_l1_cache_supported = true;

  // TODO: We might want to have these per-arch
  // Maximum number of 32-bit registers available to a thread block
  const int max_registers_per_block = 64 * 1024;

  // Maximum number of 32-bit registers available to a multiprocessor; this
  // number is shared by all thread blocks simultaneously resident on a
  // multiprocessor
  const int max_registers_per_multiprocessor = 64 * 1024;

  // Maximum number of 32-bit registers available to a thread
  const int max_registers_per_thread = 255;

  // Identifier for the architecture
  id arch_id;

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

// @brief Architecture traits
// Template function that returns the traits for an architecture with a given id.
template <id _Id>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr traits_t traits();

template <>
[[nodiscard]] _CCCL_HOST_DEVICE inline constexpr traits_t traits<id::sm_60>()
{
  traits_t __traits{};
  __traits.arch_id                              = id::sm_60;
  __traits.compute_capability_major             = 6;
  __traits.compute_capability_minor             = 0;
  __traits.compute_capability                   = 60;
  __traits.max_shared_memory_per_multiprocessor = 64 * 1024;
  __traits.max_blocks_per_multiprocessor        = 32;
  __traits.max_threads_per_multiprocessor       = 2048;
  __traits.max_warps_per_multiprocessor         = __traits.max_threads_per_multiprocessor / __traits.warp_size;
  __traits.reserved_shared_memory_per_block     = 0;
  __traits.max_shared_memory_per_block_optin    = 48 * 1024;

  __traits.cluster_supported  = false;
  __traits.redux_intrinisic   = false;
  __traits.elect_intrinsic    = false;
  __traits.cp_async_supported = false;
  __traits.tma_supported      = false;
  return __traits;
};

template <>
[[nodiscard]] _CCCL_HOST_DEVICE inline constexpr traits_t traits<id::sm_61>()
{
  traits_t __traits{};
  __traits.arch_id                              = id::sm_61;
  __traits.compute_capability_major             = 6;
  __traits.compute_capability_minor             = 1;
  __traits.compute_capability                   = 61;
  __traits.max_shared_memory_per_multiprocessor = 96 * 1024;
  __traits.max_blocks_per_multiprocessor        = 32;
  __traits.max_threads_per_multiprocessor       = 2048;
  __traits.max_warps_per_multiprocessor         = __traits.max_threads_per_multiprocessor / __traits.warp_size;
  __traits.reserved_shared_memory_per_block     = 0;
  __traits.max_shared_memory_per_block_optin    = 48 * 1024;

  __traits.cluster_supported  = false;
  __traits.redux_intrinisic   = false;
  __traits.elect_intrinsic    = false;
  __traits.cp_async_supported = false;
  __traits.tma_supported      = false;
  return __traits;
};

template <>
[[nodiscard]] _CCCL_HOST_DEVICE inline constexpr traits_t traits<id::sm_70>()
{
  traits_t __traits{};
  __traits.arch_id                              = id::sm_70;
  __traits.compute_capability_major             = 7;
  __traits.compute_capability_minor             = 0;
  __traits.compute_capability                   = 70;
  __traits.max_shared_memory_per_multiprocessor = 96 * 1024;
  __traits.max_blocks_per_multiprocessor        = 32;
  __traits.max_threads_per_multiprocessor       = 2048;
  __traits.max_warps_per_multiprocessor         = __traits.max_threads_per_multiprocessor / __traits.warp_size;
  __traits.reserved_shared_memory_per_block     = 0;
  __traits.max_shared_memory_per_block_optin =
    __traits.max_shared_memory_per_multiprocessor - __traits.reserved_shared_memory_per_block;

  __traits.cluster_supported  = false;
  __traits.redux_intrinisic   = false;
  __traits.elect_intrinsic    = false;
  __traits.cp_async_supported = false;
  __traits.tma_supported      = false;
  return __traits;
};

template <>
[[nodiscard]] _CCCL_HOST_DEVICE inline constexpr traits_t traits<id::sm_75>()
{
  traits_t __traits{};
  __traits.arch_id                              = id::sm_75;
  __traits.compute_capability_major             = 7;
  __traits.compute_capability_minor             = 5;
  __traits.compute_capability                   = 75;
  __traits.max_shared_memory_per_multiprocessor = 64 * 1024;
  __traits.max_blocks_per_multiprocessor        = 16;
  __traits.max_threads_per_multiprocessor       = 1024;
  __traits.max_warps_per_multiprocessor         = __traits.max_threads_per_multiprocessor / __traits.warp_size;
  __traits.reserved_shared_memory_per_block     = 0;
  __traits.max_shared_memory_per_block_optin =
    __traits.max_shared_memory_per_multiprocessor - __traits.reserved_shared_memory_per_block;

  __traits.cluster_supported  = false;
  __traits.redux_intrinisic   = false;
  __traits.elect_intrinsic    = false;
  __traits.cp_async_supported = false;
  __traits.tma_supported      = false;
  return __traits;
};

template <>
[[nodiscard]] _CCCL_HOST_DEVICE inline constexpr traits_t traits<id::sm_80>()
{
  traits_t __traits{};
  __traits.arch_id                              = id::sm_80;
  __traits.compute_capability_major             = 8;
  __traits.compute_capability_minor             = 0;
  __traits.compute_capability                   = 80;
  __traits.max_shared_memory_per_multiprocessor = 164 * 1024;
  __traits.max_blocks_per_multiprocessor        = 32;
  __traits.max_threads_per_multiprocessor       = 2048;
  __traits.max_warps_per_multiprocessor         = __traits.max_threads_per_multiprocessor / __traits.warp_size;
  __traits.reserved_shared_memory_per_block     = 1024;
  __traits.max_shared_memory_per_block_optin =
    __traits.max_shared_memory_per_multiprocessor - __traits.reserved_shared_memory_per_block;

  __traits.cluster_supported  = false;
  __traits.redux_intrinisic   = true;
  __traits.elect_intrinsic    = false;
  __traits.cp_async_supported = true;
  __traits.tma_supported      = false;
  return __traits;
};

template <>
[[nodiscard]] _CCCL_HOST_DEVICE inline constexpr traits_t traits<id::sm_86>()
{
  traits_t __traits{};
  __traits.arch_id                              = id::sm_86;
  __traits.compute_capability_major             = 8;
  __traits.compute_capability_minor             = 6;
  __traits.compute_capability                   = 86;
  __traits.max_shared_memory_per_multiprocessor = 100 * 1024;
  __traits.max_blocks_per_multiprocessor        = 16;
  __traits.max_threads_per_multiprocessor       = 1536;
  __traits.max_warps_per_multiprocessor         = __traits.max_threads_per_multiprocessor / __traits.warp_size;
  __traits.reserved_shared_memory_per_block     = 1024;
  __traits.max_shared_memory_per_block_optin =
    __traits.max_shared_memory_per_multiprocessor - __traits.reserved_shared_memory_per_block;

  __traits.cluster_supported  = false;
  __traits.redux_intrinisic   = true;
  __traits.elect_intrinsic    = false;
  __traits.cp_async_supported = true;
  __traits.tma_supported      = false;
  return __traits;
};

template <>
[[nodiscard]] _CCCL_HOST_DEVICE inline constexpr traits_t traits<id::sm_89>()
{
  traits_t __traits{};
  __traits.arch_id                              = id::sm_89;
  __traits.compute_capability_major             = 8;
  __traits.compute_capability_minor             = 9;
  __traits.compute_capability                   = 89;
  __traits.max_shared_memory_per_multiprocessor = 100 * 1024;
  __traits.max_blocks_per_multiprocessor        = 24;
  __traits.max_threads_per_multiprocessor       = 1536;
  __traits.max_warps_per_multiprocessor         = __traits.max_threads_per_multiprocessor / __traits.warp_size;
  __traits.reserved_shared_memory_per_block     = 1024;
  __traits.max_shared_memory_per_block_optin =
    __traits.max_shared_memory_per_multiprocessor - __traits.reserved_shared_memory_per_block;

  __traits.cluster_supported  = false;
  __traits.redux_intrinisic   = true;
  __traits.elect_intrinsic    = false;
  __traits.cp_async_supported = true;
  __traits.tma_supported      = false;
  return __traits;
};

template <>
[[nodiscard]] _CCCL_HOST_DEVICE inline constexpr traits_t traits<id::sm_90>()
{
  traits_t __traits{};
  __traits.arch_id                              = id::sm_90;
  __traits.compute_capability_major             = 9;
  __traits.compute_capability_minor             = 0;
  __traits.compute_capability                   = 90;
  __traits.max_shared_memory_per_multiprocessor = 228 * 1024;
  __traits.max_blocks_per_multiprocessor        = 32;
  __traits.max_threads_per_multiprocessor       = 2048;
  __traits.max_warps_per_multiprocessor         = __traits.max_threads_per_multiprocessor / __traits.warp_size;
  __traits.reserved_shared_memory_per_block     = 1024;
  __traits.max_shared_memory_per_block_optin =
    __traits.max_shared_memory_per_multiprocessor - __traits.reserved_shared_memory_per_block;

  __traits.cluster_supported  = true;
  __traits.redux_intrinisic   = true;
  __traits.elect_intrinsic    = true;
  __traits.cp_async_supported = true;
  __traits.tma_supported      = true;
  return __traits;
};

// No sm_90a specific fields for now.
template <>
[[nodiscard]] _CCCL_HOST_DEVICE inline constexpr traits_t traits<id::sm_90a>()
{
  return ::cuda::arch::traits<id::sm_90>();
};

template <>
[[nodiscard]] _CCCL_HOST_DEVICE inline constexpr traits_t traits<id::sm_100>()
{
  traits_t __traits{};
  __traits.arch_id                              = id::sm_100;
  __traits.compute_capability_major             = 10;
  __traits.compute_capability_minor             = 0;
  __traits.compute_capability                   = 100;
  __traits.max_shared_memory_per_multiprocessor = 228 * 1024;
  __traits.max_blocks_per_multiprocessor        = 32;
  __traits.max_threads_per_multiprocessor       = 2048;
  __traits.max_warps_per_multiprocessor         = __traits.max_threads_per_multiprocessor / __traits.warp_size;
  __traits.reserved_shared_memory_per_block     = 1024;
  __traits.max_shared_memory_per_block_optin =
    __traits.max_shared_memory_per_multiprocessor - __traits.reserved_shared_memory_per_block;

  __traits.cluster_supported  = true;
  __traits.redux_intrinisic   = true;
  __traits.elect_intrinsic    = true;
  __traits.cp_async_supported = true;
  __traits.tma_supported      = true;
  return __traits;
};

template <>
[[nodiscard]] _CCCL_HOST_DEVICE inline constexpr traits_t traits<id::sm_100a>()
{
  return ::cuda::arch::traits<id::sm_100>();
};

template <>
[[nodiscard]] _CCCL_HOST_DEVICE inline constexpr traits_t traits<id::sm_103>()
{
  traits_t __traits                 = ::cuda::arch::traits<id::sm_100>();
  __traits.arch_id                  = id::sm_103;
  __traits.compute_capability_major = 10;
  __traits.compute_capability_minor = 3;
  __traits.compute_capability       = 103;
  return __traits;
};

template <>
[[nodiscard]] _CCCL_HOST_DEVICE inline constexpr traits_t traits<id::sm_103a>()
{
  return ::cuda::arch::traits<id::sm_103>();
};

template <>
[[nodiscard]] _CCCL_HOST_DEVICE inline constexpr traits_t traits<id::sm_110>()
{
  traits_t __traits                 = ::cuda::arch::traits<id::sm_100>();
  __traits.arch_id                  = id::sm_110;
  __traits.compute_capability_major = 11;
  __traits.compute_capability_minor = 0;
  __traits.compute_capability       = 110;
  return __traits;
};

template <>
[[nodiscard]] _CCCL_HOST_DEVICE inline constexpr traits_t traits<id::sm_110a>()
{
  return ::cuda::arch::traits<id::sm_110>();
};

template <>
[[nodiscard]] _CCCL_HOST_DEVICE inline constexpr traits_t traits<id::sm_120>()
{
  traits_t __traits{};
  __traits.arch_id                              = id::sm_120;
  __traits.compute_capability_major             = 12;
  __traits.compute_capability_minor             = 0;
  __traits.compute_capability                   = 120;
  __traits.max_shared_memory_per_multiprocessor = 100 * 1024;
  __traits.max_blocks_per_multiprocessor        = 32;
  __traits.max_threads_per_multiprocessor       = 1536;
  __traits.max_warps_per_multiprocessor         = __traits.max_threads_per_multiprocessor / __traits.warp_size;
  __traits.reserved_shared_memory_per_block     = 1024;
  __traits.max_shared_memory_per_block_optin =
    __traits.max_shared_memory_per_multiprocessor - __traits.reserved_shared_memory_per_block;

  __traits.cluster_supported  = true;
  __traits.redux_intrinisic   = true;
  __traits.elect_intrinsic    = true;
  __traits.cp_async_supported = true;
  __traits.tma_supported      = true;
  return __traits;
};

template <>
[[nodiscard]] _CCCL_HOST_DEVICE inline constexpr traits_t traits<id::sm_120a>()
{
  return ::cuda::arch::traits<id::sm_120>();
};

inline constexpr int __highest_known_arch = 120;

[[nodiscard]] _CCCL_API inline constexpr traits_t traits_for_id(id __id)
{
  switch (__id)
  {
    case id::sm_60:
      return ::cuda::arch::traits<id::sm_60>();
    case id::sm_61:
      return ::cuda::arch::traits<id::sm_61>();
    case id::sm_70:
      return ::cuda::arch::traits<id::sm_70>();
    case id::sm_75:
      return ::cuda::arch::traits<id::sm_75>();
    case id::sm_80:
      return ::cuda::arch::traits<id::sm_80>();
    case id::sm_86:
      return ::cuda::arch::traits<id::sm_86>();
    case id::sm_89:
      return ::cuda::arch::traits<id::sm_89>();
    case id::sm_90:
      return ::cuda::arch::traits<id::sm_90>();
    case id::sm_90a:
      return ::cuda::arch::traits<id::sm_90a>();
    case id::sm_100:
      return ::cuda::arch::traits<id::sm_100>();
    case id::sm_100a:
      return ::cuda::arch::traits<id::sm_100a>();
    case id::sm_103:
      return ::cuda::arch::traits<id::sm_103>();
    case id::sm_103a:
      return ::cuda::arch::traits<id::sm_103a>();
    case id::sm_110:
      return ::cuda::arch::traits<id::sm_110>();
    case id::sm_110a:
      return ::cuda::arch::traits<id::sm_110a>();
    case id::sm_120:
      return ::cuda::arch::traits<id::sm_120>();
    case id::sm_120a:
      return ::cuda::arch::traits<id::sm_120a>();
    default:
      ::cuda::__throw_cuda_error(cudaErrorInvalidValue, "Traits requested for an unknown architecture");
      break;
  }
}

[[nodiscard]] _CCCL_API inline constexpr id id_for_compute_capability(int compute_capability)
{
  if (compute_capability < 60 || compute_capability > __highest_known_arch)
  {
    ::cuda::__throw_cuda_error(cudaErrorInvalidValue, "Compute capability out of range");
  }
  return static_cast<id>(compute_capability);
}

[[nodiscard]] _CCCL_API inline constexpr traits_t traits_for_compute_capability(int compute_capability)
{
  return ::cuda::arch::traits_for_id(::cuda::arch::id_for_compute_capability(compute_capability));
}

_CCCL_API inline constexpr id __special_id_for_compute_capability(int value)
{
  switch (value)
  {
    case 90:
      return id::sm_90a;
    case 100:
      return id::sm_100a;
    case 103:
      return id::sm_103a;
    case 110:
      return id::sm_110a;
    case 120:
      return id::sm_120a;
    default:
      ::cuda::__throw_cuda_error(cudaErrorInvalidValue, "Compute capability out of range");
      break;
  }
}

//! @brief Provides architecture traits of the architecture matching __CUDA_ARCH__ macro
[[nodiscard]] _CCCL_DEVICE inline constexpr arch::traits_t current_traits()
{
  // fixme: this doesn't work with nvc++ -cuda
#  ifdef __CUDA_ARCH__
#    ifdef __CUDA_ARCH_SPECIFIC__
  return ::cuda::arch::traits_for_id(::cuda::arch::__special_id_for_compute_capability(__CUDA_ARCH_SPECIFIC__ / 10));
#    else
  return ::cuda::arch::traits_for_compute_capability(__CUDA_ARCH__ / 10);
#    endif // __CUDA_ARCH_SPECIFIC__
#  else // __CUDA_ARCH__
  // Should be unreachable in __device__ function
  return ::cuda::arch::traits_t{};
#  endif // __CUDA_ARCH__
}

[[nodiscard]] inline constexpr arch::traits_t
__arch_traits_might_be_unknown(int __device, unsigned int __compute_capability)
{
  if (__compute_capability <= arch::__highest_known_arch)
  {
    return ::cuda::arch::traits_for_compute_capability(__compute_capability);
  }
  else
  {
    // If the architecture is unknown, we need to craft the arch_traits from attributes
    arch::traits_t __traits{};
    __traits.compute_capability_major = __compute_capability / 10;
    __traits.compute_capability_minor = __compute_capability % 10;
    __traits.compute_capability       = __compute_capability;
    __traits.max_shared_memory_per_multiprocessor =
      ::cuda::device_attributes::max_shared_memory_per_multiprocessor(__device);
    __traits.max_blocks_per_multiprocessor    = ::cuda::device_attributes::max_blocks_per_multiprocessor(__device);
    __traits.max_threads_per_multiprocessor   = ::cuda::device_attributes::max_threads_per_multiprocessor(__device);
    __traits.max_warps_per_multiprocessor     = __traits.max_threads_per_multiprocessor / __traits.warp_size;
    __traits.reserved_shared_memory_per_block = ::cuda::device_attributes::reserved_shared_memory_per_block(__device);
    __traits.max_shared_memory_per_block_optin =
      __traits.max_shared_memory_per_multiprocessor - __traits.reserved_shared_memory_per_block;

    __traits.cluster_supported  = __compute_capability >= 90;
    __traits.redux_intrinisic   = __compute_capability >= 80;
    __traits.elect_intrinsic    = __compute_capability >= 90;
    __traits.cp_async_supported = __compute_capability >= 80;
    __traits.tma_supported      = __compute_capability >= 90;
    return __traits;
  }
}
} // namespace arch

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___DEVICE_ARCH_TRAITS_H
