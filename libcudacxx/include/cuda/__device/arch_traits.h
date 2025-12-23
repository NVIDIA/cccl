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

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/arch_id.h>
#include <cuda/__device/compute_capability.h>
#include <cuda/__fwd/devices.h>
#include <cuda/std/__exception/cuda_error.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Architecture traits
//! This type contains information about an architecture that is constant across devices of that architecture.
struct arch_traits_t
{
  // Maximum number of threads per block
  int max_threads_per_block;

  // Maximum x-dimension of a block
  int max_block_dim_x;

  // Maximum y-dimension of a block
  int max_block_dim_y;

  // Maximum z-dimension of a block
  int max_block_dim_z;

  // Maximum x-dimension of a grid
  int max_grid_dim_x;

  // Maximum y-dimension of a grid
  int max_grid_dim_y;

  // Maximum z-dimension of a grid
  int max_grid_dim_z;

  // Maximum amount of shared memory available to a thread block in bytes
  ::cuda::std::size_t max_shared_memory_per_block;

  // Memory available on device for __constant__ variables in a CUDA C kernel in bytes
  ::cuda::std::size_t total_constant_memory;

  // Warp size in threads
  int warp_size;

  // Maximum number of concurrent grids on the device
  int max_resident_grids;

  // true if the device can concurrently copy memory between host and device
  // while executing a kernel, or false if not
  bool gpu_overlap;

  // true if the device can map host memory into CUDA address space
  bool can_map_host_memory;

  // true if the device supports executing multiple kernels within the same
  // context simultaneously, or false if not. It is not guaranteed that multiple
  // kernels will be resident on the device concurrently so this feature should
  // not be relied upon for correctness.
  bool concurrent_kernels;

  // true if the device supports stream priorities, or false if not
  bool stream_priorities_supported;

  // true if device supports caching globals in L1 cache, false if not
  bool global_l1_cache_supported;

  // true if device supports caching locals in L1 cache, false if not
  bool local_l1_cache_supported;

  // TODO: We might want to have these per-arch
  // Maximum number of 32-bit registers available to a thread block
  int max_registers_per_block;

  // Maximum number of 32-bit registers available to a multiprocessor; this
  // number is shared by all thread blocks simultaneously resident on a
  // multiprocessor
  int max_registers_per_multiprocessor;

  // Maximum number of 32-bit registers available to a thread
  int max_registers_per_thread;

  // Identifier for the architecture
  ::cuda::arch_id arch_id;

  // Major compute capability version number
  int compute_capability_major;

  // Minor compute capability version number
  int compute_capability_minor;

  // Compute capability version number in 100 * major + 10 * minor format
  ::cuda::compute_capability compute_capability;

  // Maximum amount of shared memory available to a multiprocessor in bytes;
  // this amount is shared by all thread blocks simultaneously resident on a
  // multiprocessor
  ::cuda::std::size_t max_shared_memory_per_multiprocessor;

  // Maximum number of thread blocks that can reside on a multiprocessor
  int max_blocks_per_multiprocessor;

  // Maximum resident threads per multiprocessor
  int max_threads_per_multiprocessor;

  // Maximum resident warps per multiprocessor
  int max_warps_per_multiprocessor;

  // Shared memory reserved by CUDA driver per block in bytes
  ::cuda::std::size_t reserved_shared_memory_per_block;

  // Maximum per block shared memory size on the device. This value can be opted
  // into when using dynamic_shared_memory with NonPortableSize set to true
  ::cuda::std::size_t max_shared_memory_per_block_optin;

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

[[nodiscard]] _CCCL_API constexpr arch_traits_t __common_arch_traits(arch_id __arch_id) noexcept
{
  const compute_capability __cc{__arch_id};

  arch_traits_t __traits{};
  __traits.max_threads_per_block            = 1024;
  __traits.max_block_dim_x                  = 1024;
  __traits.max_block_dim_y                  = 1024;
  __traits.max_block_dim_z                  = 64;
  __traits.max_grid_dim_x                   = ::cuda::std::numeric_limits<::cuda::std::int32_t>::max();
  __traits.max_grid_dim_y                   = 64 * 1024 - 1;
  __traits.max_grid_dim_z                   = 64 * 1024 - 1;
  __traits.max_shared_memory_per_block      = 48 * 1024;
  __traits.total_constant_memory            = 64 * 1024;
  __traits.warp_size                        = 32;
  __traits.max_resident_grids               = 128;
  __traits.gpu_overlap                      = true;
  __traits.can_map_host_memory              = true;
  __traits.concurrent_kernels               = true;
  __traits.stream_priorities_supported      = true;
  __traits.global_l1_cache_supported        = true;
  __traits.local_l1_cache_supported         = true;
  __traits.max_registers_per_block          = 64 * 1024;
  __traits.max_registers_per_multiprocessor = 64 * 1024;
  __traits.max_registers_per_thread         = 255;
  __traits.arch_id                          = __arch_id;
  __traits.compute_capability_major         = __cc.major();
  __traits.compute_capability_minor         = __cc.minor();
  __traits.compute_capability               = __cc;
  // __traits.max_shared_memory_per_multiprocessor; // set up individually
  // __traits.max_blocks_per_multiprocessor; // set up individually
  // __traits.max_threads_per_multiprocessor; // set up individually
  // __traits.max_warps_per_multiprocessor; // set up individually
  __traits.reserved_shared_memory_per_block = (__cc >= compute_capability{80}) ? 1024 : 0;
  // __traits.max_shared_memory_per_block_optin; // set up individually
  __traits.cluster_supported  = (__cc >= compute_capability{90});
  __traits.redux_intrinisic   = (__cc >= compute_capability{80});
  __traits.elect_intrinsic    = (__cc >= compute_capability{90});
  __traits.cp_async_supported = (__cc >= compute_capability{80});
  __traits.tma_supported      = (__cc >= compute_capability{90});
  return __traits;
}

//! @brief Gets the architecture traits for the given architecture id \c _Id.
template <arch_id _Id>
[[nodiscard]] _CCCL_API constexpr arch_traits_t arch_traits() noexcept;

template <>
[[nodiscard]] _CCCL_API constexpr arch_traits_t arch_traits<arch_id::sm_60>() noexcept
{
  auto __traits                                 = ::cuda::__common_arch_traits(arch_id::sm_60);
  __traits.max_shared_memory_per_multiprocessor = 64 * 1024;
  __traits.max_blocks_per_multiprocessor        = 32;
  __traits.max_threads_per_multiprocessor       = 2048;
  __traits.max_warps_per_multiprocessor         = __traits.max_threads_per_multiprocessor / __traits.warp_size;
  __traits.max_shared_memory_per_block_optin    = 48 * 1024;
  return __traits;
};

template <>
[[nodiscard]] _CCCL_API constexpr arch_traits_t arch_traits<arch_id::sm_61>() noexcept
{
  auto __traits                                 = ::cuda::__common_arch_traits(arch_id::sm_61);
  __traits.max_shared_memory_per_multiprocessor = 96 * 1024;
  __traits.max_blocks_per_multiprocessor        = 32;
  __traits.max_threads_per_multiprocessor       = 2048;
  __traits.max_warps_per_multiprocessor         = __traits.max_threads_per_multiprocessor / __traits.warp_size;
  __traits.max_shared_memory_per_block_optin    = 48 * 1024;
  return __traits;
};

template <>
[[nodiscard]] _CCCL_API constexpr arch_traits_t arch_traits<arch_id::sm_62>() noexcept
{
  auto __traits                                 = ::cuda::__common_arch_traits(arch_id::sm_62);
  __traits.max_shared_memory_per_multiprocessor = 64 * 1024;
  __traits.max_blocks_per_multiprocessor        = 32;
  __traits.max_threads_per_multiprocessor       = 2048;
  __traits.max_warps_per_multiprocessor         = __traits.max_threads_per_multiprocessor / __traits.warp_size;
  __traits.max_shared_memory_per_block_optin    = 48 * 1024;
  __traits.max_registers_per_block              = 32 * 1024;

  return __traits;
};

template <>
[[nodiscard]] _CCCL_API constexpr arch_traits_t arch_traits<arch_id::sm_70>() noexcept
{
  auto __traits                                 = ::cuda::__common_arch_traits(arch_id::sm_70);
  __traits.max_shared_memory_per_multiprocessor = 96 * 1024;
  __traits.max_blocks_per_multiprocessor        = 32;
  __traits.max_threads_per_multiprocessor       = 2048;
  __traits.max_warps_per_multiprocessor         = __traits.max_threads_per_multiprocessor / __traits.warp_size;
  __traits.reserved_shared_memory_per_block     = 0;
  __traits.max_shared_memory_per_block_optin =
    __traits.max_shared_memory_per_multiprocessor - __traits.reserved_shared_memory_per_block;
  return __traits;
};

template <>
[[nodiscard]] _CCCL_API constexpr arch_traits_t arch_traits<arch_id::sm_75>() noexcept
{
  auto __traits                                 = ::cuda::__common_arch_traits(arch_id::sm_75);
  __traits.max_shared_memory_per_multiprocessor = 64 * 1024;
  __traits.max_blocks_per_multiprocessor        = 16;
  __traits.max_threads_per_multiprocessor       = 1024;
  __traits.max_warps_per_multiprocessor         = __traits.max_threads_per_multiprocessor / __traits.warp_size;
  __traits.max_shared_memory_per_block_optin =
    __traits.max_shared_memory_per_multiprocessor - __traits.reserved_shared_memory_per_block;
  return __traits;
};

template <>
[[nodiscard]] _CCCL_API constexpr arch_traits_t arch_traits<arch_id::sm_80>() noexcept
{
  auto __traits                                 = ::cuda::__common_arch_traits(arch_id::sm_80);
  __traits.max_shared_memory_per_multiprocessor = 164 * 1024;
  __traits.max_blocks_per_multiprocessor        = 32;
  __traits.max_threads_per_multiprocessor       = 2048;
  __traits.max_warps_per_multiprocessor         = __traits.max_threads_per_multiprocessor / __traits.warp_size;
  __traits.max_shared_memory_per_block_optin =
    __traits.max_shared_memory_per_multiprocessor - __traits.reserved_shared_memory_per_block;
  return __traits;
};

template <>
[[nodiscard]] _CCCL_API constexpr arch_traits_t arch_traits<arch_id::sm_86>() noexcept
{
  auto __traits                                 = ::cuda::__common_arch_traits(arch_id::sm_86);
  __traits.max_shared_memory_per_multiprocessor = 100 * 1024;
  __traits.max_blocks_per_multiprocessor        = 16;
  __traits.max_threads_per_multiprocessor       = 1536;
  __traits.max_warps_per_multiprocessor         = __traits.max_threads_per_multiprocessor / __traits.warp_size;
  __traits.max_shared_memory_per_block_optin =
    __traits.max_shared_memory_per_multiprocessor - __traits.reserved_shared_memory_per_block;
  return __traits;
};

template <>
[[nodiscard]] _CCCL_API constexpr arch_traits_t arch_traits<arch_id::sm_87>() noexcept
{
  auto __traits                                 = ::cuda::__common_arch_traits(arch_id::sm_87);
  __traits.max_shared_memory_per_multiprocessor = 164 * 1024;
  __traits.max_blocks_per_multiprocessor        = 16;
  __traits.max_threads_per_multiprocessor       = 1536;
  __traits.max_warps_per_multiprocessor         = __traits.max_threads_per_multiprocessor / __traits.warp_size;
  __traits.max_shared_memory_per_block_optin =
    __traits.max_shared_memory_per_multiprocessor - __traits.reserved_shared_memory_per_block;
  return __traits;
};

template <>
[[nodiscard]] _CCCL_API constexpr arch_traits_t arch_traits<arch_id::sm_88>() noexcept
{
  auto __traits                     = ::cuda::arch_traits<arch_id::sm_86>();
  __traits.arch_id                  = arch_id::sm_88;
  __traits.compute_capability_major = 8;
  __traits.compute_capability_minor = 8;
  __traits.compute_capability       = compute_capability{88};
  return __traits;
};

template <>
[[nodiscard]] _CCCL_API constexpr arch_traits_t arch_traits<arch_id::sm_89>() noexcept
{
  auto __traits                                 = ::cuda::__common_arch_traits(arch_id::sm_89);
  __traits.max_shared_memory_per_multiprocessor = 100 * 1024;
  __traits.max_blocks_per_multiprocessor        = 24;
  __traits.max_threads_per_multiprocessor       = 1536;
  __traits.max_warps_per_multiprocessor         = __traits.max_threads_per_multiprocessor / __traits.warp_size;
  __traits.max_shared_memory_per_block_optin =
    __traits.max_shared_memory_per_multiprocessor - __traits.reserved_shared_memory_per_block;
  return __traits;
};

template <>
[[nodiscard]] _CCCL_API constexpr arch_traits_t arch_traits<arch_id::sm_90>() noexcept
{
  auto __traits                                 = ::cuda::__common_arch_traits(arch_id::sm_90);
  __traits.max_shared_memory_per_multiprocessor = 228 * 1024;
  __traits.max_blocks_per_multiprocessor        = 32;
  __traits.max_threads_per_multiprocessor       = 2048;
  __traits.max_warps_per_multiprocessor         = __traits.max_threads_per_multiprocessor / __traits.warp_size;
  __traits.max_shared_memory_per_block_optin =
    __traits.max_shared_memory_per_multiprocessor - __traits.reserved_shared_memory_per_block;
  return __traits;
};

// No sm_90a specific fields for now.
template <>
[[nodiscard]] _CCCL_API constexpr arch_traits_t arch_traits<arch_id::sm_90a>() noexcept
{
  auto __traits    = ::cuda::arch_traits<arch_id::sm_90>();
  __traits.arch_id = arch_id::sm_90a;
  return __traits;
};

template <>
[[nodiscard]] _CCCL_API constexpr arch_traits_t arch_traits<arch_id::sm_100>() noexcept
{
  auto __traits                                 = ::cuda::__common_arch_traits(arch_id::sm_90);
  __traits.max_shared_memory_per_multiprocessor = 228 * 1024;
  __traits.max_blocks_per_multiprocessor        = 32;
  __traits.max_threads_per_multiprocessor       = 2048;
  __traits.max_warps_per_multiprocessor         = __traits.max_threads_per_multiprocessor / __traits.warp_size;
  __traits.max_shared_memory_per_block_optin =
    __traits.max_shared_memory_per_multiprocessor - __traits.reserved_shared_memory_per_block;
  return __traits;
};

template <>
[[nodiscard]] _CCCL_API constexpr arch_traits_t arch_traits<arch_id::sm_100a>() noexcept
{
  auto __traits    = ::cuda::arch_traits<arch_id::sm_100>();
  __traits.arch_id = arch_id::sm_100a;
  return __traits;
};

template <>
[[nodiscard]] _CCCL_API constexpr arch_traits_t arch_traits<arch_id::sm_103>() noexcept
{
  auto __traits                     = ::cuda::arch_traits<arch_id::sm_100>();
  __traits.arch_id                  = arch_id::sm_103;
  __traits.compute_capability_major = 10;
  __traits.compute_capability_minor = 3;
  __traits.compute_capability       = compute_capability{103};
  return __traits;
};

template <>
[[nodiscard]] _CCCL_API constexpr arch_traits_t arch_traits<arch_id::sm_103a>() noexcept
{
  auto __traits    = ::cuda::arch_traits<arch_id::sm_103>();
  __traits.arch_id = arch_id::sm_103a;
  return __traits;
};

template <>
[[nodiscard]] _CCCL_API constexpr arch_traits_t arch_traits<arch_id::sm_110>() noexcept
{
  auto __traits                           = ::cuda::arch_traits<arch_id::sm_100>();
  __traits.arch_id                        = arch_id::sm_110;
  __traits.compute_capability_major       = 11;
  __traits.compute_capability_minor       = 0;
  __traits.compute_capability             = compute_capability{110};
  __traits.max_blocks_per_multiprocessor  = 24;
  __traits.max_threads_per_multiprocessor = 1536;
  __traits.max_warps_per_multiprocessor   = __traits.max_threads_per_multiprocessor / __traits.warp_size;
  return __traits;
};

template <>
[[nodiscard]] _CCCL_API constexpr arch_traits_t arch_traits<arch_id::sm_110a>() noexcept
{
  auto __traits    = ::cuda::arch_traits<arch_id::sm_110>();
  __traits.arch_id = arch_id::sm_110a;
  return __traits;
};

template <>
[[nodiscard]] _CCCL_API constexpr arch_traits_t arch_traits<arch_id::sm_120>() noexcept
{
  auto __traits                                 = ::cuda::__common_arch_traits(arch_id::sm_120);
  __traits.max_shared_memory_per_multiprocessor = 100 * 1024;
  __traits.max_blocks_per_multiprocessor        = 24;
  __traits.max_threads_per_multiprocessor       = 1536;
  __traits.max_warps_per_multiprocessor         = __traits.max_threads_per_multiprocessor / __traits.warp_size;
  __traits.max_shared_memory_per_block_optin =
    __traits.max_shared_memory_per_multiprocessor - __traits.reserved_shared_memory_per_block;
  return __traits;
};

template <>
[[nodiscard]] _CCCL_API constexpr arch_traits_t arch_traits<arch_id::sm_120a>() noexcept
{
  auto __traits    = ::cuda::arch_traits<arch_id::sm_120>();
  __traits.arch_id = arch_id::sm_120a;
  return __traits;
};

template <>
[[nodiscard]] _CCCL_API constexpr arch_traits_t arch_traits<arch_id::sm_121>() noexcept
{
  auto __traits                     = ::cuda::arch_traits<arch_id::sm_120>();
  __traits.arch_id                  = arch_id::sm_121;
  __traits.compute_capability_major = 12;
  __traits.compute_capability_minor = 1;
  __traits.compute_capability       = compute_capability{121};
  return __traits;
};

template <>
[[nodiscard]] _CCCL_API constexpr arch_traits_t arch_traits<arch_id::sm_121a>() noexcept
{
  auto __traits    = ::cuda::arch_traits<arch_id::sm_121>();
  __traits.arch_id = arch_id::sm_121a;
  return __traits;
};

//! @brief Gets the architecture traits for the given architecture id \c __id.
//!
//! @throws \c cuda::cuda_error if the \c __id is not a known architecture.
[[nodiscard]] _CCCL_API constexpr arch_traits_t arch_traits_for(arch_id __id)
{
  switch (__id)
  {
    case arch_id::sm_60:
      return ::cuda::arch_traits<arch_id::sm_60>();
    case arch_id::sm_61:
      return ::cuda::arch_traits<arch_id::sm_61>();
    case arch_id::sm_62:
      return ::cuda::arch_traits<arch_id::sm_62>();
    case arch_id::sm_70:
      return ::cuda::arch_traits<arch_id::sm_70>();
    case arch_id::sm_75:
      return ::cuda::arch_traits<arch_id::sm_75>();
    case arch_id::sm_80:
      return ::cuda::arch_traits<arch_id::sm_80>();
    case arch_id::sm_86:
      return ::cuda::arch_traits<arch_id::sm_86>();
    case arch_id::sm_87:
      return ::cuda::arch_traits<arch_id::sm_87>();
    case arch_id::sm_88:
      return ::cuda::arch_traits<arch_id::sm_88>();
    case arch_id::sm_89:
      return ::cuda::arch_traits<arch_id::sm_89>();
    case arch_id::sm_90:
      return ::cuda::arch_traits<arch_id::sm_90>();
    case arch_id::sm_90a:
      return ::cuda::arch_traits<arch_id::sm_90a>();
    case arch_id::sm_100:
      return ::cuda::arch_traits<arch_id::sm_100>();
    case arch_id::sm_100a:
      return ::cuda::arch_traits<arch_id::sm_100a>();
    case arch_id::sm_103:
      return ::cuda::arch_traits<arch_id::sm_103>();
    case arch_id::sm_103a:
      return ::cuda::arch_traits<arch_id::sm_103a>();
    case arch_id::sm_110:
      return ::cuda::arch_traits<arch_id::sm_110>();
    case arch_id::sm_110a:
      return ::cuda::arch_traits<arch_id::sm_110a>();
    case arch_id::sm_120:
      return ::cuda::arch_traits<arch_id::sm_120>();
    case arch_id::sm_120a:
      return ::cuda::arch_traits<arch_id::sm_120a>();
    case arch_id::sm_121:
      return ::cuda::arch_traits<arch_id::sm_121>();
    case arch_id::sm_121a:
      return ::cuda::arch_traits<arch_id::sm_121a>();
    default:
#if _CCCL_HAS_CTK()
      ::cuda::__throw_cuda_error(::cudaErrorInvalidValue, "Traits requested for an unknown architecture");
#else // ^^^ _CCCL_HAS_CTK() ^^^ / vvv !_CCCL_HAS_CTK() vvv
      ::cuda::__throw_cuda_error(/*cudaErrorInvalidValue*/ 1, "Traits requested for an unknown architecture");
#endif // ^^^ !_CCCL_HAS_CTK() ^^^
      break;
  }
}

//! @brief Gets the architecture traits for the given compute capability \c __cc.
//!
//! @throws \c cuda::cuda_error if the \c __cc doesn't have a corresponding architecture id.
[[nodiscard]] _CCCL_API constexpr arch_traits_t arch_traits_for(compute_capability __cc)
{
  return ::cuda::arch_traits_for(::cuda::to_arch_id(__cc));
}

_CCCL_END_NAMESPACE_CUDA

#if _CCCL_CUDA_COMPILATION()

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE

//! @brief Returns the \c cuda::arch_trait_t of the architecture that is currently being compiled.
//!
//!        If the current architecture is not a known architecture from \c cuda::arch_id enumeration, the compilation
//!        will fail.
//!
//! @note This API cannot be used in constexpr context when compiling with nvc++ in CUDA mode.
template <class _Dummy = void>
[[nodiscard]] _CCCL_DEVICE_API inline _CCCL_TARGET_CONSTEXPR ::cuda::arch_traits_t current_arch_traits() noexcept
{
#  if _CCCL_DEVICE_COMPILATION()
  return ::cuda::arch_traits_for(::cuda::device::current_arch_id<_Dummy>());
#  else // ^^^ _CCCL_DEVICE_COMPILATION() ^^^ / vvv !_CCCL_DEVICE_COMPILATION() vvv
  return {};
#  endif // ^^^ !_CCCL_DEVICE_COMPILATION() ^^^
}

_CCCL_END_NAMESPACE_CUDA_DEVICE

#endif // _CCCL_CUDA_COMPILATION

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___DEVICE_ARCH_TRAITS_H
