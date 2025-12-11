//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___DEVICE_ARCH_ID_H
#define _CUDA___DEVICE_ARCH_ID_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/compute_capability.h>
#include <cuda/__fwd/devices.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__utility/to_underlying.h>
#include <cuda/std/array>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Architecture identifier
//! This type identifies an architecture. It has more possible entries than just numeric values of the compute
//! capability. For example, sm_90 and sm_90a have the same compute capability, but the identifier is different.
enum class arch_id : int
{
  sm_60   = 60,
  sm_61   = 61,
  sm_62   = 62,
  sm_70   = 70,
  sm_75   = 75,
  sm_80   = 80,
  sm_86   = 86,
  sm_87   = 87,
  sm_88   = 88,
  sm_89   = 89,
  sm_90   = 90,
  sm_100  = 100,
  sm_103  = 103,
  sm_110  = 110,
  sm_120  = 120,
  sm_121  = 121,
  sm_90a  = 90 * __arch_specific_id_multiplier,
  sm_100a = 100 * __arch_specific_id_multiplier,
  sm_103a = 103 * __arch_specific_id_multiplier,
  sm_110a = 110 * __arch_specific_id_multiplier,
  sm_120a = 120 * __arch_specific_id_multiplier,
  sm_121a = 121 * __arch_specific_id_multiplier,
};

[[nodiscard]] _CCCL_API constexpr auto __all_arch_ids() noexcept
{
  return ::cuda::std::array{
    arch_id::sm_60,   arch_id::sm_61,   arch_id::sm_62,   arch_id::sm_70,   arch_id::sm_75,  arch_id::sm_80,
    arch_id::sm_86,   arch_id::sm_87,   arch_id::sm_88,   arch_id::sm_89,   arch_id::sm_90,  arch_id::sm_100,
    arch_id::sm_103,  arch_id::sm_110,  arch_id::sm_120,  arch_id::sm_121,  arch_id::sm_90a, arch_id::sm_100a,
    arch_id::sm_103a, arch_id::sm_110a, arch_id::sm_120a, arch_id::sm_121a,
  };
}

[[nodiscard]] _CCCL_API constexpr bool __is_specific_arch(arch_id __arch) noexcept
{
  return ::cuda::std::to_underlying(__arch) > __arch_specific_id_multiplier;
}

[[nodiscard]] _CCCL_API constexpr bool __has_known_arch(compute_capability __cc) noexcept
{
  switch (__cc.get())
  {
    case ::cuda::std::to_underlying(arch_id::sm_60):
    case ::cuda::std::to_underlying(arch_id::sm_61):
    case ::cuda::std::to_underlying(arch_id::sm_62):
    case ::cuda::std::to_underlying(arch_id::sm_70):
    case ::cuda::std::to_underlying(arch_id::sm_75):
    case ::cuda::std::to_underlying(arch_id::sm_80):
    case ::cuda::std::to_underlying(arch_id::sm_86):
    case ::cuda::std::to_underlying(arch_id::sm_87):
    case ::cuda::std::to_underlying(arch_id::sm_88):
    case ::cuda::std::to_underlying(arch_id::sm_89):
    case ::cuda::std::to_underlying(arch_id::sm_90):
    case ::cuda::std::to_underlying(arch_id::sm_100):
    case ::cuda::std::to_underlying(arch_id::sm_103):
    case ::cuda::std::to_underlying(arch_id::sm_110):
    case ::cuda::std::to_underlying(arch_id::sm_120):
    case ::cuda::std::to_underlying(arch_id::sm_121):
      return true;
    default:
      return false;
  }
}

[[nodiscard]] _CCCL_API constexpr bool __has_known_specific_arch(compute_capability __cc) noexcept
{
  switch (__cc.get() * __arch_specific_id_multiplier)
  {
    case ::cuda::std::to_underlying(arch_id::sm_90a):
    case ::cuda::std::to_underlying(arch_id::sm_100a):
    case ::cuda::std::to_underlying(arch_id::sm_103a):
    case ::cuda::std::to_underlying(arch_id::sm_110a):
    case ::cuda::std::to_underlying(arch_id::sm_120a):
    case ::cuda::std::to_underlying(arch_id::sm_121a):
      return true;
    default:
      return false;
  }
}

//! @brief Converts the compute capability to the architecture id.
//!
//! @param __cc The compute capability. Must have a corresponding architecture id.
//!
//! @returns The architecture id.
[[nodiscard]] _CCCL_API constexpr arch_id to_arch_id(compute_capability __cc) noexcept
{
  _CCCL_ASSERT(::cuda::__has_known_arch(__cc), "this compute capability cannot be converted to arch id");
  return static_cast<arch_id>(__cc.get());
}

//! @brief Converts the compute capability to the architecture specific id.
//!
//! @param __cc The compute capability. Must have a corresponding architecture specific id.
//!
//! @returns The architecture specific id.
[[nodiscard]] _CCCL_API constexpr arch_id to_arch_specific_id(compute_capability __cc) noexcept
{
  _CCCL_ASSERT(::cuda::__has_known_specific_arch(__cc),
               "this compute capability cannot be converted to arch specific id");
  return static_cast<arch_id>(__cc.get() * __arch_specific_id_multiplier);
}

_CCCL_END_NAMESPACE_CUDA

#if _CCCL_CUDA_COMPILATION()

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE

//! @brief This function should cause a link error. If it happens, you are trying to compile the code for an unsupported
//!        architecture (too new/old).
_CCCL_DEVICE_API ::cuda::arch_id __unknown_cuda_architecture();

//! @brief Returns the \c cuda::arch_id that is currently being compiled.
//!
//!        If the current architecture is not a known architecture from \c cuda::arch_id enumeration, the compilation
//!        will fail.
//!
//! @note This API cannot be used in constexpr context when compiling with nvc++ in CUDA mode.
template <class _Dummy = void>
[[nodiscard]] _CCCL_DEVICE_API inline _CCCL_TARGET_CONSTEXPR ::cuda::arch_id current_arch_id() noexcept
{
#  if _CCCL_CUDA_COMPILER(NVHPC)
  const auto __cc = ::cuda::device::current_compute_capability();
  if (::cuda::__is_known_arch_of(__cc))
  {
    return ::cuda::to_arch_id(__cc);
  }
  else
  {
    return ::cuda::device::__unknown_cuda_architecture();
  }
#  elif _CCCL_DEVICE_COMPILATION()
  constexpr auto __cc = ::cuda::device::current_compute_capability();
#    if defined(__CUDA_ARCH_SPECIFIC__)
  constexpr auto __is_known_cc = ::cuda::std::__always_false_v<_Dummy> || ::cuda::__has_known_specific_arch(__cc);
  static_assert(__is_known_cc, "unknown CUDA specific architecture");
  return ::cuda::to_arch_specific_id(__cc);
#    else // ^^^ __CUDA_ARCH_SPECIFIC__ ^^^ / vvv !__CUDA_ARCH_SPECIFIC__ vvv
  constexpr auto __is_known_cc = ::cuda::std::__always_false_v<_Dummy> || ::cuda::__has_known_arch(__cc);
  static_assert(__is_known_cc, "unknown CUDA architecture");
  return ::cuda::to_arch_id(__cc);
#    endif // ^^^ __CUDA_ARCH_SPECIFIC__ ^^^
#  else
  return {};
#  endif // ^^^ single-pass cuda compiler ^^^
}

_CCCL_END_NAMESPACE_CUDA_DEVICE

#endif // _CCCL_CUDA_COMPILATION()

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___DEVICE_ARCH_ID_H
