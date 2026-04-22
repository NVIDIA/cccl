//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___ARCH_ARCH_H
#define _CUDA___ARCH_ARCH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__arch/arch_id.h>
#include <cuda/__arch/arch_traits.h>
#include <cuda/__device/compute_capability.h>
#include <cuda/__fwd/arch.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

class arch
{
  arch_id __id_{};

public:
#define _CCCL_DECLARE_ARCH(_CC)          static const arch sm_##_CC;
#define _CCCL_DECLARE_SPECIFIC_ARCH(_CC) static const arch sm_##_CC##a;
  _CCCL_PP_FOR_EACH(_CCCL_DECLARE_ARCH, _CCCL_KNOWN_CUDA_ARCH_LIST)
  _CCCL_PP_FOR_EACH(_CCCL_DECLARE_SPECIFIC_ARCH, _CCCL_KNOWN_CUDA_ARCH_SPECIFIC_LIST)
#undef _CCCL_DECLARE_ARCH
#undef _CCCL_DECLARE_SPECIFIC_ARCH

  _CCCL_HIDE_FROM_ABI explicit arch() = default;

  _CCCL_API constexpr arch(arch_id __id) noexcept
      : __id_{__id}
  {}

  [[nodiscard]] _CCCL_API constexpr arch_id id() const noexcept
  {
    return __id_;
  }

  [[nodiscard]] _CCCL_API constexpr bool is_arch_specific() const noexcept
  {
    return ::cuda::__is_specific_arch(__id_);
  }

  [[nodiscard]] _CCCL_API constexpr bool is_family_specific() const noexcept
  {
    // todo: fix once we have family specific arch ids
    return false;
  }

  [[nodiscard]] _CCCL_API constexpr arch_traits_t traits() const noexcept
  {
    return ::cuda::arch_traits_for(__id_);
  }

  [[nodiscard]] _CCCL_API constexpr bool provides(arch __other) const noexcept
  {
    // If __other is arch-specific, then the arch_ids must match.
    if (__other.is_arch_specific())
    {
      return *this == __other;
    }

    const compute_capability __cc{__id_};
    const compute_capability __other_cc{__other.__id_};

    // If __other is family-specific, then both must be part of the same family.
    // Note: This id can be arch-specific.
    if (__other.is_family_specific())
    {
      if (!is_arch_specific() && !is_family_specific())
      {
        return false;
      }
      return (__cc.major_cap() == __other_cc.major_cap()) && (__cc.minor_cap() >= __other_cc.minor_cap());
    }

    // If __other is not arch/family-specific, just compare the CCs.
    return __cc >= __other_cc;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator==(arch __lhs, arch __rhs) noexcept
  {
    return __lhs.__id_ == __rhs.__id_;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(arch __lhs, arch __rhs) noexcept
  {
    return __lhs.__id_ != __rhs.__id_;
  }
};

#define _CCCL_DEFINE_ARCH(_CC)          inline constexpr arch arch::sm_##_CC{::cuda::arch_id::sm_##_CC};
#define _CCCL_DEFINE_SPECIFIC_ARCH(_CC) inline constexpr arch arch::sm_##_CC##a{::cuda::arch_id::sm_##_CC##a};
_CCCL_PP_FOR_EACH(_CCCL_DEFINE_ARCH, _CCCL_KNOWN_CUDA_ARCH_LIST)
_CCCL_PP_FOR_EACH(_CCCL_DEFINE_SPECIFIC_ARCH, _CCCL_KNOWN_CUDA_ARCH_SPECIFIC_LIST)
#undef _CCCL_DEFINE_ARCH
#undef _CCCL_DEFINE_SPECIFIC_ARCH

_CCCL_END_NAMESPACE_CUDA

#if _CCCL_CUDA_COMPILATION()

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE

template <class _Dummy = void>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_TARGET_CONSTEXPR ::cuda::arch current_arch() noexcept
{
  return {::cuda::device::current_arch_id<_Dummy>()};
}

_CCCL_END_NAMESPACE_CUDA_DEVICE

#endif // _CCCL_CUDA_COMPILATION()

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ARCH_ARCH_H
