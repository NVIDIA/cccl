//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___BIT_FFS_H
#define _CUDA___BIT_FFS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <nv/target>

#if _CCCL_COMPILER(MSVC)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC)

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_HAS_BUILTIN(__builtin_ffs) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_FFS(...)   __builtin_ffs(__VA_ARGS__)
#  define _CCCL_BUILTIN_FFSLL(...) __builtin_ffsll(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__builtin_ffs) || _CCCL_COMPILER(GCC)

_CCCL_BEGIN_NAMESPACE_CUDA

namespace __detail
{
template <class _Tp>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr int __ffs_constexpr_impl(_Tp __v) noexcept
{
  static_assert(::cuda::std::__cccl_is_unsigned_integer_v<_Tp>, "_Tp must be unsigned");
  if (__v == 0)
  {
    return 0;
  }
  int __pos = 1;
  while ((__v & 1) == 0)
  {
    __v >>= 1;
    ++__pos;
  }
  return __pos;
}

#if _CCCL_COMPILER(NVRTC)
[[nodiscard]] inline int __ffs_host32(::cuda::std::uint32_t __v) noexcept
{
  return ::cuda::__detail::__ffs_constexpr_impl(__v);
}
#else
[[nodiscard]] _CCCL_HOST_API inline int __ffs_host32(::cuda::std::uint32_t __v) noexcept
{
  if (::cuda::std::__cccl_default_is_constant_evaluated())
  {
    return ::cuda::__detail::__ffs_constexpr_impl(__v);
  }
#  if defined(_CCCL_BUILTIN_FFS)
  return _CCCL_BUILTIN_FFS(static_cast<int>(__v));
#  elif _CCCL_COMPILER(MSVC)
  unsigned long __where{};
  const unsigned char __res = ::_BitScanForward(&__where, __v);
  return __res ? static_cast<int>(__where) + 1 : 0;
#  else
  return ::cuda::__detail::__ffs_constexpr_impl(__v);
#  endif // host implementations
}
#endif // _CCCL_COMPILER(NVRTC)

#if _CCCL_COMPILER(NVRTC)
[[nodiscard]] inline int __ffs_host64(::cuda::std::uint64_t __v) noexcept
{
  return ::cuda::__detail::__ffs_constexpr_impl(__v);
}
#else
[[nodiscard]] _CCCL_HOST_API inline int __ffs_host64(::cuda::std::uint64_t __v) noexcept
{
  if (::cuda::std::__cccl_default_is_constant_evaluated())
  {
    return ::cuda::__detail::__ffs_constexpr_impl(__v);
  }
#  if defined(_CCCL_BUILTIN_FFSLL)
  return _CCCL_BUILTIN_FFSLL(static_cast<long long>(__v));
#  elif _CCCL_COMPILER(MSVC)
  unsigned long __where{};
  const unsigned char __res = ::_BitScanForward64(&__where, __v);
  return __res ? static_cast<int>(__where) + 1 : 0;
#  else
  return ::cuda::__detail::__ffs_constexpr_impl(__v);
#  endif // host implementations
}
#endif // _CCCL_COMPILER(NVRTC)

#if _CCCL_HAS_INT128()
#  if _CCCL_COMPILER(NVRTC)
[[nodiscard]] inline int __ffs_host128(__uint128_t __v) noexcept
{
  return ::cuda::__detail::__ffs_constexpr_impl(__v);
}
#  else
[[nodiscard]] _CCCL_HOST_API inline int __ffs_host128(__uint128_t __v) noexcept
{
  if (::cuda::std::__cccl_default_is_constant_evaluated())
  {
    return ::cuda::__detail::__ffs_constexpr_impl(__v);
  }
  const auto __lo = static_cast<::cuda::std::uint64_t>(__v);
  if (const int __result = ::cuda::__detail::__ffs_host64(__lo))
  {
    return __result;
  }
  const auto __hi = static_cast<::cuda::std::uint64_t>(__v >> 64);
  if (const int __result = ::cuda::__detail::__ffs_host64(__hi))
  {
    return __result + 64;
  }
  return 0;
}
#  endif // _CCCL_COMPILER(NVRTC)
#endif // _CCCL_HAS_INT128()

#if _CCCL_CUDA_COMPILATION()
[[nodiscard]] _CCCL_DEVICE_API inline int __ffs_device32(::cuda::std::uint32_t __v) noexcept
{
  return ::__ffs(static_cast<int>(__v));
}

[[nodiscard]] _CCCL_DEVICE_API inline int __ffs_device64(::cuda::std::uint64_t __v) noexcept
{
  return ::__ffsll(static_cast<long long>(__v));
}

#  if _CCCL_HAS_INT128()
[[nodiscard]] _CCCL_DEVICE_API inline int __ffs_device128(__uint128_t __v) noexcept
{
  const auto __lo = static_cast<::cuda::std::uint64_t>(__v);
  if (const int __result = ::cuda::__detail::__ffs_device64(__lo))
  {
    return __result;
  }
  const auto __hi = static_cast<::cuda::std::uint64_t>(__v >> 64);
  if (const int __result = ::cuda::__detail::__ffs_device64(__hi))
  {
    return __result + 64;
  }
  return 0;
}
#  endif // _CCCL_HAS_INT128()
#endif // _CCCL_CUDA_COMPILATION()
} // namespace __detail

template <class _Tp, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_cv_unsigned_integer_v<_Tp>, int> = 0>
[[nodiscard]] _CCCL_API constexpr int ffs(_Tp __v) noexcept
{
  using _Unsigned = ::cuda::std::remove_cv_t<_Tp>;

#if !defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  return __detail::__ffs_constexpr_impl(static_cast<_Unsigned>(__v));
#else
  if (::cuda::std::__cccl_default_is_constant_evaluated())
  {
    return __detail::__ffs_constexpr_impl(static_cast<_Unsigned>(__v));
  }

#  if _CCCL_HAS_INT128()
  if constexpr (sizeof(_Unsigned) == sizeof(__uint128_t))
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST,
                      (return __detail::__ffs_host128(static_cast<__uint128_t>(__v));),
                      (return __detail::__ffs_device128(static_cast<__uint128_t>(__v));))
  }
#  endif // _CCCL_HAS_INT128()

  if constexpr (sizeof(_Unsigned) <= sizeof(::cuda::std::uint32_t))
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST,
                      (return __detail::__ffs_host32(static_cast<::cuda::std::uint32_t>(__v));),
                      (return __detail::__ffs_device32(static_cast<::cuda::std::uint32_t>(__v));))
  }
  else
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST,
                      (return __detail::__ffs_host64(static_cast<::cuda::std::uint64_t>(__v));),
                      (return __detail::__ffs_device64(static_cast<::cuda::std::uint64_t>(__v));))
  }
#endif // !_CCCL_BUILTIN_IS_CONSTANT_EVALUATED
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___BIT_FFS_H
