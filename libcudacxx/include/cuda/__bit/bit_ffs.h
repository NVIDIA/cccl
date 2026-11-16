//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___BIT_BIT_FFS_H
#define _CUDA___BIT_BIT_FFS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/countr.h>
#include <cuda/std/__limits/numeric_limits.h> // IWYU pragma: keep
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/cstdint>

#if _CCCL_COMPILER(MSVC)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC)

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CHECK_BUILTIN(builtin_ffs) || _CCCL_COMPILER(GCC, <, 10)
#  define _CCCL_BUILTIN_FFS(...) __builtin_ffs(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_ffs)

#if _CCCL_CHECK_BUILTIN(builtin_ffsll) || _CCCL_COMPILER(GCC, <, 10)
#  define _CCCL_BUILTIN_FFSLL(...) __builtin_ffsll(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_ffsll)

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr int __bit_ffs_impl_generic(const _Tp __value) noexcept
{
  return (__value == _Tp{0}) ? 0 : ::cuda::std::countr_zero(__value) + 1;
}

#if !_CCCL_COMPILER(NVRTC)
template <class _Tp>
[[nodiscard]] _CCCL_HOST_API int __bit_ffs_impl_host(const _Tp __value) noexcept
{
#  if defined(_CCCL_BUILTIN_FFS) && defined(_CCCL_BUILTIN_FFSLL)
  if constexpr (sizeof(_Tp) <= sizeof(::cuda::std::uint32_t))
  {
    return _CCCL_BUILTIN_FFS(static_cast<int>(__value));
  }
  else
  {
    return _CCCL_BUILTIN_FFSLL(static_cast<long long>(__value));
  }
#  elif _CCCL_COMPILER(MSVC)
  unsigned long __where{};
  unsigned char __found{};
  if constexpr (sizeof(_Tp) <= sizeof(::cuda::std::uint32_t))
  {
    __found = ::_BitScanForward(&__where, static_cast<::cuda::std::uint32_t>(__value));
  }
  else
  {
    __found = ::_BitScanForward64(&__where, static_cast<::cuda::std::uint64_t>(__value));
  }
  return __found ? static_cast<int>(__where) + 1 : 0;
#  else
  return ::cuda::__bit_ffs_impl_generic(__value);
#  endif // _CCCL_BUILTIN_FFS && _CCCL_BUILTIN_FFSLL
}
#endif // !_CCCL_COMPILER(NVRTC)

#if _CCCL_CUDA_COMPILATION()
template <class _Tp>
[[nodiscard]] _CCCL_DEVICE_API int __bit_ffs_impl_device(const _Tp __v) noexcept
{
  using ::cuda::std::uint32_t;
  using ::cuda::std::uint64_t;

  if constexpr (sizeof(_Tp) <= sizeof(uint32_t))
  {
    return ::__ffs(static_cast<int>(__v));
  }
  else if constexpr (sizeof(_Tp) == sizeof(::cuda::std::uint64_t))
  {
    const auto __lo = static_cast<uint32_t>(__v);
    const auto __hi = static_cast<uint32_t>(__v >> 32);

    const bool __lo_is_zero = (__lo == 0);

    // __ffx(x) is implemented as:
    // brev.b32             tmp1, x;
    // bfind.shiftamt.u32   tmp2, tmp1;
    // add.s32              ret, tmp2, 1;
    //
    // For our improved implementation, we don't want the ret + 1 part, so we subtract 1 from the result. That leaves us
    // with only first 2 instructions. Which is what we need. We do this to avoid using inline PTX.
    int __ret = ::__ffs((__lo_is_zero) ? __hi : __lo) - 1;

    // When the first set bit is in __hi, we need to add 32 to __ret. We can take advantage of the fact that
    // bfind.shiftamt returns 0-based index < 32 and ~0 when there is no set bit. The idea is that we can simply set the
    // 6th bit (32) to set the offset. If no set bit was found, the binary or won't change the ~0 value, so we don't
    // need any special handling.
    if (__lo_is_zero)
    {
      __ret |= 32;
    }
    return __ret + 1;
  }
#  if _CCCL_HAS_INT128()
  else if constexpr (sizeof(_Tp) == sizeof(__uint128_t))
  {
    const auto __v0 = static_cast<uint32_t>(__v);
    const auto __v1 = static_cast<uint32_t>(__v >> 32);
    const auto __v2 = static_cast<uint32_t>(__v >> 64);
    const auto __v3 = static_cast<uint32_t>(__v >> 96);

    const bool __lo_is_zero   = ((__v0 | __v1) == 0);
    const auto __lo           = (__lo_is_zero) ? __v2 : __v0;
    const auto __hi           = (__lo_is_zero) ? __v3 : __v1;
    const bool __word_is_zero = (__lo == 0);
    int __ret                 = ::__ffs((__word_is_zero) ? __hi : __lo) - 1;

    // As in the 64-bit case, OR in the word offsets to preserve the ~0 sentinel for a zero input.
    if (__lo_is_zero)
    {
      __ret |= 64;
    }
    if (__word_is_zero)
    {
      __ret |= 32;
    }
    return __ret + 1;
  }
#  endif // _CCCL_HAS_INT128()
  else
  {
    return ::cuda::__bit_ffs_impl_generic(__v);
  }
}
#endif // _CCCL_CUDA_COMPILATION()

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr int __bit_ffs(const _Tp __value) noexcept
{
#if !_CCCL_TILE_COMPILATION() // error: asm statement is unsupported in tile code
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST,
                      ({
                        if constexpr (sizeof(_Tp) <= sizeof(::cuda::std::uint64_t))
                        {
                          return ::cuda::__bit_ffs_impl_host(__value);
                        }
                      }),
                      ({ return ::cuda::__bit_ffs_impl_device(__value); }))
  }
#endif // !_CCCL_TILE_COMPILATION()
  return ::cuda::__bit_ffs_impl_generic(__value);
}

// Returns one plus the index of the least significant set bit of __value, or 0 if __value is zero.
// This matches the semantics of __builtin_ffs and CUDA's __ffs. Unlike cuda::std::countr_zero, the
// result is 1-based and the zero input is well defined (it returns 0).
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(::cuda::std::__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr int bit_ffs(const _Tp __value) noexcept
{
  const auto __ret = ::cuda::__bit_ffs(__value);
  _CCCL_ASSUME(__ret >= 0 && __ret <= ::cuda::std::numeric_limits<_Tp>::digits);
  return __ret;
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___BIT_BIT_FFS_H
