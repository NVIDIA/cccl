//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___BIT_BYTESWAP_H
#define _CUDA_STD___BIT_BYTESWAP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#if _CCCL_COMPILER(MSVC)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC)

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CHECK_BUILTIN(builtin_bswap16) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_BSWAP16(...) __builtin_bswap16(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_bswap16)

#if _CCCL_CHECK_BUILTIN(builtin_bswap32) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_BSWAP32(...) __builtin_bswap32(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_bswap32)

#if _CCCL_CHECK_BUILTIN(builtin_bswap64) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_BSWAP64(...) __builtin_bswap64(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_bswap64)

#if _CCCL_CHECK_BUILTIN(builtin_bswap128) // Only available in GCC >= 11 which supports __has_builtin
#  define _CCCL_BUILTIN_BSWAP128(...) __builtin_bswap128(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_bswap128)

// nvcc doesn't support these builtins in device code
#if _CCCL_CUDA_COMPILER(NVCC) && _CCCL_DEVICE_COMPILATION()
#  undef _CCCL_BUILTIN_BSWAP16
#  undef _CCCL_BUILTIN_BSWAP32
#  undef _CCCL_BUILTIN_BSWAP64
#  undef _CCCL_BUILTIN_BSWAP128
#endif // _CCCL_CUDA_COMPILER(NVCC) && _CCCL_DEVICE_COMPILATION()

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __byteswap_impl(_Tp __val) noexcept;

template <class _Full>
[[nodiscard]] _CCCL_API constexpr _Full __byteswap_impl_recursive(_Full __val) noexcept
{
  using _Half            = __make_nbit_uint_t<numeric_limits<_Full>::digits / 2>;
  constexpr auto __shift = numeric_limits<_Half>::digits;

  if constexpr (sizeof(_Full) > 2)
  {
    return static_cast<_Full>(::cuda::std::__byteswap_impl(static_cast<_Half>(__val >> __shift)))
         | (static_cast<_Full>(::cuda::std::__byteswap_impl(static_cast<_Half>(__val))) << __shift);
  }
  else
  {
    return static_cast<_Full>((__val << __shift) | (__val >> __shift));
  }
}

#if _CCCL_CUDA_COMPILATION()

template <class _Tp>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp __byteswap_impl_device(_Tp __val) noexcept
{
  if constexpr (sizeof(_Tp) == sizeof(uint16_t))
  {
    return static_cast<uint16_t>(::__byte_perm(static_cast<uint32_t>(__val), 0u, 0x3201u));
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint32_t))
  {
    return ::__byte_perm(__val, 0u, 0x0123u);
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint64_t))
  {
    const auto __lo = ::__byte_perm(static_cast<uint32_t>(__val >> 32), 0u, 0x0123u);
    const auto __hi = ::__byte_perm(static_cast<uint32_t>(__val), 0u, 0x0123u);
    return (static_cast<uint64_t>(__hi) << 32) | static_cast<uint64_t>(__lo);
  }
  else
  {
    return ::cuda::std::__byteswap_impl_recursive(__val);
  }
}

#endif // _CCCL_CUDA_COMPILATION()

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __byteswap_impl(_Tp __val) noexcept
{
  constexpr auto __shift = numeric_limits<uint8_t>::digits;

  _Tp __result{};

  _CCCL_PRAGMA_UNROLL_FULL()
  for (size_t __i = 0; __i < sizeof(_Tp); ++__i)
  {
    __result <<= __shift;
    __result |= _Tp(__val & _Tp(numeric_limits<uint8_t>::max()));
    __val >>= __shift;
  }
  return __result;
}

[[nodiscard]] _CCCL_API constexpr uint16_t __byteswap_impl(uint16_t __val) noexcept
{
#if defined(_CCCL_BUILTIN_BSWAP16)
  return _CCCL_BUILTIN_BSWAP16(__val);
#else // ^^^ _CCCL_BUILTIN_BSWAP16 ^^^ / vvv !_CCCL_BUILTIN_BSWAP16 vvv
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
#  if _CCCL_COMPILER(MSVC)
    NV_IF_TARGET(NV_IS_HOST, return ::_byteswap_ushort(__val);)
#  endif // _CCCL_COMPILER(MSVC)
    NV_IF_TARGET(NV_IS_DEVICE, return ::cuda::std::__byteswap_impl_device(__val);)
  }
  return ::cuda::std::__byteswap_impl_recursive(__val);
#endif // !_CCCL_BUILTIN_BSWAP16
}

[[nodiscard]] _CCCL_API constexpr uint32_t __byteswap_impl(uint32_t __val) noexcept
{
#if defined(_CCCL_BUILTIN_BSWAP32)
  return _CCCL_BUILTIN_BSWAP32(__val);
#else // ^^^ _CCCL_BUILTIN_BSWAP32 ^^^ / vvv !_CCCL_BUILTIN_BSWAP32 vvv
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
#  if _CCCL_COMPILER(MSVC)
    NV_IF_TARGET(NV_IS_HOST, return ::_byteswap_ulong(__val);)
#  endif // _CCCL_COMPILER(MSVC)
    NV_IF_TARGET(NV_IS_DEVICE, return ::cuda::std::__byteswap_impl_device(__val);)
  }
  return ::cuda::std::__byteswap_impl_recursive(__val);
#endif // !_CCCL_BUILTIN_BSWAP32
}

[[nodiscard]] _CCCL_API constexpr uint64_t __byteswap_impl(uint64_t __val) noexcept
{
#if defined(_CCCL_BUILTIN_BSWAP64)
  return _CCCL_BUILTIN_BSWAP64(__val);
#else // ^^^ _CCCL_BUILTIN_BSWAP64 ^^^ / vvv !_CCCL_BUILTIN_BSWAP64 vvv
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
#  if _CCCL_COMPILER(MSVC)
    NV_IF_TARGET(NV_IS_HOST, return ::_byteswap_uint64(__val);)
#  endif // _CCCL_COMPILER(MSVC)
    NV_IF_TARGET(NV_IS_DEVICE, return ::cuda::std::__byteswap_impl_device(__val);)
  }
  return ::cuda::std::__byteswap_impl_recursive(__val);
#endif // !_CCCL_BUILTIN_BSWAP64
}

#if _CCCL_HAS_INT128()
[[nodiscard]] _CCCL_API constexpr __uint128_t __byteswap_impl(__uint128_t __val) noexcept
{
#  if defined(_CCCL_BUILTIN_BSWAP128)
  // nvcc fails to use this builtin in constexpr context
#    if _CCCL_CUDA_COMPILER(NVCC)
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
#    endif // _CCCL_CUDA_COMPILER(NVCC)
  {
    return _CCCL_BUILTIN_BSWAP128(__val);
  }
#  endif // _CCCL_BUILTIN_BSWAP128
  return ::cuda::std::__byteswap_impl_recursive(__val);
}
#endif // _CCCL_HAS_INT128()

_CCCL_TEMPLATE(class _Integer)
_CCCL_REQUIRES(is_integral_v<_Integer>)
[[nodiscard]] _CCCL_API constexpr _Integer byteswap(_Integer __val) noexcept
{
  if constexpr (sizeof(_Integer) > 1)
  {
    return static_cast<_Integer>(::cuda::std::__byteswap_impl(::cuda::std::__to_unsigned_like(__val)));
  }
  else
  {
    return __val;
  }
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___BIT_BYTESWAP_H
