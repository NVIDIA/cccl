//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___BIT_BIT_TRUNC_H
#define _CUDA___BIT_BIT_TRUNC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__bit/shl.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/cstdint>

#include <nv/target>

#if _CCCL_HOST_ARCH_FEAT(X86_64, BMI2)
#  if (_CCCL_HAS_BUILTIN(__builtin_ia32_bzhi_si) && _CCCL_HAS_BUILTIN(__builtin_ia32_bzhi_di)) || _CCCL_COMPILER(GCC)
#    define _CCCL_BUILTIN_IA32_BZHI_SI(...) __builtin_ia32_bzhi_si(__VA_ARGS__)
#    define _CCCL_BUILTIN_IA32_BZHI_DI(...) __builtin_ia32_bzhi_di(__VA_ARGS__)
#  elif _CCCL_COMPILER(MSVC)
#    include <intrin.h>
#    define _CCCL_BUILTIN_IA32_BZHI_SI(...) ::_bzhi_u32(__VA_ARGS__)
#    define _CCCL_BUILTIN_IA32_BZHI_DI(...) ::_bzhi_u64(__VA_ARGS__)
#  endif
#endif // _CCCL_HOST_ARCH_FEAT(X86_64, BMI2)

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __bit_trunc_impl_generic(const _Tp __v, const int __n) noexcept
{
  return static_cast<_Tp>(__v & (::cuda::std::shl(_Tp{1}, static_cast<unsigned>(__n)) - 1));
}

#if _CCCL_HOST_COMPILATION()
template <class _Tp>
[[nodiscard]] _CCCL_HOST_API _Tp __bit_trunc_impl_host(const _Tp __v, const int __n) noexcept
{
  using ::cuda::std::uint32_t;
  using ::cuda::std::uint64_t;

#  if defined(_CCCL_BUILTIN_IA32_BZHI_SI) && defined(_CCCL_BUILTIN_IA32_BZHI_DI)
  const auto __un = static_cast<unsigned>(__n);

  if constexpr (sizeof(_Tp) <= sizeof(uint32_t))
  {
    // BZHI only uses the low eight bits of the width.
    const auto __width = ::cuda::std::min(__un, unsigned{::cuda::std::__num_bits_v<_Tp>});
    return static_cast<_Tp>(_CCCL_BUILTIN_IA32_BZHI_SI(uint32_t{__v}, __width));
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint64_t))
  {
    // BZHI only uses the low eight bits of the width.
    const auto __width = ::cuda::std::min(__un, unsigned{::cuda::std::__num_bits_v<_Tp>});
    return static_cast<_Tp>(_CCCL_BUILTIN_IA32_BZHI_DI(__v, __width));
  }
#    if _CCCL_HAS_INT128()
  else if constexpr (sizeof(_Tp) == sizeof(__uint128_t))
  {
    const auto __lo_width = ::cuda::std::min(__un, 64u);
    const auto __hi_width = static_cast<unsigned>(::cuda::std::clamp(__n - 64, 0, 64));
    const auto __lo       = _CCCL_BUILTIN_IA32_BZHI_DI(static_cast<uint64_t>(__v), __lo_width);
    const auto __hi       = _CCCL_BUILTIN_IA32_BZHI_DI(static_cast<uint64_t>(__v >> 64), __hi_width);
    return _Tp{__lo} | (_Tp{__hi} << 64);
  }
#    endif // _CCCL_HAS_INT128()
  else
#  endif // _CCCL_BUILTIN_IA32_BZHI_SI && _CCCL_BUILTIN_IA32_BZHI_DI
  {
    return ::cuda::__bit_trunc_impl_generic(__v, __n);
  }
}
#endif // _CCCL_HOST_COMPILATION()

#if _CCCL_DEVICE_COMPILATION()
template <class = void>
[[nodiscard]]
_CCCL_DEVICE_API ::cuda::std::uint32_t
__szext_clamp_sm70(const ::cuda::std::uint32_t __v, const ::cuda::std::uint32_t __n) noexcept
{
  ::cuda::std::uint32_t __ret;
  asm("szext.clamp.u32 %0, %1, %2;" : "=r"(__ret) : "r"(__v), "r"(__n));
  return __ret;
}

template <class _Tp>
[[nodiscard]] _CCCL_DEVICE_API _Tp __bit_trunc_impl_sm70(const _Tp __v, const int __n) noexcept
{
  using ::cuda::std::uint32_t;
  using ::cuda::std::uint64_t;

  const auto __un = static_cast<unsigned>(__n);

  if constexpr (sizeof(_Tp) <= sizeof(uint32_t))
  {
    return static_cast<_Tp>(::cuda::__szext_clamp_sm70(uint32_t{__v}, __un));
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint64_t))
  {
    const auto __lo = ::cuda::__szext_clamp_sm70(static_cast<uint32_t>(__v), __un);
    const auto __hi = ::cuda::__szext_clamp_sm70(
      static_cast<uint32_t>(__v >> 32), static_cast<uint32_t>(::cuda::std::max(__n - 32, 0)));
    return _Tp{__lo} | (_Tp{__hi} << 32);
  }
#  if _CCCL_HAS_INT128()
  else if constexpr (sizeof(_Tp) == sizeof(__uint128_t))
  {
    const auto __w0 = ::cuda::__szext_clamp_sm70(static_cast<uint32_t>(__v), __un);
    const auto __w1 = ::cuda::__szext_clamp_sm70(
      static_cast<uint32_t>(__v >> 32), static_cast<uint32_t>(::cuda::std::max(__n - 32, 0)));
    const auto __w2 = ::cuda::__szext_clamp_sm70(
      static_cast<uint32_t>(__v >> 64), static_cast<uint32_t>(::cuda::std::max(__n - 64, 0)));
    const auto __w3 = ::cuda::__szext_clamp_sm70(
      static_cast<uint32_t>(__v >> 96), static_cast<uint32_t>(::cuda::std::max(__n - 96, 0)));
    return _Tp{__w0} | (_Tp{__w1} << 32) | (_Tp{__w2} << 64) | (_Tp{__w3} << 96);
  }
#  endif // _CCCL_HAS_INT128()
  else
  {
    return ::cuda::__bit_trunc_impl_generic(__v, __n);
  }
}
#endif // _CCCL_DEVICE_COMPILATION()

//! @brief Retains the least significant \p __n bits of an unsigned integer and clears the remaining bits.
//! @tparam _Tp An unsigned integer type.
//! @param[in] __v The value to truncate.
//! @param[in] __n The number of least significant bits to retain.
//! @pre \p __n is nonnegative.
//! @return The truncated value. Returns zero if \p __n is zero, or \p __v if \p __n is at least the bit width of
//! \p _Tp.
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(::cuda::std::__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]]
_CCCL_API constexpr _Tp bit_trunc(const _Tp __v, const int __n) noexcept
{
  _CCCL_ASSERT(__n >= 0, "__n must be nonnegative");

  // When __n is a constant, use the fallback path.
  if (!_CCCL_IS_CONSTANT_P(__n))
  {
#if !_CCCL_TILE_COMPILATION() // error: asm statement is unsupported in tile code
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      NV_IF_TARGET(NV_IS_HOST, (return ::cuda::__bit_trunc_impl_host(__v, __n);))
      NV_IF_TARGET(NV_PROVIDES_SM_70, (return ::cuda::__bit_trunc_impl_sm70(__v, __n);))
    }
#endif // !_CCCL_TILE_COMPILATION()
  }
  return ::cuda::__bit_trunc_impl_generic(__v, __n);
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___BIT_BIT_TRUNC_H
