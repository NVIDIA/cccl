//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___BIT_BIT_COMPRESS_H
#define _CUDA_STD___BIT_BIT_COMPRESS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__bit/bitmask.h>
#include <cuda/__ptx/instructions/bfind.h>
#include <cuda/std/__bit/bit_reverse.h>
#include <cuda/std/__bit/countl.h>
#include <cuda/std/__bit/popcount.h>
#include <cuda/std/__bit/shl.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/cstdint>

// clang-23 implements __builtin_elementwise_pext which is portable.
// On x86_64, the __builtin_ia32_pext_Xi/_pext_uXX builtins can be used for targets that support BMI2.
#if _CCCL_HAS_BUILTIN(__builtin_elementwise_pext)
#  define _CCCL_BUILTIN_ELEMENTWISE_PEXT(...) __builtin_elementwise_pext(__VA_ARGS__)
#elif _CCCL_HOST_ARCH_FEAT(X86_64, BMI2)
#  if (_CCCL_HAS_BUILTIN(__builtin_ia32_pext_si) && _CCCL_HAS_BUILTIN(__builtin_ia32_pext_di)) || _CCCL_COMPILER(GCC)
#    define _CCCL_BUILTIN_IA32_PEXT_SI(...) __builtin_ia32_pext_si(__VA_ARGS__)
#    define _CCCL_BUILTIN_IA32_PEXT_DI(...) __builtin_ia32_pext_di(__VA_ARGS__)
#  elif _CCCL_COMPILER(MSVC)
#    include <intrin.h>
#    define _CCCL_BUILTIN_IA32_PEXT_SI(...) ::_pext_u32(__VA_ARGS__)
#    define _CCCL_BUILTIN_IA32_PEXT_DI(...) ::_pext_u64(__VA_ARGS__)
#  endif
#endif

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __bit_compress_impl_generic(_Tp __v, const _Tp __mask) noexcept
{
  if (__mask == static_cast<_Tp>(~_Tp{0}))
  {
    return __v;
  }

  // Work on reversed mask, so we can use __builtin_clz.
  auto __mask_rev = ::cuda::std::bit_reverse(__mask);

  _Tp __ret{0};
  int __offset = 0;

  for (auto __skip = ::cuda::std::countl_zero(__mask_rev); __skip != __num_bits_v<_Tp>;
       __skip      = ::cuda::std::countl_zero(__mask_rev))
  {
    // Skip leading zeros in the mask.
    __mask_rev <<= __skip;
    __v >>= __skip;

    // Find out how many consecutive bits we can write.
    const auto __n = ::cuda::std::countl_one(__mask_rev);

    // Write __n consecutive bits.
    const auto __segment = static_cast<_Tp>(__v & ::cuda::bitmask<_Tp>(0, __n));
    __ret                = static_cast<_Tp>(__ret | static_cast<_Tp>(__segment << __offset));
    __offset += __n;

    // Remove written bits from __v and __mask_rev.
    __mask_rev <<= __n;
    __v >>= __n;
  }
  return __ret;
}

#if _CCCL_HOST_COMPILATION()
template <class _Tp>
[[nodiscard]] _CCCL_HOST_API _Tp __bit_compress_impl_host(const _Tp __v, const _Tp __mask) noexcept
{
#  if defined(_CCCL_BUILTIN_ELEMENTWISE_PEXT)
  return _CCCL_BUILTIN_ELEMENTWISE_PEXT(__v, __mask);
#  elif defined(_CCCL_BUILTIN_IA32_PEXT_SI) && defined(_CCCL_BUILTIN_IA32_PEXT_DI)
  if constexpr (sizeof(_Tp) <= sizeof(uint32_t))
  {
    return static_cast<_Tp>(_CCCL_BUILTIN_IA32_PEXT_SI(uint32_t{__v}, uint32_t{__mask}));
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint64_t))
  {
    return _CCCL_BUILTIN_IA32_PEXT_DI(__v, __mask);
  }
#    if _CCCL_HAS_INT128()
  else if constexpr (sizeof(_Tp) == sizeof(__uint128_t))
  {
    const auto __v_lo    = static_cast<uint64_t>(__v);
    const auto __v_hi    = static_cast<uint64_t>(__v >> 64);
    const auto __mask_lo = static_cast<uint64_t>(__mask);
    const auto __mask_hi = static_cast<uint64_t>(__mask >> 64);

    const auto __lower_bits = _CCCL_BUILTIN_IA32_PEXT_DI(__v_lo, __mask_lo);
    const auto __upper_bits = _CCCL_BUILTIN_IA32_PEXT_DI(__v_hi, __mask_hi);
    return (_Tp{__upper_bits} << ::cuda::std::popcount(__mask_lo)) | __lower_bits;
  }
#    endif // _CCCL_HAS_INT128()
  else
  {
    return ::cuda::std::__bit_compress_impl_generic(__v, __mask);
  }
#  else // ^^^ has pext builtin ^^^ / vvv no pext builtin vvv
  return ::cuda::std::__bit_compress_impl_generic(__v, __mask);
#  endif // ^^^ no pext builtin ^^^
}
#endif // _CCCL_HOST_COMPILATION()

#if _CCCL_CUDA_COMPILATION()
template <class _Tp>
_CCCL_DEVICE_API _Tp __bit_compress_impl_device_prepend(_Tp __v, uint32_t __prefix, uint32_t __n) noexcept
{
  if constexpr (sizeof(_Tp) == sizeof(uint32_t))
  {
    return ::__funnelshift_lc(__prefix, __v, __n);
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint64_t))
  {
    const auto __lo = static_cast<uint32_t>(__v);
    const auto __hi = static_cast<uint32_t>(__v >> 32);

    const auto __ret_lo = ::__funnelshift_lc(__prefix, __lo, __n);
    const auto __ret_hi = ::__funnelshift_lc(__lo, __hi, __n);
    return (_Tp{__ret_hi} << 32) | __ret_lo;
  }
#  if _CCCL_HAS_INT128()
  else if constexpr (sizeof(_Tp) == sizeof(__uint128_t))
  {
    const auto __v0 = static_cast<uint32_t>(__v);
    const auto __v1 = static_cast<uint32_t>(__v >> 32);
    const auto __v2 = static_cast<uint32_t>(__v >> 64);
    const auto __v3 = static_cast<uint32_t>(__v >> 96);

    const auto __ret0 = ::__funnelshift_lc(__prefix, __v0, __n);
    const auto __ret1 = ::__funnelshift_lc(__v0, __v1, __n);
    const auto __ret2 = ::__funnelshift_lc(__v1, __v2, __n);
    const auto __ret3 = ::__funnelshift_lc(__v2, __v3, __n);
    return (_Tp{__ret3} << 96) | (_Tp{__ret2} << 64) | (_Tp{__ret1} << 32) | __ret0;
  }
#  endif // _CCCL_HAS_INT128()
  else
  {
    static_assert(__always_false_v<_Tp>, "Unsupported _Tp");
  }
}

template <class _Tp>
_CCCL_DEVICE_API void __bit_compress_impl_device_process_word(_Tp& __ret, uint32_t __v, uint32_t __mask) noexcept
{
  for (auto __skip = ::cuda::ptx::bfind_shiftamt(__mask); __skip != ~0u; __skip = ::cuda::ptx::bfind_shiftamt(__mask))
  {
    // Skip leading zeros in the mask.
    __mask <<= __skip;
    __v <<= __skip;

    // Find out how many consecutive bits we can write.
    const auto __n = ::cuda::ptx::bfind_shiftamt(~__mask);

    // Write __n consecutive bits.
    __ret = ::cuda::std::__bit_compress_impl_device_prepend(__ret, __v, __n);

    // Remove written bits from __v and __mask_rev.
    __mask = ::cuda::std::shl(__mask, __n);
    __v    = ::cuda::std::shl(__v, __n);
  }
}

template <class _Tp>
[[nodiscard]] _CCCL_DEVICE_API _Tp __bit_compress_impl_device(const _Tp __v, const _Tp __mask) noexcept
{
  if constexpr (sizeof(_Tp) <= sizeof(uint32_t))
  {
    uint32_t __ret{0};
    ::cuda::std::__bit_compress_impl_device_process_word(__ret, __v, __mask);
    return static_cast<_Tp>(__ret);
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint64_t))
  {
    const auto __v_lo = static_cast<uint32_t>(__v);
    const auto __v_hi = static_cast<uint32_t>(__v >> 32);
    const auto __m_lo = static_cast<uint32_t>(__mask);
    const auto __m_hi = static_cast<uint32_t>(__mask >> 32);

    _Tp __ret{0};
    ::cuda::std::__bit_compress_impl_device_process_word(__ret, __v_hi, __m_hi);
    ::cuda::std::__bit_compress_impl_device_process_word(__ret, __v_lo, __m_lo);
    return __ret;
  }
#  if _CCCL_HAS_INT128()
  else if constexpr (sizeof(_Tp) == sizeof(__uint128_t))
  {
    const auto __v0 = static_cast<uint32_t>(__v);
    const auto __v1 = static_cast<uint32_t>(__v >> 32);
    const auto __v2 = static_cast<uint32_t>(__v >> 64);
    const auto __v3 = static_cast<uint32_t>(__v >> 96);

    const auto __m0 = static_cast<uint32_t>(__mask);
    const auto __m1 = static_cast<uint32_t>(__mask >> 32);
    const auto __m2 = static_cast<uint32_t>(__mask >> 64);
    const auto __m3 = static_cast<uint32_t>(__mask >> 96);

    _Tp __ret{0};
    ::cuda::std::__bit_compress_impl_device_process_word(__ret, __v3, __m3);
    ::cuda::std::__bit_compress_impl_device_process_word(__ret, __v2, __m2);
    ::cuda::std::__bit_compress_impl_device_process_word(__ret, __v1, __m1);
    ::cuda::std::__bit_compress_impl_device_process_word(__ret, __v0, __m0);
    return __ret;
  }
#  endif // _CCCL_HAS_INT128()
  else
  {
    return ::cuda::std::__bit_compress_impl_generic(__v, __mask);
  }
}
#endif // _CCCL_CUDA_COMPILATION()

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Tp bit_compress(const _Tp __v, const _Tp __mask) noexcept
{
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    NV_IF_ELSE_TARGET(NV_IS_DEVICE, ({ return ::cuda::std::__bit_compress_impl_device(__v, __mask); }), ({
                        return ::cuda::std::__bit_compress_impl_host(__v, __mask);
                      }))
  }
  return ::cuda::std::__bit_compress_impl_generic(__v, __mask);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___BIT_BIT_COMPRESS_H
