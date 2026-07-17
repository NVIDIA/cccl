//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_DETAIL_UTILITY_ATOMIC_CUH
#define _CUDAX___CUCO_DETAIL_UTILITY_ATOMIC_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::detail
{
#if _CCCL_CUDA_COMPILATION()
template <::cuda::std::size_t _Size>
struct __atomic_word;

template <>
struct __atomic_word<1>
{
  using __type = ::cuda::std::uint8_t;
};

template <>
struct __atomic_word<2>
{
  using __type = ::cuda::std::uint16_t;
};

template <>
struct __atomic_word<4>
{
  using __type = unsigned int;
};

template <>
struct __atomic_word<8>
{
  using __type = unsigned long long;
};

template <class _Tp>
using __atomic_word_t = typename __atomic_word<sizeof(_Tp)>::__type;

template <::cuda::thread_scope _Scope>
inline constexpr bool __is_cuda_atomic_scope =
  _Scope == ::cuda::thread_scope_thread || _Scope == ::cuda::thread_scope_block || _Scope == ::cuda::thread_scope_device
  || _Scope == ::cuda::thread_scope_system;

template <::cuda::thread_scope _Scope, class _Word>
[[nodiscard]] _CCCL_DEVICE_API _Word __atomic_cas_word(_Word* __address, _Word __compare, _Word __value) noexcept
{
  static_assert(__is_cuda_atomic_scope<_Scope>, "cuCO atomics support thread, block, device, and system thread scopes");
  static_assert(sizeof(_Word) == 4 || sizeof(_Word) == 8, "CUDA atomicCAS requires a 32-bit or 64-bit word");

  if constexpr (_Scope == ::cuda::thread_scope_thread)
  {
    const _Word __old = *__address;
    if (__old == __compare)
    {
      *__address = __value;
    }
    return __old;
  }
  else if constexpr (_Scope == ::cuda::thread_scope_block)
  {
    return ::atomicCAS_block(__address, __compare, __value);
  }
  else if constexpr (_Scope == ::cuda::thread_scope_device)
  {
    return ::atomicCAS(__address, __compare, __value);
  }
  else
  {
    return ::atomicCAS_system(__address, __compare, __value);
  }
}

template <::cuda::thread_scope _Scope, class _Word>
[[nodiscard]] _CCCL_DEVICE_API _Word __atomic_exchange_word(_Word* __address, _Word __value) noexcept
{
  static_assert(__is_cuda_atomic_scope<_Scope>, "cuCO atomics support thread, block, device, and system thread scopes");
  static_assert(sizeof(_Word) == 4 || sizeof(_Word) == 8, "CUDA atomicExch requires a 32-bit or 64-bit word");

  if constexpr (_Scope == ::cuda::thread_scope_thread)
  {
    const _Word __old = *__address;
    *__address        = __value;
    return __old;
  }
  else if constexpr (_Scope == ::cuda::thread_scope_block)
  {
    return ::atomicExch_block(__address, __value);
  }
  else if constexpr (_Scope == ::cuda::thread_scope_device)
  {
    return ::atomicExch(__address, __value);
  }
  else
  {
    return ::atomicExch_system(__address, __value);
  }
}

template <class _Tp>
_CCCL_DEVICE_API constexpr void __validate_atomic_type() noexcept
{
  static_assert(::cuda::std::is_trivially_copyable_v<_Tp>, "cuCO atomics require a trivially copyable type");
  static_assert(sizeof(_Tp) == 1 || sizeof(_Tp) == 2 || sizeof(_Tp) == 4 || sizeof(_Tp) == 8,
                "cuCO atomics require a 1, 2, 4, or 8 byte type");
}

template <::cuda::thread_scope _Scope, class _Tp>
[[nodiscard]] _CCCL_DEVICE_API bool __atomic_compare_exchange(_Tp* __address, _Tp& __expected, _Tp __desired) noexcept
{
  ::cuda::experimental::cuco::detail::__validate_atomic_type<_Tp>();

  using __word_type           = __atomic_word_t<_Tp>;
  const __word_type __compare = ::cuda::std::bit_cast<__word_type>(__expected);
  const __word_type __value   = ::cuda::std::bit_cast<__word_type>(__desired);

  if constexpr (sizeof(_Tp) <= 2)
  {
    constexpr ::cuda::std::uintptr_t __align_mask = sizeof(unsigned int) - 1;
    constexpr unsigned int __size_mask            = (1u << (sizeof(_Tp) * 8)) - 1;
    const auto __address_value                    = reinterpret_cast<::cuda::std::uintptr_t>(__address);
    auto* const __aligned                         = reinterpret_cast<unsigned int*>(__address_value & ~__align_mask);
    const auto __offset                           = static_cast<unsigned int>((__address_value & __align_mask) * 8);
    const auto __value_mask                       = __size_mask << __offset;
    const auto __window_mask                      = ~__value_mask;
    const auto __compare_bits                     = static_cast<unsigned int>(__compare) << __offset;
    const auto __value_bits                       = static_cast<unsigned int>(__value) << __offset;

    auto __old = ::cuda::experimental::cuco::detail::__atomic_cas_word<_Scope>(__aligned, 0u, 0u);
    while ((__old & __value_mask) == __compare_bits)
    {
      const auto __attempt = (__old & __window_mask) | __value_bits;
      const auto __result  = ::cuda::experimental::cuco::detail::__atomic_cas_word<_Scope>(__aligned, __old, __attempt);
      if (__result == __old)
      {
        return true;
      }
      __old = __result;
    }
    __expected = ::cuda::std::bit_cast<_Tp>(static_cast<__word_type>((__old & __value_mask) >> __offset));
    return false;
  }
  else
  {
    auto* const __word_address = reinterpret_cast<__word_type*>(__address);
    const auto __old =
      ::cuda::experimental::cuco::detail::__atomic_cas_word<_Scope>(__word_address, __compare, __value);
    const bool __success = __old == __compare;
    if (!__success)
    {
      __expected = ::cuda::std::bit_cast<_Tp>(__old);
    }
    return __success;
  }
}

template <::cuda::thread_scope _Scope, class _Tp>
[[nodiscard]] _CCCL_DEVICE_API _Tp __atomic_load(const _Tp* __address) noexcept
{
  ::cuda::experimental::cuco::detail::__validate_atomic_type<_Tp>();
  static_assert(sizeof(_Tp) == 4 || sizeof(_Tp) == 8, "cuCO atomic load requires a 4 or 8 byte type");

  using __word_type            = __atomic_word_t<_Tp>;
  auto* const __word_address   = reinterpret_cast<__word_type*>(const_cast<_Tp*>(__address));
  constexpr __word_type __zero = 0;
  const auto __old = ::cuda::experimental::cuco::detail::__atomic_cas_word<_Scope>(__word_address, __zero, __zero);
  return ::cuda::std::bit_cast<_Tp>(__old);
}

template <::cuda::thread_scope _Scope, class _Tp>
_CCCL_DEVICE_API void __atomic_store(_Tp* __address, _Tp __value) noexcept
{
  ::cuda::experimental::cuco::detail::__validate_atomic_type<_Tp>();
  static_assert(sizeof(_Tp) == 4 || sizeof(_Tp) == 8, "cuCO atomic store requires a 4 or 8 byte type");

  using __word_type           = __atomic_word_t<_Tp>;
  auto* const __word_address  = reinterpret_cast<__word_type*>(__address);
  const __word_type __desired = ::cuda::std::bit_cast<__word_type>(__value);
  (void) ::cuda::experimental::cuco::detail::__atomic_exchange_word<_Scope>(__word_address, __desired);
}

template <::cuda::thread_scope _Scope, class _Tp>
[[nodiscard]] _CCCL_DEVICE_API _Tp __atomic_fetch_add(_Tp* __address, _Tp __value) noexcept
{
  ::cuda::experimental::cuco::detail::__validate_atomic_type<_Tp>();
  static_assert(::cuda::std::is_integral_v<_Tp> && !::cuda::std::is_same_v<_Tp, bool>,
                "cuCO atomic add requires a non-bool integral type");
  static_assert(sizeof(_Tp) == 4 || sizeof(_Tp) == 8, "cuCO atomic add requires a 4 or 8 byte type");

  using __word_type          = __atomic_word_t<_Tp>;
  auto* const __word_address = reinterpret_cast<__word_type*>(__address);
  const auto __addend        = ::cuda::std::bit_cast<__word_type>(__value);
  __word_type __old;
  if constexpr (_Scope == ::cuda::thread_scope_thread)
  {
    __old           = *__word_address;
    *__word_address = __old + __addend;
  }
  else if constexpr (_Scope == ::cuda::thread_scope_block)
  {
    __old = ::atomicAdd_block(__word_address, __addend);
  }
  else if constexpr (_Scope == ::cuda::thread_scope_device)
  {
    __old = ::atomicAdd(__word_address, __addend);
  }
  else if constexpr (_Scope == ::cuda::thread_scope_system)
  {
    __old = ::atomicAdd_system(__word_address, __addend);
  }
  else
  {
    static_assert(__is_cuda_atomic_scope<_Scope>,
                  "cuCO atomics support thread, block, device, and system thread scopes");
  }
  return ::cuda::std::bit_cast<_Tp>(__old);
}

template <::cuda::thread_scope _Scope, class _Tp>
[[nodiscard]] _CCCL_DEVICE_API _Tp __atomic_fetch_max(_Tp* __address, _Tp __value) noexcept
{
  ::cuda::experimental::cuco::detail::__validate_atomic_type<_Tp>();
  static_assert(::cuda::std::is_integral_v<_Tp> && !::cuda::std::is_same_v<_Tp, bool>,
                "cuCO atomic max requires a non-bool integral type");
  static_assert(sizeof(_Tp) == 4 || sizeof(_Tp) == 8, "cuCO atomic max requires a 4 or 8 byte type");

  using __native_type =
    ::cuda::std::conditional_t<sizeof(_Tp) == 4,
                               ::cuda::std::conditional_t<::cuda::std::is_signed_v<_Tp>, int, unsigned int>,
                               ::cuda::std::conditional_t<::cuda::std::is_signed_v<_Tp>, long long, unsigned long long>>;
  auto* const __native_address = reinterpret_cast<__native_type*>(__address);
  const auto __native_value    = ::cuda::std::bit_cast<__native_type>(__value);
  __native_type __old;
  if constexpr (_Scope == ::cuda::thread_scope_thread)
  {
    __old = *__native_address;
    if (__old < __native_value)
    {
      *__native_address = __native_value;
    }
  }
  else if constexpr (_Scope == ::cuda::thread_scope_block)
  {
    __old = ::atomicMax_block(__native_address, __native_value);
  }
  else if constexpr (_Scope == ::cuda::thread_scope_device)
  {
    __old = ::atomicMax(__native_address, __native_value);
  }
  else if constexpr (_Scope == ::cuda::thread_scope_system)
  {
    __old = ::atomicMax_system(__native_address, __native_value);
  }
  else
  {
    static_assert(__is_cuda_atomic_scope<_Scope>,
                  "cuCO atomics support thread, block, device, and system thread scopes");
  }
  return ::cuda::std::bit_cast<_Tp>(__old);
}
#endif // _CCCL_CUDA_COMPILATION()
} // namespace cuda::experimental::cuco::detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_DETAIL_UTILITY_ATOMIC_CUH
