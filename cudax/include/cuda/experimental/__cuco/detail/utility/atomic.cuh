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

#include <cuda/__memory/is_aligned.h>
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

//! Relaxed atomics on top of the CUDA atomic intrinsics.
//!
//! Until CTK 13.5, `cuda::atomic_ref` lowers to volatile inline PTX that the compiler cannot see through, e.g.
//! a result-producing `ATOM` instead of `RED` when the result is unused (NVIDIA/cccl#11246). These helpers restore
//! the codegen of the intrinsics and can be replaced by `cuda::atomic_ref` once CTK 13.5 is the oldest supported
//! toolkit (NVIDIA/cccl#9983).
//!
//! There are no cluster-scope intrinsics, so `thread_scope_cluster` uses the device-scope ones, which are stronger.
//! `__atomic_load` and `__atomic_store` are volatile accesses, which the PTX memory model treats as relaxed
//! system-scope operations.

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

template <class _Tp>
_CCCL_DEVICE_API constexpr void __validate_atomic_type() noexcept
{
  static_assert(::cuda::std::is_trivially_copyable_v<_Tp>, "cuCO atomics require a trivially copyable type");
  static_assert(sizeof(_Tp) == 1 || sizeof(_Tp) == 2 || sizeof(_Tp) == 4 || sizeof(_Tp) == 8,
                "cuCO atomics require a 1, 2, 4, or 8 byte type");
}

template <::cuda::thread_scope _Scope, class _Word>
[[nodiscard]] _CCCL_DEVICE_API _Word __atomic_cas_word(_Word* __address, _Word __compare, _Word __value) noexcept
{
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
  else if constexpr (_Scope == ::cuda::thread_scope_cluster || _Scope == ::cuda::thread_scope_device)
  {
    return ::atomicCAS(__address, __compare, __value);
  }
  else
  {
    static_assert(_Scope == ::cuda::thread_scope_system, "Unsupported thread scope");
    return ::atomicCAS_system(__address, __compare, __value);
  }
}

template <class _Tp>
[[nodiscard]] _CCCL_DEVICE_API _Tp __atomic_load(const _Tp* __address) noexcept
{
  __validate_atomic_type<_Tp>();

  using __word_type        = __atomic_word_t<_Tp>;
  const __word_type __bits = *reinterpret_cast<const volatile __word_type*>(__address);
  return ::cuda::std::bit_cast<_Tp>(__bits);
}

template <class _Word>
[[nodiscard]] _CCCL_DEVICE_API _Word __weak_load_word(const _Word* __address) noexcept
{
  static_assert(sizeof(_Word) == 4 || sizeof(_Word) == 8, "A weak load requires a 32-bit or 64-bit word");

  // The memory clobber keeps the compiler from narrowing, merging, or dropping the load. It is not volatile, which
  // would keep the compiler from optimizing the code around it, so `ptxas` may hoist the load out of a loop that has
  // no atomic operation.
  _Word __word;
  if constexpr (sizeof(_Word) == 4)
  {
    asm("ld.b32 %0,[%1];" : "=r"(__word) : "l"(__address) : "memory");
  }
  else
  {
    asm("ld.b64 %0,[%1];" : "=l"(__word) : "l"(__address) : "memory");
  }
  return __word;
}

//! @brief Reads the word that `__atomic_compare_exchange` swaps for the object at `__address` while other threads may
//! be swapping it.
//!
//! A plain load would let the compiler narrow, widen, merge, or drop the read. This is one weak PTX load of exactly the
//! swapped word instead. `__atomic_load` is not used because a strong load is several times slower when many threads
//! probe the same slots.
//!
//! The load still races with the swap, so the value may already be stale when the caller sees it. Re-read it only
//! after an atomic operation: a loop that only waits on this load may never observe a change, so wait with
//! `__atomic_load` instead.
//!
//! @tparam _Tp Type of the object, 1, 2, 4, or 8 bytes in size
//!
//! @param __address Address of the object
//!
//! @return The object held by the swapped word
template <class _Tp>
[[nodiscard]] _CCCL_DEVICE_API _Tp __weak_load(const _Tp* __address) noexcept
{
  static_assert(sizeof(_Tp) == 1 || sizeof(_Tp) == 2 || sizeof(_Tp) == 4 || sizeof(_Tp) == 8,
                "cuCO atomics require a 1, 2, 4, or 8 byte type");

  using __word_type = __atomic_word_t<_Tp>;
  if constexpr (sizeof(_Tp) <= 2)
  {
    _CCCL_ASSERT(::cuda::is_aligned(__address, sizeof(_Tp)), "cuCO atomics require a naturally aligned address");
    constexpr ::cuda::std::uintptr_t __align_mask = sizeof(unsigned int) - 1;
    const auto __address_value                    = reinterpret_cast<::cuda::std::uintptr_t>(__address);
    const auto* const __aligned = reinterpret_cast<const unsigned int*>(__address_value & ~__align_mask);
    const auto __offset         = static_cast<unsigned int>((__address_value & __align_mask) * 8);
    return ::cuda::std::bit_cast<_Tp>(static_cast<__word_type>(__weak_load_word(__aligned) >> __offset));
  }
  else
  {
    return ::cuda::std::bit_cast<_Tp>(__weak_load_word(reinterpret_cast<const __word_type*>(__address)));
  }
}

template <class _Tp>
_CCCL_DEVICE_API void __atomic_store(_Tp* __address, _Tp __value) noexcept
{
  __validate_atomic_type<_Tp>();

  using __word_type                                   = __atomic_word_t<_Tp>;
  *reinterpret_cast<volatile __word_type*>(__address) = ::cuda::std::bit_cast<__word_type>(__value);
}

template <::cuda::thread_scope _Scope, class _Tp>
[[nodiscard]] _CCCL_DEVICE_API bool __atomic_compare_exchange(_Tp* __address, _Tp& __expected, _Tp __desired) noexcept
{
  __validate_atomic_type<_Tp>();

  using __word_type           = __atomic_word_t<_Tp>;
  const __word_type __compare = ::cuda::std::bit_cast<__word_type>(__expected);
  const __word_type __value   = ::cuda::std::bit_cast<__word_type>(__desired);

  if constexpr (sizeof(_Tp) <= 2)
  {
    // Swap the enclosing aligned 32-bit word as `cuda::atomic_ref` does: there is no 8-bit atomicCAS and the 16-bit one
    // has no block or system scope variant. The other bytes are compared and written back unchanged, so a concurrent
    // update to them fails the swap instead of being lost, but the whole word must be addressable.
    _CCCL_ASSERT(::cuda::is_aligned(__address, sizeof(_Tp)), "cuCO atomics require a naturally aligned address");
    constexpr ::cuda::std::uintptr_t __align_mask = sizeof(unsigned int) - 1;
    constexpr unsigned int __size_mask            = (1u << (sizeof(_Tp) * 8)) - 1;
    const auto __address_value                    = reinterpret_cast<::cuda::std::uintptr_t>(__address);
    auto* const __aligned                         = reinterpret_cast<unsigned int*>(__address_value & ~__align_mask);
    const auto __offset                           = static_cast<unsigned int>((__address_value & __align_mask) * 8);
    const auto __value_mask                       = __size_mask << __offset;
    const auto __compare_bits                     = static_cast<unsigned int>(__compare) << __offset;
    const auto __value_bits                       = static_cast<unsigned int>(__value) << __offset;

    auto __old = __atomic_load(__aligned);
    while ((__old & __value_mask) == __compare_bits)
    {
      const auto __result = __atomic_cas_word<_Scope>(__aligned, __old, (__old & ~__value_mask) | __value_bits);
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
    const auto __old     = __atomic_cas_word<_Scope>(reinterpret_cast<__word_type*>(__address), __compare, __value);
    const bool __success = __old == __compare;
    if (!__success)
    {
      __expected = ::cuda::std::bit_cast<_Tp>(__old);
    }
    return __success;
  }
}

template <::cuda::thread_scope _Scope, class _Tp>
[[nodiscard]] _CCCL_DEVICE_API _Tp __atomic_fetch_add(_Tp* __address, _Tp __value) noexcept
{
  static_assert(::cuda::std::is_integral_v<_Tp> && !::cuda::std::is_same_v<_Tp, bool>,
                "cuCO atomic add requires a non-bool integral type");
  static_assert(sizeof(_Tp) == 4 || sizeof(_Tp) == 8, "cuCO atomic add requires a 4 or 8 byte type");

  using __word_type          = __atomic_word_t<_Tp>;
  auto* const __word_address = reinterpret_cast<__word_type*>(__address);
  const auto __addend        = static_cast<__word_type>(__value);
  if constexpr (_Scope == ::cuda::thread_scope_thread)
  {
    const auto __old = *__word_address;
    *__word_address  = __old + __addend;
    return static_cast<_Tp>(__old);
  }
  else if constexpr (_Scope == ::cuda::thread_scope_block)
  {
    return static_cast<_Tp>(::atomicAdd_block(__word_address, __addend));
  }
  else if constexpr (_Scope == ::cuda::thread_scope_cluster || _Scope == ::cuda::thread_scope_device)
  {
    return static_cast<_Tp>(::atomicAdd(__word_address, __addend));
  }
  else
  {
    static_assert(_Scope == ::cuda::thread_scope_system, "Unsupported thread scope");
    return static_cast<_Tp>(::atomicAdd_system(__word_address, __addend));
  }
}

template <::cuda::thread_scope _Scope, class _Tp>
[[nodiscard]] _CCCL_DEVICE_API _Tp __atomic_fetch_max(_Tp* __address, _Tp __value) noexcept
{
  static_assert(::cuda::std::is_integral_v<_Tp> && !::cuda::std::is_same_v<_Tp, bool>,
                "cuCO atomic max requires a non-bool integral type");
  static_assert(sizeof(_Tp) == 4 || sizeof(_Tp) == 8, "cuCO atomic max requires a 4 or 8 byte type");

  using __native_type =
    ::cuda::std::conditional_t<::cuda::std::is_signed_v<_Tp>,
                               ::cuda::std::conditional_t<sizeof(_Tp) == 4, int, long long>,
                               __atomic_word_t<_Tp>>;
  auto* const __native_address = reinterpret_cast<__native_type*>(__address);
  const auto __native_value    = static_cast<__native_type>(__value);
  if constexpr (_Scope == ::cuda::thread_scope_thread)
  {
    const auto __old = *__native_address;
    if (__old < __native_value)
    {
      *__native_address = __native_value;
    }
    return static_cast<_Tp>(__old);
  }
  else if constexpr (_Scope == ::cuda::thread_scope_block)
  {
    return static_cast<_Tp>(::atomicMax_block(__native_address, __native_value));
  }
  else if constexpr (_Scope == ::cuda::thread_scope_cluster || _Scope == ::cuda::thread_scope_device)
  {
    return static_cast<_Tp>(::atomicMax(__native_address, __native_value));
  }
  else
  {
    static_assert(_Scope == ::cuda::thread_scope_system, "Unsupported thread scope");
    return static_cast<_Tp>(::atomicMax_system(__native_address, __native_value));
  }
}
#endif // _CCCL_CUDA_COMPILATION()
} // namespace cuda::experimental::cuco::detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_DETAIL_UTILITY_ATOMIC_CUH
