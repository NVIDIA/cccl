//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___RANDOM_MWC_ENGINE_H
#define _CUDA___RANDOM_MWC_ENGINE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__random/is_seed_sequence.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>

#if _CCCL_HOSTED()
#  include <istream>
#  include <ostream>
#endif // _CCCL_HOSTED()

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE(cuda::experimental)

// The engine implementation follows Sebastiano Vigna's MWC64X/32 and MWC128X/32:
// https://github.com/vigna/MWCPRNG
template <int _Lag, ::cuda::std::uint32_t _Multiplier>
class __mwc_engine
{
  static_assert(_Lag > 0, "Lag must be positive");

  using __state_t _CCCL_NODEBUG = ::cuda::std::uint32_t;
  using __state _CCCL_NODEBUG   = ::cuda::std::array<__state_t, _Lag + 1>;

  __state __state_{};

  [[nodiscard]] _CCCL_API static constexpr ::cuda::std::uint64_t __splitmix(::cuda::std::uint64_t& __seed) noexcept
  {
    ::cuda::std::uint64_t __value = (__seed += 0x9E3779B97F4A7C15ull);
    __value                       = (__value ^ (__value >> 30)) * 0xBF58476D1CE4E5B9ull;
    __value                       = (__value ^ (__value >> 27)) * 0x94D049BB133111EBull;
    return __value ^ (__value >> 31);
  }

  _CCCL_API constexpr void __map_carry() noexcept
  {
    __state_[_Lag] = 1 + static_cast<result_type>((::cuda::std::uint64_t{__state_[_Lag]} * (_Multiplier - 2)) >> 32);
  }

  // (consteval) Compute modulo m = a * 2^(32*lag) - 1 without native 128-bit types
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL __state __modulus() noexcept
  {
    __state __mod{};
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Lag; ++__i)
    {
      __mod[__i] = ::cuda::std::numeric_limits<result_type>::max();
    }
    __mod[_Lag] = _Multiplier - 1;
    return __mod;
  }

  [[nodiscard]] _CCCL_API static constexpr bool __less(const __state& __lhs, const __state& __rhs) noexcept
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = _Lag; __i >= 0; --__i)
    {
      if (__lhs[__i] < __rhs[__i])
      {
        return true;
      }
      if (__lhs[__i] > __rhs[__i])
      {
        return false;
      }
    }
    return false;
  }

  // Return lhs * rhs / R modulo m, where b = 2^32 and R = b^(lag + 1).
  [[nodiscard]] _CCCL_API static constexpr __state
  __montgomery_multiply(const __state& __lhs, const __state& __rhs) noexcept
  {
    constexpr int __words = _Lag + 1;
    constexpr int __bits  = ::cuda::std::numeric_limits<__state_t>::digits;
    __state_t __product[2 * __words + 1]{};

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < __words; ++__i)
    {
      ::cuda::std::uint64_t __carry = 0;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int __j = 0; __j < __words; ++__j)
      {
        const auto __sum     = (::cuda::std::uint64_t{__lhs[__i]} * __rhs[__j]) + __product[__i + __j] + __carry;
        __product[__i + __j] = static_cast<__state_t>(__sum);
        __carry              = __sum >> __bits;
      }
      __product[__i + __words] = static_cast<__state_t>(__carry);
    }

    // Since m = a*b^lag - 1, adding u*m cancels the low word u
    // Skip that word and add u*a at the lag offset. After all rounds the low half is discarded
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < __words; ++__i)
    {
      const auto __sum      = (::cuda::std::uint64_t{__product[__i]} * _Multiplier) + __product[__i + _Lag];
      __product[__i + _Lag] = static_cast<__state_t>(__sum);
      auto __carry          = __sum >> __bits;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int __j = __i + __words; __j <= 2 * __words; ++__j)
      {
        const auto __word = __product[__j] + __carry;
        __product[__j]    = static_cast<__state_t>(__word);
        __carry           = __word >> __bits;
      }
    }

    // The reduced product is below 2*m, so at most one subtraction is needed
    __state __result{};
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < __words; ++__i)
    {
      __result[__i] = __product[__words + __i];
    }
    constexpr auto __mod = __modulus();
    if (__product[2 * __words] != 0 || !__less(__result, __mod))
    {
      bool __borrow = false;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int __i = 0; __i < __words; ++__i)
      {
        const auto __sub = ::cuda::std::uint64_t{__mod[__i]} + __borrow;
        __borrow         = __result[__i] < __sub;
        __result[__i]    = static_cast<__state_t>(__result[__i] - __sub);
      }
    }
    return __result;
  }

public:
  //! @brief unsigned 32-bit output type
  using result_type _CCCL_NODEBUG = __state_t;

  //! @brief The default seed
  static constexpr ::cuda::std::uint64_t default_seed = 1;

  //! @brief Returns the smallest possible output
  [[nodiscard]] _CCCL_API static constexpr result_type min() noexcept
  {
    return 0;
  }

  //! @brief Returns the largest possible output
  [[nodiscard]] _CCCL_API static constexpr result_type max() noexcept
  {
    return ::cuda::std::numeric_limits<result_type>::max();
  }

  //! @brief Initializes the engine with @c default_seed
  _CCCL_API constexpr __mwc_engine() noexcept
      : __mwc_engine(default_seed)
  {}

  //! @brief Initializes the whole state with @p __seed
  _CCCL_API constexpr explicit __mwc_engine(const ::cuda::std::uint64_t __seed) noexcept
  {
    seed(__seed);
  }

  //! @brief Initializes the state from a seed sequence
  _CCCL_TEMPLATE(class _Sseq)
  _CCCL_REQUIRES(::cuda::std::__is_seed_sequence<_Sseq, __mwc_engine>)
  _CCCL_API constexpr explicit __mwc_engine(_Sseq& __seq)
  {
    seed(__seq);
  }

  //! @brief Seeds the state with SplitMix64, mapping the carry to [1, multiplier - 2]
  _CCCL_API constexpr void seed(::cuda::std::uint64_t __seed = default_seed) noexcept
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i <= _Lag; ++__i)
    {
      __state_[__i] = static_cast<result_type>(__splitmix(__seed));
    }
    __map_carry();
  }

  //! @brief Generates lag + 1 words from @p __seq and maps the last word to a valid carry
  _CCCL_TEMPLATE(class _Sseq)
  _CCCL_REQUIRES(::cuda::std::__is_seed_sequence<_Sseq, __mwc_engine>)
  _CCCL_API constexpr void seed(_Sseq& __seq)
  {
    __state __words{};
    __seq.generate(__words.begin(), __words.end());
    __state_ = __words;
    __map_carry();
  }

  //! @brief Returns the most recent XOR result, then advances the state
  _CCCL_API constexpr result_type operator()() noexcept
  {
    const result_type __result         = __state_[_Lag - 1] ^ __state_[_Lag];
    const ::cuda::std::uint64_t __next = ::cuda::std::uint64_t{_Multiplier} * __state_[0] + __state_[_Lag];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i < _Lag - 1; ++__i)
    {
      __state_[__i] = __state_[__i + 1];
    }
    __state_[_Lag - 1] = static_cast<result_type>(__next);
    __state_[_Lag]     = static_cast<result_type>(__next >> 32);
    return __result;
  }

  //! @brief Advances by @p __count outputs in O(log(__count)) time
  _CCCL_API constexpr void discard(unsigned long long __count) noexcept
  {
    // Direct stepping is cheaper for short skips, especially for the wider Montgomery product.
    constexpr unsigned __jump_threshold = _Lag == 1 ? 64u : 256u;
    if (__count < __jump_threshold)
    {
      while (__count--)
      {
        static_cast<void>((*this)());
      }
      return;
    }

    // Keep the state in ordinary form and the jump powers in Montgomery form.
    // The first power represents one transition: b^-1 * R = b^lag.
    __state __power{};
    __power[_Lag] = 1;
    while (__count)
    {
      if (__count & 1)
      {
        __state_ = __montgomery_multiply(__state_, __power);
      }
      __count >>= 1;
      if (__count)
      {
        __power = __montgomery_multiply(__power, __power);
      }
    }
  }

  //! @brief Compares the engine states
  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const __mwc_engine& __lhs, const __mwc_engine& __rhs) noexcept
  {
    return __lhs.__state_ == __rhs.__state_;
  }

#if _CCCL_STD_VER <= 2017
  //! @brief Returns whether the engine states differ
  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const __mwc_engine& __lhs, const __mwc_engine& __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }
#endif // _CCCL_STD_VER <= 2017

#if _CCCL_HOSTED()
  //! @brief Writes the digits, least significant first, followed by the carry in decimal
  template <class _CharT, class _Traits>
  _CCCL_HOST_API friend ::std::basic_ostream<_CharT, _Traits>&
  operator<<(::std::basic_ostream<_CharT, _Traits>& __os, const __mwc_engine& __engine)
  {
    using __ios _CCCL_NODEBUG = ::std::ios_base;
    const auto __flags        = __os.flags();
    const auto __fill         = __os.fill();
    __os.flags(__ios::dec | __ios::fixed | __ios::left);
    __os.fill(__os.widen(' '));
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i <= _Lag; ++__i)
    {
      if (__i)
      {
        __os << __os.widen(' ');
      }
      __os << __engine.__state_[__i];
    }
    __os.flags(__flags);
    __os.fill(__fill);
    return __os;
  }

  //! @brief Restores a state. Invalid input sets failbit and leaves the engine unchanged.
  template <class _CharT, class _Traits>
  _CCCL_HOST_API friend ::std::basic_istream<_CharT, _Traits>&
  operator>>(::std::basic_istream<_CharT, _Traits>& __is, __mwc_engine& __engine)
  {
    using __ios _CCCL_NODEBUG = ::std::ios_base;
    const auto __flags        = __is.flags();
    __is.flags(__ios::dec | __ios::skipws);
    __state __words{};
    bool __nonzero = false;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __i = 0; __i <= _Lag; ++__i)
    {
      __is >> __words[__i];
      __nonzero = __nonzero || __words[__i] != 0;
    }
    __is.flags(__flags);
    if (!__is.fail())
    {
      if (__nonzero && __less(__words, __modulus()))
      {
        __engine.__state_ = __words;
      }
      else
      {
        __is.setstate(__ios::failbit);
      }
    }
    return __is;
  }
#endif // _CCCL_HOSTED()
};

//! @brief Experimental scrambled multiply-with-carry engine with 64-bit state and 32-bit output
//! The period is approximately 2^63. Prefer discard-based stream partitioning
using mwc64x32 = __mwc_engine<1, 0xFFF145F8u>;

//! @brief Experimental scrambled multiply-with-carry engine with 128-bit state and 32-bit output
//! The period is approximately 2^127
using mwc128x32 = __mwc_engine<3, 0xFFCCCC66u>;
_CCCL_END_NAMESPACE(cuda::experimental)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___RANDOM_MWC_ENGINE_H
