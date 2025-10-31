//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___RANDOM_PHILOX_ENGINE_H
#define _CUDA_STD___RANDOM_PHILOX_ENGINE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/mul_hi.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__random/is_seed_sequence.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/array>
#include <cuda/std/cstddef> // for size_t
#include <cuda/std/cstdint>

#if !_CCCL_COMPILER(NVRTC)
#  include <ios>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

//! @class philox_engine
//! @brief A philox_engine random number engine produces unsigned integer
//!        random numbers using a Philox counter-based random number generation algorithm
//!        as described in: Salmon, John K., et al. "Parallel random numbers: as easy as 1, 2, 3." Proceedings of 2011
//!        international conference for high performance computing, networking, storage and analysis. 2011.
//!
//!
//! @tparam _UIntType The type of unsigned integer to produce.
//! @tparam _WordSize The word size
//! @tparam _WordCount The buffer size
//! @tparam _NumRounds The number of rounds
//! @tparam _Constants The constants used in the generation algorithm.
//!
//! @note Users should not use this class template directly.  Instead, use
//!       philox4x32 or philox4x64 .
//!
//! The following code snippet shows examples of use of a philox_engine instance:
//!
//! .. code-block:: c++
//!
//!    #include <cuda/std/random>
//!    #include <iostream>
//!
//!    int main()
//!    {
//!      // create a philox4x64 object, which is an instance of philox_engine
//!      cuda::std::philox4x64 rng1;
//!      cuda::std::philox4x64 rng2;
//!      // Create two different streams of random numbers
//!      // The counter is set as a big integer with the least significant word last.
//!      // Each counter increment produces 4 new values
//!      rng1.set_counter({0, 0, 0, 0});
//!      rng2.set_counter({0, 0, 1, 0}); // rng2 is now 4*2^w values ahead of rng1
//!
//!      // Relation between discard and set_counter
//!      cuda::std::philox4x64 rng3;
//!      rng3.set_counter({0, 0, 0, 100});
//!      const int n = 4;
//!      rng1.discard(100*n); // rng1 is now at the same position as rng3
//!      std::cout << (rng1() == rng3()) << std::endl; // 1
//!
//!      return 0;
//!    }
//!
//!
//! @see cuda::std::philox4x32
//! @see cuda::std::philox4x64
template <typename _UIntType,
          ::cuda::std::size_t _WordSize,
          ::cuda::std::size_t _WordCount,
          ::cuda::std::size_t _NumRounds,
          _UIntType... _Constants>
class philox_engine
{
  static_assert(::cuda::std::__cccl_is_unsigned_integer_v<_UIntType>,
                "philox_engine: _UIntType must be an unsigned integer type");
  static_assert(_WordCount == 2 || _WordCount == 4, "N argument must be either 2 or 4");
  static_assert(sizeof...(_Constants) == _WordCount, "consts array must be of length N");
  static_assert(_NumRounds > 0, "rounds must be a strictly positive number");
  static_assert((0 < _WordSize && _WordSize <= ::cuda::std::numeric_limits<_UIntType>::digits),
                "Word size w must satisfy 0 < w <= numeric_limits<_UIntType>::digits");

  [[nodiscard]] _CCCL_API static constexpr auto __multipliers() noexcept
  {
    constexpr _UIntType __constants[] = {_Constants...};
    if constexpr (_WordCount == 2)
    {
      return ::cuda::std::array<_UIntType, 1>{__constants[0]};
    }
    else
    {
      return ::cuda::std::array<_UIntType, 2>{__constants[0], __constants[2]};
    }
  }

  [[nodiscard]] _CCCL_API static constexpr auto __round_consts() noexcept
  {
    constexpr _UIntType __constants[] = {_Constants...};
    if constexpr (_WordCount == 2)
    {
      return ::cuda::std::array<_UIntType, 1>{__constants[1]};
    }
    else
    {
      return ::cuda::std::array<_UIntType, 2>{__constants[1], __constants[3]};
    }
  }

public:
  using result_type                                = _UIntType;
  static constexpr ::cuda::std::size_t word_size   = _WordSize;
  static constexpr ::cuda::std::size_t word_count  = _WordCount;
  static constexpr ::cuda::std::size_t round_count = _NumRounds;
  static constexpr auto multipliers                = __multipliers();
  static constexpr auto round_consts               = __round_consts();
  static constexpr result_type default_seed        = 20111115u;

  //! The smallest value this engine may potentially produce.
  [[nodiscard]] _CCCL_API static constexpr result_type min() noexcept
  {
    return 0;
  }
  //! The largest value this engine may potentially produce.
  [[nodiscard]] _CCCL_API static constexpr result_type max() noexcept
  {
    return ((result_type{1} << (word_size - 1)) | ((result_type{1} << (word_size - 1)) - 1));
  }

  //! This constructor, which optionally accepts a seed, initializes a new
  //! philox_engine.
  //!
  //! @param s The seed used to initialize this philox_engine's state.
  _CCCL_API constexpr philox_engine() noexcept
  {
    seed(default_seed);
  }
  _CCCL_API constexpr explicit philox_engine(const result_type __seed) noexcept
  {
    seed(__seed);
  }

  _CCCL_TEMPLATE(class _Sseq)
  _CCCL_REQUIRES(__is_seed_sequence<_Sseq, philox_engine>)
  _CCCL_API constexpr explicit philox_engine(_Sseq& __seq)
  {
    seed(__seq);
  }

  //! This method initializes this philox_engine's state, and optionally accepts
  //! a seed value.  If none is provided uses ``default_seed``.
  //!
  //! @param __s The seed used to initializes this philox_engine's state.
  _CCCL_API constexpr void seed(result_type __s = default_seed) noexcept
  {
    __x_    = {};
    __y_    = {};
    __k_    = {};
    __k_[0] = __s & max();
    __j_    = word_count - 1;
  }

  // Prevent this overload if Sseq is convertible to result_type
  _CCCL_TEMPLATE(class _Sseq)
  _CCCL_REQUIRES(__is_seed_sequence<_Sseq, philox_engine>)
  _CCCL_API constexpr void seed(_Sseq& __seq)
  {
    __x_                                                              = {};
    __y_                                                              = {};
    __k_                                                              = {};
    __j_                                                              = word_count - 1;
    constexpr auto __p                                                = (word_size - 1) / 32 + 1;
    ::cuda::std::array<std::uint_least32_t, word_count / 2 * __p> __a = {};
    __seq.generate(__a.begin(), __a.end());
    for (::cuda::std::size_t __k = 0; __k < word_count / 2; ++__k)
    {
      result_type __sum = 0;
      for (::cuda::std::size_t __i = 0; __i < __p; ++__i)
      {
        const size_t __current = __k * __p;
        __sum += static_cast<result_type>(__a[__current + __i]) << (32 * __i);
      }
      __k_[__k] = __sum & max();
    }
  }

  //! This method sets the internal counter. Each increment of the counter produces n new values. The array counter
  //! can be thought of as a big integer. The n-1'th counter value is the least significant and the 0'th counter value
  //! is the most significant. set_counter is related but distinct from discard:
  //! - set_counter sets the engine's absolute position, while discard increments the engine.
  //! - Each increment of the counter always produces n new values, while discard can increment by any number of values
  //! equivalent to calling operator(). i.e. The sub-counter j is always set to n-1 after calling set_counter.
  //! - set_counter exposes the full period of the engine as a big integer, while discard is limited by its word size
  //! argument.
  //!
  //! set_counter is commonly used to initialize different streams of random numbers in parallel applications.
  //!
  //! .. code-block:: c++
  //!
  //!    Engine e1; // some philox_engine
  //!    Engine e2;
  //!    e1.set_counter({0, 0, 0, 100});
  //!    e2.set_counter({0, 0, 1, 100}); // e2 is now 4*2^w values ahead of e1
  //!
  //! @param __counter The counter.
  _CCCL_API constexpr void set_counter(const ::cuda::std::array<result_type, word_count>& __counter) noexcept
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (::cuda::std::size_t __j = 0; __j < word_count; ++__j)
    {
      __x_[__j] = __counter[word_count - 1 - __j] & max();
    }
    __j_ = word_count - 1;
  }

  // generating functions

  //! This member function produces a new random value and updates this philox_engine's state.
  //! @return A new random number.
  _CCCL_API constexpr result_type operator()() noexcept
  {
    ++__j_;
    if (__j_ == word_count)
    {
      __philox();
      __increment_counter();
      __j_ = 0;
    }
    return __y_[__j_];
  }

  //! This member function advances this philox_engine's state a given number of times
  //! and discards the results. philox_engine is a counter-based engine, therefore can discard with O(1) complexity.
  //!
  //! @param __z The number of random values to discard.
  _CCCL_API constexpr void discard(unsigned long long __z) noexcept
  {
    // Advance __j_ until we are at n - 1
    auto __advance = ::cuda::std::min(__z, static_cast<unsigned long long>(word_count - 1 - __j_));
    __j_ += static_cast<::cuda::std::size_t>(__advance);
    __z -= __advance;

    // Increment the big integer counter, handling overflow
    unsigned long long __increment       = __z / word_count;
    const unsigned long long __remainder = __z % word_count;
    ::cuda::std::size_t __carry          = 0;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (::cuda::std::size_t __j = 0; __j < word_count; ++__j)
    {
      if (__increment == 0 && __carry == 0)
      {
        break;
      }
      result_type __new_x_j = (__x_[__j] + (__increment & max()) + __carry) & max();
      __carry               = (__new_x_j < __x_[__j]) ? 1 : 0;
      __x_[__j]             = __new_x_j;
      if constexpr (word_size < 64)
      {
        __increment >>= word_size;
      }
      else
      {
        __increment = 0;
      }
    }

    // Advance the output buffer position by the remainder
    if (__remainder > 0)
    {
      __philox();
      __increment_counter();
      __j_ = static_cast<::cuda::std::size_t>(__remainder - 1);
    }
  }

  //! This function checks two philox_engines for equality.
  //! @param lhs The first philox_engine to test.
  //! @param rhs The second philox_engine to test.
  //! @return true if lhs is equal to rhs; false, otherwise.
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const philox_engine& __lhs, const philox_engine& __rhs) noexcept
  {
    if (__lhs.__x_ != __rhs.__x_)
    {
      return false;
    }
    // Only check the y buffer if not __j_ != word_count-1
    // If __j_ == word_count-1, then __y_ is not valid
    if (__lhs.__j_ != word_count - 1 && __lhs.__y_ != __rhs.__y_)
    {
      return false;
    }
    if (__lhs.__k_ != __rhs.__k_)
    {
      return false;
    }
    return __lhs.__j_ == __rhs.__j_;
  }

#if _CCCL_STD_VER == 2017
  //! This function checks two philox_engines for inequality.
  //! @param lhs The first philox_engine to test.
  //! @param rhs The second philox_engine to test.
  //! @return true if lhs is not equal to rhs; false, otherwise.
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const philox_engine& __lhs, const philox_engine& __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }
#endif

#if !_CCCL_COMPILER(NVRTC)
  //! This function streams a philox_engine to a std::basic_ostream.
  //! @param os The basic_ostream to stream out to.
  //! @param e The philox_engine to stream out.
  //! @return os
  template <typename _CharT, typename _Traits>
  _CCCL_API friend ::std::basic_ostream<_CharT, _Traits>&
  operator<<(::std::basic_ostream<_CharT, _Traits>& __os, const philox_engine& __e)
  {
    using ostream_type = ::std::basic_ostream<_CharT, _Traits>;
    using ios_base     = typename ostream_type::ios_base;

    // save old flags & fill character
    const typename ios_base::fmtflags __flags = __os.flags();
    const _CharT __fill                       = __os.fill();

    __os.flags(ios_base::dec | ios_base::fixed | ios_base::left);
    __os.fill(__os.widen(' '));

    // output counter array (__x_)
    for (::cuda::std::size_t __i = 0; __i < word_count; ++__i)
    {
      __os << __e.__x_[__i];
      if (__i < word_count - 1)
      {
        __os << __os.widen(' ');
      }
    }
    __os << __os.widen(' ');

    // output key array (__k_)
    for (::cuda::std::size_t __i = 0; __i < word_count / 2; ++__i)
    {
      __os << __e.__k_[__i];
      if (__i < word_count / 2 - 1)
      {
        __os << __os.widen(' ');
      }
    }
    __os << __os.widen(' ');

    // output output buffer (__y_)
    for (::cuda::std::size_t __i = 0; __i < word_count; ++__i)
    {
      __os << __e.__y_[__i];
      if (__i < word_count - 1)
      {
        __os << __os.widen(' ');
      }
    }
    __os << __os.widen(' ');

    // output current position
    __os << __e.__j_;

    // restore flags & fill character
    __os.flags(__flags);
    __os.fill(__fill);

    return __os;
  }

  //! This function streams a philox_engine in from a std::basic_istream.
  //! @param is The basic_istream to stream from.
  //! @param e The philox_engine to stream in.
  //! @return is
  template <typename _CharT, typename _Traits>
  _CCCL_API friend ::std::basic_istream<_CharT, _Traits>&
  operator>>(::std::basic_istream<_CharT, _Traits>& __is, philox_engine& __e)
  {
    using istream_type = ::std::basic_istream<_CharT, _Traits>;
    using ios_base     = typename istream_type::ios_base;

    // save old flags
    const typename ios_base::fmtflags __flags = __is.flags();

    __is.flags(ios_base::dec);

    // input counter array (__x_)
    for (::cuda::std::size_t __i = 0; __i < word_count; ++__i)
    {
      __is >> __e.__x_[__i];
    }

    // input key array (__k_)
    for (::cuda::std::size_t __i = 0; __i < word_count / 2; ++__i)
    {
      __is >> __e.__k_[__i];
    }

    // input output buffer (__y_)
    for (::cuda::std::size_t __i = 0; __i < word_count; ++__i)
    {
      __is >> __e.__y_[__i];
    }

    // input current position
    __is >> __e.__j_;

    // restore flags
    __is.flags(__flags);

    return __is;
  }
#endif // !_CCCL_COMPILER(NVRTC)

private:
  _CCCL_API constexpr void __increment_counter() noexcept
  {
    // Increment the big integer __x_ by 1, handling overflow.
    _CCCL_PRAGMA_UNROLL_FULL()
    for (::cuda::std::size_t __i = 0; __i < word_count; ++__i)
    {
      __x_[__i] = (__x_[__i] + 1) & max();
      if (__x_[__i] != 0)
      {
        break;
      }
    }
  }

  [[nodiscard]] static _CCCL_API constexpr auto __mulhilo_fallback(result_type __a, result_type __b) noexcept
  {
    // Generic slow implementation
    constexpr result_type __w_half  = word_size / 2;
    constexpr result_type __lo_mask = (result_type{1} << __w_half) - 1;

    result_type __lo  = __a * __b;
    result_type __ahi = __a >> __w_half;
    result_type __alo = __a & __lo_mask;
    result_type __bhi = __b >> __w_half;
    result_type __blo = __b & __lo_mask;

    result_type __ahbl = __ahi * __blo;
    result_type __albh = __alo * __bhi;

    result_type __ahbl_albh = ((__ahbl & __lo_mask) + (__albh & __lo_mask));
    result_type __hi        = __ahi * __bhi + (__ahbl >> __w_half) + (__albh >> __w_half);
    __hi += __ahbl_albh >> __w_half;
    __hi += ((__lo >> __w_half) < (__ahbl_albh & __lo_mask));

    return ::cuda::std::pair{__hi & max(), __lo & max()};
  }

  static _CCCL_API constexpr auto __mulhilo(result_type __a, result_type __b) noexcept
  {
    if constexpr (word_size == 32 || word_size == 64)
    {
      using _Up       = ::cuda::std::__make_nbit_uint_t<word_size>;
      auto __hi       = static_cast<result_type>(::cuda::mul_hi(static_cast<_Up>(__a), static_cast<_Up>(__b)));
      const auto __lo = (__a * __b) & max();
      return ::cuda::std::pair{__hi, __lo};
    }
    else
    {
      return __mulhilo_fallback(__a, __b);
    }
  }

  _CCCL_API constexpr void __philox() noexcept
  {
    // Only two variants are allowed, n=2 or n=4
    ::cuda::std::array<result_type, word_count> __S     = __x_;
    ::cuda::std::array<result_type, word_count / 2> __K = __k_;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (::cuda::std::size_t __j = 0; __j < round_count; ++__j)
    {
      // Only two variants are allowed, n=2 or n=4
      if constexpr (word_count == 2)
      {
        auto [__hi, __lo] = __mulhilo(__S[0], multipliers[0]);
        __S[0]            = __hi ^ __K[0] ^ __S[1];
        __S[1]            = __lo;
        __K[0]            = (__K[0] + round_consts[0]) & max();
      }
      else // word_count == 4
      {
        ::cuda::std::array<result_type, word_count> __V = {__S[2], __S[1], __S[0], __S[3]};
        auto [__hi0, __lo0]                             = __mulhilo(__V[0], multipliers[0]);
        auto [__hi2, __lo2]                             = __mulhilo(__V[2], multipliers[1]);
        __S[0]                                          = __hi0 ^ __K[0] ^ __V[1];
        __S[1]                                          = __lo0;
        __S[2]                                          = __hi2 ^ __K[1] ^ __V[3];
        __S[3]                                          = __lo2;
        __K[0]                                          = (__K[0] + round_consts[0]) & max();
        __K[1]                                          = (__K[1] + round_consts[1]) & max();
      }
    }
    __y_ = __S;
  }

  // The counter X, a big integer stored as word_count w-bit words.
  // The least significant word is __x_[0].
  ::cuda::std::array<result_type, word_count> __x_ = {};
  // K is the "Key", storing the seed
  ::cuda::std::array<result_type, word_count / 2> __k_ = {};
  // The output buffer Y
  // Each time __j_ reaches word_count, we generate word_count new values and store them in __y_.
  ::cuda::std::array<result_type, word_count> __y_ = {};
  // Each generation produces n random numbers, which are returned one at a time.
  // __j_ cycles through [0, n-1].
  ::cuda::std::size_t __j_ = 0;

}; // end philox_engine

//! @typedef philox4x32
//! @brief A random number engine with predefined parameters which implements the
//!        Philox counter based random number generation algorithm.
//! @note The 10000th consecutive invocation of a default-constructed object of type philox4x32
//!       shall produce the value 1955073260.
using philox4x32 = philox_engine<::cuda::std::uint_fast32_t, 32, 4, 10, 0xCD9E8D57, 0x9E3779B9, 0xD2511F53, 0xBB67AE85>;

//! @typedef philox4x64
//! @brief A random number engine with predefined parameters which implements the
//!        Philox counter based random number generation algorithm.
//! @note The 10000th consecutive invocation of a default-constructed object of type philox4x64
//!       shall produce the value 3409172418970261260.
using philox4x64 =
  philox_engine<::cuda::std::uint_fast64_t,
                64,
                4,
                10,
                0xCA5A826395121157,
                0x9E3779B97F4A7C15,
                0xD2E7470EE14C6C93,
                0xBB67AE8584CAA73B>;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANDOM_PHILOX_ENGINE_H
