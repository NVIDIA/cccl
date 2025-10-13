//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
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

#include <cuda/__cmath/fast_modulo_division.h>
#include <cuda/std/array>
#include <cuda/std/cstddef> // for size_t
#include <cuda/std/cstdint>

#if !_CCCL_COMPILER(NVRTC)
#  include <iostream>
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
//! @tparam UIntType The type of unsigned integer to produce.
//! @tparam w The word size
//! @tparam n The buffer size
//! @tparam r The number of rounds
//! @tparam consts The constants used in the generation algorithm.
//!
//! @note Inexperienced users should not use this class template directly.  Instead, use
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
template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
class philox_engine
{
  static_assert(n == 2 || n == 4, "N argument must be either 2 or 4");
  static_assert(sizeof...(consts) == n, "consts array must be of length N");
  static_assert(0 < r, "rounds must be a natural number");
  static_assert((0 < w && w <= std::numeric_limits<UIntType>::digits),
                "Word size w must satisfy 0 < w <= numeric_limits<UIntType>::digits");

public:
  using result_type = UIntType;

  //! The smallest value this engine may potentially produce.
  static const result_type min = 0;
  //! The largest value this engine may potentially produce.
  static const result_type max = ((1ull << (w - 1)) | ((1ull << (w - 1)) - 1));

  //! The default seed.
  static constexpr result_type default_seed = 20111115u;

  //! This constructor, which optionally accepts a seed, initializes a new
  //! philox_engine.
  //!
  //! @param s The seed used to initialize this philox_engine's state.
  _CCCL_HOST_DEVICE explicit philox_engine(result_type s = default_seed)
  {
    seed(s);
  }

  //! This method initializes this philox_engine's state, and optionally accepts
  //! a seed value.
  //!
  //! @param s The seed used to initializes this philox_engine's state.
  _CCCL_HOST_DEVICE void seed(result_type s = default_seed)
  {
    m_x    = {};
    m_y    = {};
    m_k    = {};
    m_k[0] = s & max;
    m_j    = n - 1;
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
  //! @param counter The counter.
  _CCCL_HOST_DEVICE void set_counter(const ::cuda::std::array<result_type, n>& counter)
  {
    for (size_t j = 0; j < n; ++j)
    {
      m_x[j] = counter[n - 1 - j] & max;
    }
    m_j = n - 1;
  }

  // generating functions

  //! This member function produces a new random value and updates this philox_engine's state.
  //! @return A new random number.
  _CCCL_HOST_DEVICE result_type operator()(void)
  {
    m_j++;
    if (m_j == n)
    {
      this->philox();
      this->increment_counter();
      m_j = 0;
    }
    return m_y[m_j];
  }

  //! This member function advances this philox_engine's state a given number of times
  //! and discards the results. philox_engine is a counter-based engine, therefore can discard with O(1) complexity.
  //!
  //! @param z The number of random values to discard.
  _CCCL_HOST_DEVICE void discard(unsigned long long z)
  {
    // Advance m_j until we are at n - 1
    while (m_j != n - 1 && z > 0)
    {
      (*this)();
      z--;
    }

    // Increment the big integer counter, handling overflow
    auto increment    = z / n;
    std::size_t carry = 0;
    for (std::size_t j = 0; j < n; ++j)
    {
      if (increment == 0 && carry == 0)
      {
        break;
      }
      UIntType new_m_x_j = (m_x[j] + (increment & max) + carry) & max;
      carry              = (new_m_x_j < m_x[j]) ? 1 : 0;
      m_x[j]             = new_m_x_j;
      if constexpr (w < 64)
      {
        increment >>= w;
      }
      else
      {
        increment = 0;
      }
    }

    // Advance the output buffer position by the remainder
    const auto remainder = z % n;
    for (std::size_t j = 0; j < remainder; ++j)
    {
      (*this)();
    }
  }

private:
  template <typename UIntType_, size_t w_, size_t n_, size_t r_, UIntType_... consts_>
  friend _CCCL_HOST_DEVICE bool operator==(const philox_engine<UIntType_, w_, n_, r_, consts_...>& lhs,
                                           const philox_engine<UIntType_, w_, n_, r_, consts_...>& rhs);
  template <typename CharT, typename Traits, typename UIntType_, size_t w_, size_t n_, size_t r_, UIntType_... consts_>
  friend ::std::basic_istream<CharT, Traits>&
  operator>>(::std::basic_istream<CharT, Traits>& is, philox_engine<UIntType_, w_, n_, r_, consts_...>& e);

  template <typename CharT, typename Traits, typename UIntType_, size_t w_, size_t n_, size_t r_, UIntType_... consts_>
  friend ::std::basic_ostream<CharT, Traits>&
  operator<<(::std::basic_ostream<CharT, Traits>& os, const philox_engine<UIntType_, w_, n_, r_, consts_...>& e);

  _CCCL_HOST_DEVICE void increment_counter()
  {
    // Increment the big integer m_x by 1, handling overflow.
    std::size_t i = 0;
    do
    {
      m_x[i] = (m_x[i] + 1) & max;
      ++i;
    } while (i < n && !m_x[i - 1]);
  }

  _CCCL_HOST_DEVICE void mulhilo(result_type a, result_type b, result_type& hi, result_type& lo) const
  {
    if constexpr (w == 32)
    {
      // std::uint_fast32_t can actually be 64 bits so cast to 32 bits
      hi = static_cast<UIntType>(
        ::cuda::__multiply_extract_higher_bits(static_cast<std::uint32_t>(a), static_cast<std::uint32_t>(b)));
      lo = (a * b) & max;
    }
    else if constexpr (w == 64)
    {
      hi = static_cast<UIntType>(
        ::cuda::__multiply_extract_higher_bits(static_cast<std::uint64_t>(a), static_cast<std::uint64_t>(b)));
      lo = (a * b) & max;
    }
    else
    {
      // Generic slow implementation
      constexpr UIntType w_half  = w / 2;
      constexpr UIntType lo_mask = (((UIntType) 1) << w_half) - 1;

      lo           = a * b;
      UIntType ahi = a >> w_half;
      UIntType alo = a & lo_mask;
      UIntType bhi = b >> w_half;
      UIntType blo = b & lo_mask;

      UIntType ahbl = ahi * blo;
      UIntType albh = alo * bhi;

      UIntType ahbl_albh = ((ahbl & lo_mask) + (albh & lo_mask));
      hi                 = ahi * bhi + (ahbl >> w_half) + (albh >> w_half);
      hi += ahbl_albh >> w_half;
      hi += ((lo >> w_half) < (ahbl_albh & lo_mask));
    }
  }

  _CCCL_HOST_DEVICE void philox()
  {
    // Only two variants are allowed, n=2 or n=4
    const UIntType consts_array[n] = {consts...};
    if constexpr (n == 2)
    {
      ::cuda::std::array<result_type, n> S     = this->m_x;
      ::cuda::std::array<result_type, n / 2> K = this->m_k;
      for (size_t j = 0; j < r; ++j)
      {
        result_type hi, lo;
        this->mulhilo(S[0], consts_array[0], hi, lo);
        S[0] = hi ^ K[0] ^ S[1];
        S[1] = lo;
        K[0] = (K[0] + consts_array[1]) & max;
      }
      this->m_y = S;
    }
    else if constexpr (n == 4)
    {
      ::cuda::std::array<result_type, n> S     = this->m_x;
      ::cuda::std::array<result_type, n / 2> K = this->m_k;
      for (size_t j = 0; j < r; ++j)
      {
        ::cuda::std::array<result_type, n> V = {S[2], S[1], S[0], S[3]};
        result_type hi0, lo0;
        this->mulhilo(V[0], consts_array[2], hi0, lo0);
        result_type hi2, lo2;
        this->mulhilo(V[2], consts_array[0], hi2, lo2);

        S[0] = hi0 ^ K[0] ^ V[1];
        S[1] = lo0;

        S[2] = hi2 ^ K[1] ^ V[3];

        S[3] = lo2;

        K[0] = (K[0] + consts_array[1]) & max;
        K[1] = (K[1] + consts_array[3]) & max;
      }

      this->m_y = S;
    }
  }

  // The counter X, a big integer stored as n w-bit words.
  // The least significant word is m_x[0].
  ::cuda::std::array<UIntType, n> m_x = {};
  // K is the "Key", storing the seed
  ::cuda::std::array<UIntType, n / 2> m_k = {};
  // The output buffer Y
  // Each time m_j reaches n, we generate n new values and store them in m_y.
  ::cuda::std::array<UIntType, n> m_y = {};
  // Each generation produces n random numbers, which are returned one at a time.
  // m_j cycles through [0, n-1].
  unsigned long long m_j = 0;

}; // end philox_engine

//! This function checks two philox_engines for equality.
//! @param lhs The first philox_engine to test.
//! @param rhs The second philox_engine to test.
//! @return true if lhs is equal to rhs; false, otherwise.
template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
_CCCL_HOST_DEVICE bool operator==(const philox_engine<UIntType, w, n, r, consts...>& lhs,
                                  const philox_engine<UIntType, w, n, r, consts...>& rhs)
{
  if (lhs.m_x != rhs.m_x)
  {
    return false;
  }
  // Only check the y buffer if not m_j != n-1
  // If m_j == n-1, then m_y is not valid
  if (lhs.m_j != n - 1 && lhs.m_y != rhs.m_y)
  {
    return false;
  }
  if (lhs.m_k != rhs.m_k)
  {
    return false;
  }
  return lhs.m_j == rhs.m_j;
}

//! This function checks two philox_engines for inequality.
//! @param lhs The first philox_engine to test.
//! @param rhs The second philox_engine to test.
//! @return true if lhs is not equal to rhs; false, otherwise.
template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
_CCCL_HOST_DEVICE bool operator!=(const philox_engine<UIntType, w, n, r, consts...>& lhs,
                                  const philox_engine<UIntType, w, n, r, consts...>& rhs)
{
  return !(lhs == rhs);
}

//! This function streams a philox_engine to a std::basic_ostream.
//! @param os The basic_ostream to stream out to.
//! @param e The philox_engine to stream out.
//! @return os
template <typename CharT, typename Traits, typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
::std::basic_ostream<CharT, Traits>&
operator<<(::std::basic_ostream<CharT, Traits>& os, const philox_engine<UIntType, w, n, r, consts...>& e)
{
  using ostream_type = ::std::basic_ostream<CharT, Traits>;
  using ios_base     = typename ostream_type::ios_base;

  // save old flags & fill character
  const typename ios_base::fmtflags flags = os.flags();
  const CharT fill                        = os.fill();

  os.flags(ios_base::dec | ios_base::fixed | ios_base::left);
  os.fill(os.widen(' '));

  // output counter array (m_x)
  for (size_t i = 0; i < n; ++i)
  {
    os << e.m_x[i];
    if (i < n - 1)
    {
      os << os.widen(' ');
    }
  }
  os << os.widen(' ');

  // output key array (m_k)
  for (size_t i = 0; i < n / 2; ++i)
  {
    os << e.m_k[i];
    if (i < n / 2 - 1)
    {
      os << os.widen(' ');
    }
  }
  os << os.widen(' ');

  // output output buffer (m_y)
  for (size_t i = 0; i < n; ++i)
  {
    os << e.m_y[i];
    if (i < n - 1)
    {
      os << os.widen(' ');
    }
  }
  os << os.widen(' ');

  // output current position
  os << e.m_j;

  // restore flags & fill character
  os.flags(flags);
  os.fill(fill);

  return os;
}

//! This function streams a philox_engine in from a std::basic_istream.
//! @param is The basic_istream to stream from.
//! @param e The philox_engine to stream in.
//! @return is
template <typename CharT, typename Traits, typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
::std::basic_istream<CharT, Traits>&
operator>>(::std::basic_istream<CharT, Traits>& is, philox_engine<UIntType, w, n, r, consts...>& e)
{
  using istream_type = ::std::basic_istream<CharT, Traits>;
  using ios_base     = typename istream_type::ios_base;

  // save old flags
  const typename ios_base::fmtflags flags = is.flags();

  is.flags(ios_base::dec);

  // input counter array (m_x)
  for (size_t i = 0; i < n; ++i)
  {
    is >> e.m_x[i];
  }

  // input key array (m_k)
  for (size_t i = 0; i < n / 2; ++i)
  {
    is >> e.m_k[i];
  }

  // input output buffer (m_y)
  for (size_t i = 0; i < n; ++i)
  {
    is >> e.m_y[i];
  }

  // input current position
  is >> e.m_j;

  // restore flags
  is.flags(flags);

  return is;
}

//! @typedef philox4x32
//! @brief A random number engine with predefined parameters which implements the
//!        Philox counter based random number generation algorithm.
//! @note The 10000th consecutive invocation of a default-constructed object of type philox4x32
//!       shall produce the value 1955073260.
using philox4x32 = philox_engine<std::uint_fast32_t, 32, 4, 10, 0xD2511F53, 0x9E3779B9, 0xCD9E8D57, 0xBB67AE85>;

//! @typedef philox4x64
//! @brief A random number engine with predefined parameters which implements the
//!        Philox counter based random number generation algorithm.
//! @note The 10000th consecutive invocation of a default-constructed object of type philox4x64
//!       shall produce the value 3409172418970261260.
using philox4x64 =
  philox_engine<std::uint_fast64_t, 64, 4, 10, 0xD2E7470EE14C6C93, 0x9E3779B97F4A7C15, 0xCA5A826395121157, 0xBB67AE8584CAA73B>;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANDOM_PHILOX_ENGINE_H
