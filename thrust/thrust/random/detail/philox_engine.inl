/*
 *  Copyright 2025 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
#pragma once

#include <thrust/detail/config.h>

#include <thrust/random/detail/random_core_access.h>
#include <thrust/random/philox_engine.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

THRUST_NAMESPACE_BEGIN

namespace random
{
template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
_CCCL_HOST_DEVICE philox_engine<UIntType, w, n, r, consts...>::philox_engine(result_type s)
{
  seed(s);
}

template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
_CCCL_HOST_DEVICE void philox_engine<UIntType, w, n, r, consts...>::seed(result_type s)
{
  m_x    = {};
  m_y    = {};
  m_k    = {};
  m_k[0] = s & max;
  m_j    = n - 1;
}

template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
_CCCL_HOST_DEVICE void
philox_engine<UIntType, w, n, r, consts...>::set_counter(const ::cuda::std::array<result_type, n>& counter)
{
  for (size_t j = 0; j < n; ++j)
  {
    m_x[j] = counter[n - 1 - j] & max;
  }
  m_j = n - 1;
}

template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
_CCCL_HOST_DEVICE void philox_engine<UIntType, w, n, r, consts...>::increment_counter()
{
  // Increment the big integer m_x by 1, handling overflow.
  std::size_t i = 0;
  do
  {
    m_x[i] = (m_x[i] + 1) & max;
    ++i;
  } while (i < n && !m_x[i - 1]);
}

template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
_CCCL_HOST_DEVICE void philox_engine<UIntType, w, n, r, consts...>::discard(unsigned long long z)
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

template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
_CCCL_HOST_DEVICE inline void philox_engine<UIntType, w, n, r, consts...>::mulhilo(
  result_type a, result_type b, result_type& hi, result_type& lo) const
{
  if constexpr (w == 32)
  {
    // 32 bit is easy - just use 64 bit multiplication and extract the hi/lo parts
    const auto ab = static_cast<std::uint_fast64_t>(a) * static_cast<std::uint_fast64_t>(b);
    hi            = static_cast<UIntType>(ab >> w);
    lo            = static_cast<UIntType>(ab) & max;
  }
  else
  {
    // 64 bit multiplication is more difficult. The generic implementation is slow, so try to use platform specific
    if constexpr (w == 64)
    {
      // CUDA
#ifdef __CUDA_ARCH__
      hi = static_cast<UIntType>(__umul64hi(a, b));
      lo = static_cast<UIntType>(a * b);
#elif defined(_MSC_VER)
      // MSVC x64
      lo = static_cast<UIntType>(a * b);
      hi = static_cast<UIntType>(__umulh(a, b));
#elif defined(__GNUC__)
      // GCC
      lo = static_cast<UIntType>(a * b);
      hi = static_cast<UIntType>(__uint128_t(a) * __uint128_t(b) >> 64);
#endif
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
}

template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
_CCCL_HOST_DEVICE void philox_engine<UIntType, w, n, r, consts...>::philox()
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

template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
_CCCL_HOST_DEVICE typename philox_engine<UIntType, w, n, r, consts...>::result_type
philox_engine<UIntType, w, n, r, consts...>::operator()(void)
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

template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
_CCCL_HOST_DEVICE bool
philox_engine<UIntType, w, n, r, consts...>::equal(const philox_engine<UIntType, w, n, r, consts...>& rhs) const
{
  if (m_x != rhs.m_x)
  {
    return false;
  }
  // Only check the y buffer if not m_j != n-1
  // If m_j == n-1, then m_y is not valid
  if (m_j != n - 1 && m_y != rhs.m_y)
  {
    return false;
  }
  if (m_k != rhs.m_k)
  {
    return false;
  }
  return m_j == rhs.m_j;
}

template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
philox_engine<UIntType, w, n, r, consts...>::stream_out(std::basic_ostream<CharT, Traits>& os) const
{
  using ostream_type = std::basic_ostream<CharT, Traits>;
  using ios_base     = typename ostream_type::ios_base;

  // save old flags & fill character
  const typename ios_base::fmtflags flags = os.flags();
  const CharT fill                        = os.fill();

  os.flags(ios_base::dec | ios_base::fixed | ios_base::left);
  os.fill(os.widen(' '));

  // output counter array (m_x)
  for (size_t i = 0; i < n; ++i)
  {
    os << m_x[i];
    if (i < n - 1)
    {
      os << os.widen(' ');
    }
  }
  os << os.widen(' ');

  // output key array (m_k)
  for (size_t i = 0; i < n / 2; ++i)
  {
    os << m_k[i];
    if (i < n / 2 - 1)
    {
      os << os.widen(' ');
    }
  }
  os << os.widen(' ');

  // output output buffer (m_y)
  for (size_t i = 0; i < n; ++i)
  {
    os << m_y[i];
    if (i < n - 1)
    {
      os << os.widen(' ');
    }
  }
  os << os.widen(' ');

  // output current position
  os << m_j;

  // restore flags & fill character
  os.flags(flags);
  os.fill(fill);

  return os;
}

template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
template <typename CharT, typename Traits>
std::basic_istream<CharT, Traits>&
philox_engine<UIntType, w, n, r, consts...>::stream_in(std::basic_istream<CharT, Traits>& is)
{
  using istream_type = std::basic_istream<CharT, Traits>;
  using ios_base     = typename istream_type::ios_base;

  // save old flags
  const typename ios_base::fmtflags flags = is.flags();

  is.flags(ios_base::dec);

  // input counter array (m_x)
  for (size_t i = 0; i < n; ++i)
  {
    is >> m_x[i];
  }

  // input key array (m_k)
  for (size_t i = 0; i < n / 2; ++i)
  {
    is >> m_k[i];
  }

  // input output buffer (m_y)
  for (size_t i = 0; i < n; ++i)
  {
    is >> m_y[i];
  }

  // input current position
  is >> m_j;

  // restore flags
  is.flags(flags);

  return is;
}

template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
_CCCL_HOST_DEVICE bool operator==(const philox_engine<UIntType, w, n, r, consts...>& lhs,
                                  const philox_engine<UIntType, w, n, r, consts...>& rhs)
{
  return detail::random_core_access::equal(lhs, rhs);
}

template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
_CCCL_HOST_DEVICE bool operator!=(const philox_engine<UIntType, w, n, r, consts...>& lhs,
                                  const philox_engine<UIntType, w, n, r, consts...>& rhs)
{
  return !(lhs == rhs);
}

template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts, typename CharT, typename Traits>
std::basic_istream<CharT, Traits>&
operator>>(std::basic_istream<CharT, Traits>& is, philox_engine<UIntType, w, n, r, consts...>& e)
{
  return detail::random_core_access::stream_in(is, e);
}

template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts, typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& os, const philox_engine<UIntType, w, n, r, consts...>& e)
{
  return detail::random_core_access::stream_out(os, e);
}

} // namespace random

THRUST_NAMESPACE_END
