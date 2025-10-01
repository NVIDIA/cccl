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
  for (size_t j = 0; j < n; ++j)
  {
    m_x[j] = 0;
    m_y[j] = 0;
  }
  for (size_t l = 0; l < n / 2; ++l)
  {
    m_k[l] = 0;
  }
  m_k[0] = s & max;
  m_j    = n - 1;
}

template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
_CCCL_HOST_DEVICE void
philox_engine<UIntType, w, n, r, consts...>::set_counter(const std::array<result_type, n>& counter)
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
  for (unsigned long long i = 0; i < z; ++i)
  {
    (*this)();
  }
}

template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
_CCCL_HOST_DEVICE inline void philox_engine<UIntType, w, n, r, consts...>::mulhilo(
  result_type a, result_type b, result_type& hi, result_type& lo) const
{
  static_assert(w == 32 || w == 64, "Only w=32 or w=64 are supported in philox_engine");
  if constexpr (w == 32)
  {
    using upgraded_type = std::uint_fast64_t;

    const upgraded_type ab = static_cast<upgraded_type>(a) * static_cast<upgraded_type>(b);
    hi                     = static_cast<result_type>(ab >> w);
    lo                     = static_cast<result_type>(ab) & this->max;
  }
  else
  {
    std::uint_fast64_t u1 = (a & 0xffffffff);
    std::uint_fast64_t v1 = (b & 0xffffffff);
    std::uint_fast64_t t  = (u1 * v1);
    std::uint_fast64_t w3 = (t & 0xffffffff);
    std::uint_fast64_t k  = (t >> 32);

    a >>= 32;
    t                     = (a * v1) + k;
    k                     = (t & 0xffffffff);
    std::uint_fast64_t w1 = (t >> 32);

    b >>= 32;
    t = (u1 * b) + k;
    k = (t >> 32);

    hi = static_cast<result_type>((a * b) + w1 + k);
    lo = static_cast<result_type>((t << 32) + w3) & this->max;
  }
}

template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
_CCCL_HOST_DEVICE void philox_engine<UIntType, w, n, r, consts...>::philox()
{
  // Only two variants are allowed, n=2 or n=4
  const UIntType consts_array[n] = {consts...};
  if constexpr (n == 2)
  {
    result_type S0 = this->m_x[0];
    result_type S1 = this->m_x[1];
    result_type K0 = this->m_k[0];
    for (size_t j = 0; j < r; ++j)
    {
      result_type hi, lo;
      this->mulhilo(S0, consts_array[0], hi, lo);
      S0 = hi ^ K0 ^ S1;
      S1 = lo;
      K0 = (K0 + consts_array[1]) & max;
    }
    this->m_y[0] = S0;
    this->m_y[1] = S1;
  }
  else if constexpr (n == 4)
  {
    result_type S0 = this->m_x[0];
    result_type S1 = this->m_x[1];
    result_type S2 = this->m_x[2];
    result_type S3 = this->m_x[3];
    result_type K0 = this->m_k[0];
    result_type K1 = this->m_k[1];
    for (size_t j = 0; j < r; ++j)
    {
      result_type V0 = S2;
      result_type V1 = S1;
      result_type V2 = S0;
      result_type V3 = S3;

      result_type hi0, lo0;
      this->mulhilo(V0, consts_array[2], hi0, lo0);
      result_type hi2, lo2;
      this->mulhilo(V2, consts_array[0], hi2, lo2);

      S0 = hi0 ^ K0 ^ V1;
      S1 = lo0;

      S2 = hi2 ^ K1 ^ V3;

      S3 = lo2;

      K0 = (K0 + consts_array[1]) & max;
      K1 = (K1 + consts_array[3]) & max;
    }

    this->m_y[0] = S0;
    this->m_y[1] = S1;
    this->m_y[2] = S2;
    this->m_y[3] = S3;
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
  // Compare all state: counter (m_x), key (m_k), output buffer (m_y), and position (m_j)
  for (size_t i = 0; i < n; ++i)
  {
    if (m_x[i] != rhs.m_x[i] || m_y[i] != rhs.m_y[i])
    {
      return false;
    }
  }

  for (size_t i = 0; i < n / 2; ++i)
  {
    if (m_k[i] != rhs.m_k[i])
    {
      return false;
    }
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
