// SPDX-FileCopyrightText: Copyright (c) 2008-2021, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/random/discard_block_engine.h>

THRUST_NAMESPACE_BEGIN

namespace random
{
template <typename Engine, size_t P, size_t R>
_CCCL_HOST_DEVICE discard_block_engine<Engine, P, R>::discard_block_engine()
    : m_e()
    , m_n(0)
{}

template <typename Engine, size_t P, size_t R>
_CCCL_HOST_DEVICE discard_block_engine<Engine, P, R>::discard_block_engine(result_type s)
    : m_e(s)
    , m_n(0)
{}

template <typename Engine, size_t P, size_t R>
_CCCL_HOST_DEVICE discard_block_engine<Engine, P, R>::discard_block_engine(const base_type& urng)
    : m_e(urng)
    , m_n(0)
{}

template <typename Engine, size_t P, size_t R>
_CCCL_HOST_DEVICE void discard_block_engine<Engine, P, R>::seed()
{
  m_e.seed();
  m_n = 0;
}

template <typename Engine, size_t P, size_t R>
_CCCL_HOST_DEVICE void discard_block_engine<Engine, P, R>::seed(result_type s)
{
  m_e.seed(s);
  m_n = 0;
}

template <typename Engine, size_t P, size_t R>
_CCCL_HOST_DEVICE typename discard_block_engine<Engine, P, R>::result_type
discard_block_engine<Engine, P, R>::operator()()
{
  if (m_n >= used_block)
  {
    m_e.discard(block_size - m_n);
    //    for(; m_n < block_size; ++m_n)
    //      m_e();
    m_n = 0;
  }

  ++m_n;

  return m_e();
}

template <typename Engine, size_t P, size_t R>
_CCCL_HOST_DEVICE void discard_block_engine<Engine, P, R>::discard(unsigned long long z)
{
  // XXX this should be accelerated
  for (; z > 0; --z)
  {
    this->operator()();
  } // end for
}

template <typename Engine, size_t P, size_t R>
_CCCL_HOST_DEVICE const typename discard_block_engine<Engine, P, R>::base_type&
discard_block_engine<Engine, P, R>::base() const
{
  return m_e;
}

template <typename Engine, size_t P, size_t R>
template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
discard_block_engine<Engine, P, R>::stream_out(std::basic_ostream<CharT, Traits>& os) const
{
  using ostream_type = std::basic_ostream<CharT, Traits>;
  using ios_base     = typename ostream_type::ios_base;

  // save old flags & fill character
  const typename ios_base::fmtflags flags = os.flags();
  const CharT fill                        = os.fill();

  const CharT space = os.widen(' ');
  os.flags(ios_base::dec | ios_base::fixed | ios_base::left);
  os.fill(space);

  // output the base engine followed by n
  os << m_e << space << m_n;

  // restore flags & fill character
  os.flags(flags);
  os.fill(fill);

  return os;
}

template <typename Engine, size_t P, size_t R>
template <typename CharT, typename Traits>
std::basic_istream<CharT, Traits>& discard_block_engine<Engine, P, R>::stream_in(std::basic_istream<CharT, Traits>& is)
{
  using istream_type = std::basic_istream<CharT, Traits>;
  using ios_base     = typename istream_type::ios_base;

  // save old flags
  const typename ios_base::fmtflags flags = is.flags();

  is.flags(ios_base::skipws);

  // input the base engine and then n
  is >> m_e >> m_n;

  // restore old flags
  is.flags(flags);
  return is;
}

template <typename Engine, size_t P, size_t R>
_CCCL_HOST_DEVICE bool discard_block_engine<Engine, P, R>::equal(const discard_block_engine<Engine, P, R>& rhs) const
{
  return (m_e == rhs.m_e) && (m_n == rhs.m_n);
}

template <typename Engine, size_t P, size_t R, typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& os, const discard_block_engine<Engine, P, R>& e)
{
  return thrust::random::detail::random_core_access::stream_out(os, e);
}

template <typename Engine, size_t P, size_t R, typename CharT, typename Traits>
std::basic_istream<CharT, Traits>&
operator>>(std::basic_istream<CharT, Traits>& is, discard_block_engine<Engine, P, R>& e)
{
  return thrust::random::detail::random_core_access::stream_in(is, e);
}

template <typename Engine, size_t P, size_t R>
_CCCL_HOST_DEVICE bool
operator==(const discard_block_engine<Engine, P, R>& lhs, const discard_block_engine<Engine, P, R>& rhs)
{
  return thrust::random::detail::random_core_access::equal(lhs, rhs);
}

template <typename Engine, size_t P, size_t R>
_CCCL_HOST_DEVICE bool
operator!=(const discard_block_engine<Engine, P, R>& lhs, const discard_block_engine<Engine, P, R>& rhs)
{
  return !(lhs == rhs);
}
} // namespace random

THRUST_NAMESPACE_END
