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

#include <thrust/random/detail/mod.h>
#include <thrust/random/detail/random_core_access.h>
#include <thrust/random/linear_congruential_engine.h>

THRUST_NAMESPACE_BEGIN

namespace random
{
template <typename UIntType, UIntType A, UIntType C, UIntType M>
_CCCL_HOST_DEVICE linear_congruential_engine<UIntType, A, C, M>::linear_congruential_engine(result_type s)
{
  seed(s);
} // end linear_congruential_engine::linear_congruential_engine()

template <typename UIntType, UIntType A, UIntType C, UIntType M>
_CCCL_HOST_DEVICE void linear_congruential_engine<UIntType, A, C, M>::seed(result_type s)
{
  if ((detail::mod<UIntType, 1, 0, M>(C) == 0) && (detail::mod<UIntType, 1, 0, M>(s) == 0))
  {
    m_x = detail::mod<UIntType, 1, 0, M>(1);
  }
  else
  {
    m_x = detail::mod<UIntType, 1, 0, M>(s);
  }
} // end linear_congruential_engine::seed()

template <typename UIntType, UIntType A, UIntType C, UIntType M>
_CCCL_HOST_DEVICE typename linear_congruential_engine<UIntType, A, C, M>::result_type
linear_congruential_engine<UIntType, A, C, M>::operator()()
{
  m_x = detail::mod<UIntType, A, C, M>(m_x);
  return m_x;
} // end linear_congruential_engine::operator()()

template <typename UIntType, UIntType A, UIntType C, UIntType M>
_CCCL_HOST_DEVICE void linear_congruential_engine<UIntType, A, C, M>::discard(unsigned long long z)
{
  thrust::random::detail::linear_congruential_engine_discard::discard(*this, z);
} // end linear_congruential_engine::discard()

template <typename UIntType, UIntType A, UIntType C, UIntType M>
template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
linear_congruential_engine<UIntType, A, C, M>::stream_out(std::basic_ostream<CharT, Traits>& os) const
{
  using ostream_type = std::basic_ostream<CharT, Traits>;
  using ios_base     = typename ostream_type::ios_base;

  // save old flags & fill character
  const typename ios_base::fmtflags flags = os.flags();
  const CharT fill                        = os.fill();

  os.flags(ios_base::dec | ios_base::fixed | ios_base::left);
  os.fill(os.widen(' '));

  // output one word of state
  os << m_x;

  // restore flags & fill character
  os.flags(flags);
  os.fill(fill);

  return os;
}

template <typename UIntType, UIntType A, UIntType C, UIntType M>
template <typename CharT, typename Traits>
std::basic_istream<CharT, Traits>&
linear_congruential_engine<UIntType, A, C, M>::stream_in(std::basic_istream<CharT, Traits>& is)
{
  using istream_type = std::basic_istream<CharT, Traits>;
  using ios_base     = typename istream_type::ios_base;

  // save old flags
  const typename ios_base::fmtflags flags = is.flags();

  is.flags(ios_base::dec);

  // input one word of state
  is >> m_x;

  // restore flags
  is.flags(flags);

  return is;
}

template <typename UIntType, UIntType A, UIntType C, UIntType M>
_CCCL_HOST_DEVICE bool
linear_congruential_engine<UIntType, A, C, M>::equal(const linear_congruential_engine<UIntType, A, C, M>& rhs) const
{
  return m_x == rhs.m_x;
}

template <typename UIntType, UIntType A, UIntType C, UIntType M>
_CCCL_HOST_DEVICE bool operator==(const linear_congruential_engine<UIntType, A, C, M>& lhs,
                                  const linear_congruential_engine<UIntType, A, C, M>& rhs)
{
  return detail::random_core_access::equal(lhs, rhs);
}

template <typename UIntType, UIntType A, UIntType C, UIntType M>
_CCCL_HOST_DEVICE bool operator!=(const linear_congruential_engine<UIntType, A, C, M>& lhs,
                                  const linear_congruential_engine<UIntType, A, C, M>& rhs)
{
  return !(lhs == rhs);
}

template <typename UIntType, UIntType A, UIntType C, UIntType M, typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& os, const linear_congruential_engine<UIntType, A, C, M>& e)
{
  return detail::random_core_access::stream_out(os, e);
}

template <typename UIntType, UIntType A, UIntType C, UIntType M, typename CharT, typename Traits>
std::basic_istream<CharT, Traits>&
operator>>(std::basic_istream<CharT, Traits>& is, linear_congruential_engine<UIntType, A, C, M>& e)
{
  return detail::random_core_access::stream_in(is, e);
}
} // namespace random

THRUST_NAMESPACE_END
