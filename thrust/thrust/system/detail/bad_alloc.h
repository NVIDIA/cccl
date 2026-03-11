// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
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

#include <new>
#include <string>

THRUST_NAMESPACE_BEGIN
namespace system::detail
{
// define our own bad_alloc so we can set its .what()
class bad_alloc : public std::bad_alloc
{
public:
  inline bad_alloc(const std::string& w)
      : std::bad_alloc()
      , m_what()
  {
    m_what = std::bad_alloc::what();
    m_what += ": ";
    m_what += w;
  } // end bad_alloc()

  inline virtual ~bad_alloc() noexcept {}

  inline virtual const char* what() const noexcept
  {
    return m_what.c_str();
  } // end what()

private:
  std::string m_what;
}; // end bad_alloc
} // namespace system::detail
THRUST_NAMESPACE_END
