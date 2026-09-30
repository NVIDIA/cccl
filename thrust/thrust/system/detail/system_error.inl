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

#include <thrust/system/system_error.h>

THRUST_NAMESPACE_BEGIN

namespace system
{
system_error ::system_error(error_code ec, const std::string& what_arg)
    : std::runtime_error(what_arg)
    , m_error_code(ec)
{} // end system_error::system_error()

system_error ::system_error(error_code ec, const char* what_arg)
    : std::runtime_error(what_arg)
    , m_error_code(ec)
{
  ;
} // end system_error::system_error()

system_error ::system_error(error_code ec)
    : std::runtime_error("")
    , m_error_code(ec)
{
  ;
} // end system_error::system_error()

system_error ::system_error(int ev, const error_category& ecat, const std::string& what_arg)
    : std::runtime_error(what_arg)
    , m_error_code(ev, ecat)
{
  ;
} // end system_error::system_error()

system_error ::system_error(int ev, const error_category& ecat, const char* what_arg)
    : std::runtime_error(what_arg)
    , m_error_code(ev, ecat)
{
  ;
} // end system_error::system_error()

system_error ::system_error(int ev, const error_category& ecat)
    : std::runtime_error("")
    , m_error_code(ev, ecat)
{
  ;
} // end system_error::system_error()

const error_code& system_error ::code() const noexcept
{
  return m_error_code;
} // end system_error::code()

const char* system_error ::what() const noexcept
{
  if (m_what.empty())
  {
    try
    {
      m_what = this->std::runtime_error::what();
      if (m_error_code)
      {
        if (!m_what.empty())
        {
          m_what += ": ";
        }
        m_what += m_error_code.message();
      }
    }
    catch (...)
    {
      return std::runtime_error::what();
    }
  }

  return m_what.c_str();
} // end system_error::what()
} // namespace system

THRUST_NAMESPACE_END
