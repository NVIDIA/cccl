// SPDX-FileCopyrightText: Copyright (c) 2008-2021, NVIDIA Corporation
// SPDX-FileCopyrightText: Copyright (c) 2013, Filipe RNC Maia
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#include <thrust/complex.h>

THRUST_NAMESPACE_BEGIN
template <typename ValueType, class CharT, class Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os, const complex<ValueType>& z)
{
  os << '(' << z.real() << ',' << z.imag() << ')';
  return os;
}

template <typename ValueType, typename CharT, class Traits>
std::basic_istream<CharT, Traits>& operator>>(std::basic_istream<CharT, Traits>& is, complex<ValueType>& z)
{
  ValueType re, im;

  CharT ch;
  is >> ch;

  if (ch == '(')
  {
    is >> re >> ch;
    if (ch == ',')
    {
      is >> im >> ch;
      if (ch == ')')
      {
        z = complex<ValueType>(re, im);
      }
      else
      {
        is.setstate(std::ios_base::failbit);
      }
    }
    else if (ch == ')')
    {
      z = re;
    }
    else
    {
      is.setstate(std::ios_base::failbit);
    }
  }
  else
  {
    is.putback(ch);
    is >> re;
    z = re;
  }
  return is;
}

THRUST_NAMESPACE_END
