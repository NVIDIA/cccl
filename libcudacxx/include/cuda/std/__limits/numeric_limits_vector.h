//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___LIMITS_NUMERIC_LIMITS_VECTOR_H
#define _LIBCUDACXX___LIMITS_NUMERIC_LIMITS_VECTOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__limits/numeric_limits.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

/***********************************************************************************************************************
 * Vector X2 numeric_limits
 **********************************************************************************************************************/

template <typename _VecTp, typename _Tp, __numeric_limits_type __type>
class __numeric_limits_vector_X2_impl : public __numeric_limits_impl<_Tp, __type>
{
public:
  using type = _VecTp;

  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type min() noexcept
  {
    return _VecTp{__numeric_limits_impl<_Tp>::min(), __numeric_limits_impl<_Tp>::min()};
  }

  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type max() noexcept
  {
    return _VecTp{__numeric_limits_impl<_Tp>::max(), __numeric_limits_impl<_Tp>::max()};
  }

  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type lowest() noexcept
  {
    return _VecTp{__numeric_limits_impl<_Tp>::lowest(), __numeric_limits_impl<_Tp>::lowest()};
  }
};

template <>
class __numeric_limits_impl<char2, __numeric_limits_type::__integral>
    : public __numeric_limits_vector_X2_impl<char2, signed char, __numeric_limits_type::__integral>
{};

template <>
class __numeric_limits_impl<uchar2, __numeric_limits_type::__integral>
    : public __numeric_limits_vector_X2_impl<uchar2, unsigned char, __numeric_limits_type::__integral>
{};

template <>
class __numeric_limits_impl<short2, __numeric_limits_type::__integral>
    : public __numeric_limits_vector_X2_impl<short2, signed short, __numeric_limits_type::__integral>
{};

template <>
class __numeric_limits_impl<ushort2, __numeric_limits_type::__integral>
    : public __numeric_limits_vector_X2_impl<ushort2, unsigned short, __numeric_limits_type::__integral>
{};

template <>
class __numeric_limits_impl<int2, __numeric_limits_type::__integral>
    : public __numeric_limits_vector_X2_impl<int2, int, __numeric_limits_type::__integral>
{};

template <>
class __numeric_limits_impl<uint2, __numeric_limits_type::__integral>
    : public __numeric_limits_vector_X2_impl<uint2, unsigned, __numeric_limits_type::__integral>
{};

template <>
class __numeric_limits_impl<long2, __numeric_limits_type::__integral>
    : public __numeric_limits_vector_X2_impl<long2, long, __numeric_limits_type::__integral>
{};

template <>
class __numeric_limits_impl<ulong2, __numeric_limits_type::__integral>
    : public __numeric_limits_vector_X2_impl<ulong2, long unsigned, __numeric_limits_type::__integral>
{};

template <>
class __numeric_limits_impl<longlong2, __numeric_limits_type::__integral>
    : public __numeric_limits_vector_X2_impl<longlong2, long long, __numeric_limits_type::__integral>
{};

template <>
class __numeric_limits_impl<ulonglong2, __numeric_limits_type::__integral>
    : public __numeric_limits_vector_X2_impl<ulonglong2, long long unsigned, __numeric_limits_type::__integral>
{};

/***********************************************************************************************************************
 * Vector X4 numeric_limits
 **********************************************************************************************************************/

template <typename _VecTp, typename _Tp, __numeric_limits_type __type>
class __numeric_limits_vector_X4_impl : public __numeric_limits_impl<_Tp, __type>
{
public:
  using type = _VecTp;

  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type min() noexcept
  {
    return _VecTp{__numeric_limits_impl<_Tp>::min(),
                  __numeric_limits_impl<_Tp>::min(),
                  __numeric_limits_impl<_Tp>::min(),
                  __numeric_limits_impl<_Tp>::min()};
  }

  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type max() noexcept
  {
    return _VecTp{__numeric_limits_impl<_Tp>::max(),
                  __numeric_limits_impl<_Tp>::max(),
                  __numeric_limits_impl<_Tp>::max(),
                  __numeric_limits_impl<_Tp>::max()};
  }

  _LIBCUDACXX_HIDE_FROM_ABI static constexpr type lowest() noexcept
  {
    return _VecTp{__numeric_limits_impl<_Tp>::lowest(),
                  __numeric_limits_impl<_Tp>::lowest(),
                  __numeric_limits_impl<_Tp>::lowest(),
                  __numeric_limits_impl<_Tp>::lowest()};
  }
};

template <>
class __numeric_limits_impl<char4, __numeric_limits_type::__integral>
    : public __numeric_limits_vector_X4_impl<char4, signed char, __numeric_limits_type::__integral>
{};

template <>
class __numeric_limits_impl<uchar4, __numeric_limits_type::__integral>
    : public __numeric_limits_vector_X4_impl<uchar2, unsigned char, __numeric_limits_type::__integral>
{};

template <>
class __numeric_limits_impl<short4, __numeric_limits_type::__integral>
    : public __numeric_limits_vector_X4_impl<short4, signed short, __numeric_limits_type::__integral>
{};

template <>
class __numeric_limits_impl<ushort4, __numeric_limits_type::__integral>
    : public __numeric_limits_vector_X4_impl<ushort4, unsigned short, __numeric_limits_type::__integral>
{};

template <>
class __numeric_limits_impl<int4, __numeric_limits_type::__integral>
    : public __numeric_limits_vector_X4_impl<int4, int, __numeric_limits_type::__integral>
{};

template <>
class __numeric_limits_impl<uint4, __numeric_limits_type::__integral>
    : public __numeric_limits_vector_X4_impl<uint4, unsigned, __numeric_limits_type::__integral>
{};

template <>
class __numeric_limits_impl<long4, __numeric_limits_type::__integral>
    : public __numeric_limits_vector_X4_impl<long4, long, __numeric_limits_type::__integral>
{};

template <>
class __numeric_limits_impl<ulong4, __numeric_limits_type::__integral>
    : public __numeric_limits_vector_X4_impl<ulong4, long unsigned, __numeric_limits_type::__integral>
{};

template <>
class __numeric_limits_impl<longlong4, __numeric_limits_type::__integral>
    : public __numeric_limits_vector_X4_impl<longlong4, long long, __numeric_limits_type::__integral>
{};

template <>
class __numeric_limits_impl<ulonglong4, __numeric_limits_type::__integral>
    : public __numeric_limits_vector_X4_impl<ulonglong4, long long unsigned, __numeric_limits_type::__integral>
{};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___LIMITS_NUMERIC_LIMITS_VECTOR_H
