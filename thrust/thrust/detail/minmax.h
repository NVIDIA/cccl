/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

THRUST_NAMESPACE_BEGIN

template<typename T, typename BinaryPredicate>
_CCCL_HOST_DEVICE
  T min THRUST_PREVENT_MACRO_SUBSTITUTION (const T &lhs, const T &rhs, BinaryPredicate comp)
{
  return comp(rhs, lhs) ? rhs : lhs;
} // end min()

template<typename T>
_CCCL_HOST_DEVICE
  T min THRUST_PREVENT_MACRO_SUBSTITUTION (const T &lhs, const T &rhs)
{
  return rhs < lhs ? rhs : lhs;
} // end min()

template<typename T, typename BinaryPredicate>
_CCCL_HOST_DEVICE
  T max THRUST_PREVENT_MACRO_SUBSTITUTION (const T &lhs, const T &rhs, BinaryPredicate comp)
{
  return comp(lhs,rhs) ? rhs : lhs;
} // end max()

template<typename T>
_CCCL_HOST_DEVICE
  T max THRUST_PREVENT_MACRO_SUBSTITUTION (const T &lhs, const T &rhs)
{
  return lhs < rhs ? rhs : lhs;
} // end max()

THRUST_NAMESPACE_END
