/*
 *  Copyright 2008-2021 NVIDIA Corporation
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

#include <thrust/functional.h>

THRUST_NAMESPACE_BEGIN

_CCCL_SUPPRESS_DEPRECATED_PUSH
template <typename Predicate>
_CCCL_HOST_DEVICE unary_negate<Predicate> not1(const Predicate& pred)
{
  return unary_negate<Predicate>(pred);
} // end not1()
_CCCL_SUPPRESS_DEPRECATED_POP

_CCCL_SUPPRESS_DEPRECATED_PUSH
template <typename BinaryPredicate>
_CCCL_HOST_DEVICE binary_negate<BinaryPredicate> not2(const BinaryPredicate& pred)
{
  return binary_negate<BinaryPredicate>(pred);
} // end not2()
_CCCL_SUPPRESS_DEPRECATED_POP

THRUST_NAMESPACE_END
