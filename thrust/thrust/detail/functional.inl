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

namespace detail
{

template <typename Operation>
struct unary_traits_imp;

template <typename Operation>
struct unary_traits_imp<Operation*>
{
  using function_type = Operation;
  using param_type    = const function_type&;
  using result_type   = typename Operation::result_type;
  using argument_type = typename Operation::argument_type;
}; // end unary_traits_imp

template <typename Result, typename Argument>
struct unary_traits_imp<Result (*)(Argument)>
{
  using function_type = Result (*)(Argument);
  using param_type    = Result (*)(Argument);
  using result_type   = Result;
  using argument_type = Argument;
}; // end unary_traits_imp

template <typename Operation>
struct binary_traits_imp;

template <typename Operation>
struct binary_traits_imp<Operation*>
{
  using function_type        = Operation;
  using param_type           = const function_type&;
  using result_type          = typename Operation::result_type;
  using first_argument_type  = typename Operation::first_argument_type;
  using second_argument_type = typename Operation::second_argument_type;
}; // end binary_traits_imp

template <typename Result, typename Argument1, typename Argument2>
struct binary_traits_imp<Result (*)(Argument1, Argument2)>
{
  using function_type        = Result (*)(Argument1, Argument2);
  using param_type           = Result (*)(Argument1, Argument2);
  using result_type          = Result;
  using first_argument_type  = Argument1;
  using second_argument_type = Argument2;
}; // end binary_traits_imp

} // namespace detail

_CCCL_SUPPRESS_DEPRECATED_PUSH
template <typename Operation>
struct unary_traits
{
  using function_type = typename detail::unary_traits_imp<Operation*>::function_type;
  using param_type    = typename detail::unary_traits_imp<Operation*>::param_type;
  using result_type   = typename detail::unary_traits_imp<Operation*>::result_type;
  using argument_type = typename detail::unary_traits_imp<Operation*>::argument_type;
}; // end unary_traits
_CCCL_SUPPRESS_DEPRECATED_POP

_CCCL_SUPPRESS_DEPRECATED_PUSH
template <typename Result, typename Argument>
struct unary_traits<Result (*)(Argument)>
{
  using function_type = Result (*)(Argument);
  using param_type    = Result (*)(Argument);
  using result_type   = Result;
  using argument_type = Argument;
}; // end unary_traits
_CCCL_SUPPRESS_DEPRECATED_POP

_CCCL_SUPPRESS_DEPRECATED_PUSH
template <typename Operation>
struct binary_traits
{
  using function_type        = typename detail::binary_traits_imp<Operation*>::function_type;
  using param_type           = typename detail::binary_traits_imp<Operation*>::param_type;
  using result_type          = typename detail::binary_traits_imp<Operation*>::result_type;
  using first_argument_type  = typename detail::binary_traits_imp<Operation*>::first_argument_type;
  using second_argument_type = typename detail::binary_traits_imp<Operation*>::second_argument_type;
}; // end binary_traits
_CCCL_SUPPRESS_DEPRECATED_POP

_CCCL_SUPPRESS_DEPRECATED_PUSH
template <typename Result, typename Argument1, typename Argument2>
struct binary_traits<Result (*)(Argument1, Argument2)>
{
  using function_type        = Result (*)(Argument1, Argument2);
  using param_type           = Result (*)(Argument1, Argument2);
  using result_type          = Result;
  using first_argument_type  = Argument1;
  using second_argument_type = Argument2;
}; // end binary_traits
_CCCL_SUPPRESS_DEPRECATED_POP

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
