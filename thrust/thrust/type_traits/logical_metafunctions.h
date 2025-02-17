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

#include <cuda/std/__type_traits/conjunction.h>
#include <cuda/std/__type_traits/disjunction.h>
#include <cuda/std/__type_traits/negation.h>

THRUST_NAMESPACE_BEGIN

using ::cuda::std::conjunction;
using ::cuda::std::disjunction;
using ::cuda::std::negation;
#if !defined(_CCCL_NO_VARIABLE_TEMPLATES)
using ::cuda::std::conjunction_v;
using ::cuda::std::disjunction_v;
using ::cuda::std::negation_v;
#endif // !_CCCL_NO_VARIABLE_TEMPLATES

template <bool... Bs>
using conjunction_value CCCL_DEPRECATED_BECAUSE("Use: cuda::std::bool_constant<(Bs && ...)>") =
  conjunction<::cuda::std::bool_constant<Bs>...>;

template <bool... Bs>
using disjunction_value CCCL_DEPRECATED_BECAUSE("Use: cuda::std::bool_constant<(Bs || ...)>") =
  disjunction<::cuda::std::bool_constant<Bs>...>;

template <bool B>
using negation_value CCCL_DEPRECATED_BECAUSE("Use cuda::std::bool_constant<!B>") = ::cuda::std::bool_constant<!B>;

#if _CCCL_STD_VER >= 2014
_CCCL_SUPPRESS_DEPRECATED_PUSH
template <bool... Bs>
constexpr bool
  conjunction_value_v CCCL_DEPRECATED_BECAUSE("Use a fold expression: Bs && ...") = conjunction_value<Bs...>::value;

template <bool... Bs>
constexpr bool
  disjunction_value_v CCCL_DEPRECATED_BECAUSE("Use a fold expression: Bs || ...") = disjunction_value<Bs...>::value;

template <bool B>
constexpr bool negation_value_v CCCL_DEPRECATED_BECAUSE("Use a plain negation !B") = negation_value<B>::value;
_CCCL_SUPPRESS_DEPRECATED_POP
#endif

THRUST_NAMESPACE_END
