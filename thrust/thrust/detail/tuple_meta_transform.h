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

#include <thrust/type_traits/integer_sequence.h>

#include <cuda/std/tuple>

THRUST_NAMESPACE_BEGIN

namespace detail
{
// introduce an intermediate type tuple_meta_transform_WAR_NVCC
// rather than directly specializing tuple_meta_transform with
// default argument IndexSequence = thrust::make_index_sequence<cuda::std::tuple_size<Tuple>::value>
// to workaround nvcc 11.0 compiler bug
template <typename Tuple, template <typename> class UnaryMetaFunction, typename IndexSequence>
struct tuple_meta_transform_WAR_NVCC;

template <typename Tuple, template <typename> class UnaryMetaFunction, size_t... Is>
struct tuple_meta_transform_WAR_NVCC<Tuple, UnaryMetaFunction, thrust::index_sequence<Is...>>
{
  using type =
    ::cuda::std::tuple<typename UnaryMetaFunction<typename ::cuda::std::tuple_element<Is, Tuple>::type>::type...>;
};

template <typename Tuple, template <typename> class UnaryMetaFunction>
struct tuple_meta_transform
{
  using type =
    typename tuple_meta_transform_WAR_NVCC<Tuple,
                                           UnaryMetaFunction,
                                           thrust::make_index_sequence<::cuda::std::tuple_size<Tuple>::value>>::type;
};
} // namespace detail

THRUST_NAMESPACE_END
