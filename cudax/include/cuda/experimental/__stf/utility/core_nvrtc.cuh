//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/** @file
 * @brief Widely used artifacts used by most of the library (subset that is compatible with nvrtc)
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/__stf/utility/cuda_attributes.cuh>
#include <cuda/experimental/__stf/utility/each.cuh>
#include <cuda/experimental/__stf/utility/mv.cuh>

namespace cuda::experimental::stf
{

template <typename T, typename... P>
constexpr auto cuda_tuple_prepend(T&& prefix, ::cuda::std::tuple<P...> tuple)
{
  return ::cuda::std::apply(
    [&](auto&&... p) {
      return ::cuda::std::tuple(::std::forward<T>(prefix), ::std::forward<decltype(p)>(p)...);
    },
    mv(tuple));
}

namespace reserved
{

inline constexpr auto make_cuda_tuple()
{
  return ::cuda::std::tuple<>();
}

template <typename T, typename... P>
constexpr auto make_cuda_tuple([[maybe_unused]] T t, P... p)
{
  if constexpr (::cuda::std::is_same_v<const T, const decltype(::cuda::std::ignore)>)
  {
    // Recurse skipping the first parameter
    return make_cuda_tuple(mv(p)...);
  }
  else
  {
    // Keep first parameter, concatenate with recursive call
    return cuda_tuple_prepend(mv(t), make_cuda_tuple(mv(p)...));
  }
}

} // end namespace reserved

template <size_t n, typename F, size_t... i>
constexpr auto make_cuda_tuple_indexwise(F&& f, ::cuda::std::index_sequence<i...> = ::cuda::std::index_sequence<>())
{
  if constexpr (sizeof...(i) != n)
  {
    return make_cuda_tuple_indexwise<n>(::std::forward<F>(f), ::cuda::std::make_index_sequence<n>());
  }
  else
  {
    return reserved::make_cuda_tuple(f(::cuda::std::integral_constant<size_t, i>())...);
  }
}

} // namespace cuda::experimental::stf
