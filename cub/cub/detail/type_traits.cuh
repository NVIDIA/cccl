/******************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * Wrappers and extensions around <type_traits> utilities.
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/util_cpp_dialect.cuh>
#include <cub/util_namespace.cuh>

#include <cuda/std/__concepts/concept_macros.h> // IWYU pragma: keep
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/functional>
#include <cuda/std/mdspan>
#include <cuda/std/span>
#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN
namespace detail
{

template <typename Invokable, typename... Args>
using invoke_result_t = _CUDA_VSTD::invoke_result_t<Invokable, Args...>;

template <typename T, typename... TArgs>
inline constexpr bool is_one_of_v = (_CCCL_TRAIT(_CUDA_VSTD::is_same, T, TArgs) || ...);

template <typename T, typename V, typename = void>
struct has_binary_call_operator : _CUDA_VSTD::false_type
{};

template <typename T, typename V>
struct has_binary_call_operator<
  T,
  V,
  _CUDA_VSTD::void_t<decltype(_CUDA_VSTD::declval<T>()(_CUDA_VSTD::declval<V>(), _CUDA_VSTD::declval<V>()))>>
    : _CUDA_VSTD::true_type
{};

/***********************************************************************************************************************
 * Array-like type traits
 **********************************************************************************************************************/

template <typename T>
inline constexpr bool is_fixed_size_random_access_range_v = false;

template <typename T, size_t N>
inline constexpr bool is_fixed_size_random_access_range_v<T[N]> = true;

template <typename T, size_t N>
inline constexpr bool is_fixed_size_random_access_range_v<_CUDA_VSTD::array<T, N>> = true;

template <typename T, size_t N>
inline constexpr bool is_fixed_size_random_access_range_v<_CUDA_VSTD::span<T, N>> = N != _CUDA_VSTD::dynamic_extent;

template <typename T, typename E, typename L, typename A>
inline constexpr bool is_fixed_size_random_access_range_v<_CUDA_VSTD::mdspan<T, E, L, A>> =
  E::rank == 1 && E::rank_dynamic() == 0;

/***********************************************************************************************************************
 * static_size: a type trait that returns the number of elements in an Array-like type
 **********************************************************************************************************************/

template <typename T>
inline constexpr int static_size_v = _CUDA_VSTD::enable_if_t<_CUDA_VSTD::__always_false_v<T>>{};

template <typename T, size_t N>
inline constexpr int static_size_v<T[N]> = N;

template <typename T, size_t N>
inline constexpr int static_size_v<_CUDA_VSTD::array<T, N>> = N;

template <typename T, size_t N>
inline constexpr int static_size_v<_CUDA_VSTD::span<T, N>> =
  _CUDA_VSTD::enable_if_t<N != _CUDA_VSTD::dynamic_extent, int>{N};

template <typename T, typename E, typename L, typename A>
inline constexpr int static_size_v<_CUDA_VSTD::mdspan<T, E, L, A>> =
  _CUDA_VSTD::enable_if_t<E::rank == 1 && E::rank_dynamic() == 0, int>{E::static_extent(0)};

template <typename T>
using implicit_prom_t = decltype(+T{});

} // namespace detail

CUB_NAMESPACE_END
