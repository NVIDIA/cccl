/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

namespace detail
{

template <class... >
using void_t = void;

template <bool... Bs>
struct logic_helper_t;

template <bool>
struct false_t
{
  static constexpr bool value = false;
};

template <bool>
struct true_t
{
  static constexpr bool value = true;
};

template <bool... Bs>
using all_t = ::cuda::std::is_same<logic_helper_t<Bs...>, logic_helper_t<true_t<Bs>::value...>>;

template <bool... Bs>
using none_t = ::cuda::std::is_same<logic_helper_t<Bs...>, logic_helper_t<false_t<Bs>::value...>>;

// True if `T` and `U` are ordered with `operator<`
template <class T, class U, class = void>
struct ordered : ::cuda::std::false_type
{};

template <class T, class U>
struct ordered<T,
               U,
               void_t<decltype(::cuda::std::declval<T>() < ::cuda::std::declval<U>()
                               && ::cuda::std::declval<U>() < ::cuda::std::declval<T>())>> : ::cuda::std::true_type
{};

// True if `T` and `U` can be compared at compile time with `operator<`
template <class T, class U, class = void>
struct statically_ordered : ::cuda::std::false_type
{};

template <class T, class U>
struct statically_ordered<T, U, typename ::cuda::std::enable_if<true_t<T{} < U{} || U{} < T{}>::value>::type>
    : ::cuda::std::true_type
{};

// True if `T{} < U{}` is true and can be computed at compile time
template <class T, class U, class = void>
struct statically_less : ::cuda::std::false_type
{};

template <class T, class U>
struct statically_less<T, U, typename ::cuda::std::enable_if<T{} < U{}>::type> : ::cuda::std::true_type
{};

// True if `!(T{} < U{}) && !(T{} < T{})` and the expression can be computed at compile time
template <class T, class U>
struct statically_equal
    : ::cuda::std::integral_constant<
        bool,
        statically_ordered<T, U>::value && !statically_less<T, U>::value && !statically_less<U, T>::value>
{};

} // namespace detail

CUB_NAMESPACE_END
