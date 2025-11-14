/*
 *  Copyright 2008-2018 NVIDIA Corporation
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

#include <thrust/detail/raw_reference_cast.h>
#include <thrust/detail/reference_forward_declaration.h>
#include <thrust/tuple.h>

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/pair.h>
#include <cuda/std/tuple>

THRUST_NAMESPACE_BEGIN

namespace detail
{
template <typename... Ts>
class tuple_of_iterator_references;

template <class U, class T>
struct maybe_unwrap_nested
{
  _CCCL_HOST_DEVICE U operator()(const T& t) const
  {
    return t;
  }
};

template <class... Us, class... Ts>
struct maybe_unwrap_nested<tuple<Us...>, tuple_of_iterator_references<Ts...>>
{
  _CCCL_HOST_DEVICE tuple<Us...> operator()(const tuple_of_iterator_references<Ts...>& t) const
  {
    return t.template __to_tuple<Us...>(typename ::cuda::std::__make_tuple_indices<sizeof...(Ts)>::type{});
  }
};

template <typename... Ts>
class tuple_of_iterator_references : public tuple<Ts...>
{
public:
  using super_t = tuple<Ts...>;
  using super_t::super_t;

  tuple_of_iterator_references() = default;

  // allow implicit construction from tuple<refs>
  _CCCL_HOST_DEVICE tuple_of_iterator_references(const super_t& other)
      : super_t(other)
  {}

  _CCCL_HOST_DEVICE tuple_of_iterator_references(super_t&& other)
      : super_t(::cuda::std::move(other))
  {}

  // allow assignment from tuples
  // XXX might be worthwhile to guard this with an enable_if is_assignable
  _CCCL_EXEC_CHECK_DISABLE
  template <typename... Us>
  _CCCL_HOST_DEVICE tuple_of_iterator_references& operator=(const tuple<Us...>& other)
  {
    super_t::operator=(other);
    return *this;
  }

  // allow assignment from pairs
  // XXX might be worthwhile to guard this with an enable_if is_assignable
  _CCCL_EXEC_CHECK_DISABLE
  template <typename U1, typename U2>
  _CCCL_HOST_DEVICE tuple_of_iterator_references& operator=(const ::cuda::std::pair<U1, U2>& other)
  {
    super_t::operator=(other);
    return *this;
  }

  // allow assignment from reference<tuple>
  // XXX perhaps we should generalize to reference<T> we could captures reference<pair> this way
  _CCCL_EXEC_CHECK_DISABLE
  template <typename Pointer, typename Derived, typename... Us>
  _CCCL_HOST_DEVICE tuple_of_iterator_references& operator=(const reference<tuple<Us...>, Pointer, Derived>& other)
  {
    using tuple_type = tuple<Us...>;

    // XXX perhaps this could be accelerated
    super_t::operator=(tuple_type{other});
    return *this;
  }

  template <class... Us, ::cuda::std::enable_if_t<sizeof...(Us) == sizeof...(Ts), int> = 0>
  _CCCL_HOST_DEVICE constexpr operator tuple<Us...>() const
  {
    return __to_tuple<Us...>(typename ::cuda::std::__make_tuple_indices<sizeof...(Ts)>::type{});
  }

  // this overload of swap() permits swapping tuple_of_iterator_references returned as temporaries from
  // iterator dereferences
  template <class... Us>
  _CCCL_HOST_DEVICE friend void swap(tuple_of_iterator_references&& x, tuple_of_iterator_references<Us...>&& y)
  {
    x.swap(y);
  }

  template <class... Us, size_t... Id>
  _CCCL_HOST_DEVICE constexpr tuple<Us...> __to_tuple(::cuda::std::__tuple_indices<Id...>) const
  {
    return {maybe_unwrap_nested<Us, Ts>{}(get<Id>(*this))...};
  }
};
} // namespace detail

THRUST_NAMESPACE_END

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class... Ts>
inline constexpr bool
  __is_tuple_of_iterator_references_v<THRUST_NS_QUALIFIER::detail::tuple_of_iterator_references<Ts...>> = true;

// define tuple_size, tuple_element, etc.
template <class... Ts>
struct tuple_size<THRUST_NS_QUALIFIER::detail::tuple_of_iterator_references<Ts...>>
    : integral_constant<size_t, sizeof...(Ts)>
{};

template <size_t Id, class... Ts>
struct tuple_element<Id, THRUST_NS_QUALIFIER::detail::tuple_of_iterator_references<Ts...>>
    : ::cuda::std::tuple_element<Id, ::cuda::std::tuple<Ts...>>
{};

_CCCL_END_NAMESPACE_CUDA_STD

// structured bindings support
#if !_CCCL_COMPILER(NVRTC)
namespace std
{
template <class... Ts>
struct tuple_size<THRUST_NS_QUALIFIER::detail::tuple_of_iterator_references<Ts...>>
    : integral_constant<size_t, sizeof...(Ts)>
{};

template <size_t Id, class... Ts>
struct tuple_element<Id, THRUST_NS_QUALIFIER::detail::tuple_of_iterator_references<Ts...>>
    : ::cuda::std::tuple_element<Id, ::cuda::std::tuple<Ts...>>
{};
} // namespace std
#endif // !_CCCL_COMPILER(NVRTC)
