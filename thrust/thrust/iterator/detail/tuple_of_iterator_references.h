// SPDX-FileCopyrightText: Copyright (c) 2008-2018, NVIDIA Corporation. All rights reserved.
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

#include <thrust/detail/raw_reference_cast.h>
#include <thrust/detail/reference_forward_declaration.h>

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/pair.h>
#include <cuda/std/tuple>

THRUST_NAMESPACE_BEGIN

namespace detail
{
template <typename... Ts>
class tuple_of_iterator_references;

// is_compatible_tuple_normalize:
//   device_reference<T> --> T
//   tuple_of_iterator_references<Ts...> --> tuple<Ts...>
//   T& --> T

template <typename T>
struct is_compatible_tuple_normalize
{
  using type = T;
};

template <typename... Ts>
struct is_compatible_tuple_normalize<tuple_of_iterator_references<Ts...>>
{
  using type = ::cuda::std::tuple<Ts...>;
};

template <typename T>
struct is_compatible_tuple_normalize<thrust::device_reference<T>>
{
  using type = T;
};

template <typename T>
struct is_compatible_tuple_normalize<T&>
{
  using type = T;
};

template <typename T>
using is_compatible_tuple_normalize_t = typename is_compatible_tuple_normalize<T>::type;

// is_compatible_tuple_v:
//  - checks if the tuple structure matches
//  - rather than just testing the top-level size, this handles nesting with length-1 tuples,

// is_compatible_tuple_v:
//  - case of two non-tuple types are compatible
//  - case of mixing tuples is not compatible
template <typename U, typename T>
inline constexpr bool is_compatible_tuple_v = ::cuda::std::__tuple_like<U> == ::cuda::std::__tuple_like<T>;

// is_compatible_tuple_helper_v: verifies that the outer-most tuple_size matches prior to recursing further
//  - case1: non-viable, sizes don't even match, do not recurse
template <typename U, typename T, bool TupleSizeMatches>
inline constexpr bool is_compatible_tuple_helper_v = false;

// is_compatible_tuple_helper_v: viable, sizes match, recurse further but unwrap references
template <template <class...> class Tuple1, template <class...> class Tuple2, typename... Ts, typename... Us>
inline constexpr bool is_compatible_tuple_helper_v<Tuple1<Us...>, Tuple2<Ts...>, true> =
  (is_compatible_tuple_v<is_compatible_tuple_normalize_t<Us>, is_compatible_tuple_normalize_t<Ts>> && ...);

// is_compatible_tuple_v: recurse via is_compatible_tuple_helper_v to see if the two tuples are compatible
template <template <class...> class Tuple1, template <class...> class Tuple2, typename... Ts, typename... Us>
inline constexpr bool is_compatible_tuple_v<Tuple1<Us...>, Tuple2<Ts...>> =
  is_compatible_tuple_helper_v<Tuple1<Us...>, Tuple2<Ts...>, sizeof...(Us) == sizeof...(Ts)>;

// is_compatible_tuple_v: recurse via is_compatible_tuple_helper_v to see if the two tuples are compatible
template <typename... Us, typename... Ts>
inline constexpr bool is_compatible_tuple_v<::cuda::std::tuple<Us...>, ::cuda::std::tuple<Ts...>> =
  is_compatible_tuple_helper_v<::cuda::std::tuple<Us...>, ::cuda::std::tuple<Ts...>, sizeof...(Us) == sizeof...(Ts)>;

template <class U, class T, class Enable = void>
struct maybe_unwrap_nested
{
  _CCCL_HOST_DEVICE U operator()(const T& t) const
  {
    return t;
  }
};

template <class... Us, class... Ts>
struct maybe_unwrap_nested<
  ::cuda::std::tuple<Us...>,
  tuple_of_iterator_references<Ts...>,
  ::cuda::std::enable_if_t<is_compatible_tuple_v<::cuda::std::tuple<Us...>, ::cuda::std::tuple<Ts...>>, int>>
{
  _CCCL_HOST_DEVICE ::cuda::std::tuple<Us...> operator()(const tuple_of_iterator_references<Ts...>& t) const
  {
    return t.template __to_tuple<Us...>(typename ::cuda::std::__make_tuple_indices<sizeof...(Ts)>::type{});
  }
};

template <typename... Ts>
class tuple_of_iterator_references : public ::cuda::std::tuple<Ts...>
{
public:
  using super_t = ::cuda::std::tuple<Ts...>;
  using super_t::super_t;

  tuple_of_iterator_references() = default;

  // allow implicit construction from cuda::std::tuple<refs>
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
  _CCCL_HOST_DEVICE tuple_of_iterator_references& operator=(const ::cuda::std::tuple<Us...>& other)
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
  _CCCL_HOST_DEVICE tuple_of_iterator_references&
  operator=(const reference<::cuda::std::tuple<Us...>, Pointer, Derived>& other)
  {
    using tuple_type = ::cuda::std::tuple<Us...>;

    // XXX perhaps this could be accelerated
    super_t::operator=(tuple_type{other});
    return *this;
  }

  template <
    class... Us,
    ::cuda::std::enable_if_t<is_compatible_tuple_v<::cuda::std::tuple<Us...>, ::cuda::std::tuple<Ts...>>, int> = 0>
  _CCCL_HOST_DEVICE constexpr operator ::cuda::std::tuple<Us...>() const
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

  template <
    class... Us,
    size_t... Id,
    ::cuda::std::enable_if_t<is_compatible_tuple_v<::cuda::std::tuple<Us...>, ::cuda::std::tuple<Ts...>>, int> = 0>
  _CCCL_HOST_DEVICE constexpr ::cuda::std::tuple<Us...> __to_tuple(::cuda::std::__tuple_indices<Id...>) const
  {
    return {maybe_unwrap_nested<Us, Ts>{}(::cuda::std::get<Id>(*this))...};
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
