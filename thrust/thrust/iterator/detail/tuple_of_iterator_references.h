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

#include <cuda/std/type_traits>
#include <cuda/std/tuple>

#include <thrust/tuple.h>
#include <thrust/pair.h>
#include <thrust/detail/reference_forward_declaration.h>
#include <thrust/detail/raw_reference_cast.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{

template<
  typename... Ts
>
  class tuple_of_iterator_references : public thrust::tuple<Ts...>
{
  public:
    using super_t = thrust::tuple<Ts...>;
    using super_t::super_t;

    inline __host__ __device__
    tuple_of_iterator_references()
      : super_t()
    {}

    // allow implicit construction from tuple<refs>
    inline __host__ __device__
    tuple_of_iterator_references(const super_t& other)
      : super_t(other)
    {}

    inline __host__ __device__
    tuple_of_iterator_references(super_t&& other)
      : super_t(::cuda::std::move(other))
    {}

    // allow assignment from tuples
    // XXX might be worthwhile to guard this with an enable_if is_assignable
    __thrust_exec_check_disable__
    template<typename... Us>
    inline __host__ __device__
    tuple_of_iterator_references &operator=(const thrust::tuple<Us...> &other)
    {
      super_t::operator=(other);
      return *this;
    }

    // allow assignment from pairs
    // XXX might be worthwhile to guard this with an enable_if is_assignable
    __thrust_exec_check_disable__
    template<typename U1, typename U2>
    inline __host__ __device__
    tuple_of_iterator_references &operator=(const thrust::pair<U1,U2> &other)
    {
      super_t::operator=(other);
      return *this;
    }

    // allow assignment from reference<tuple>
    // XXX perhaps we should generalize to reference<T>
    //     we could captures reference<pair> this way
    __thrust_exec_check_disable__
    template<typename Pointer, typename Derived, typename... Us>
    inline __host__ __device__
    tuple_of_iterator_references&
    operator=(const thrust::reference<thrust::tuple<Us...>, Pointer, Derived> &other)
    {
      typedef thrust::tuple<Us...> tuple_type;

      // XXX perhaps this could be accelerated
      super_t::operator=(tuple_type{other});
      return *this;
    }

    template<class... Us, ::cuda::std::__enable_if_t<sizeof...(Us) == sizeof...(Ts), int> = 0>
    inline __host__ __device__
    constexpr operator thrust::tuple<Us...>() const {
      return to_tuple<Us...>(typename ::cuda::std::__make_tuple_indices<sizeof...(Ts)>::type{});
    }

    // this overload of swap() permits swapping tuple_of_iterator_references returned as temporaries from
    // iterator dereferences
    template<class... Us>
    inline __host__ __device__
    friend void swap(tuple_of_iterator_references&& x, tuple_of_iterator_references<Us...>&& y)
    {
      x.swap(y);
    }

private:
    template<class... Us, size_t... Id>
    inline __host__ __device__
    constexpr thrust::tuple<Us...> to_tuple(::cuda::std::__tuple_indices<Id...>) const {
      return {get<Id>(*this)...};
    }
};

} // end detail

THRUST_NAMESPACE_END

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// define tuple_size, tuple_element, etc.
template <class... Ts>
struct tuple_size<THRUST_NS_QUALIFIER::detail::tuple_of_iterator_references<Ts...>>
    : integral_constant<size_t, sizeof...(Ts)>
{};

template <size_t Id, class... Ts>
struct tuple_element<Id, THRUST_NS_QUALIFIER::detail::tuple_of_iterator_references<Ts...>>
    : _CUDA_VSTD::tuple_element<Id, _CUDA_VSTD::tuple<Ts...>>
{};

_LIBCUDACXX_END_NAMESPACE_STD

// structured bindings suppport
namespace std {

template <class... Ts>
struct tuple_size<THRUST_NS_QUALIFIER::detail::tuple_of_iterator_references<Ts...>>
    : integral_constant<size_t, sizeof...(Ts)>
{};

template <size_t Id, class... Ts>
struct tuple_element<Id, THRUST_NS_QUALIFIER::detail::tuple_of_iterator_references<Ts...>>
    : _CUDA_VSTD::tuple_element<Id, _CUDA_VSTD::tuple<Ts...>>
{};

} // namespace std
