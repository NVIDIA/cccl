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

// allocator_traits::rebind_alloc and allocator::rebind_traits are from libc++,
// dual licensed under the MIT and the University of Illinois Open Source
// Licenses.

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/iterator/iterator_traits.h>

#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__memory/allocator_traits.h>
#include <cuda/std/__type_traits/add_lvalue_reference.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{
// forward declaration for has_member_system
template <typename Alloc>
struct allocator_system;

namespace allocator_traits_detail
{
template <class T, class = void>
inline constexpr bool has_system_type = false;
template <class T>
inline constexpr bool has_system_type<T, ::cuda::std::void_t<typename T::system_type>> = true;

template <class T>
struct nested_system_type
{
  using type = typename T::system_type;
};

template <class T, class = void>
inline constexpr bool has_member_system = false;
template <class T>
inline constexpr bool has_member_system<T, ::cuda::std::void_t<decltype(::cuda::std::declval<T>().system())>> =
  ::cuda::std::is_same_v<decltype(::cuda::std::declval<T>().system()), typename allocator_system<T>::type&>;

template <typename Alloc>
[[nodiscard]] _CCCL_API decltype(auto) system(Alloc& a)
{
  if constexpr (has_member_system<Alloc>)
  { // return the allocator's system
    return a.system();
  }
  else
  { // return a copy of a value-initialized system
    return typename allocator_system<Alloc>::type{};
  }
}
} // namespace allocator_traits_detail

// XXX consider moving this non-standard functionality inside allocator_traits
template <typename Alloc>
struct allocator_system
{
  // the type of the allocator's system
  using type = typename eval_if<allocator_traits_detail::has_system_type<Alloc>,
                                allocator_traits_detail::nested_system_type<Alloc>,
                                thrust::iterator_system<typename ::cuda::std::allocator_traits<Alloc>::pointer>>::type;

  // the type that get returns
  using get_result_type =
    typename eval_if<allocator_traits_detail::has_member_system<Alloc>,
                     ::cuda::std::add_lvalue_reference<type>,
                     ::cuda::std::type_identity<type>>::type;

  _CCCL_HOST_DEVICE inline static get_result_type get(Alloc& a)
  {
    return allocator_traits_detail::system(a);
  }
};
} // namespace detail
THRUST_NAMESPACE_END
