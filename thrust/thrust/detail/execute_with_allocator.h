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

#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/detail/execute_with_allocator_fwd.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/detail/type_traits/pointer_traits.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__utility/pair.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{
template <typename T, typename Allocator, template <typename> class BaseSystem>
_CCCL_HOST ::cuda::std::pair<T*, std::ptrdiff_t>
get_temporary_buffer(thrust::detail::execute_with_allocator<Allocator, BaseSystem>& system, std::ptrdiff_t n)
{
  using naked_allocator = ::cuda::std::remove_reference_t<Allocator>;
  using alloc_traits    = typename thrust::detail::allocator_traits<naked_allocator>;
  using void_pointer    = typename alloc_traits::void_pointer;
  using size_type       = typename alloc_traits::size_type;
  using value_type      = typename alloc_traits::value_type;

  // How many elements of type value_type do we need to accommodate n elements
  // of type T?
  const size_type num_elements = static_cast<size_type>(::cuda::ceil_div(sizeof(T) * n, sizeof(value_type)));

  void_pointer ptr = alloc_traits::allocate(system.get_allocator(), num_elements);

  // Return the pointer and the number of elements of type T allocated.
  return ::cuda::std::make_pair(thrust::reinterpret_pointer_cast<T*>(ptr), n);
}

template <typename Pointer, typename Allocator, template <typename> class BaseSystem>
_CCCL_HOST void return_temporary_buffer(
  thrust::detail::execute_with_allocator<Allocator, BaseSystem>& system, Pointer p, std::ptrdiff_t n)
{
  using naked_allocator = ::cuda::std::remove_reference_t<Allocator>;
  using alloc_traits    = typename thrust::detail::allocator_traits<naked_allocator>;
  using pointer         = typename alloc_traits::pointer;
  using size_type       = typename alloc_traits::size_type;
  using value_type      = typename alloc_traits::value_type;
  using T               = typename thrust::detail::pointer_traits<Pointer>::element_type;

  size_type num_elements = ::cuda::ceil_div(sizeof(T) * n, sizeof(value_type));

  pointer to_ptr = thrust::reinterpret_pointer_cast<pointer>(p);
  alloc_traits::deallocate(system.get_allocator(), to_ptr, num_elements);
}
} // namespace detail

THRUST_NAMESPACE_END
