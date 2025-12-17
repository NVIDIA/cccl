/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

#include <thrust/detail/allocator/malloc_allocator.h>
#include <thrust/detail/allocator/tagged_allocator.h>
#include <thrust/detail/malloc_and_free.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/system/detail/bad_alloc.h>
#include <thrust/system/detail/generic/select_system.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{
template <typename T, typename System, typename Pointer>
class malloc_allocator : public tagged_allocator<T, System, Pointer>
{
private:
  using super_t = tagged_allocator<T, System, Pointer>;

public:
  using pointer   = typename super_t::pointer;
  using size_type = typename super_t::size_type;

  pointer allocate(size_type cnt)
  {
    using thrust::system::detail::generic::select_system;

    // XXX should use a hypothetical thrust::static_pointer_cast here
    System system;

    pointer result = thrust::malloc<T>(select_system(system), cnt);

    if (result.get() == 0)
    {
      throw thrust::system::detail::bad_alloc("malloc_allocator::allocate: malloc failed");
    } // end if

    return result;
  }

  void deallocate(pointer p, size_type n) noexcept
  {
    using thrust::system::detail::generic::select_system;

    System system;
    thrust::free(select_system(system), p);
  }
};
} // namespace detail
THRUST_NAMESPACE_END
