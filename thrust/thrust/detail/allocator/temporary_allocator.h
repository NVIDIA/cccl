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

#include <thrust/detail/allocator/allocator_traits.h>
#include <thrust/detail/allocator/tagged_allocator.h>
#include <thrust/detail/allocator/temporary_allocator.h>
#include <thrust/detail/execution_policy.h>
#include <thrust/detail/temporary_buffer.h>
#include <thrust/memory.h>
#include <thrust/system/detail/bad_alloc.h>

#include <cuda/std/__utility/pair.h>
#include <cuda/std/cassert>

#include <nv/target>

#if _CCCL_CUDA_COMPILATION() && _CCCL_DEVICE_COMPILATION()
#  include <thrust/system/cuda/detail/terminate.h>
#endif // _CCCL_CUDA_COMPILATION() && _CCCL_DEVICE_COMPILATION()

THRUST_NAMESPACE_BEGIN
namespace detail
{
// XXX the pointer parameter given to tagged_allocator should be related to
//     the type of the expression get_temporary_buffer(system, n).first
//     without decltype, compromise on pointer<T,System>
template <typename T, typename System>
class temporary_allocator : public thrust::detail::tagged_allocator<T, System, thrust::pointer<T, System>>
{
private:
  using super_t = thrust::detail::tagged_allocator<T, System, thrust::pointer<T, System>>;

  System& m_system;

public:
  using pointer   = typename super_t::pointer;
  using size_type = typename super_t::size_type;

  inline _CCCL_HOST_DEVICE temporary_allocator(const temporary_allocator& other)
      : super_t()
      , m_system(other.m_system)
  {}

  inline _CCCL_HOST_DEVICE explicit temporary_allocator(thrust::execution_policy<System>& system)
      : super_t()
      , m_system(thrust::detail::derived_cast(system))
  {}

  _CCCL_HOST_DEVICE pointer allocate(size_type cnt)
  {
    pointer_and_size result = thrust::get_temporary_buffer<T>(system(), cnt);

    // handle failure
    if (result.second < cnt)
    {
      // deallocate and throw
      // note that we pass cnt to deallocate, not a value derived from result.second
      deallocate(result.first, cnt);

#if _CCCL_CUDA_COMPILATION()
      NV_IF_TARGET(
        NV_IS_HOST,
        (throw thrust::system::detail::bad_alloc("temporary_buffer::allocate: get_temporary_buffer failed");),
        ( // NV_IS_DEVICE
          thrust::system::cuda::detail::terminate_with_message("temporary_buffer::allocate: "
                                                               "get_temporary_buffer failed");));
#else
      throw thrust::system::detail::bad_alloc("temporary_buffer::allocate: get_temporary_buffer failed");
#endif
    } // end if

    return result.first;
  }

  _CCCL_HOST_DEVICE void deallocate(pointer p, size_type n) noexcept
  {
    return thrust::return_temporary_buffer(system(), p, n);
  }

  _CCCL_HOST_DEVICE inline System& system()
  {
    return m_system;
  } // end system()

private:
  using pointer_and_size = ::cuda::std::pair<pointer, size_type>;
}; // end temporary_allocator
} // namespace detail
THRUST_NAMESPACE_END
