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

#include <thrust/detail/malloc_and_free_fwd.h>
#include <thrust/detail/static_assert.h>
#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/system/detail/adl/malloc_and_free.h>

#include <cuda/std/__type_traits/is_void.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{

template <typename DerivedPolicy, typename Size>
_CCCL_HOST_DEVICE void malloc(thrust::execution_policy<DerivedPolicy>&, Size)
{
  static_assert(thrust::detail::depend_on_instantiation<Size, false>::value, "unimplemented for this system");
}

template <typename T, typename DerivedPolicy>
_CCCL_HOST_DEVICE thrust::pointer<T, DerivedPolicy> malloc(thrust::execution_policy<DerivedPolicy>& exec, std::size_t n)
{
  if constexpr (::cuda::std::is_void_v<T>)
  {
    // We cannot determine sizeof(void), but if void is the target type we are allocating bytes anyway
    return pointer<void, DerivedPolicy>(thrust::malloc(exec, n).get());
  }
  else
  {
    return pointer<T, DerivedPolicy>(static_cast<T*>(thrust::malloc(exec, sizeof(T) * n).get()));
  }

} // end malloc()

template <typename DerivedPolicy, typename Pointer>
_CCCL_HOST_DEVICE void free(thrust::execution_policy<DerivedPolicy>&, Pointer)
{
  static_assert(thrust::detail::depend_on_instantiation<Pointer, false>::value, "unimplemented for this system");
}

template <typename DerivedPolicy, typename Pointer1, typename Pointer2>
_CCCL_HOST_DEVICE void assign_value(thrust::execution_policy<DerivedPolicy>&, Pointer1, Pointer2)
{
  static_assert(thrust::detail::depend_on_instantiation<Pointer1, false>::value, "unimplemented for this system");
}

template <typename DerivedPolicy, typename Pointer>
_CCCL_HOST_DEVICE void get_value(thrust::execution_policy<DerivedPolicy>&, Pointer)
{
  static_assert(thrust::detail::depend_on_instantiation<Pointer, false>::value, "unimplemented for this system");
}

template <typename DerivedPolicy, typename Pointer1, typename Pointer2>
_CCCL_HOST_DEVICE void iter_swap(thrust::execution_policy<DerivedPolicy>&, Pointer1, Pointer2)
{
  static_assert(thrust::detail::depend_on_instantiation<Pointer1, false>::value, "unimplemented for this system");
}

} // namespace generic
} // namespace detail
} // namespace system
THRUST_NAMESPACE_END
