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

#include <thrust/detail/execution_policy.h>
#include <thrust/detail/malloc_and_free_fwd.h>
#include <thrust/detail/pointer.h>
#include <thrust/detail/raw_pointer_cast.h>

// Include all active backend system implementations (generic, sequential, host and device)
#include <thrust/system/detail/generic/memory.h>
#include <thrust/system/detail/sequential/malloc_and_free.h>
#include __THRUST_HOST_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(malloc_and_free.h)
#include __THRUST_DEVICE_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(malloc_and_free.h)

// Some build systems need a hint to know which files we could include
#if 0
#  include <thrust/system/cpp/detail/malloc_and_free.h>
#  include <thrust/system/cuda/detail/malloc_and_free.h>
#  include <thrust/system/omp/detail/malloc_and_free.h>
#  include <thrust/system/tbb/detail/malloc_and_free.h>
#endif

THRUST_NAMESPACE_BEGIN

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy>
_CCCL_HOST_DEVICE pointer<void, DerivedPolicy>
malloc(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, std::size_t n)
{
  using thrust::system::detail::generic::malloc;

  // XXX should use a hypothetical thrust::static_pointer_cast here
  void* raw_ptr = static_cast<void*>(
    thrust::raw_pointer_cast(malloc(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), n)));

  return pointer<void, DerivedPolicy>(raw_ptr);
}

_CCCL_EXEC_CHECK_DISABLE
template <typename T, typename DerivedPolicy>
_CCCL_HOST_DEVICE pointer<T, DerivedPolicy>
malloc(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, std::size_t n)
{
  using thrust::system::detail::generic::malloc;

  T* raw_ptr = static_cast<T*>(
    thrust::raw_pointer_cast(malloc<T>(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), n)));

  return pointer<T, DerivedPolicy>(raw_ptr);
}

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename Pointer>
_CCCL_HOST_DEVICE void free(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, Pointer ptr)
{
  using thrust::system::detail::generic::free;

  free(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), ptr);
}

// XXX consider another form of free which does not take a system argument and
// instead infers the system from the pointer

THRUST_NAMESPACE_END
