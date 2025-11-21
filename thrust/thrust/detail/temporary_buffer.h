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

#include <thrust/detail/execute_with_allocator.h>
#include <thrust/detail/execution_policy.h>
#include <thrust/detail/pointer.h>
#include <thrust/detail/raw_pointer_cast.h>

#include <cuda/std/__utility/pair.h>

// Include all active backend system implementations (generic, sequential, host and device)
#include <thrust/system/detail/generic/temporary_buffer.h>
#include <thrust/system/detail/sequential/temporary_buffer.h>
#include __THRUST_HOST_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(temporary_buffer.h)
#include __THRUST_DEVICE_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(temporary_buffer.h)

// Some build systems need a hint to know which files we could include
#if 0
#  include <thrust/system/cpp/detail/temporary_buffer.h>
#  include <thrust/system/cuda/detail/temporary_buffer.h>
#  include <thrust/system/omp/detail/temporary_buffer.h>
#  include <thrust/system/tbb/detail/temporary_buffer.h>
#endif

THRUST_NAMESPACE_BEGIN
namespace detail
{
template <typename T, typename DerivedPolicy, typename Pair>
_CCCL_HOST_DEVICE ::cuda::std::pair<thrust::pointer<T, DerivedPolicy>,
                                    typename thrust::pointer<T, DerivedPolicy>::difference_type>
down_cast_pair(Pair p)
{
  // XXX should use a hypothetical thrust::static_pointer_cast here
  thrust::pointer<T, DerivedPolicy> ptr =
    thrust::pointer<T, DerivedPolicy>(static_cast<T*>(thrust::raw_pointer_cast(p.first)));

  using result_type =
    ::cuda::std::pair<thrust::pointer<T, DerivedPolicy>, typename thrust::pointer<T, DerivedPolicy>::difference_type>;
  return result_type(ptr, p.second);
} // end down_cast_pair()
} // namespace detail

_CCCL_EXEC_CHECK_DISABLE
template <typename T, typename DerivedPolicy>
_CCCL_HOST_DEVICE ::cuda::std::pair<thrust::pointer<T, DerivedPolicy>,
                                    typename thrust::pointer<T, DerivedPolicy>::difference_type>
get_temporary_buffer(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                     typename thrust::pointer<T, DerivedPolicy>::difference_type n)
{
  using thrust::detail::get_temporary_buffer; // execute_with_allocator
  using thrust::system::detail::generic::get_temporary_buffer;

  return thrust::detail::down_cast_pair<T, DerivedPolicy>(
    get_temporary_buffer<T>(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), n));
} // end get_temporary_buffer()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename Pointer>
_CCCL_HOST_DEVICE void
return_temporary_buffer(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, Pointer p, std::ptrdiff_t n)
{
  using thrust::detail::return_temporary_buffer; // execute_with_allocator
  using thrust::system::detail::generic::return_temporary_buffer;

  return return_temporary_buffer(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), p, n);
} // end return_temporary_buffer()

THRUST_NAMESPACE_END
