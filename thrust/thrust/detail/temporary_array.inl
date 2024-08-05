/*
 *  Copyright 2008-2021 NVIDIA Corporation
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

#include <thrust/detail/temporary_array.h>
#include <thrust/detail/type_traits.h>
#include <thrust/distance.h>
#include <thrust/system/detail/generic/select_system.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{
namespace temporary_array_detail
{

template <typename T>
struct avoid_initialization : ::cuda::std::is_trivially_copy_constructible<T>
{};

template <typename T, typename TemporaryArray, typename Size>
_CCCL_HOST_DEVICE ::cuda::std::__enable_if_t<avoid_initialization<T>::value> construct_values(TemporaryArray&, Size)
{
  // avoid the overhead of initialization
} // end construct_values()

template <typename T, typename TemporaryArray, typename Size>
_CCCL_HOST_DEVICE ::cuda::std::__enable_if_t<!avoid_initialization<T>::value> construct_values(TemporaryArray& a, Size n)
{
  a.value_initialize_n(a.begin(), n);
} // end construct_values()

} // namespace temporary_array_detail

template <typename T, typename System>
_CCCL_HOST_DEVICE temporary_array<T, System>::temporary_array(thrust::execution_policy<System>& system)
    : super_t(alloc_type(temporary_allocator<T, System>(system)))
{} // end temporary_array::temporary_array()

template <typename T, typename System>
_CCCL_HOST_DEVICE temporary_array<T, System>::temporary_array(thrust::execution_policy<System>& system, size_type n)
    : super_t(n, alloc_type(temporary_allocator<T, System>(system)))
{
  temporary_array_detail::construct_values<T>(*this, n);
} // end temporary_array::temporary_array()

template <typename T, typename System>
_CCCL_HOST_DEVICE temporary_array<T, System>::temporary_array(int, thrust::execution_policy<System>& system, size_type n)
    : super_t(n, alloc_type(temporary_allocator<T, System>(system)))
{
  // avoid initialization
  ;
} // end temporary_array::temporary_array()

template <typename T, typename System>
template <typename InputIterator>
_CCCL_HOST_DEVICE
temporary_array<T, System>::temporary_array(thrust::execution_policy<System>& system, InputIterator first, size_type n)
    : super_t(alloc_type(temporary_allocator<T, System>(system)))
{
  super_t::allocate(n);

  super_t::uninitialized_copy_n(system, first, n, super_t::begin());
} // end temporary_array::temporary_array()

template <typename T, typename System>
template <typename InputIterator, typename InputSystem>
_CCCL_HOST_DEVICE temporary_array<T, System>::temporary_array(
  thrust::execution_policy<System>& system,
  thrust::execution_policy<InputSystem>& input_system,
  InputIterator first,
  size_type n)
    : super_t(alloc_type(temporary_allocator<T, System>(system)))
{
  super_t::allocate(n);

  super_t::uninitialized_copy_n(input_system, first, n, super_t::begin());
} // end temporary_array::temporary_array()

template <typename T, typename System>
template <typename InputIterator>
_CCCL_HOST_DEVICE temporary_array<T, System>::temporary_array(
  thrust::execution_policy<System>& system, InputIterator first, InputIterator last)
    : super_t(alloc_type(temporary_allocator<T, System>(system)))
{
  super_t::allocate(thrust::distance(first, last));

  super_t::uninitialized_copy(system, first, last, super_t::begin());
} // end temporary_array::temporary_array()

template <typename T, typename System>
template <typename InputSystem, typename InputIterator>
_CCCL_HOST_DEVICE temporary_array<T, System>::temporary_array(
  thrust::execution_policy<System>& system,
  thrust::execution_policy<InputSystem>& input_system,
  InputIterator first,
  InputIterator last)
    : super_t(alloc_type(temporary_allocator<T, System>(system)))
{
  super_t::allocate(thrust::distance(first, last));

  super_t::uninitialized_copy(input_system, first, last, super_t::begin());
} // end temporary_array::temporary_array()

template <typename T, typename System>
_CCCL_HOST_DEVICE temporary_array<T, System>::~temporary_array()
{
  // note that super_t::destroy will ignore trivial destructors automatically
  super_t::destroy(super_t::begin(), super_t::end());
} // end temporary_array::~temporary_array()

} // namespace detail

THRUST_NAMESPACE_END
