// SPDX-FileCopyrightText: Copyright (c) 2008-2021, NVIDIA Corporation. All rights reserved.
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

#include <thrust/detail/temporary_array.h>
#include <thrust/detail/type_traits.h>
#include <thrust/system/detail/generic/select_system.h>

#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__type_traits/is_trivially_copy_constructible.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{
template <typename T, typename System>
_CCCL_HOST_DEVICE temporary_array<T, System>::temporary_array(thrust::execution_policy<System>& system)
    : super_t(alloc_type(temporary_allocator<T, System>(system)))
{} // end temporary_array::temporary_array()

template <typename T, typename System>
_CCCL_HOST_DEVICE temporary_array<T, System>::temporary_array(thrust::execution_policy<System>& system, size_type n)
    : super_t(n, alloc_type(temporary_allocator<T, System>(system)))
{
  if constexpr (!::cuda::std::is_trivially_copy_constructible_v<T>)
  {
    super_t::value_initialize_n(super_t::begin(), n);
  }
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
  super_t::allocate(::cuda::std::distance(first, last));

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
  super_t::allocate(::cuda::std::distance(first, last));

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
