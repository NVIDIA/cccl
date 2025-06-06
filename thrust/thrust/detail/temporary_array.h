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

/*! \file temporary_array.h
 *  \brief Container-like class temporary storage inside algorithms.
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

THRUST_NAMESPACE_BEGIN
namespace detail
{

// Forward declare temporary_array, as it's used by the CUDA copy backend, which
// is included in contiguous_storage's definition.
template <typename T, typename System>
class temporary_array;

} // namespace detail
THRUST_NAMESPACE_END

#include <thrust/detail/allocator/no_throw_allocator.h>
#include <thrust/detail/allocator/temporary_allocator.h>
#include <thrust/detail/contiguous_storage.h>
#include <thrust/detail/memory_wrapper.h>
#include <thrust/iterator/detail/tagged_iterator.h>
#include <thrust/iterator/iterator_traits.h>

#include <cuda/std/type_traits>

THRUST_NAMESPACE_BEGIN
namespace detail
{

template <typename T, typename System>
class temporary_array : public contiguous_storage<T, no_throw_allocator<temporary_allocator<T, System>>>
{
private:
  using super_t = contiguous_storage<T, no_throw_allocator<temporary_allocator<T, System>>>;

  // to help out the constructor
  using alloc_type = no_throw_allocator<temporary_allocator<T, System>>;

public:
  using size_type = typename super_t::size_type;

  _CCCL_HOST_DEVICE temporary_array(thrust::execution_policy<System>& system);

  _CCCL_HOST_DEVICE temporary_array(thrust::execution_policy<System>& system, size_type n);

  // provide a kill-switch to explicitly avoid initialization
  _CCCL_HOST_DEVICE temporary_array(int uninit, thrust::execution_policy<System>& system, size_type n);

  template <typename InputIterator>
  _CCCL_HOST_DEVICE temporary_array(thrust::execution_policy<System>& system, InputIterator first, size_type n);

  template <typename InputIterator, typename InputSystem>
  _CCCL_HOST_DEVICE temporary_array(
    thrust::execution_policy<System>& system,
    thrust::execution_policy<InputSystem>& input_system,
    InputIterator first,
    size_type n);

  template <typename InputIterator>
  _CCCL_HOST_DEVICE temporary_array(thrust::execution_policy<System>& system, InputIterator first, InputIterator last);

  template <typename InputSystem, typename InputIterator>
  _CCCL_HOST_DEVICE temporary_array(
    thrust::execution_policy<System>& system,
    thrust::execution_policy<InputSystem>& input_system,
    InputIterator first,
    InputIterator last);

  _CCCL_HOST_DEVICE ~temporary_array();
}; // end temporary_array

// XXX eliminate this when we do ranges for real
template <typename Iterator, typename System>
class tagged_iterator_range
{
public:
  using iterator = thrust::detail::tagged_iterator<Iterator, System>;

  template <typename Ignored1, typename Ignored2>
  tagged_iterator_range(const Ignored1&, const Ignored2&, Iterator first, Iterator last)
      : m_begin(first)
      , m_end(last)
  {}

  iterator begin() const
  {
    return m_begin;
  }
  iterator end() const
  {
    return m_end;
  }

private:
  iterator m_begin, m_end;
};

// if FromSystem is convertible to ToSystem, then just make a shallow
// copy of the range. else, use a temporary_array
// note that the resulting iterator is explicitly tagged with ToSystem either way
template <typename Iterator, typename FromSystem, typename ToSystem>
struct move_to_system_base
    : public eval_if<::cuda::std::is_convertible<FromSystem, ToSystem>::value,
                     ::cuda::std::type_identity<tagged_iterator_range<Iterator, ToSystem>>,
                     ::cuda::std::type_identity<temporary_array<thrust::detail::it_value_t<Iterator>, ToSystem>>>
{};

template <typename Iterator, typename FromSystem, typename ToSystem>
class move_to_system : public move_to_system_base<Iterator, FromSystem, ToSystem>::type
{
  using super_t = typename move_to_system_base<Iterator, FromSystem, ToSystem>::type;

public:
  move_to_system(thrust::execution_policy<FromSystem>& from_system,
                 thrust::execution_policy<ToSystem>& to_system,
                 Iterator first,
                 Iterator last)
      : super_t(to_system, from_system, first, last)
  {}
};

} // namespace detail
THRUST_NAMESPACE_END

#include <thrust/detail/temporary_array.inl>
