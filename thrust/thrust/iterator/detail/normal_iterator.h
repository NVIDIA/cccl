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

//! \file normal_iterator.h
//! \brief Defines the interface to an iterator class which adapts a pointer type.

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
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/type_traits/is_contiguous_iterator.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{
template <typename Pointer>
class normal_iterator : public iterator_adaptor<normal_iterator<Pointer>, Pointer>
{
  using super_t = iterator_adaptor<normal_iterator<Pointer>, Pointer>;

public:
  _CCCL_HIDE_FROM_ABI normal_iterator() = default;

  _CCCL_HOST_DEVICE normal_iterator(Pointer p)
      : super_t(p)
  {}

  template <typename OtherPointer>
  _CCCL_HOST_DEVICE
  normal_iterator(const normal_iterator<OtherPointer>& other, enable_if_convertible_t<OtherPointer, Pointer>* = 0)
      : super_t(other.base())
  {}
};

template <typename Pointer>
_CCCL_HOST_DEVICE normal_iterator<Pointer> make_normal_iterator(Pointer ptr)
{
  return normal_iterator<Pointer>(ptr);
}
} // namespace detail

template <typename T>
struct proclaim_contiguous_iterator<detail::normal_iterator<T>> : true_type
{};

THRUST_NAMESPACE_END
