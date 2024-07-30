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

_CCCL_IMPLICIT_SYSTEM_HEADER
#include <thrust/detail/pointer.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/tagged_iterator.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{

// we can retag an iterator if FromTag converts to ToTag
// or vice versa
template <typename FromTag, typename ToTag>
struct is_retaggable
    : integral_constant<bool,
                        (::cuda::std::is_convertible<FromTag, ToTag>::value
                         || ::cuda::std::is_convertible<ToTag, FromTag>::value)>
{};

template <typename FromTag, typename ToTag, typename Result>
struct enable_if_retaggable : ::cuda::std::enable_if<is_retaggable<FromTag, ToTag>::value, Result>
{}; // end enable_if_retaggable

} // namespace detail

template <typename Tag, typename Iterator>
_CCCL_HOST_DEVICE thrust::detail::tagged_iterator<Iterator, Tag> reinterpret_tag(Iterator iter)
{
  return thrust::detail::tagged_iterator<Iterator, Tag>(iter);
} // end reinterpret_tag()

// specialization for raw pointer
template <typename Tag, typename T>
_CCCL_HOST_DEVICE thrust::pointer<T, Tag> reinterpret_tag(T* ptr)
{
  return thrust::pointer<T, Tag>(ptr);
} // end reinterpret_tag()

// specialization for thrust::pointer
template <typename Tag, typename T, typename OtherTag, typename Reference, typename Derived>
_CCCL_HOST_DEVICE thrust::pointer<T, Tag> reinterpret_tag(thrust::pointer<T, OtherTag, Reference, Derived> ptr)
{
  return reinterpret_tag<Tag>(ptr.get());
} // end reinterpret_tag()

// avoid deeply-nested tagged_iterator
template <typename Tag, typename BaseIterator, typename OtherTag>
_CCCL_HOST_DEVICE thrust::detail::tagged_iterator<BaseIterator, Tag>
reinterpret_tag(thrust::detail::tagged_iterator<BaseIterator, OtherTag> iter)
{
  return reinterpret_tag<Tag>(iter.base());
} // end reinterpret_tag()

template <typename Tag, typename Iterator>
_CCCL_HOST_DEVICE
typename thrust::detail::enable_if_retaggable<typename thrust::iterator_system<Iterator>::type,
                                              Tag,
                                              thrust::detail::tagged_iterator<Iterator, Tag>>::type
retag(Iterator iter)
{
  return reinterpret_tag<Tag>(iter);
} // end retag()

// specialization for raw pointer
template <typename Tag, typename T>
_CCCL_HOST_DEVICE typename thrust::detail::
  enable_if_retaggable<typename thrust::iterator_system<T*>::type, Tag, thrust::pointer<T, Tag>>::type
  retag(T* ptr)
{
  return reinterpret_tag<Tag>(ptr);
} // end retag()

// specialization for thrust::pointer
template <typename Tag, typename T, typename OtherTag>
_CCCL_HOST_DEVICE typename thrust::detail::enable_if_retaggable<OtherTag, Tag, thrust::pointer<T, Tag>>::type
retag(thrust::pointer<T, OtherTag> ptr)
{
  return reinterpret_tag<Tag>(ptr);
} // end retag()

// avoid deeply-nested tagged_iterator
template <typename Tag, typename BaseIterator, typename OtherTag>
_CCCL_HOST_DEVICE
typename thrust::detail::enable_if_retaggable<OtherTag, Tag, thrust::detail::tagged_iterator<BaseIterator, Tag>>::type
retag(thrust::detail::tagged_iterator<BaseIterator, OtherTag> iter)
{
  return reinterpret_tag<Tag>(iter);
} // end retag()

THRUST_NAMESPACE_END
