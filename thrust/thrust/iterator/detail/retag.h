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
#include <thrust/detail/pointer.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/tagged_iterator.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{
// we can retag an iterator if FromTag converts to ToTag or vice versa
template <typename FromTag, typename ToTag>
inline constexpr bool is_retaggable =
  ::cuda::std::is_convertible_v<FromTag, ToTag> || ::cuda::std::is_convertible_v<ToTag, FromTag>;
} // namespace detail

template <typename Tag, typename Iterator>
_CCCL_HOST_DEVICE detail::tagged_iterator<Iterator, Tag> reinterpret_tag(Iterator iter)
{
  return detail::tagged_iterator<Iterator, Tag>(iter);
}

// specialization for raw pointer
template <typename Tag, typename T>
_CCCL_HOST_DEVICE pointer<T, Tag> reinterpret_tag(T* ptr)
{
  return pointer<T, Tag>(ptr);
}

// specialization for pointer
template <typename Tag, typename T, typename OtherTag, typename Reference, typename Derived>
_CCCL_HOST_DEVICE pointer<T, Tag> reinterpret_tag(pointer<T, OtherTag, Reference, Derived> ptr)
{
  return reinterpret_tag<Tag>(ptr.get());
}

// avoid deeply-nested tagged_iterator
template <typename Tag, typename BaseIterator, typename OtherTag>
_CCCL_HOST_DEVICE detail::tagged_iterator<BaseIterator, Tag>
reinterpret_tag(detail::tagged_iterator<BaseIterator, OtherTag> iter)
{
  return reinterpret_tag<Tag>(iter.base());
}

template <typename Tag,
          typename Iterator,
          ::cuda::std::enable_if_t<detail::is_retaggable<iterator_system_t<Iterator>, Tag>, int> = 0>
_CCCL_HOST_DEVICE detail::tagged_iterator<Iterator, Tag> retag(Iterator iter)
{
  return reinterpret_tag<Tag>(iter);
}

// specialization for raw pointer
template <typename Tag, typename T, ::cuda::std::enable_if_t<detail::is_retaggable<iterator_system_t<T*>, Tag>, int> = 0>
_CCCL_HOST_DEVICE pointer<T, Tag> retag(T* ptr)
{
  return reinterpret_tag<Tag>(ptr);
}

// specialization for pointer
template <typename Tag,
          typename T,
          typename OtherTag,
          ::cuda::std::enable_if_t<detail::is_retaggable<OtherTag, Tag>, int> = 0>
_CCCL_HOST_DEVICE pointer<T, Tag> retag(pointer<T, OtherTag> ptr)
{
  return reinterpret_tag<Tag>(ptr);
}

// avoid deeply-nested tagged_iterator
template <typename Tag,
          typename BaseIterator,
          typename OtherTag,
          ::cuda::std::enable_if_t<detail::is_retaggable<OtherTag, Tag>, int> = 0>
_CCCL_HOST_DEVICE detail::tagged_iterator<BaseIterator, Tag> retag(detail::tagged_iterator<BaseIterator, OtherTag> iter)
{
  return reinterpret_tag<Tag>(iter);
}

THRUST_NAMESPACE_END
