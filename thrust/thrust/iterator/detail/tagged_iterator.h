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
#include <thrust/detail/use_default.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/type_traits/is_contiguous_iterator.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{
template <typename, typename>
class tagged_iterator;

template <typename Iterator, typename Tag>
using make_tagged_iterator_base =
  iterator_adaptor<tagged_iterator<Iterator, Tag>,
                   Iterator,
                   it_value_t<Iterator>,
                   Tag,
                   typename iterator_traversal<Iterator>::type,
                   it_reference_t<Iterator>,
                   it_difference_t<Iterator>>;

template <typename Iterator, typename Tag>
class tagged_iterator : public make_tagged_iterator_base<Iterator, Tag>
{
  using super_t = make_tagged_iterator_base<Iterator, Tag>;

public:
  tagged_iterator() = default;

  _CCCL_HOST_DEVICE explicit tagged_iterator(Iterator x)
      : super_t(x)
  {}
};

//! \p make_tagged_iterator creates a \p tagged_iterator from a \c Iterator with system tag \c Tag.
//!
//! \tparam Tag Any system tag.
//! \tparam Iterator Any iterator type.
//! \param iter The iterator of interest.
//! \return An iterator whose system tag is \p Tag and whose behavior is otherwise equivalent to \p iter.
template <typename Tag, typename Iterator>
auto make_tagged_iterator(Iterator iter) -> tagged_iterator<Iterator, Tag>
{
  return tagged_iterator<Iterator, Tag>(iter);
}
} // namespace detail

// tagged_iterator is trivial if its base iterator is.
template <typename BaseIterator, typename Tag>
struct proclaim_contiguous_iterator<detail::tagged_iterator<BaseIterator, Tag>> : is_contiguous_iterator<BaseIterator>
{};

THRUST_NAMESPACE_END
