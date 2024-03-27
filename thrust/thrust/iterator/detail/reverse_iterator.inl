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

#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/iterator_traits.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{

_CCCL_EXEC_CHECK_DISABLE
template<typename Iterator>
  _CCCL_HOST_DEVICE
  Iterator prior(Iterator x)
{
  return --x;
} // end prior()

} // end detail

template<typename BidirectionalIterator>
  _CCCL_HOST_DEVICE
  typename reverse_iterator<BidirectionalIterator>::super_t::reference
    reverse_iterator<BidirectionalIterator>
      ::dereference() const
{
  return *thrust::detail::prior(this->base());
} // end reverse_iterator::increment()

template<typename BidirectionalIterator>
  _CCCL_HOST_DEVICE
  void reverse_iterator<BidirectionalIterator>
    ::increment()
{
  --this->base_reference();
} // end reverse_iterator::increment()

template<typename BidirectionalIterator>
  _CCCL_HOST_DEVICE
  void reverse_iterator<BidirectionalIterator>
    ::decrement()
{
  ++this->base_reference();
} // end reverse_iterator::decrement()

template<typename BidirectionalIterator>
  _CCCL_HOST_DEVICE
  void reverse_iterator<BidirectionalIterator>
    ::advance(typename super_t::difference_type n)
{
  this->base_reference() += -n;
} // end reverse_iterator::advance()

template<typename BidirectionalIterator>
  template<typename OtherBidirectionalIterator>
    _CCCL_HOST_DEVICE
    typename reverse_iterator<BidirectionalIterator>::super_t::difference_type
      reverse_iterator<BidirectionalIterator>
        ::distance_to(reverse_iterator<OtherBidirectionalIterator> const &y) const
{
  return this->base_reference() - y.base();
} // end reverse_iterator::distance_to()

template<typename BidirectionalIterator>
_CCCL_HOST_DEVICE
reverse_iterator<BidirectionalIterator> make_reverse_iterator(BidirectionalIterator x)
{
  return reverse_iterator<BidirectionalIterator>(x);
} // end make_reverse_iterator()


THRUST_NAMESPACE_END

