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
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{


template<typename RandomAccessIterator,
         typename BinaryPredicate = thrust::equal_to<typename thrust::iterator_value<RandomAccessIterator>::type>,
         typename ValueType = bool,
         typename IndexType = typename thrust::iterator_difference<RandomAccessIterator>::type>
  class tail_flags
{
  // XXX WAR cudafe bug
  //private:
  public:
    struct tail_flag_functor
    {
      BinaryPredicate binary_pred; // this must be the first member for performance reasons
      RandomAccessIterator iter;
      IndexType n;

      typedef ValueType result_type;

      _CCCL_HOST_DEVICE
      tail_flag_functor(RandomAccessIterator first, RandomAccessIterator last)
        : binary_pred(), iter(first), n(last - first)
      {}

      _CCCL_HOST_DEVICE
      tail_flag_functor(RandomAccessIterator first, RandomAccessIterator last, BinaryPredicate binary_pred)
        : binary_pred(binary_pred), iter(first), n(last - first)
      {}

      _CCCL_HOST_DEVICE __thrust_forceinline__
      result_type operator()(const IndexType &i)
      {
        return (i == (n - 1) || !binary_pred(iter[i], iter[i+1]));
      }
    };

    typedef thrust::counting_iterator<IndexType> counting_iterator;

  public:
    typedef thrust::transform_iterator<
      tail_flag_functor,
      counting_iterator
    > iterator;

    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_HOST_DEVICE
    tail_flags(RandomAccessIterator first, RandomAccessIterator last)
      : m_begin(thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0),
                                                tail_flag_functor(first, last))),
        m_end(m_begin + (last - first))
    {}

    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_HOST_DEVICE
    tail_flags(RandomAccessIterator first, RandomAccessIterator last, BinaryPredicate binary_pred)
      : m_begin(thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0),
                                                tail_flag_functor(first, last, binary_pred))),
        m_end(m_begin + (last - first))
    {}

    _CCCL_HOST_DEVICE
    iterator begin() const
    {
      return m_begin;
    }

    _CCCL_HOST_DEVICE
    iterator end() const
    {
      return m_end;
    }

    template<typename OtherIndex>
    _CCCL_HOST_DEVICE
    typename iterator::reference operator[](OtherIndex i)
    {
      return *(begin() + i);
    }

  private:
    iterator m_begin, m_end;
};


template<typename RandomAccessIterator, typename BinaryPredicate>
_CCCL_HOST_DEVICE
tail_flags<RandomAccessIterator, BinaryPredicate>
  make_tail_flags(RandomAccessIterator first, RandomAccessIterator last, BinaryPredicate binary_pred)
{
  return tail_flags<RandomAccessIterator, BinaryPredicate>(first, last, binary_pred);
}


template<typename RandomAccessIterator>
_CCCL_HOST_DEVICE
tail_flags<RandomAccessIterator>
  make_tail_flags(RandomAccessIterator first, RandomAccessIterator last)
{
  return tail_flags<RandomAccessIterator>(first, last);
}


} // end detail
THRUST_NAMESPACE_END

