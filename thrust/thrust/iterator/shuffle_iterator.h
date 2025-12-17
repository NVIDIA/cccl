/*
 *  Copyright 2025 NVIDIA Corporation
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

//! \file thrust/iterator/shuffle_iterator.h
//! \brief An output iterator which generates a sequence of values representing a random permutation
#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/random_bijection.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_traits.h>

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/cstdint>

THRUST_NAMESPACE_BEGIN

template <class IndexType, class BijectionFunc>
class shuffle_iterator;

namespace detail
{
template <class IndexType, class BijectionFunc>
struct make_shuffle_iterator_base
{
  static_assert(::cuda::std::is_integral_v<IndexType>, "IndexType must be an integral type");

  using system     = any_system_tag;
  using traversal  = random_access_traversal_tag;
  using difference = ::cuda::std::_If<sizeof(IndexType) < sizeof(int), int, ::cuda::std::ptrdiff_t>;

  using type =
    iterator_adaptor<shuffle_iterator<IndexType, BijectionFunc>,
                     IndexType,
                     IndexType,
                     system,
                     traversal,
                     IndexType,
                     difference>;
};
} // namespace detail

//! \p shuffle_iterator is an iterator which generates a sequence of values representing a random permutation.
//! \tparam IndexType The type of the index to shuffle.
//! \tparam BijectionFunc The bijection to use. This should be a bijective function that maps [0..n) -> [0..n). It must
//! be deterministic and stateless.
//!
//! \addtogroup iterators
//! \{

//! \addtogroup fancyiterator Fancy Iterators
//! \ingroup iterators
//! \{

//! \p shuffle_iterator is an iterator which generates a sequence of values representing a random permutation. This
//! iterator is useful for working with random permutations of a range without explicitly storing them in memory. The
//! shuffle iterator is also useful for sampling from a range by selecting only a subset of the elements in the
//! permutation.
//!
//! The following code snippet demonstrates how to create a \p shuffle_iterator which generates a random permutation of
//! a vector.
//!
//! \code
//! #include <thrust/iterator/shuffle_iterator.h>
//! ...
//! // create a shuffle iterator
//! thrust::shuffle_iterator<int> iterator(4, thrust::default_random_engine(0xDEADBEEF));
//! // iterator[0] returns 1
//! // iterator[1] returns 3
//! // iterator[2] returns 2
//! // iterator[3] returns 0
//!
//! thrust::device_vector<int> vec = {0, 10, 20, 30};
//! thrust::device_vector<int> shuffled(4);
//! thrust::gather(iterator, iterator + 4, vec.begin(), shuffled.begin());
//! // shuffled returns {10, 30, 20, 0}
//! \endcode
//!
//! This next example demonstrates how to use a \p shuffle_iterator to randomly sample from a vector.
//!
//! \code
//! #include <thrust/iterator/shuffle_iterator.h>
//! ...
//! // create a shuffle iterator
//! thrust::shuffle_iterator<int> iterator(100, thrust::default_random_engine(0xDEADBEEF));
//!
//! // iterator[0] returns 38
//! // iterator[1] returns 50
//! // iterator[2] returns 18
//! // iterator[3] returns 12
//!
//! // create a vector of size 100
//! thrust::device_vector<int> vec(100);
//! thrust::device_vector<int> sample(4);
//!
//! // fill vec with random values
//! thrust::sequence(vec.begin(), vec.end(), 100);
//!
//! // sample 4 random values from vec
//! thrust::gather(iterator, iterator + 4, vec.begin(), sample.begin());
//! // sample returns {138, 150, 118, 112}
//! \endcode
//!
//! \see make_shuffle_iterator
template <class IndexType, class BijectionFunc = thrust::detail::random_bijection<IndexType>>
class shuffle_iterator
#ifndef _CCCL_DOXYGEN_INVOKED // Doxygen breaks here for whatever reason
    : public detail::make_shuffle_iterator_base<IndexType, BijectionFunc>::type
#endif // _CCCL_DOXYGEN_INVOKED
{
  //! \cond
  using super_t = typename detail::make_shuffle_iterator_base<IndexType, BijectionFunc>::type;
  friend class iterator_core_access;
  //! \endcond

public:
  //! \brief Constructs a \p shuffle_iterator with a given number of elements and a \c URBG. The parameters will be
  //! forwarded to the bijection constructor.
  //! \param n The number of elements in the permutation.
  //! \param g The \c URBG used to generate the random permutation. This is only invoked during construction of the \p
  //! shuffle_iterator.
  template <class URBG,
            class Enable = ::cuda::std::enable_if_t<::cuda::std::is_constructible_v<BijectionFunc, IndexType, URBG&&>>>
  _CCCL_HOST_DEVICE shuffle_iterator(IndexType n, URBG&& g)
      : super_t(IndexType{0})
      , bijection(n, ::cuda::std::forward<URBG>(g))
  {}

  //! \brief Constructs a \p shuffle_iterator with a given bijection.
  //! \param bijection The bijection to use.
  _CCCL_HOST_DEVICE shuffle_iterator(BijectionFunc bijection)
      : super_t(IndexType{0})
      , bijection(std::move(bijection))
  {}

private:
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
  _CCCL_HOST_DEVICE IndexType dereference() const
  {
    assert(this->base() < bijection.size());
    return bijection(this->base());
  }

  BijectionFunc bijection;
#endif // _CCCL_DOXYGEN_INVOKED
};

//! \p make_shuffle_iterator creates a \p shuffle_iterator from an \c IndexType and \c URBG.
//!
//! \param n The number of elements in the permutation.
//! \param g The \c URBG used to generate the random permutation.
//! \return A new \p shuffle_iterator which generates a random permutation of the input range.
//! \see shuffle_iterator
template <class IndexType, class URBG>
_CCCL_HOST_DEVICE shuffle_iterator<IndexType> make_shuffle_iterator(IndexType n, URBG&& g)
{
  return shuffle_iterator<IndexType>(n, ::cuda::std::forward<URBG>(g));
} // end make_shuffle_iterator

//! \} // end fancyiterators
//! \} // end iterators

THRUST_NAMESPACE_END
