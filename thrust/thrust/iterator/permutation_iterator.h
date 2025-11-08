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

//! \file thrust/iterator/permutation_iterator.h
//! \brief An iterator which performs a gather or scatter operation when dereferenced

/*
 * (C) Copyright Toon Knapen    2001.
 * (C) Copyright David Abrahams 2003.
 * (C) Copyright Roland Richter 2003.
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
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

#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/minimum_system.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_traits.h>

THRUST_NAMESPACE_BEGIN

template <typename, typename>
class permutation_iterator;

namespace detail
{
template <typename ElementIterator, typename IndexIterator>
struct make_permutation_iterator_base
{
  using System1 = iterator_system_t<ElementIterator>;
  using System2 = iterator_system_t<IndexIterator>;

  using type =
    iterator_adaptor<permutation_iterator<ElementIterator, IndexIterator>,
                     IndexIterator,
                     it_value_t<ElementIterator>,
                     minimum_system_t<System1, System2>,
                     use_default,
                     it_reference_t<ElementIterator>>;
};
} // namespace detail

//! \addtogroup iterators
//! \{

//! \addtogroup fancyiterator Fancy Iterators
//! \ingroup iterators
//! \{

//! \p permutation_iterator is an iterator which represents a pointer into a reordered view of a given range. \p
//! permutation_iterator is an imprecise name; the reordered view need not be a strict permutation. This iterator is
//! useful for fusing a scatter or gather operation with other algorithms.
//!
//! This iterator takes two arguments:
//!
//!   - an iterator to the range \c V on which the "permutation" will be applied
//!   - the reindexing scheme that defines how the elements of \c V will be permuted.
//!
//! Note that \p permutation_iterator is not limited to strict permutations of the given range \c V. The distance
//! between begin and end of the reindexing iterators is allowed to be smaller compared to the size of the range \c V,
//! in which case the \p permutation_iterator only provides a "permutation" of a subrange of \c V. The indices neither
//! need to be unique. In this same context, it must be noted that the past-the-end \p permutation_iterator is
//! completely defined by means of the past-the-end iterator to the indices.
//!
//! The following code snippet demonstrates how to create a \p permutation_iterator which represents a reordering of the
//! contents of a \p device_vector.
//!
//! \code
//! #include <thrust/iterator/permutation_iterator.h>
//! #include <thrust/device_vector.h>
//! ...
//! thrust::device_vector<float> values{10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
//! thrust::device_vector<int> indices{2, 6, 1, 3};
//!
//! using ElementIterator = thrust::device_vector<float>::iterator;
//! using IndexIterator = thrust::device_vector<int>::iterator  ;
//!
//! thrust::permutation_iterator<ElementIterator,IndexIterator> iter(values.begin(), indices.begin());
//!
//! *iter;   // returns 30.0f;
//! iter[0]; // returns 30.0f;
//! iter[1]; // returns 70.0f;
//! iter[2]; // returns 20.0f;
//! iter[3]; // returns 40.0f;
//!
//! // iter[4] is an out-of-bounds error
//!
//! *iter   = -1.0f; // sets values[2] to -1.0f;
//! iter[0] = -1.0f; // sets values[2] to -1.0f;
//! iter[1] = -1.0f; // sets values[6] to -1.0f;
//! iter[2] = -1.0f; // sets values[1] to -1.0f;
//! iter[3] = -1.0f; // sets values[3] to -1.0f;
//!
//! // values is now {10, -1, -1, -1, 50, 60, -1, 80}
//! \endcode
//!
//! \see make_permutation_iterator
template <typename ElementIterator, typename IndexIterator>
class _CCCL_DECLSPEC_EMPTY_BASES permutation_iterator
    : public detail::make_permutation_iterator_base<ElementIterator, IndexIterator>::type
{
  //! \cond

  using super_t = typename detail::make_permutation_iterator_base<ElementIterator, IndexIterator>::type;

  friend class iterator_core_access;
  //! \endcond

public:
  //! Null constructor calls the null constructor of this \p permutation_iterator's element iterator.
  permutation_iterator() = default;

  //! Constructor accepts an \c ElementIterator into a range of values and an \c IndexIterator into a range of indices
  //! defining the indexing scheme on the values.
  //!
  //! \param x An \c ElementIterator pointing this \p permutation_iterator's range of values.
  //! \param y An \c IndexIterator pointing to an indexing scheme to use on \p x.
  _CCCL_HOST_DEVICE explicit permutation_iterator(ElementIterator x, IndexIterator y)
      : super_t(y)
      , m_element_iterator(x)
  {}

  //! Copy constructor accepts a related \p permutation_iterator. \param rhs A compatible \p permutation_iterator to
  //! copy from.
  template <typename OtherElementIterator,
            typename OtherIndexIterator,
            detail::enable_if_convertible_t<OtherElementIterator, ElementIterator, int> = 0,
            detail::enable_if_convertible_t<OtherIndexIterator, IndexIterator, int>     = 0>
  _CCCL_HOST_DEVICE permutation_iterator(permutation_iterator<OtherElementIterator, OtherIndexIterator> const& rhs)
      : super_t(rhs.base())
      , m_element_iterator(rhs.m_element_iterator)
  {}

  //! \cond

private:
  // MSVC incorrectly warning about returning a reference to a local/temporary here.
  // NVHPC breaks with push / pop within a class
#if _CCCL_COMPILER(MSVC)
  _CCCL_DIAG_PUSH
  _CCCL_DIAG_SUPPRESS_MSVC(4172)
#endif // _CCCL_COMPILER(MSVC)

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE typename super_t::reference dereference() const
  {
    return *(m_element_iterator + *this->base());
  }

  // make friends for the copy constructor
  template <typename, typename>
  friend class permutation_iterator;

  ElementIterator m_element_iterator;
  //! \endcond
};

//! \p make_permutation_iterator creates a \p permutation_iterator from an \c ElementIterator pointing to a range of
//! elements to "permute" and an \c IndexIterator pointing to a range of indices defining an indexing scheme on the
//! values.
//!
//! \param e An \c ElementIterator pointing to a range of values.
//! \param i An \c IndexIterator pointing to an indexing scheme to use on \p e.
//! \return A new \p permutation_iterator which permutes the range \p e by \p i.
//! \see permutation_iterator
template <typename ElementIterator, typename IndexIterator>
_CCCL_HOST_DEVICE permutation_iterator<ElementIterator, IndexIterator>
make_permutation_iterator(ElementIterator e, IndexIterator i)
{
  return permutation_iterator<ElementIterator, IndexIterator>(e, i);
}

//! \} // end fancyiterators
//! \} // end iterators

THRUST_NAMESPACE_END
