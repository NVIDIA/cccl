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

//! \file thrust/iterator/zip_iterator.h
//! \brief An iterator which returns a tuple of the result of dereferencing a tuple of iterators when dereferenced

/*
 * Copyright David Abrahams and Thomas Becker 2000-2006.
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

#include <thrust/advance.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/minimum_system.h>
#include <thrust/iterator/detail/tuple_of_iterator_references.h>
#include <thrust/iterator/iterator_facade.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/type_traits/integer_sequence.h>

#include <cuda/std/tuple>

THRUST_NAMESPACE_BEGIN

template <typename IteratorTuple>
class zip_iterator;

namespace detail
{
template <typename IteratorTuple>
struct make_zip_iterator_base
{
  static_assert(!sizeof(IteratorTuple), "thrust::zip_iterator only supports cuda::std::tuple");
};

template <typename... Its>
struct make_zip_iterator_base<::cuda::std::tuple<Its...>>
{
  // We need this to make proxy iterators work because those have a void reference type
  template <class Iter>
  using zip_iterator_reference_t =
    ::cuda::std::conditional_t<::cuda::std::is_same_v<it_reference_t<Iter>, void>,
                               decltype(*::cuda::std::declval<Iter>()),
                               it_reference_t<Iter>>;

  // reference type is the type of the tuple obtained from the iterator's reference types.
  using reference = tuple_of_iterator_references<zip_iterator_reference_t<Its>...>;

  // Boost's Value type is the same as reference type. using value_type = reference;
  using value_type = ::cuda::std::tuple<it_value_t<Its>...>;

  // Difference type is the first iterator's difference type
  using difference_type = it_difference_t<::cuda::std::tuple_element_t<0, ::cuda::std::tuple<Its...>>>;

  // Iterator system is the minimum system tag in the iterator tuple
  using system = minimum_system_t<iterator_system_t<Its>...>;

  // Traversal category is the minimum traversal category in the iterator tuple
  using traversal_category = minimum_type<iterator_traversal_t<Its>...>;

  // The iterator facade type from which the zip iterator will be derived.
  using type =
    iterator_facade<zip_iterator<::cuda::std::tuple<Its...>>,
                    value_type,
                    system,
                    traversal_category,
                    reference,
                    difference_type>;
};
} // namespace detail

//! \addtogroup iterators
//! \{

//! \addtogroup fancyiterator Fancy Iterators
//! \ingroup iterators
//! \{

//! \p zip_iterator is an iterator which represents a pointer into a range of \p tuples whose elements are themselves
//! taken from a \p tuple of input iterators. This iterator is useful for creating a virtual array of structures while
//! achieving the same performance and bandwidth as the structure of arrays idiom. \p zip_iterator also facilitates
//! kernel fusion by providing a convenient means of amortizing the execution of the same operation over multiple
//! ranges.
//!
//! The following code snippet demonstrates how to create a \p zip_iterator which represents the result of "zipping"
//! multiple ranges together.
//!
//! \code
//! #include <thrust/iterator/zip_iterator.h>
//! #include <thrust/tuple.h>
//! #include <thrust/device_vector.h>
//! ...
//! thrust::device_vector<int> int_v{0, 1, 2};
//! thrust::device_vector<float> float_v{0.0f, 1.0f, 2.0f};
//! thrust::device_vector<char> char_v{'a', 'b', 'c'};
//!
//! // aliases for iterators
//! using IntIterator = thrust::device_vector<int>::iterator;
//! using FloatIterator = thrust::device_vector<float>::iterator;
//! using CharIterator = thrust::device_vector<char>::iterator;
//!
//! // alias for a tuple of these iterators
//! using IteratorTuple = thrust::tuple<IntIterator, FloatIterator, CharIterator>;
//!
//! // alias the zip_iterator of this tuple
//! using ZipIterator = thrust::zip_iterator<IteratorTuple>;
//!
//! // finally, create the zip_iterator
//! ZipIterator iter(thrust::make_tuple(int_v.begin(), float_v.begin(), char_v.begin()));
//!
//! *iter;   // returns (0, 0.0f, 'a')
//! iter[0]; // returns (0, 0.0f, 'a')
//! iter[1]; // returns (1, 1.0f, 'b')
//! iter[2]; // returns (2, 2.0f, 'c')
//!
//! thrust::get<0>(iter[2]); // returns 2
//! thrust::get<1>(iter[0]); // returns 0.0f
//! thrust::get<2>(iter[1]); // returns 'b'
//!
//! // iter[3] is an out-of-bounds error
//! \endcode
//!
//! Defining the type of a \p zip_iterator can be complex. The next code example demonstrates how to use the \p
//! make_zip_iterator function with the \p make_tuple function to avoid explicitly specifying the type of the \p
//! zip_iterator. This example shows how to use \p zip_iterator to copy multiple ranges with a single call to \p
//! thrust::copy.
//!
//! \code
//! #include <thrust/zip_iterator.h>
//! #include <thrust/tuple.h>
//! #include <thrust/device_vector.h>
//!
//! int main()
//! {
//!   thrust::device_vector<int> int_in{0, 1, 2}, int_out(3);
//!   thrust::device_vector<float> float_in{0.0f, 10.0f, 20.0f}, float_out(3);
//!
//!   thrust::copy(thrust::make_zip_iterator(int_in.begin(), float_in.begin()),
//!                thrust::make_zip_iterator(int_in.end(),   float_in.end()),
//!                thrust::make_zip_iterator(int_out.begin(),float_out.begin()));
//!
//!   // int_out is now [0, 1, 2]
//!   // float_out is now [0.0f, 10.0f, 20.0f]
//!
//!   return 0;
//! }
//! \endcode
//!
//! \see make_zip_iterator
//! \see make_tuple
//! \see tuple
//! \see get
template <typename IteratorTuple>
class _CCCL_DECLSPEC_EMPTY_BASES zip_iterator : public detail::make_zip_iterator_base<IteratorTuple>::type
{
public:
  //! The underlying iterator tuple type. Alias to zip_iterator's first template argument.
  using iterator_tuple = IteratorTuple;

  zip_iterator() = default;

  //! This constructor creates a new \p zip_iterator from a \p tuple of iterators.
  //!
  //! \param iterator_tuple The \p tuple of iterators to copy from.
  inline _CCCL_HOST_DEVICE zip_iterator(IteratorTuple iterator_tuple)
      : m_iterator_tuple(iterator_tuple)
  {}

  //! This constructor creates a new \p zip_iterator from multiple iterators.
  //!
  //! \param iterators The iterators to zip.
  template <class... Iterators,
            ::cuda::std::enable_if_t<(sizeof...(Iterators) != 0), int>                                  = 0,
            ::cuda::std::enable_if_t<::cuda::std::is_constructible_v<IteratorTuple, Iterators...>, int> = 0>
  inline _CCCL_HOST_DEVICE zip_iterator(Iterators&&... iterators)
      : m_iterator_tuple(::cuda::std::forward<Iterators>(iterators)...)
  {}

  //! This copy constructor creates a new \p zip_iterator from another \p zip_iterator.
  //!
  //! \param other The \p zip_iterator to copy.
  template <typename OtherIteratorTuple, detail::enable_if_convertible_t<OtherIteratorTuple, IteratorTuple, int> = 0>
  inline _CCCL_HOST_DEVICE zip_iterator(const zip_iterator<OtherIteratorTuple>& other)
      : m_iterator_tuple(other.get_iterator_tuple())
  {}

  //! This method returns a \c const reference to this \p zip_iterator's
  //! \p tuple of iterators.
  //!
  //! \return A \c const reference to this \p zip_iterator's \p tuple  of iterators.
  inline _CCCL_HOST_DEVICE const IteratorTuple& get_iterator_tuple() const
  {
    return m_iterator_tuple;
  }

  //! \cond

private:
  using super_t = typename detail::make_zip_iterator_base<IteratorTuple>::type;

  friend class iterator_core_access;

  using index_seq = make_index_sequence<::cuda::std::tuple_size_v<IteratorTuple>>;

  _CCCL_EXEC_CHECK_DISABLE
  template <size_t... Is>
  _CCCL_HOST_DEVICE typename super_t::reference dereference_impl(index_sequence<Is...>) const
  {
    return {*::cuda::std::get<Is>(m_iterator_tuple)...};
  }

  // Dereferencing returns a tuple built from the dereferenced iterators in the iterator tuple.
  _CCCL_HOST_DEVICE typename super_t::reference dereference() const
  {
    return dereference_impl(index_seq{});
  }

  // Two zip_iterators are equal if the two first iterators of the tuple are equal. Note this differs from Boost's
  // implementation, which considers the entire tuple.
  _CCCL_EXEC_CHECK_DISABLE
  template <typename OtherIteratorTuple>
  inline _CCCL_HOST_DEVICE bool equal(const zip_iterator<OtherIteratorTuple>& other) const
  {
    return get<0>(get_iterator_tuple()) == get<0>(other.get_iterator_tuple());
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <size_t... Is>
  inline _CCCL_HOST_DEVICE void advance_impl(typename super_t::difference_type n, index_sequence<Is...>)
  {
    (..., ::cuda::std::advance(::cuda::std::get<Is>(m_iterator_tuple), n));
  }

  // Advancing a zip_iterator means to advance all iterators in the tuple
  inline _CCCL_HOST_DEVICE void advance(typename super_t::difference_type n)
  {
    advance_impl(n, index_seq{});
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <size_t... Is>
  inline _CCCL_HOST_DEVICE void increment_impl(index_sequence<Is...>)
  {
    (..., ++::cuda::std::get<Is>(m_iterator_tuple));
  }

  // Incrementing a zip iterator means to increment all iterators in the tuple
  inline _CCCL_HOST_DEVICE void increment()
  {
    increment_impl(index_seq{});
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <size_t... Is>
  inline _CCCL_HOST_DEVICE void decrement_impl(index_sequence<Is...>)
  {
    (..., --::cuda::std::get<Is>(m_iterator_tuple));
  }

  // Decrementing a zip iterator means to decrement all iterators in the tuple
  inline _CCCL_HOST_DEVICE void decrement()
  {
    decrement_impl(index_seq{});
  }

  // Distance is calculated using the first iterator in the tuple.
  template <typename OtherIteratorTuple>
  inline _CCCL_HOST_DEVICE typename super_t::difference_type
  distance_to(const zip_iterator<OtherIteratorTuple>& other) const
  {
    return get<0>(other.get_iterator_tuple()) - get<0>(get_iterator_tuple());
  }

  // The iterator tuple.
  IteratorTuple m_iterator_tuple;

  //! \endcond
};

#ifndef _CCCL_DOXYGEN_INVOKED
template <class... Iterators>
_CCCL_HOST_DEVICE zip_iterator(Iterators...) -> zip_iterator<::cuda::std::tuple<Iterators...>>;
#endif // _CCCL_DOXYGEN_INVOKED

//! \p make_zip_iterator creates a \p zip_iterator from a \p tuple of iterators.
//!
//! \param t The \p tuple of iterators to copy.
//! \return A newly created \p zip_iterator which zips the iterators encapsulated in \p t.
//! \see zip_iterator
template <typename... Iterators>
inline _CCCL_HOST_DEVICE zip_iterator<::cuda::std::tuple<Iterators...>>
make_zip_iterator(::cuda::std::tuple<Iterators...> t)
{
  return zip_iterator<::cuda::std::tuple<Iterators...>>{t};
}

//! \p make_zip_iterator creates a \p zip_iterator from
//! iterators.
//!
//! \param its The iterators to copy.
//! \return A newly created \p zip_iterator which zips the iterators.
//!
//! \see zip_iterator
template <typename... Iterators>
inline _CCCL_HOST_DEVICE zip_iterator<::cuda::std::tuple<Iterators...>> make_zip_iterator(Iterators... its)
{
  return zip_iterator<::cuda::std::tuple<Iterators...>>{its...};
}

//! \} // end fancyiterators
//! \} // end iterators

THRUST_NAMESPACE_END

// libcu++ iterator traits fail for complex zip_iterators in C++17, see e.g.: https://godbolt.org/z/7jb4qG3bb
// The reason is that libcu++ backported the C++20 range iterator machinery to C++17, but C++17 has slightly different
// language rules, especially regarding `void`. We deemed to it too hard to work around the issues.
#if _CCCL_STD_VER < 2020
_CCCL_BEGIN_NAMESPACE_CUDA_STD
template <typename IteratorTuple>
struct iterator_traits<THRUST_NS_QUALIFIER::zip_iterator<IteratorTuple>>
{
  using It                = THRUST_NS_QUALIFIER::zip_iterator<IteratorTuple>;
  using value_type        = typename It::value_type;
  using reference         = typename It::reference;
  using pointer           = void;
  using iterator_category = typename It::iterator_category;
  using difference_type   = typename It::difference_type;
};
_CCCL_END_NAMESPACE_CUDA_STD
#endif // _CCCL_STD_VER < 2020
