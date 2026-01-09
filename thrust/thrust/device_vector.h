/*
 *  Copyright 2008-2018 NVIDIA Corporation
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

/*! \file
 *  \brief A dynamically-sizable array of elements which resides in memory
 *         accessible to devices.
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
#include <thrust/detail/vector_base.h>
#include <thrust/device_allocator.h>

#include <cuda/std/__utility/move.h>
#include <cuda/std/initializer_list>

#include <vector>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup containers Containers
 *  \{
 */

/*! A \p device_vector is a container that supports random access to elements,
 *  constant time removal of elements at the end, and linear time insertion
 *  and removal of elements at the beginning or in the middle. The number of
 *  elements in a \p device_vector may vary dynamically; memory management is
 *  automatic. The memory associated with a \p device_vector resides in the
 *  memory accessible to devices.
 *
 *  \see https://en.cppreference.com/w/cpp/container/vector
 *  \see device_allocator
 *  \see host_vector
 *  \see universal_vector
 */
template <typename T, typename Alloc = thrust::device_allocator<T>>
class device_vector : public detail::vector_base<T, Alloc>
{
private:
  using Parent = detail::vector_base<T, Alloc>;

public:
  /*! \cond
   */
  using size_type  = typename Parent::size_type;
  using value_type = typename Parent::value_type;
  /*! \endcond
   */

  /*! This constructor creates an empty \p device_vector.
   */
  device_vector()
      : Parent()
  {}

  /*! This constructor creates an empty \p device_vector.
   *  \param alloc The allocator to use by this device_vector.
   */
  device_vector(const Alloc& alloc)
      : Parent(alloc)
  {}

  /*! The destructor erases the elements.
   */
  //  Define an empty destructor to explicitly specify
  //  its execution space qualifier, as a workaround for nvcc warning
  ~device_vector() {}

  /*! This constructor creates a \p device_vector with the given
   *  size.
   *  \param n The number of elements to initially create.
   */
  explicit device_vector(size_type n)
      : Parent(n)
  {}

  //! This constructor creates a \p device_vector with the given size, performing only default-initialization instead of
  //! value-initialization.
  //! \param n The number of elements to initially create.
  device_vector(size_type n, default_init_t)
      : Parent(n, default_init_t{})
  {}

  //! This constructor creates a \p device_vector with the given size, without initializing elements. It mandates that
  //! the element type is trivially default-constructible.
  //! \param n The number of elements to initially create.
  device_vector(size_type n, no_init_t)
      : Parent(n, no_init_t{})
  {}

  /*! This constructor creates a \p device_vector with the given
   *  size.
   *  \param n The number of elements to initially create.
   *  \param alloc The allocator to use by this device_vector.
   */
  explicit device_vector(size_type n, const Alloc& alloc)
      : Parent(n, alloc)
  {}

  /*! This constructor creates a \p device_vector with copies
   *  of an exemplar element.
   *  \param n The number of elements to initially create.
   *  \param value An element to copy.
   */
  explicit device_vector(size_type n, const value_type& value)
      : Parent(n, value)
  {}

  /*! This constructor creates a \p device_vector with copies
   *  of an exemplar element.
   *  \param n The number of elements to initially create.
   *  \param value An element to copy.
   *  \param alloc The allocator to use by this device_vector.
   */
  explicit device_vector(size_type n, const value_type& value, const Alloc& alloc)
      : Parent(n, value, alloc)
  {}

  /*! Copy constructor copies from an exemplar \p device_vector.
   *  \param v The \p device_vector to copy.
   */
  device_vector(const device_vector& v)
      : Parent(v)
  {}

  /*! Copy constructor copies from an exemplar \p device_vector.
   *  \param v The \p device_vector to copy.
   *  \param alloc The allocator to use by this device_vector.
   */
  device_vector(const device_vector& v, const Alloc& alloc)
      : Parent(v, alloc)
  {}

  /*! Move constructor moves from another \p device_vector.
   *  \param v The device_vector to move.
   */
  device_vector(device_vector&& v)
      : Parent(::cuda::std::move(v))
  {}

  /*! Move constructor moves from another \p device_vector.
   *  \param v The device_vector to move.
   *  \param alloc The allocator to use by this device_vector.
   */
  device_vector(device_vector&& v, const Alloc& alloc)
      : Parent(::cuda::std::move(v), alloc)
  {}

  /*! Copy assign operator copies another \p device_vector with the same type.
   *  \param v The \p device_vector to copy.
   */
  device_vector& operator=(const device_vector& v)
  {
    Parent::operator=(v);
    return *this;
  }

  /*! Move assign operator moves from another \p device_vector.
   *  \param v The device_vector to move.
   */
  device_vector& operator=(device_vector&& v)
  {
    Parent::operator=(::cuda::std::move(v));
    return *this;
  }

  /*! Copy constructor copies from an exemplar \p device_vector with different type.
   *  \param v The \p device_vector to copy.
   */
  template <typename OtherT, typename OtherAlloc>
  explicit device_vector(const device_vector<OtherT, OtherAlloc>& v)
      : Parent(v)
  {}

  /*! Assign operator copies from an exemplar \p device_vector with different type.
   *  \param v The \p device_vector to copy.
   */
  template <typename OtherT, typename OtherAlloc>
  device_vector& operator=(const device_vector<OtherT, OtherAlloc>& v)
  {
    Parent::operator=(v);
    return *this;
  }

  /*! Copy constructor copies from an exemplar \c std::vector.
   *  \param v The <tt>std::vector</tt> to copy.
   */
  template <typename OtherT, typename OtherAlloc>
  device_vector(const std::vector<OtherT, OtherAlloc>& v)
      : Parent(v)
  {}

  /*! Assign operator copies from an exemplar <tt>std::vector</tt>.
   *  \param v The <tt>std::vector</tt> to copy.
   */
  template <typename OtherT, typename OtherAlloc>
  device_vector& operator=(const std::vector<OtherT, OtherAlloc>& v)
  {
    Parent::operator=(v);
    return *this;
  }

  /*! Copy construct from a \p vector_base whose element type is convertible
   *  to \c T.
   *
   *  \param v The \p vector_base to copy.
   */
  template <typename OtherT, typename OtherAlloc>
  device_vector(const detail::vector_base<OtherT, OtherAlloc>& v)
      : Parent(v)
  {}

  /*! Assign a \p vector_base whose element type is convertible to \c T.
   *  \param v The \p vector_base to copy.
   */
  template <typename OtherT, typename OtherAlloc>
  device_vector& operator=(const detail::vector_base<OtherT, OtherAlloc>& v)
  {
    Parent::operator=(v);
    return *this;
  }

  /*! This constructor builds a \p device_vector from an intializer_list.
   *  \param il The intializer_list.
   */
  device_vector(::cuda::std::initializer_list<T> il)
      : Parent(il)
  {}

  /*! This constructor builds a \p device_vector from an intializer_list.
   *  \param il The intializer_list.
   *  \param alloc The allocator to use by this device_vector.
   */
  device_vector(::cuda::std::initializer_list<T> il, const Alloc& alloc)
      : Parent(il, alloc)
  {}

  /*! Assign an \p intializer_list with a matching element type
   *  \param il The intializer_list.
   */
  device_vector& operator=(::cuda::std::initializer_list<T> il)
  {
    Parent::operator=(il);
    return *this;
  }

  /*! This constructor builds a \p device_vector from a range.
   *  \param first The beginning of the range.
   *  \param last The end of the range.
   */
  template <typename InputIterator>
  device_vector(InputIterator first, InputIterator last)
      : Parent(first, last)
  {}

  /*! This constructor builds a \p device_vector from a range.
   *  \param first The beginning of the range.
   *  \param last The end of the range.
   *  \param alloc The allocator to use by this device_vector.
   */
  template <typename InputIterator>
  device_vector(InputIterator first, InputIterator last, const Alloc& alloc)
      : Parent(first, last, alloc)
  {}

  /*! Exchanges the values of two vectors.
   *  \p x The first \p device_vector of interest.
   *  \p y The second \p device_vector of interest.
   */
  friend void swap(device_vector& a, device_vector& b) noexcept(noexcept(a.swap(b)))
  {
    a.swap(b);
  }

// declare these members for the purpose of Doxygenating them
// they actually exist in a base class
#if _CCCL_DOXYGEN_INVOKED
  /*! \brief Resizes this vector to the specified number of elements.
   *  \param new_size Number of elements this vector should contain.
   *  \param x Data with which new elements should be populated.
   *  \throw std::length_error If n exceeds max_size().
   *
   *  This method will resize this vector to the specified number of
   *  elements.  If the number is smaller than this vector's current
   *  size this vector is truncated, otherwise this vector is
   *  extended and new elements are populated with given data.
   */
  void resize(size_type new_size, const value_type& x = value_type());

  //! \brief Resizes this vector to the specified number of elements, performing default-initialization instead of
  //!         value-initialization.
  //! \param new_size Number of elements this vector should contain.
  //! \throw std::length_error If n exceeds max_size().
  void resize(size_type new_size, default_init_t);

  //! \brief Resizes this vector_base to the specified number of elements, without initializing elements. It mandates
  //! that the element type is trivially default-constructible.
  //! \param new_size Number of elements this vector_base should contain.
  //! \throw std::length_error If n exceeds max_size().
  void resize(size_type new_size, no_init_t);

  /*! Returns the number of elements in this vector.
   */
  size_type size() const;

  /*! Returns the size() of the largest possible vector.
   *  \return The largest possible return value of size().
   */
  size_type max_size() const;

  /*! \brief If n is less than or equal to capacity(), this call has no effect.
   *         Otherwise, this method is a request for allocation of additional memory. If
   *         the request is successful, then capacity() is greater than or equal to
   *         n; otherwise, capacity() is unchanged. In either case, size() is unchanged.
   *  \throw std::length_error If n exceeds max_size().
   */
  void reserve(size_type n);

  /*! Returns the number of elements which have been reserved in this
   *  vector.
   */
  size_type capacity() const;

  /*! This method shrinks the capacity of this vector to exactly
   *  fit its elements.
   */
  void shrink_to_fit();

  /*! \brief Subscript access to the data contained in this vector_dev.
   *  \param n The index of the element for which data should be accessed.
   *  \return Read/write reference to data.
   *
   *  This operator allows for easy, array-style, data access.
   *  Note that data access with this operator is unchecked and
   *  out_of_range lookups are not defined.
   */
  reference operator[](size_type n);

  /*! \brief Subscript read access to the data contained in this vector_dev.
   *  \param n The index of the element for which data should be accessed.
   *  \return Read reference to data.
   *
   *  This operator allows for easy, array-style, data access.
   *  Note that data access with this operator is unchecked and
   *  out_of_range lookups are not defined.
   */
  const_reference operator[](size_type n) const;

  /*! This method returns an iterator pointing to the beginning of
   *  this vector.
   *  \return mStart
   */
  iterator begin();

  /*! This method returns a const_iterator pointing to the beginning
   *  of this vector.
   *  \return mStart
   */
  const_iterator begin() const;

  /*! This method returns a const_iterator pointing to the beginning
   *  of this vector.
   *  \return mStart
   */
  const_iterator cbegin() const;

  /*! This method returns a reverse_iterator pointing to the beginning of
   *  this vector's reversed sequence.
   *  \return A reverse_iterator pointing to the beginning of this
   *          vector's reversed sequence.
   */
  reverse_iterator rbegin();

  /*! This method returns a const_reverse_iterator pointing to the beginning of
   *  this vector's reversed sequence.
   *  \return A const_reverse_iterator pointing to the beginning of this
   *          vector's reversed sequence.
   */
  const_reverse_iterator rbegin() const;

  /*! This method returns a const_reverse_iterator pointing to the beginning of
   *  this vector's reversed sequence.
   *  \return A const_reverse_iterator pointing to the beginning of this
   *          vector's reversed sequence.
   */
  const_reverse_iterator crbegin() const;

  /*! This method returns an iterator pointing to one element past the
   *  last of this vector.
   *  \return begin() + size().
   */
  iterator end();

  /*! This method returns a const_iterator pointing to one element past the
   *  last of this vector.
   *  \return begin() + size().
   */
  const_iterator end() const;

  /*! This method returns a const_iterator pointing to one element past the
   *  last of this vector.
   *  \return begin() + size().
   */
  const_iterator cend() const;

  /*! This method returns a reverse_iterator pointing to one element past the
   *  last of this vector's reversed sequence.
   *  \return rbegin() + size().
   */
  reverse_iterator rend();

  /*! This method returns a const_reverse_iterator pointing to one element past the
   *  last of this vector's reversed sequence.
   *  \return rbegin() + size().
   */
  const_reverse_iterator rend() const;

  /*! This method returns a const_reverse_iterator pointing to one element past the
   *  last of this vector's reversed sequence.
   *  \return rbegin() + size().
   */
  const_reverse_iterator crend() const;

  /*! This method returns a const_reference referring to the first element of this
   *  vector.
   *  \return The first element of this vector.
   */
  const_reference front() const;

  /*! This method returns a reference pointing to the first element of this
   *  vector.
   *  \return The first element of this vector.
   */
  reference front();

  /*! This method returns a const reference pointing to the last element of
   *  this vector.
   *  \return The last element of this vector.
   */
  const_reference back() const;

  /*! This method returns a reference referring to the last element of
   *  this vector_dev.
   *  \return The last element of this vector.
   */
  reference back();

  /*! This method returns a pointer to this vector's first element.
   *  \return A pointer to the first element of this vector.
   */
  pointer data();

  /*! This method returns a const_pointer to this vector's first element.
   *  \return a const_pointer to the first element of this vector.
   */
  const_pointer data() const;

  /*! This method resizes this vector to 0.
   */
  void clear();

  /*! This method returns true iff size() == 0.
   *  \return true if size() == 0; false, otherwise.
   */
  bool empty() const;

  /*! This method appends the given element to the end of this vector.
   *  \param x The element to append.
   */
  void push_back(const value_type& x);

  /*! This method erases the last element of this vector, invalidating
   *  all iterators and references to it.
   */
  void pop_back();

  /*! This method swaps the contents of this device_vector with another vector.
   *  \param v The vector with which to swap.
   */
  void swap(device_vector& v);

  /*! This method removes the element at position pos.
   *  \param pos The position of the element of interest.
   *  \return An iterator pointing to the new location of the element that followed the element
   *          at position pos.
   */
  iterator erase(iterator pos);

  /*! This method removes the range of elements [first,last) from this vector.
   *  \param first The beginning of the range of elements to remove.
   *  \param last The end of the range of elements to remove.
   *  \return An iterator pointing to the new location of the element that followed the last
   *          element in the sequence [first,last).
   */
  iterator erase(iterator first, iterator last);

  /*! This method inserts a single copy of a given exemplar value at the
   *  specified position in this vector.
   *  \param position The insertion position.
   *  \param x The exemplar element to copy & insert.
   *  \return An iterator pointing to the newly inserted element.
   */
  iterator insert(iterator position, const T& x);

  /*! This method inserts a copy of an exemplar value to a range at the
   *  specified position in this vector.
   *  \param position The insertion position
   *  \param n The number of insertions to perform.
   *  \param x The value to replicate and insert.
   */
  void insert(iterator position, size_type n, const T& x);

  /*! This method inserts a copy of an input range at the specified position
   *  in this vector.
   *  \param position The insertion position.
   *  \param first The beginning of the range to copy.
   *  \param last  The end of the range to copy.
   *
   *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/iterator/input_iterator>Input
   * Iterator</a>, and \p InputIterator's \c value_type is a model of <a
   * href="https://en.cppreference.com/w/cpp/named_req/CopyAssignable">Assignable</a>.
   */
  template <typename InputIterator>
  void insert(iterator position, InputIterator first, InputIterator last);

  /*! This version of \p assign replicates a given exemplar
   *  \p n times into this vector.
   *  \param n The number of times to copy \p x.
   *  \param x The exemplar element to replicate.
   */
  void assign(size_type n, const T& x);

  /*! This version of \p assign makes this vector a copy of a given input range.
   *  \param first The beginning of the range to copy.
   *  \param last  The end of the range to copy.
   *
   *  \tparam InputIterator is a model of <a href="https://en.cppreference.com/w/cpp/named_req/InputIterator">Input
   * Iterator</a>.
   */
  template <typename InputIterator>
  void assign(InputIterator first, InputIterator last);

  /*! This method returns a copy of this vector's allocator.
   *  \return A copy of the allocator used by this vector.
   */
  allocator_type get_allocator() const;
#endif // end doxygen-only members
};

/*! \} // containres
 */

THRUST_NAMESPACE_END
