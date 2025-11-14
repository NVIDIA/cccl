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

/*! \file vector_base.h
 *  \brief Defines the interface to a base class for
 *         host_vector & device_vector.
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

#include <thrust/detail/contiguous_storage.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/normal_iterator.h>
#include <thrust/iterator/iterator_traits.h>

#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__iterator/reverse_iterator.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/initializer_list>

#include <vector>

THRUST_NAMESPACE_BEGIN

struct default_init_t
{};
struct no_init_t
{};

//! Tag to indicate that a vector's elements should be default initialized
inline constexpr default_init_t default_init;

//! Tag to indicate that a vector's elements should not be initialized
inline constexpr no_init_t no_init;

namespace detail
{
template <typename T, typename Alloc>
class vector_base
{
private:
  using storage_type = thrust::detail::contiguous_storage<T, Alloc>;

public:
  // aliases
  using value_type      = typename storage_type::value_type;
  using pointer         = typename storage_type::pointer;
  using const_pointer   = typename storage_type::const_pointer;
  using reference       = typename storage_type::reference;
  using const_reference = typename storage_type::const_reference;
  using size_type       = typename storage_type::size_type;
  using difference_type = typename storage_type::difference_type;
  using allocator_type  = typename storage_type::allocator_type;

  using iterator       = typename storage_type::iterator;
  using const_iterator = typename storage_type::const_iterator;

  using reverse_iterator       = ::cuda::std::reverse_iterator<iterator>;
  using const_reverse_iterator = ::cuda::std::reverse_iterator<const_iterator>;

  /*! This constructor creates an empty vector_base.
   */
  vector_base();

  /*! This constructor creates an empty vector_base.
   *  \param alloc The allocator to use by this vector_base.
   */
  explicit vector_base(const Alloc& alloc);

  /*! This constructor creates a vector_base with value-initialized elements.
   *  \param n The number of elements to create.
   */
  explicit vector_base(size_type n);

  //! This constructor creates a vector_base with default-initialized elements.
  //! \param n The number of elements to create.
  explicit vector_base(size_type n, default_init_t);

  //! This constructor creates a vector_base without initializing elements. It mandates that the element type is
  //! trivially default-constructible.
  //! \param n The number of elements to create.
  template <typename T2 = T>
  explicit vector_base(size_type n, no_init_t);

  /*! This constructor creates a vector_base with value-initialized elements.
   *  \param n The number of elements to create.
   *  \param alloc The allocator to use by this vector_base.
   */
  explicit vector_base(size_type n, const Alloc& alloc);

  /*! This constructor creates a vector_base with copies
   *  of an exemplar element.
   *  \param n The number of elements to initially create.
   *  \param value An element to copy.
   */
  explicit vector_base(size_type n, const value_type& value);

  /*! This constructor creates a vector_base with copies
   *  of an exemplar element.
   *  \param n The number of elements to initially create.
   *  \param value An element to copy.
   *  \param alloc The allocator to use by this vector_base.
   */
  explicit vector_base(size_type n, const value_type& value, const Alloc& alloc);

  /*! Copy constructor copies from an exemplar vector_base.
   *  \param v The vector_base to copy.
   */
  vector_base(const vector_base& v);

  /*! Copy constructor copies from an exemplar vector_base.
   *  \param v The vector_base to copy.
   *  \param alloc The allocator to use by this vector_base.
   */
  vector_base(const vector_base& v, const Alloc& alloc);

  /*! Move constructor moves from another vector_base.
   *  \param v The vector_base to move.
   */
  vector_base(vector_base&& v);

  // FIXME: the internal Thrust machinery in range_init doesn't work with move
  // iterators, which is necessary for the following constructor to be implemented
  // correctly
  // vector_base(vector_base &&v, const Alloc &alloc);

  /*! Copy assign operator copies from another vector_base.
   *  \param v The vector_base to copy.
   */
  vector_base& operator=(const vector_base& v);

  /*! Move assign operator moves from another vector_base.
   *  \param v The vector_base to move.
   */
  vector_base& operator=(vector_base&& v);

  /*! This constructor builds a \p vector_base from an intializer_list.
   *  \param il The intializer_list.
   */
  vector_base(::cuda::std::initializer_list<T> il);

  /*! This constructor builds a \p vector_base from an intializer_list.
   *  \param il The intializer_list.
   *  \param alloc The allocator to use by this device_vector.
   */
  vector_base(::cuda::std::initializer_list<T> il, const Alloc& alloc);

  /*! Assign operator copies from an initializer_list
   *  \param il The initializer_list.
   */
  vector_base& operator=(::cuda::std::initializer_list<T> il);

  /*! Copy constructor copies from an exemplar vector_base with different
   *  type.
   *  \param v The vector_base to copy.
   */
  template <typename OtherT, typename OtherAlloc>
  vector_base(const vector_base<OtherT, OtherAlloc>& v);

  /*! assign operator makes a copy of an exemplar vector_base with different
   *  type.
   *  \param v The vector_base to copy.
   */
  template <typename OtherT, typename OtherAlloc>
  vector_base& operator=(const vector_base<OtherT, OtherAlloc>& v);

  /*! Copy constructor copies from an exemplar std::vector.
   *  \param v The std::vector to copy.
   *  XXX TODO: Make this method redundant with a properly templatized constructor.
   *            We would like to copy from a vector whose element type is anything
   *            assignable to value_type.
   */
  template <typename OtherT, typename OtherAlloc>
  vector_base(const std::vector<OtherT, OtherAlloc>& v);

  /*! assign operator makes a copy of an exemplar std::vector.
   *  \param v The vector to copy.
   *  XXX TODO: Templatize this assign on the type of the vector to copy from.
   *            We would like to copy from a vector whose element type is anything
   *            assignable to value_type.
   */
  template <typename OtherT, typename OtherAlloc>
  vector_base& operator=(const std::vector<OtherT, OtherAlloc>& v);

  /*! This constructor builds a vector_base from a range.
   *  \param first The beginning of the range.
   *  \param last The end of the range.
   */
  template <typename InputIterator, ::cuda::std::enable_if_t<::cuda::std::__has_input_traversal<InputIterator>, int> = 0>
  vector_base(InputIterator first, InputIterator last);

  /*! This constructor builds a vector_base from a range.
   *  \param first The beginning of the range.
   *  \param last The end of the range.
   *  \param alloc The allocator to use by this vector_base.
   */
  template <typename InputIterator, ::cuda::std::enable_if_t<::cuda::std::__has_input_traversal<InputIterator>, int> = 0>
  vector_base(InputIterator first, InputIterator last, const Alloc& alloc);

  /*! The destructor erases the elements.
   */
  ~vector_base();

  /*! \brief Resizes this vector_base to the specified number of elements.
   *  \param new_size Number of elements this vector_base should contain.
   *  \throw std::length_error If n exceeds max_size().
   *
   *  This method will resize this vector_base to the specified number of
   *  elements. If the number is smaller than this vector_base's current
   *  size this vector_base is truncated, otherwise this vector_base is
   *  extended and new elements are value initialized.
   */
  void resize(size_type new_size);

  //! \brief Resizes this vector_base to the specified number of elements, performing default-initialization instead of
  //!         value-initialization.
  //! \param new_size Number of elements this vector_base should contain.
  //! \throw std::length_error If n exceeds max_size().
  void resize(size_type new_size, default_init_t);

  //! \brief Resizes this vector_base to the specified number of elements, without initializing elements. It mandates
  //! that the element type is trivially default-constructible.
  //! \param new_size Number of elements this vector_base should contain.
  //! \throw std::length_error If n exceeds max_size().
  template <typename T2 = T>
  void resize(size_type new_size, no_init_t);

  /*! \brief Resizes this vector_base to the specified number of elements.
   *  \param new_size Number of elements this vector_base should contain.
   *  \param x Data with which new elements should be populated.
   *  \throw std::length_error If n exceeds max_size().
   *
   *  This method will resize this vector_base to the specified number of
   *  elements.  If the number is smaller than this vector_base's current
   *  size this vector_base is truncated, otherwise this vector_base is
   *  extended and new elements are populated with given data.
   */
  void resize(size_type new_size, const value_type& x);

  /*! Returns the number of elements in this vector_base.
   */
  _CCCL_HOST_DEVICE size_type size() const;

  /*! Returns the size() of the largest possible vector_base.
   *  \return The largest possible return value of size().
   */
  _CCCL_HOST_DEVICE size_type max_size() const;

  /*! \brief If n is less than or equal to capacity(), this call has no effect.
   *         Otherwise, this method is a request for allocation of additional memory. If
   *         the request is successful, then capacity() is greater than or equal to
   *         n; otherwise, capacity() is unchanged. In either case, size() is unchanged.
   *  \throw std::length_error If n exceeds max_size().
   */
  void reserve(size_type n);

  /*! Returns the number of elements which have been reserved in this
   *  vector_base.
   */
  _CCCL_HOST_DEVICE size_type capacity() const;

  /*! This method shrinks the capacity of this vector_base to exactly
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
  _CCCL_HOST_DEVICE reference operator[](size_type n);

  /*! \brief Subscript read access to the data contained in this vector_dev.
   *  \param n The index of the element for which data should be accessed.
   *  \return Read reference to data.
   *
   *  This operator allows for easy, array-style, data access.
   *  Note that data access with this operator is unchecked and
   *  out_of_range lookups are not defined.
   */
  _CCCL_HOST_DEVICE const_reference operator[](size_type n) const;

  /*! This method returns an iterator pointing to the beginning of
   *  this vector_base.
   *  \return mStart
   */
  _CCCL_HOST_DEVICE iterator begin();

  /*! This method returns a const_iterator pointing to the beginning
   *  of this vector_base.
   *  \return mStart
   */
  _CCCL_HOST_DEVICE const_iterator begin() const;

  /*! This method returns a const_iterator pointing to the beginning
   *  of this vector_base.
   *  \return mStart
   */
  _CCCL_HOST_DEVICE const_iterator cbegin() const;

  /*! This method returns a reverse_iterator pointing to the beginning of
   *  this vector_base's reversed sequence.
   *  \return A reverse_iterator pointing to the beginning of this
   *          vector_base's reversed sequence.
   */
  _CCCL_HOST_DEVICE reverse_iterator rbegin();

  /*! This method returns a const_reverse_iterator pointing to the beginning of
   *  this vector_base's reversed sequence.
   *  \return A const_reverse_iterator pointing to the beginning of this
   *          vector_base's reversed sequence.
   */
  _CCCL_HOST_DEVICE const_reverse_iterator rbegin() const;

  /*! This method returns a const_reverse_iterator pointing to the beginning of
   *  this vector_base's reversed sequence.
   *  \return A const_reverse_iterator pointing to the beginning of this
   *          vector_base's reversed sequence.
   */
  _CCCL_HOST_DEVICE const_reverse_iterator crbegin() const;

  /*! This method returns an iterator pointing to one element past the
   *  last of this vector_base.
   *  \return begin() + size().
   */
  _CCCL_HOST_DEVICE iterator end();

  /*! This method returns a const_iterator pointing to one element past the
   *  last of this vector_base.
   *  \return begin() + size().
   */
  _CCCL_HOST_DEVICE const_iterator end() const;

  /*! This method returns a const_iterator pointing to one element past the
   *  last of this vector_base.
   *  \return begin() + size().
   */
  _CCCL_HOST_DEVICE const_iterator cend() const;

  /*! This method returns a reverse_iterator pointing to one element past the
   *  last of this vector_base's reversed sequence.
   *  \return rbegin() + size().
   */
  _CCCL_HOST_DEVICE reverse_iterator rend();

  /*! This method returns a const_reverse_iterator pointing to one element past the
   *  last of this vector_base's reversed sequence.
   *  \return rbegin() + size().
   */
  _CCCL_HOST_DEVICE const_reverse_iterator rend() const;

  /*! This method returns a const_reverse_iterator pointing to one element past the
   *  last of this vector_base's reversed sequence.
   *  \return rbegin() + size().
   */
  _CCCL_HOST_DEVICE const_reverse_iterator crend() const;

  /*! This method returns a const_reference referring to the first element of this
   *  vector_base.
   *  \return The first element of this vector_base.
   */
  _CCCL_HOST_DEVICE const_reference front() const;

  /*! This method returns a reference pointing to the first element of this
   *  vector_base.
   *  \return The first element of this vector_base.
   */
  _CCCL_HOST_DEVICE reference front();

  /*! This method returns a const reference pointing to the last element of
   *  this vector_base.
   *  \return The last element of this vector_base.
   */
  _CCCL_HOST_DEVICE const_reference back() const;

  /*! This method returns a reference referring to the last element of
   *  this vector_dev.
   *  \return The last element of this vector_base.
   */
  _CCCL_HOST_DEVICE reference back();

  /*! This method returns a pointer to this vector_base's first element.
   *  \return A pointer to the first element of this vector_base.
   */
  _CCCL_HOST_DEVICE pointer data();

  /*! This method returns a const_pointer to this vector_base's first element.
   *  \return a const_pointer to the first element of this vector_base.
   */
  _CCCL_HOST_DEVICE const_pointer data() const;

  /*! This method resizes this vector_base to 0.
   */
  void clear();

  /*! This method returns true iff size() == 0.
   *  \return true if size() == 0; false, otherwise.
   */
  _CCCL_HOST_DEVICE bool empty() const;

  /*! This method appends the given element to the end of this vector_base.
   *  \param x The element to append.
   */
  void push_back(const value_type& x);

  /*! This method erases the last element of this vector_base, invalidating
   *  all iterators and references to it.
   */
  void pop_back();

  /*! This method swaps the contents of this vector_base with another vector_base.
   *  \param v The vector_base with which to swap.
   */
  void swap(vector_base& v)
  {
    using ::cuda::std::swap;
    swap(m_storage, v.m_storage);
    swap(m_size, v.m_size);
  }

  /*! This method removes the element at position pos.
   *  \param pos The position of the element of interest.
   *  \return An iterator pointing to the new location of the element that followed the element
   *          at position pos.
   */
  iterator erase(iterator pos);

  /*! This method removes the range of elements [first,last) from this vector_base.
   *  \param first The beginning of the range of elements to remove.
   *  \param last The end of the range of elements to remove.
   *  \return An iterator pointing to the new location of the element that followed the last
   *          element in the sequence [first,last).
   */
  iterator erase(iterator first, iterator last);

  /*! This method inserts a single copy of a given exemplar value at the
   *  specified position in this vector_base.
   *  \param position The insertion position.
   *  \param x The exemplar element to copy & insert.
   *  \return An iterator pointing to the newly inserted element.
   */
  iterator insert(iterator position, const T& x);

  /*! This method inserts a copy of an exemplar value to a range at the
   *  specified position in this vector_base.
   *  \param position The insertion position
   *  \param n The number of insertions to perform.
   *  \param x The value to replicate and insert.
   */
  void insert(iterator position, size_type n, const T& x);

  /*! This method inserts a copy of an input range at the specified position
   *  in this vector_base.
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
   *  \p n times into this vector_base.
   *  \param n The number of times to copy \p x.
   *  \param x The exemplar element to replicate.
   */
  void assign(size_type n, const T& x);

  /*! This version of \p assign makes this vector_base a copy of a given input range.
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

  _CCCL_SYNTHESIZE_SEQUENCE_ACCESS(vector_base, const_iterator)
  _CCCL_SYNTHESIZE_SEQUENCE_REVERSE_ACCESS(vector_base, const_reverse_iterator)

protected:
  // Our storage
  storage_type m_storage;

  // The size of this vector_base, in number of elements.
  size_type m_size;

private:
  template <typename InputIterator>
  void range_init(InputIterator first, InputIterator last);

  void value_init(size_type n);

  void fill_init(size_type n, const T& x);

  // these methods resolve the ambiguity of the insert() template of form (iterator, InputIterator, InputIterator)
  template <typename InputIteratorOrIntegralType>
  void
  insert_dispatch(iterator position, InputIteratorOrIntegralType first, InputIteratorOrIntegralType last, false_type);

  // these methods resolve the ambiguity of the insert() template of form (iterator, InputIterator, InputIterator)
  template <typename InputIteratorOrIntegralType>
  void insert_dispatch(iterator position, InputIteratorOrIntegralType n, InputIteratorOrIntegralType x, true_type);

  // this method appends n value-initialized elements at the end
  template <bool SkipInit = false>
  void append(size_type n);

  // this method performs insertion from a fill value
  void fill_insert(iterator position, size_type n, const T& x);

  // this method performs insertion from a range
  template <typename InputIterator>
  void copy_insert(iterator position, InputIterator first, InputIterator last);

  // this method performs assignment from a range
  template <typename InputIterator>
  void range_assign(InputIterator first, InputIterator last);

  // this method performs assignment from a fill value
  void fill_assign(size_type n, const T& x);

  // this method allocates new storage and construct copies the given range
  template <typename ForwardIterator>
  void
  allocate_and_copy(size_type requested_size, ForwardIterator first, ForwardIterator last, storage_type& new_storage);

  /*! This function assigns the contents of vector a to vector b and the
   *  contents of vector b to vector a.
   *
   *  \param a The first vector of interest. After completion, the contents
   *           of b will be returned here.
   *  \param b The second vector of interest. After completion, the contents
   *           of a will be returned here.
   */
  friend void swap(vector_base& a, vector_base& b) noexcept(noexcept(a.swap(b)))
  {
    a.swap(b);
  }
}; // end vector_base

/*! This operator allows comparison between two vectors.
 *  \param lhs The first \p vector to compare.
 *  \param rhs The second \p vector to compare.
 *  \return \c true if and only if each corresponding element in either
 *          \p vector equals the other; \c false, otherwise.
 */
template <typename T1, typename Alloc1, typename T2, typename Alloc2>
bool operator==(const vector_base<T1, Alloc1>& lhs, const vector_base<T2, Alloc2>& rhs);

template <typename T1, typename Alloc1, typename T2, typename Alloc2>
bool operator==(const vector_base<T1, Alloc1>& lhs, const std::vector<T2, Alloc2>& rhs);

template <typename T1, typename Alloc1, typename T2, typename Alloc2>
bool operator==(const std::vector<T1, Alloc1>& lhs, const vector_base<T2, Alloc2>& rhs);

/*! This operator allows comparison between two vectors.
 *  \param lhs The first \p vector to compare.
 *  \param rhs The second \p vector to compare.
 *  \return \c false if and only if each corresponding element in either
 *          \p vector equals the other; \c true, otherwise.
 */
template <typename T1, typename Alloc1, typename T2, typename Alloc2>
bool operator!=(const vector_base<T1, Alloc1>& lhs, const vector_base<T2, Alloc2>& rhs);

template <typename T1, typename Alloc1, typename T2, typename Alloc2>
bool operator!=(const vector_base<T1, Alloc1>& lhs, const std::vector<T2, Alloc2>& rhs);

template <typename T1, typename Alloc1, typename T2, typename Alloc2>
bool operator!=(const std::vector<T1, Alloc1>& lhs, const vector_base<T2, Alloc2>& rhs);
} // namespace detail

THRUST_NAMESPACE_END

#include <thrust/detail/vector_base.inl>
