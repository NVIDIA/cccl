// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file
 * Random-access iterator types
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/thread/thread_load.cuh>
#include <cub/thread/thread_store.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/iterator/iterator_facade.h>

#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__utility/declval.h>

#if !_CCCL_COMPILER(NVRTC)
#  include <ostream>
#endif // !_CCCL_COMPILER(NVRTC)

CUB_NAMESPACE_BEGIN

/**
 * @brief A random-access input wrapper for dereferencing array values using a PTX cache load
 *        modifier.
 *
 * @par Overview
 * - CacheModifiedInputIterator is a random-access input iterator that wraps a native
 *   device pointer of type <tt>ValueType*</tt>. @p ValueType references are
 *   made by reading @p ValueType values through loads modified by @p MODIFIER.
 * - Can be used to load any data type from memory using PTX cache load modifiers (e.g., "LOAD_LDG",
 *   "LOAD_CG", "LOAD_CA", "LOAD_CS", "LOAD_CV", etc.).
 * - Can be constructed, manipulated, and exchanged within and between host and device
 *   functions, but can only be dereferenced within device functions.
 * - Compatible with Thrust API v1.7 or newer.
 *
 * @par Snippet
 * The code snippet below illustrates the use of @p CacheModifiedInputIterator to
 * dereference a device array of double using the "ldg" PTX load modifier
 * (i.e., load values through texture cache).
 * @par
 * @code
 * #include <cub/cub.cuh>   // or equivalently <cub/iterator/cache_modified_input_iterator.cuh>
 *
 * // Declare, allocate, and initialize a device array
 * double *d_in;            // e.g., [8.0, 6.0, 7.0, 5.0, 3.0, 0.0, 9.0]
 *
 * // Create an iterator wrapper
 * cub::CacheModifiedInputIterator<cub::LOAD_LDG, double> itr(d_in);
 *
 * // Within device code:
 * printf("%f\n", itr[0]);  // 8.0
 * printf("%f\n", itr[1]);  // 6.0
 * printf("%f\n", itr[6]);  // 9.0
 *
 * @endcode
 *
 * @tparam CacheLoadModifier
 *   The cub::CacheLoadModifier to use when accessing data
 *
 * @tparam ValueType
 *   The value type of this iterator
 *
 * @tparam OffsetT
 *   The difference type of this iterator (Default: @p ptrdiff_t)
 */
template <CacheLoadModifier MODIFIER, typename ValueType, typename OffsetT = ptrdiff_t>
class CacheModifiedInputIterator
{
public:
  // Required iterator traits

  /// My own type
  using self_type = CacheModifiedInputIterator;

  /// Type to express the result of subtracting one iterator from another
  using difference_type = OffsetT;

  /// The type of the element the iterator can point to
  using value_type = ValueType;

  /// The type of a pointer to an element the iterator can point to
  using pointer = ValueType*;

  /// The type of a reference to an element the iterator can point to
  using reference = ValueType;

#if _CCCL_COMPILER(NVRTC)
  using iterator_category = ::cuda::std::random_access_iterator_tag;
#else // ^^^ _CCCL_COMPILER(NVRTC) ^^^ // vvv !_CCCL_COMPILER(NVRTC) vvv
  using iterator_category =
    THRUST_NS_QUALIFIER::detail::iterator_facade_category_t<THRUST_NS_QUALIFIER::device_system_tag,
                                                            THRUST_NS_QUALIFIER::random_access_traversal_tag>;
#endif // _CCCL_COMPILER(NVRTC)

public:
  /// Wrapped native pointer
  ValueType* ptr;

  /// Constructor
  template <typename QualifiedValueType>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE CacheModifiedInputIterator(QualifiedValueType* ptr) ///< Native pointer to wrap
      : ptr(const_cast<::cuda::std::remove_cv_t<QualifiedValueType>*>(ptr))
  {}

  /// Postfix increment
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_type operator++(int)
  {
    self_type retval = *this;
    ptr++;
    return retval;
  }

  /// Prefix increment
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_type operator++()
  {
    ptr++;
    return *this;
  }

  /// Indirection
  _CCCL_DEVICE _CCCL_FORCEINLINE reference operator*() const
  {
    return ThreadLoad<MODIFIER>(ptr);
  }

  /// Addition
  template <typename Distance>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_type operator+(Distance n) const
  {
    self_type retval(ptr + n);
    return retval;
  }

  /// Addition assignment
  template <typename Distance>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_type& operator+=(Distance n)
  {
    ptr += n;
    return *this;
  }

  /// Subtraction
  template <typename Distance>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_type operator-(Distance n) const
  {
    self_type retval(ptr - n);
    return retval;
  }

  /// Subtraction assignment
  template <typename Distance>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE self_type& operator-=(Distance n)
  {
    ptr -= n;
    return *this;
  }

  /// Distance
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE difference_type operator-(self_type other) const
  {
    return ptr - other.ptr;
  }

  /// Array subscript
  template <typename Distance>
  _CCCL_DEVICE _CCCL_FORCEINLINE reference operator[](Distance n) const
  {
    return ThreadLoad<MODIFIER>(ptr + n);
  }

  /// Structure dereference
  _CCCL_DEVICE _CCCL_FORCEINLINE pointer operator->()
  {
    return &ThreadLoad<MODIFIER>(ptr);
  }

  /// Equal to
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator==(const self_type& rhs) const
  {
    return (ptr == rhs.ptr);
  }

  /// Not equal to
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator!=(const self_type& rhs) const
  {
    return (ptr != rhs.ptr);
  }

  /// ostream operator
#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const self_type& /*itr*/)
  {
    return os;
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

namespace detail
{
template <CacheLoadModifier LoadModifier, typename Iterator>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto try_make_cache_modified_iterator(Iterator it)
{
  if constexpr (::cuda::std::contiguous_iterator<Iterator>)
  {
    return CacheModifiedInputIterator<LoadModifier, it_value_t<Iterator>, it_difference_t<Iterator>>{
      THRUST_NS_QUALIFIER::raw_pointer_cast(&*it)};
  }
  else
  {
    return it;
  }
}

template <CacheLoadModifier LoadModifier, typename Iterator>
using try_make_cache_modified_iterator_t =
  decltype(try_make_cache_modified_iterator<LoadModifier>(::cuda::std::declval<Iterator>()));
} // namespace detail

CUB_NAMESPACE_END
