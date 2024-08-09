//===----------------------------------------------------------------------===//
//
// Part of the CUDA Toolkit, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__CONTAINERS_VECTOR_H
#define __CUDAX__CONTAINERS_VECTOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory_resource/properties.h>
#include <cuda/__memory_resource/resource_ref.h>
#include <cuda/std/__algorithm/copy.h>
#include <cuda/std/__algorithm/equal.h>
#include <cuda/std/__algorithm/fill.h>
#include <cuda/std/__algorithm/lexicographical_compare.h>
#include <cuda/std/__algorithm/move.h>
#include <cuda/std/__algorithm/move_backward.h>
#include <cuda/std/__algorithm/remove.h>
#include <cuda/std/__algorithm/remove_if.h>
#include <cuda/std/__algorithm/rotate.h>
#include <cuda/std/__algorithm/swap_ranges.h>
#include <cuda/std/__concepts/_One_of.h>
#include <cuda/std/__exception/terminate.h>
#include <cuda/std/__iterator/advance.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__iterator/iter_move.h>
#include <cuda/std/__iterator/next.h>
#include <cuda/std/__iterator/reverse_iterator.h>
#include <cuda/std/__memory/align.h>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__memory/uninitialized_algorithms.h>
#include <cuda/std/__new/bad_alloc.h>
#include <cuda/std/__new/launder.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__ranges/unwrap_end.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_copy_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_swappable.h>
#include <cuda/std/__type_traits/is_trivial.h>
#include <cuda/std/__type_traits/is_trivially_copy_assignable.h>
#include <cuda/std/__type_traits/is_trivially_copy_constructible.h>
#include <cuda/std/__type_traits/is_trivially_destructible.h>
#include <cuda/std/__type_traits/is_trivially_move_assignable.h>
#include <cuda/std/__type_traits/is_trivially_move_constructible.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstdint>
#include <cuda/std/detail/libcxx/include/__assert> // all public C++ headers provide the assertion handler
#include <cuda/std/detail/libcxx/include/stdexcept>
#include <cuda/std/initializer_list>
#include <cuda/std/limits>

#include <cuda/experimental/__container/heterogeneous_iterator.h>
#include <cuda/experimental/__container/uninitialized_buffer.h>
#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__utility/select_execution_space.h>

#if _CCCL_STD_VER >= 2014 && !defined(_CCCL_COMPILER_MSVC_2017) \
  && defined(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

_CCCL_PUSH_MACROS

//! @file The \c vector class provides a container of contiguous memory
namespace cuda::experimental
{

//! @rst
//! .. _cudax-containers-vector:
//!
//! vector
//! -------
//!
//! ``vector`` is a container that provides resizable typed storage allocated from a given :ref:`memory resource
//! <libcudacxx-extended-api-memory-resources-resource>`. It handles alignment, release and growth of the allocation.
//! The elements are initialized during construction, if the storage is not user provided through a
//! :ref:`uninitialized_buffer <cudax-containers-uninitialized-buffer>` with matching properties.
//!
//! In addition to being type-safe, ``vector`` also takes a set of :ref:`properties
//! <libcudacxx-extended-api-memory-resources-properties>` to ensure that e.g. execution space constraints are checked
//! at compile time. However, only stateless properties can be forwarded. To use a stateful property,
//! implement :ref:`get_property(const vector&, Property) <libcudacxx-extended-api-memory-resources-properties>`.
//!
//! .. warning::
//!
//!    ``vector`` stores a reference to the provided memory :ref:`memory resource
//!    <libcudacxx-extended-api-memory-resources-resource>`. It is the user's responsibility to ensure the lifetime of
//!    the resource exceeds the lifetime of the vector.
//!
//! @endrst
//! @tparam _Tp the type to be stored in the buffer
//! @tparam _Properties... The properties the allocated memory satisfies
template <class _Tp, class... _Properties>
class vector
{
public:
  using value_type             = _Tp;
  using reference              = _Tp&;
  using const_reference        = const _Tp&;
  using pointer                = _Tp*;
  using const_pointer          = const _Tp*;
  using iterator               = heterogeneous_iterator<_Tp, false, _Properties...>;
  using const_iterator         = heterogeneous_iterator<_Tp, true, _Properties...>;
  using reverse_iterator       = _CUDA_VSTD::reverse_iterator<iterator>;
  using const_reverse_iterator = _CUDA_VSTD::reverse_iterator<const_iterator>;
  using size_type              = _CUDA_VSTD::size_t;
  using difference_type        = _CUDA_VSTD::ptrdiff_t;
  using __resource_ref         = _CUDA_VMR::resource_ref<_Properties...>;

  // Doxygen cannot handle the macro expansions
  template <class _Range>
  static constexpr bool __compatible_range = _CUDA_VRANGES::__container_compatible_range<_Range, _Tp>;

private:
  uninitialized_buffer<_Tp, _Properties...> __buf_;
  size_type __size_ = 0; // initialized to 0 in case initialization of the elements might throw

  //! @brief If an exception is thrown, adjusts the size of the vector so that the previously constructed elements
  //! are accounted for
  struct _Adjust_size
  {
    vector* __obj_;
    iterator& __first_;
    iterator __current_;

    constexpr _Adjust_size(vector* __obj, iterator& __first, iterator& __current) noexcept
        : __obj_(__obj)
        , __first_(__first)
        , __current_(__current)
    {}

    void operator()() const noexcept
    {
      __obj_->__size_ += static_cast<size_type>(__current_ - __first_);
    }
  };

  //! @brief Destroy the elements in the range `[__first, end())` and adjusts the size.
  //! @param __first Iterator to the first element to be destroyed.
  //! No destructor is run if the `_Tp` is trivially destructible.
  void __destroy_from(iterator __first) noexcept
  {
    const auto __last = end();
    _CCCL_IF_CONSTEXPR (!_CCCL_TRAIT(_CUDA_VSTD::is_trivially_destructible, _Tp))
    {
      _CUDA_VSTD::__destroy(__first, __last);
    }
    __size_ -= static_cast<size_type>(__last - __first);
  }

  //! @brief Value-initializes the elements in the range `[__first, __first + __count)` and adjusts the size.
  //! @param __first Iterator to the first element to be value-initialized.
  //! @param __size The number of elements to be value-initialized.
  _LIBCUDACXX_TEMPLATE(bool _IsNothrow = _CCCL_TRAIT(_CUDA_VSTD::is_nothrow_default_constructible, _Tp))
  _LIBCUDACXX_REQUIRES(_IsNothrow)
  void __uninitialized_value_construct_n(iterator __first, const size_type __size) noexcept
  {
    size_type __count = 0;
    for (; __count < __size; ++__first, (void) ++__count)
    {
      ::new (_CUDA_VSTD::__voidify(*__first)) _Tp();
    }
    __size_ += __size;
  }

  //! @brief Value-initializes the elements in the range `[__first, __first + __count)` and adjusts the size.
  //! @param __first Iterator to the first element to be value-initialized.
  //! @param __size The number of elements to be value-initialized.
  //! If an exception is thrown it updates the size so that all constructed elements are accounted for.
  _LIBCUDACXX_TEMPLATE(bool _IsNothrow = _CCCL_TRAIT(_CUDA_VSTD::is_nothrow_default_constructible, _Tp))
  _LIBCUDACXX_REQUIRES((!_IsNothrow))
  void __uninitialized_value_construct_n(iterator __first, const size_type __size)
  {
    size_type __count    = 0;
    iterator __old_first = __first;
    auto __guard         = _CUDA_VSTD::__make_exception_guard(_Adjust_size{this, __old_first, __first});
    for (; __count < __size; ++__first, (void) ++__count)
    {
      ::new (_CUDA_VSTD::__voidify(*__first)) _Tp();
    }
    __guard.__complete();
    __size_ += __size;
  }

  //! @brief Copy-constructs the elements in the range `[__first, __last)` from \p __value and adjusts the size.
  //! @param __first Iterator to the first element to be copy-constructed.
  //! @param __size The number of elements to be copy-constructed.
  //! @param __value The element to be copied.
  _LIBCUDACXX_TEMPLATE(bool _IsNothrow = _CCCL_TRAIT(_CUDA_VSTD::is_nothrow_copy_constructible, _Tp))
  _LIBCUDACXX_REQUIRES(_IsNothrow)
  void __uninitialized_fill_n(iterator __first, const size_type __size, const _Tp& __value) noexcept
  {
    size_type __count = 0;
    for (; __count < __size; ++__first, (void) ++__count)
    {
      ::new (_CUDA_VSTD::__voidify(*__first)) _Tp(__value);
    }
    __size_ += __size;
  }

  //! @brief Copy-constructs the elements in the range `[__first, __last)` from \p __value and adjusts the size.
  //! @param __first Iterator to the first element to be copy-constructed.
  //! @param __size The number of elements to be copy-constructed.
  //! @param __value The element to be copied.
  //! If an exception is thrown it updates the size so that all constructed elements are accounted for.
  _LIBCUDACXX_TEMPLATE(bool _IsNothrow = _CCCL_TRAIT(_CUDA_VSTD::is_nothrow_copy_constructible, _Tp))
  _LIBCUDACXX_REQUIRES((!_IsNothrow))
  void __uninitialized_fill_n(iterator __first, const size_type __size, const _Tp& __value)
  {
    size_type __count    = 0;
    iterator __old_first = __first;
    auto __guard         = _CUDA_VSTD::__make_exception_guard(_Adjust_size{this, __old_first, __first});
    for (; __count < __size; ++__first, (void) ++__count)
    {
      ::new (_CUDA_VSTD::__voidify(*__first)) _Tp(__value);
    }
    __guard.__complete();
    __size_ += __size;
  }

  //! @brief Copy-constructs the elements after \p __dest from the range `[__first, __last)` and adjusts the size.
  //! @param __first Iterator to the first element to be copied.
  //! @param __last Iterator to the element after the last element to be copied.
  //! @param __dest Iterator to the first element to be copy-constructed.
  _LIBCUDACXX_TEMPLATE(class _Iter, bool _IsNothrow = _CCCL_TRAIT(_CUDA_VSTD::is_nothrow_copy_constructible, _Tp))
  _LIBCUDACXX_REQUIRES(_IsNothrow)
  void __uninitialized_copy(_Iter __first, _Iter __last, iterator __dest) noexcept
  {
    iterator __curr = __dest;
    for (; __first != __last; ++__curr, (void) ++__first)
    {
      ::new (_CUDA_VSTD::__voidify(*__curr)) _Tp(*__first);
    }
    __size_ += static_cast<size_type>(__curr - __dest);
  }

  //! @brief Copy-constructs the elements after \p __dest from the range `[__first, __last)` and adjusts the size.
  //! @param __first Iterator to the first element to be copied.
  //! @param __last Iterator to the element after the last element to be copied.
  //! @param __dest Iterator to the first element to be copy-constructed.
  //! If an exception is thrown it updates the size so that all constructed elements are accounted for.
  _LIBCUDACXX_TEMPLATE(class _Iter, bool _IsNothrow = _CCCL_TRAIT(_CUDA_VSTD::is_nothrow_copy_constructible, _Tp))
  _LIBCUDACXX_REQUIRES((!_IsNothrow))
  void __uninitialized_copy(_Iter __first, _Iter __last, iterator __dest)
  {
    iterator __curr = __dest;
    auto __guard    = _CUDA_VSTD::__make_exception_guard(_Adjust_size{this, __dest, __curr});
    for (; __first != __last; ++__curr, (void) ++__first)
    {
      ::new (_CUDA_VSTD::__voidify(*__curr)) _Tp(*__first);
    }
    __guard.__complete();
    __size_ += static_cast<size_type>(__curr - __dest);
  }

  //! @brief Move-constructs the elements after \p __dest from the range `[__first, __last)` and adjusts the size.
  //! @param __first Iterator to the first element to be moved.
  //! @param __last Iterator to the element after the last element to be moved.
  //! @param __dest Iterator to the first element to be move-constructed.
  _LIBCUDACXX_TEMPLATE(class _Iter, bool _IsNothrow = _CCCL_TRAIT(_CUDA_VSTD::is_nothrow_move_constructible, _Tp))
  _LIBCUDACXX_REQUIRES(_IsNothrow)
  void __uninitialized_move(_Iter __first, _Iter __last, iterator __dest) noexcept
  {
    iterator __curr = __dest;
    for (; __first != __last; ++__curr, (void) ++__first)
    {
#  if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)
      ::new (_CUDA_VSTD::__voidify(*__curr)) _Tp(_CUDA_VRANGES::iter_move(__first));
#  else // ^^^ C++17 ^^^ / vvv C++14 vvv
      ::new (_CUDA_VSTD::__voidify(*__curr)) _Tp(_CUDA_VSTD::move(*__first));
#  endif // _CCCL_STD_VER <= 2014 || _CCCL_COMPILER_MSVC_2017
    }
    __size_ += static_cast<size_type>(__curr - __dest);
  }

  //! @brief Move-constructs the elements after \p __dest from the range `[__first, __last)` and adjusts the size.
  //! @param __first Iterator to the first element to be moved.
  //! @param __last Iterator to the element after the last element to be moved.
  //! @param __dest Iterator to the first element to be move-constructed.
  //! If an exception is thrown it updates the size so that all constructed elements are accounted for.
  _LIBCUDACXX_TEMPLATE(class _Iter, bool _IsNothrow = _CCCL_TRAIT(_CUDA_VSTD::is_nothrow_move_constructible, _Tp))
  _LIBCUDACXX_REQUIRES((!_IsNothrow))
  void __uninitialized_move(_Iter __first, _Iter __last, iterator __dest)
  {
    iterator __curr = __dest;
    auto __guard    = _CUDA_VSTD::__make_exception_guard(_Adjust_size{this, __dest, __curr});
    for (; __first != __last; ++__curr, (void) ++__first)
    {
#  if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)
      ::new (_CUDA_VSTD::__voidify(*__curr)) _Tp(_CUDA_VRANGES::iter_move(__first));
#  else // ^^^ C++17 ^^^ / vvv C++14 vvv
      ::new (_CUDA_VSTD::__voidify(*__curr)) _Tp(_CUDA_VSTD::move(*__first));
#  endif // _CCCL_STD_VER <= 2014 || _CCCL_COMPILER_MSVC_2017
    }
    __guard.__complete();
    __size_ += static_cast<size_type>(__curr - __dest);
  }

public:
  //! @addtogroup construction
  //! @{

  //! @brief Copy-constructs a vector
  //! @param __other The other vector.
  //! The new vector has capacity of \p __other.size() which is potentially less than \p __other.capacity().
  //! @note No memory is allocated if \p __other is empty
  vector(const vector& __other)
      : __buf_(__other.resource(), __other.__size_)
  {
    if (__other.__size_ != 0)
    {
      this->__uninitialized_copy(__other.begin(), __other.end(), begin());
    }
  }

  //! @brief Move-constructs a vector
  //! @param __other The other vector.
  //! The new vector takes ownership of the allocation of \p __other and resets the other vector.
  vector(vector&& __other) noexcept
      : __buf_(_CUDA_VSTD::move(__other.__buf_))
      , __size_(_CUDA_VSTD::exchange(__other.__size_, 0))
  {}

  //! @brief Copy-assigns a vector
  //! @param __other The other vector.
  //! @note Even if the old vector would have enough storage available, we have to reallocate if the stored memory
  //! resource is not equal to the new one. In that case no memory is allocated if \p __other is empty.
  vector& operator=(const vector& __other)
  {
    // There is sufficient space in the allocation and the resources are compatible
    if (resource() == __other.resource() && capacity() >= __other.__size_)
    {
      if (__size_ >= __other.__size_)
      {
        const auto __res = _CUDA_VSTD::copy(__other.begin(), __other.end(), begin());
        this->__destroy_from(__res);
      }
      else
      {
        const auto __res = _CUDA_VSTD::copy(__other.begin(), __other.begin() + __size_, begin());
        this->__uninitialized_copy(__other.begin() + __size_, __other.end(), __res);
      }
      return *this;
    }

    // We need to reallocate and copy. Note we do not change the size of the current vector until the copy is done
    uninitialized_buffer<_Tp, _Properties...> __new_buf{__other.resource(), __other.__size_};
    _CUDA_VSTD::uninitialized_copy(__other.begin(), __other.end(), __new_buf.begin());

    // Now that everything is set up bring over the new data
    this->clear();
    __buf_ = _CUDA_VSTD::move(__new_buf);

    // The above call to destroy has set the size of this vector to 0 so we need set it correctly
    __size_ = __other.__size_;
    return *this;
  }

  //! @brief Move-assigns a vector
  //! @param __other The other vector.
  //! Clears the vector and swaps the contents with \p __other.
  vector& operator=(vector&& __other) noexcept
  {
    if (this == _CUDA_VSTD::addressof(__other))
    {
      return *this;
    }

    this->clear();
    __buf_  = _CUDA_VSTD::move(__other.__buf_);
    __size_ = _CUDA_VSTD::exchange(__other.__size_, 0);
    return *this;
  }

  vector& operator=(_CUDA_VSTD::initializer_list<_Tp> __ilist)
  {
    const auto __count = __ilist.size();
    if (capacity() < __count)
    {
      _CUDA_VSTD::__throw_bad_alloc();
    }

    if (__count < __size_)
    {
      const iterator __new_end = _CUDA_VSTD::copy(__ilist.begin(), __ilist.end(), begin());
      this->__destroy_from(__new_end);
    }
    else
    {
      _CUDA_VSTD::copy(__ilist.begin(), __ilist.begin() + __size_, begin());
      this->__uninitialized_copy(__ilist.begin() + __size_, __ilist.end(), end());
    }
    return *this;
  }

  //! @brief Destroys the \c vector and deallocates the storage after destroying all elements
  //! @note Does not destroy elements if `is_trivially_destructible_v<_Tp>` holds.
  ~vector() noexcept
  {
    this->__destroy_from(begin());
  }

  //! @brief Constructs a vector of size \p __size using a memory resource and value-initializes \p __size elements
  //! @param __mr The memory resource to allocate the vector with.
  //! @param __size The size of the vector. Defaults to zero
  //! @note If `__size == 0` then no memory is allocated.
  vector(__resource_ref __mr, const size_type __size = 0)
      : __buf_(__mr, __size)
  {
    if (__size != 0)
    {
      this->__uninitialized_value_construct_n(begin(), __size);
    }
  }

  //! @brief Constructs a vector of size \p __size using a memory resource and copy-constructs \p __size elements from
  //! \p __value
  //! @param __mr The memory resource to allocate the vector with.
  //! @param __size The size of the vector.
  //! @param __value The value all elements are copied from.
  //! @note If `__size == 0` then no memory is allocated.
  vector(__resource_ref __mr, const size_type __size, const _Tp& __value)
      : __buf_(__mr, __size)
  {
    if (__size != 0)
    {
      this->__uninitialized_fill_n(begin(), __size, __value);
    }
  }

  //! @brief Constructs a vector of size \p __size using a memory and leaves all elements uninitialized
  //! @param __mr The memory resource to allocate the vector with.
  //! @param __size The size of the vector.
  //! @warning This constructor does *NOT* initialize any elements. It is the user's responsibility to ensure that the
  //! elements within `[vec.begin(), vec.end())` are properly initialized, e.g with `cuda::std::uninitialized_copy`
  //! At the destruction of the \c vector all elements in the range `[vec.begin(), vec.end())` will be destroyed
  vector(__resource_ref __mr, const size_type __size, ::cuda::experimental::uninit_t)
      : __buf_(__mr, __size)
      , __size_(__size)
  {}

  //! @brief Constructs a vector using a memory resource and copy-constructs all elements from the input range
  //! ``[__first, __last)``
  //! @param __mr The memory resource to allocate the vector with.
  //! @param __first The start of the input sequence.
  //! @param __last The end of the input sequence.
  //! @note If `__first == __last` then no memory is allocated. Might allocate multiple times
  _LIBCUDACXX_TEMPLATE(class _Iter)
  _LIBCUDACXX_REQUIRES(_CUDA_VSTD::__is_cpp17_input_iterator<_Iter>::value _LIBCUDACXX_AND(
    !_CUDA_VSTD::__is_cpp17_forward_iterator<_Iter>::value))
  vector(__resource_ref __mr, _Iter __first, _Iter __last)
      : __buf_(__mr, 0)
  {
    for (; __first != __last; ++__first)
    {
      emplace_back(*__first);
    }
  }

  //! @brief Constructs a vector using a memory resource and copy-constructs all elements from the forward range
  //! ``[__first, __last)``
  //! @param __mr The memory resource to allocate the vector with.
  //! @param __first The start of the input sequence.
  //! @param __last The end of the input sequence.
  //! @note If `__first == __last` then no memory is allocated
  _LIBCUDACXX_TEMPLATE(class _Iter)
  _LIBCUDACXX_REQUIRES(_CUDA_VSTD::__is_cpp17_forward_iterator<_Iter>::value)
  vector(__resource_ref __mr, _Iter __first, _Iter __last)
      : __buf_(__mr, static_cast<size_type>(_CUDA_VSTD::distance(__first, __last)))
  {
    if (capacity() > 0)
    {
      this->__uninitialized_copy(__first, __last, __unwrapped_begin());
    }
  }

  //! @brief Constructs a vector using a memory resource and copy-constructs all elements from \p __ilist
  //! @param __mr The memory resource to allocate the vector with.
  //! @param __ilist The initializer_list being copied into the vector.
  //! @note If `__ilist.size() == 0` then no memory is allocated
  vector(__resource_ref __mr, _CUDA_VSTD::initializer_list<_Tp> __ilist)
      : __buf_(__mr, __ilist.size())
  {
    if (capacity() > 0)
    {
      this->__uninitialized_copy(__ilist.begin(), __ilist.end(), begin());
    }
  }

#  if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)
  _LIBCUDACXX_TEMPLATE(class _Range)
  _LIBCUDACXX_REQUIRES(__compatible_range<_Range> _LIBCUDACXX_AND(!_CUDA_VRANGES::forward_range<_Range>))
  vector(__resource_ref __mr, _Range&& __range)
      : __buf_(__mr, 0)
  {
    auto __first = _CUDA_VRANGES::begin(__range);
    auto __last  = _CUDA_VRANGES::end(__range);
    for (; __first != __last; ++__first)
    {
      emplace_back(_CUDA_VRANGES::iter_move(__first));
    }
  }

#    ifndef DOXYGEN_SHOULD_SKIP_THIS // doxygen conflates the overloads
  _LIBCUDACXX_TEMPLATE(class _Range)
  _LIBCUDACXX_REQUIRES(__compatible_range<_Range> _LIBCUDACXX_AND _CUDA_VRANGES::forward_range<_Range> _LIBCUDACXX_AND
                         _CUDA_VRANGES::sized_range<_Range>)
  vector(__resource_ref __mr, _Range&& __range)
      : __buf_(__mr, static_cast<size_type>(_CUDA_VRANGES::size(__range)))
  {
    if (capacity() > 0)
    {
      this->__uninitialized_move(_CUDA_VRANGES::begin(__range), _CUDA_VRANGES::__unwrap_end(__range), begin());
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Range)
  _LIBCUDACXX_REQUIRES(__compatible_range<_Range> _LIBCUDACXX_AND _CUDA_VRANGES::forward_range<_Range> _LIBCUDACXX_AND(
    !_CUDA_VRANGES::sized_range<_Range>))
  vector(__resource_ref __mr, _Range&& __range)
      : __buf_(
          __mr,
          static_cast<size_type>(_CUDA_VRANGES::distance(_CUDA_VRANGES::begin(__range), _CUDA_VRANGES::end(__range))))
  {
    if (capacity() > 0)
    {
      this->__uninitialized_move(_CUDA_VRANGES::begin(__range), _CUDA_VRANGES::__unwrap_end(__range), begin());
    }
  }
#    endif // DOXYGEN_SHOULD_SKIP_THIS
#  endif // _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)
  //! @}

  //! @addtogroup iterators
  //! @{
  //! @brief Returns an iterator to the first element of the vector. If the vector is empty, the returned iterator will
  //! be equal to end().
  _CCCL_NODISCARD iterator begin() noexcept
  {
    return iterator{__buf_.data()};
  }

  //! @brief Returns an immutable iterator to the first element of the vector. If the vector is empty, the returned
  //! iterator will be equal to end().
  _CCCL_NODISCARD const_iterator begin() const noexcept
  {
    return const_iterator{__buf_.data()};
  }

  //! @brief Returns an immutable iterator to the first element of the vector. If the vector is empty, the returned
  //! iterator will be equal to end().
  _CCCL_NODISCARD const_iterator cbegin() const noexcept
  {
    return const_iterator{__buf_.data()};
  }

  //! @brief Returns an iterator to the element following the last element of the vector. This element acts as a
  //! placeholder; attempting to access it results in undefined behavior.
  _CCCL_NODISCARD iterator end() noexcept
  {
    return iterator{__buf_.data() + __size_};
  }

  //! @brief Returns an immutable iterator to the element following the last element of the vector. This element acts as
  //! a placeholder; attempting to access it results in undefined behavior.
  _CCCL_NODISCARD const_iterator end() const noexcept
  {
    return const_iterator{__buf_.data() + __size_};
  }

  //! @brief Returns an immutable iterator to the element following the last element of the vector. This element acts as
  //! a placeholder; attempting to access it results in undefined behavior.
  _CCCL_NODISCARD const_iterator cend() const noexcept
  {
    return const_iterator{__buf_.data() + __size_};
  }

  //! @brief Returns a reverse iterator to the first element of the reversed vector. It corresponds to the last element
  //! of the non-reversed vector. If the vector is empty, the returned iterator is equal to rend().
  _CCCL_NODISCARD reverse_iterator rbegin() noexcept
  {
    return reverse_iterator{end()};
  }

  //! @brief Returns an immutable reverse iterator to the first element of the reversed vector. It corresponds to the
  //! last element of the non-reversed vector. If the vector is empty, the returned iterator is equal to rend().
  _CCCL_NODISCARD const_reverse_iterator rbegin() const noexcept
  {
    return const_reverse_iterator{end()};
  }

  //! @brief Returns an immutable reverse iterator to the first element of the reversed vector. It corresponds to the
  //! last element of the non-reversed vector. If the vector is empty, the returned iterator is equal to rend().
  _CCCL_NODISCARD const_reverse_iterator crbegin() const noexcept
  {
    return const_reverse_iterator{end()};
  }

  //! @brief Returns a reverse iterator to the element following the last element of the reversed vector. It corresponds
  //! to the element preceding the first element of the non-reversed vector. This element acts as a placeholder,
  //! attempting to access it results in undefined behavior.
  _CCCL_NODISCARD reverse_iterator rend() noexcept
  {
    return reverse_iterator{begin()};
  }

  //! @brief Returns an immutable reverse iterator to the element following the last element of the reversed vector. It
  //! corresponds to the element preceding the first element of the non-reversed vector. This element acts as a
  //! placeholder, attempting to access it results in undefined behavior.
  _CCCL_NODISCARD const_reverse_iterator rend() const noexcept
  {
    return const_reverse_iterator{begin()};
  }

  //! @brief Returns an immutable reverse iterator to the element following the last element of the reversed vector. It
  //! corresponds to the element preceding the first element of the non-reversed vector. This element acts as a
  //! placeholder, attempting to access it results in undefined behavior.
  _CCCL_NODISCARD const_reverse_iterator crend() const noexcept
  {
    return const_reverse_iterator{begin()};
  }

  //! @brief Returns a pointer to the first element of the vector. If the vector has not allocated memory the pointer
  //! will be null.
  _CCCL_NODISCARD pointer data() noexcept
  {
    return __buf_.data();
  }

  //! @brief Returns a pointer to the first element of the vector. If the vector has not allocated memory the pointer
  //! will be null.
  _CCCL_NODISCARD const_pointer data() const noexcept
  {
    return __buf_.data();
  }

  //! @brief Returns a pointer to the first element of the vector. If the vector is empty, the returned
  //! pointer will be null.
  _CCCL_NODISCARD pointer __unwrapped_begin() noexcept
  {
    return __buf_.data();
  }

  //! @brief Returns a const pointer to the first element of the vector. If the vector is empty, the returned
  //! pointer will be null.
  _CCCL_NODISCARD const_pointer __unwrapped_begin() const noexcept
  {
    return __buf_.data();
  }

  //! @brief Returns a pointer to the element following the last element of the vector. This element acts as a
  //! placeholder; attempting to access it results in undefined behavior.
  _CCCL_NODISCARD pointer __unwrapped_end() noexcept
  {
    return __buf_.data() + __size_;
  }

  //! @brief Returns a const pointer to the element following the last element of the vector. This element acts as
  //! a placeholder; attempting to access it results in undefined behavior.
  _CCCL_NODISCARD const_pointer __unwrapped_end() const noexcept
  {
    return __buf_.data() + __size_;
  }

  //! @}

  //! @addtogroup access
  //! @{
  //! @brief Returns a reference to the \p __n 'th element of the vector
  //! @param __n The index of the element we want to access
  _CCCL_NODISCARD reference operator[](const size_type __n) noexcept
  {
    return begin()[__n];
  }

  //! @brief Returns a reference to the \p __n 'th element of the vector
  //! @param __n The index of the element we want to access
  _CCCL_NODISCARD const_reference operator[](const size_type __n) const noexcept
  {
    return begin()[__n];
  }

  //! @brief Returns a reference to the first element of the vector
  _CCCL_NODISCARD reference first() noexcept
  {
    return begin()[0];
  }

  //! @brief Returns a reference to the first element of the vector
  _CCCL_NODISCARD const_reference first() const noexcept
  {
    return begin()[0];
  }

  //! @brief Returns a reference to the last element of the vector
  _CCCL_NODISCARD reference back() noexcept
  {
    return begin()[__size_ - 1];
  }

  //! @brief Returns a reference to the last element of the vector
  _CCCL_NODISCARD const_reference back() const noexcept
  {
    return begin()[__size_ - 1];
  }
  //! @}

  //! @addtogroup capacity
  //! @{
  //! @brief Returns the current number of elements stored in the vector.
  _CCCL_NODISCARD size_type size() const noexcept
  {
    return __size_;
  }

  //! @brief Returns true if the vector is empty.
  _CCCL_NODISCARD bool empty() const noexcept
  {
    return __size_ == 0;
  }

  //! @brief Returns the capacity of the current allocation of the vector..
  _CCCL_NODISCARD size_type capacity() const noexcept
  {
    return static_cast<size_type>(__buf_.size());
  }

  //! @brief Returns the maximal size of the vector.
  _CCCL_NODISCARD size_type max_size() const noexcept
  {
    return static_cast<size_type>((_CUDA_VSTD::numeric_limits<difference_type>::max)());
  }

  //! @rst
  //! Returns the :ref:`resource_ref <libcudacxx-extended-api-memory-resources-resource-ref>` used to allocate
  //! the buffer.
  //! @endrst
  _CCCL_NODISCARD _CUDA_VMR::resource_ref<_Properties...> resource() const noexcept
  {
    return __buf_.resource();
  }
  //! @}

  //! @addtogroup modification
  //! @{

  iterator insert(const_iterator __cpos, const _Tp& __value)
  {
    return emplace(__cpos, __value);
  }

  iterator insert(const_iterator __cpos, _Tp&& __value)
  {
    return emplace(__cpos, _CUDA_VSTD::move(__value));
  }

  iterator insert(const_iterator __cpos, const size_type __count, const _Tp& __value)
  {
    const iterator __pos = __cpos.__to_mutable();
    const iterator __end = end();
    if (__size_ + __count > capacity())
    {
      _CUDA_VSTD::__throw_bad_alloc();
    }
    else if (__pos < begin() || __end < __pos)
    {
      _CUDA_VSTD::__throw_out_of_range("inplace_vector::insert(const_iterator, size_type, T)");
    }

    if (__count == 0)
    {
      return __pos;
    }

    if (__pos == __end)
    {
      this->__uninitialized_fill(__end, __end + __count, __value);
      return __pos;
    }

    const iterator __middle = __pos + __count;
    if (__end <= __middle)
    { // all existing elements are pushed into uninitialized storage
      this->__uninitialized_fill(__end, __middle, __value);
      this->__uninitialized_move(__pos, __end, __middle);
      _CUDA_VSTD::fill(__pos, __end, __value);
    }
    else
    { // some elements get copied into existing storage
      this->__uninitialized_move(__end - __count, __end, __end);
      _CUDA_VSTD::move_backward(__pos, __end - __count, __end);
      _CUDA_VSTD::fill(__pos, __middle, __value);
    }

    return __pos;
  }

  _LIBCUDACXX_TEMPLATE(class _Iter)
  _LIBCUDACXX_REQUIRES(_CUDA_VSTD::__is_cpp17_input_iterator<_Iter>::value _LIBCUDACXX_AND(
    !_CUDA_VSTD::__is_cpp17_forward_iterator<_Iter>::value))
  iterator insert(const_iterator __cpos, _Iter __first, _Iter __last)
  {
    // add all new elements to the back then rotate
    const iterator __old_end = end();
    for (; __first != __last; ++__first)
    {
      emplace_back(*__first);
    }

    const iterator __pos = __cpos.__to_mutable();
    _CUDA_VSTD::rotate(__pos, __old_end, end());
    return __pos;
  }

  _LIBCUDACXX_TEMPLATE(class _Iter)
  _LIBCUDACXX_REQUIRES(_CUDA_VSTD::__is_cpp17_forward_iterator<_Iter>::value)
  iterator insert(const_iterator __cpos, _Iter __first, _Iter __last)
  {
    const iterator __pos = __cpos.__to_mutable();
    const iterator __end = end();
    const auto __count   = _CUDA_VSTD::distance(__first, __last);
    if (__size_ + __count > capacity())
    {
      _CUDA_VSTD::__throw_bad_alloc();
    }
    else if (__pos < begin() || __end < __pos)
    {
      _CUDA_VSTD::__throw_out_of_range("inplace_vector::insert(const_iterator, Iter, Iter)");
    }

    if (__count == 0)
    {
      return __pos;
    }

    if (__pos == __end)
    {
      this->__uninitialized_copy(__first, __last, __end);
      return __pos;
    }

    const iterator __middle = __pos + __count;
    const auto __to_copy    = __end - __pos;
    if (__end <= __middle)
    { // all existing elements are pushed into uninitialized storage
      _Iter __imiddle = _CUDA_VSTD::next(__first, __to_copy);
      this->__uninitialized_copy(__imiddle, __last, __end);
      this->__uninitialized_move(__pos, __end, __middle);
      _CUDA_VSTD::copy(__first, __imiddle, __pos);
    }
    else
    { // all new elements get copied into existing storage
      this->__uninitialized_move(__end - __count, __end, __end);
      _CUDA_VSTD::move_backward(__pos, __end - __count, __end);
      _CUDA_VSTD::copy(__first, __last, __pos);
    }

    return __pos;
  }

  iterator insert(const_iterator __cpos, _CUDA_VSTD::initializer_list<_Tp> __ilist)
  {
    const iterator __pos = __cpos.__to_mutable();
    const iterator __end = end();
    const auto __count   = __ilist.size();
    if (__size_ + __count > capacity())
    {
      _CUDA_VSTD::__throw_bad_alloc();
    }
    else if (__pos < begin() || __end < __pos)
    {
      _CUDA_VSTD::__throw_out_of_range("inplace_vector::insert(const_iterator, initializer_list)");
    }

    if (__count == 0)
    {
      return __pos;
    }

    if (__pos == __end)
    {
      this->__uninitialized_copy(__ilist.begin(), __ilist.end(), __end);
      return __pos;
    }

    const iterator __middle = __pos + __count;
    const auto __to_copy    = __end - __pos;
    if (__end <= __middle)
    { // all existing elements are pushed into uninitialized storage
      auto __imiddel = __ilist.begin() + __to_copy;
      this->__uninitialized_copy(__imiddel, __ilist.end(), __end);
      this->__uninitialized_move(__pos, __end, __middle);
      _CUDA_VSTD::copy(__ilist.begin(), __imiddel, __pos);
    }
    else
    { // all new elements get copied into existing storage
      this->__uninitialized_move(__end - __count, __end, __end);
      _CUDA_VSTD::move_backward(__pos, __end - __count, __end);
      _CUDA_VSTD::copy(__ilist.begin(), __ilist.end(), __pos);
    }

    return __pos;
  }

#  if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)
  _LIBCUDACXX_TEMPLATE(class _Range)
  _LIBCUDACXX_REQUIRES(__compatible_range<_Range> _LIBCUDACXX_AND(!_CUDA_VRANGES::forward_range<_Range>))
  iterator insert_range(const_iterator __cpos, _Range&& __range)
  {
    // add all new elements to the back then rotate
    auto __first             = _CUDA_VRANGES::begin(__range);
    auto __last              = _CUDA_VRANGES::end(__range);
    const iterator __old_end = end();
    for (; __first != __last; ++__first)
    {
      emplace_back(*__first);
    }

    const iterator __pos = __cpos.__to_mutable();
    _CUDA_VSTD::rotate(__pos, __old_end, end());
    return __pos;
  }

#    ifndef DOXYGEN_SHOULD_SKIP_THIS // doxygen conflates both overloads
  _LIBCUDACXX_TEMPLATE(class _Range)
  _LIBCUDACXX_REQUIRES(__compatible_range<_Range> _LIBCUDACXX_AND _CUDA_VRANGES::forward_range<_Range>)
  iterator insert_range(const_iterator __cpos, _Range&& __range)
  {
    auto __first = _CUDA_VRANGES::begin(__range);
    return insert(__cpos, __first, _CUDA_VRANGES::__unwrap_end(__range));
  }
#    endif // DOXYGEN_SHOULD_SKIP_THIS

  _LIBCUDACXX_TEMPLATE(class _Range)
  _LIBCUDACXX_REQUIRES(__compatible_range<_Range> _LIBCUDACXX_AND(!_CUDA_VRANGES::forward_range<_Range>))
  void append_range(_Range&& __range)
  {
    auto __first = _CUDA_VRANGES::begin(__range);
    auto __last  = _CUDA_VRANGES::end(__range);
    for (; __first != __last; ++__first)
    {
      emplace_back(*__first);
    }
  }

#    ifndef DOXYGEN_SHOULD_SKIP_THIS // doxygen conflates both overloads
  _LIBCUDACXX_TEMPLATE(class _Range)
  _LIBCUDACXX_REQUIRES(__compatible_range<_Range> _LIBCUDACXX_AND _CUDA_VRANGES::forward_range<_Range>)
  void append_range(_Range&& __range)
  {
    auto __first = _CUDA_VRANGES::begin(__range);
    insert(end(), __first, _CUDA_VRANGES::__unwrap_end(__range));
  }
#    endif // DOXYGEN_SHOULD_SKIP_THIS
#  endif // _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

  template <class... _Args>
  iterator emplace(const_iterator __cpos, _Args&&... __args)
  {
    const iterator __pos = __cpos.__to_mutable();
    const iterator __end = end();
    if (__size_ == capacity())
    {
      _CUDA_VSTD::__throw_bad_alloc();
    }
    else if (__pos < begin() || __end < __pos)
    {
      _CUDA_VSTD::__throw_out_of_range("inplace_vector::emplace(const_iterator, Args...)");
    }

    if (__pos == __end)
    {
      unchecked_emplace_back(_CUDA_VSTD::forward<_Args>(__args)...);
    }
    else
    {
      _Tp __temp{_CUDA_VSTD::forward<_Args>(__args)...};
      unchecked_emplace_back(_CUDA_VSTD::move(*(__end - 1)));
      _CUDA_VSTD::move_backward(__pos, __end - 1, __end);
      *__pos = _CUDA_VSTD::move(__temp);
    }

    return __pos;
  }

  template <class... _Args>
  reference emplace_back(_Args&&... __args)
  {
    if (__size_ == capacity())
    {
      _CUDA_VSTD::__throw_bad_alloc();
    }

    return unchecked_emplace_back(_CUDA_VSTD::forward<_Args>(__args)...);
  }

  reference push_back(const _Tp& __value)
  {
    if (__size_ == capacity())
    {
      _CUDA_VSTD::__throw_bad_alloc();
    }

    return unchecked_emplace_back(__value);
  }

  reference push_back(_Tp&& __value)
  {
    if (__size_ == capacity())
    {
      _CUDA_VSTD::__throw_bad_alloc();
    }

    return unchecked_emplace_back(_CUDA_VSTD::move(__value));
  }

  template <class... _Args>
  pointer try_emplace_back(_Args&&... __args)
#  ifndef DOXYGEN_SHOULD_SKIP_THIS // doxygen breaks with the noexcept
    noexcept(_CCCL_TRAIT(_CUDA_VSTD::is_nothrow_constructible, _Tp, _Args...))
#  endif // DOXYGEN_SHOULD_SKIP_THIS
  {
    if (__size_ == capacity())
    {
      return nullptr;
    }

    return _CUDA_VSTD::addressof(unchecked_emplace_back(_CUDA_VSTD::forward<_Args>(__args)...));
  }

  pointer try_push_back(const _Tp& __value)
#  ifndef DOXYGEN_SHOULD_SKIP_THIS // doxygen breaks with the noexcept
    noexcept(_CCCL_TRAIT(_CUDA_VSTD::is_nothrow_copy_constructible, _Tp))
#  endif // DOXYGEN_SHOULD_SKIP_THIS
  {
    if (__size_ == capacity())
    {
      return nullptr;
    }

    return _CUDA_VSTD::addressof(unchecked_emplace_back(__value));
  }

  pointer try_push_back(_Tp&& __value)
#  ifndef DOXYGEN_SHOULD_SKIP_THIS // doxygen breaks with the noexcept
    noexcept(_CCCL_TRAIT(_CUDA_VSTD::is_nothrow_move_constructible, _Tp))
#  endif // DOXYGEN_SHOULD_SKIP_THIS
  {
    if (__size_ == capacity())
    {
      return nullptr;
    }

    return _CUDA_VSTD::addressof(unchecked_emplace_back(_CUDA_VSTD::move(__value)));
  }

#  if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)
  _LIBCUDACXX_TEMPLATE(class _Range)
  _LIBCUDACXX_REQUIRES(__compatible_range<_Range> _LIBCUDACXX_AND(!_CUDA_VRANGES::forward_range<_Range>))
  _CUDA_VRANGES::iterator_t<_Range>
  try_append_range(_Range&& __range) noexcept(_CCCL_TRAIT(_CUDA_VSTD::is_nothrow_move_constructible, _Tp))
  {
    auto __first = _CUDA_VRANGES::begin(__range);
    auto __last  = _CUDA_VRANGES::end(__range);
    for (; size() != capacity() && __first != __last; ++__first)
    {
      emplace_back(*__first);
    }
    return __first;
  }

#    ifndef DOXYGEN_SHOULD_SKIP_THIS // doxygen conflates both overloads
  _LIBCUDACXX_TEMPLATE(class _Range)
  _LIBCUDACXX_REQUIRES(__compatible_range<_Range> _LIBCUDACXX_AND _CUDA_VRANGES::forward_range<_Range> _LIBCUDACXX_AND
                         _CUDA_VRANGES::sized_range<_Range>)
  _CUDA_VRANGES::iterator_t<_Range>
  try_append_range(_Range&& __range) noexcept(_CCCL_TRAIT(_CUDA_VSTD::is_nothrow_move_constructible, _Tp))
  {
    const auto __capacity = capacity() - size();
    const auto __size     = _CUDA_VRANGES::size(__range);
    const auto __diff     = __size < __capacity ? __size : __capacity;

    auto __first  = _CUDA_VRANGES::begin(__range);
    auto __middle = _CUDA_VRANGES::next(__first, __diff);
    this->__uninitialized_move(__first, __middle, end());
    return __middle;
  }

  _LIBCUDACXX_TEMPLATE(class _Range)
  _LIBCUDACXX_REQUIRES(__compatible_range<_Range> _LIBCUDACXX_AND _CUDA_VRANGES::forward_range<_Range> _LIBCUDACXX_AND(
    !_CUDA_VRANGES::sized_range<_Range>))
  _CUDA_VRANGES::iterator_t<_Range>
  try_append_range(_Range&& __range) noexcept(_CCCL_TRAIT(_CUDA_VSTD::is_nothrow_move_constructible, _Tp))
  {
    const auto __capacity = static_cast<ptrdiff_t>(capacity() - size());
    auto __first          = _CUDA_VRANGES::begin(__range);
    const auto __size = static_cast<ptrdiff_t>(_CUDA_VRANGES::distance(__first, _CUDA_VRANGES::__unwrap_end(__range)));
    const ptrdiff_t __diff = __size < __capacity ? __size : __capacity;

    auto __middle = _CUDA_VRANGES::next(__first, __diff);
    this->__uninitialized_move(__first, __middle, end());
    return __middle;
  }
#    endif // DOXYGEN_SHOULD_SKIP_THIS
#  endif // _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

  template <class... _Args>
  reference unchecked_emplace_back(_Args&&... __args)
#  ifndef DOXYGEN_SHOULD_SKIP_THIS // doxygen breaks with the noexcept
    noexcept(_CCCL_TRAIT(_CUDA_VSTD::is_nothrow_constructible, _Tp, _Args...))
#  endif // DOXYGEN_SHOULD_SKIP_THIS
  {
    auto __final = _CUDA_VSTD::__construct_at(__unwrapped_end(), _CUDA_VSTD::forward<_Args>(__args)...);
    ++__size_;
    return *__final;
  }

  reference unchecked_push_back(const _Tp& __value)
#  ifndef DOXYGEN_SHOULD_SKIP_THIS // doxygen breaks with the noexcept
    noexcept(_CCCL_TRAIT(_CUDA_VSTD::is_nothrow_copy_constructible, _Tp))
#  endif // DOXYGEN_SHOULD_SKIP_THIS
  {
    return unchecked_emplace_back(__value);
  }

  reference unchecked_push_back(_Tp&& __value)
#  ifndef DOXYGEN_SHOULD_SKIP_THIS // doxygen breaks with the noexcept
    noexcept(_CCCL_TRAIT(_CUDA_VSTD::is_nothrow_move_constructible, _Tp))
#  endif // DOXYGEN_SHOULD_SKIP_THIS
  {
    return unchecked_emplace_back(_CUDA_VSTD::move(__value));
  }

  void pop_back() noexcept
  {
    this->__destroy_from(--end());
  }

  iterator erase(const_iterator __cpos) noexcept(_CCCL_TRAIT(_CUDA_VSTD::is_nothrow_move_assignable, _Tp))
  {
    const iterator __pos = __cpos.__to_mutable();
    const iterator __end = end();
    if (__pos == __end)
    {
      return __pos;
    }

    if (__size_ == 0 || __pos < begin() || __end < __pos)
    {
      _CUDA_VSTD_NOVERSION::terminate();
    }

    _CUDA_VSTD::move(__pos + 1, __end, __pos);
    this->__destroy_from(__end - 1);
    return __pos;
  }

  iterator erase(const_iterator __cfirst,
                 const_iterator __clast) noexcept(_CCCL_TRAIT(_CUDA_VSTD::is_nothrow_move_assignable, _Tp))
  {
    const iterator __first = __cfirst.__to_mutable();
    const iterator __last  = __clast.__to_mutable();
    const iterator __end   = end();
    if (__first == __last)
    {
      return __last;
    }

    if (__first < begin() || __end < __last)
    {
      _CUDA_VSTD_NOVERSION::terminate();
    }

    const auto __new_end = _CUDA_VSTD::move(__last, __end, __first);
    this->__destroy_from(__new_end);
    return __first;
  }

  // [containers.sequences.vector.erasure]
  size_type __erase(const _Tp& __value) noexcept(_CCCL_TRAIT(_CUDA_VSTD::is_nothrow_move_assignable, _Tp))
  {
    const pointer __old_end = __unwrapped_end();
    const pointer __new_end = _CUDA_VSTD::remove(__unwrapped_begin(), __old_end, __value);
    this->__destroy_from(__new_end);
    return static_cast<size_type>(__old_end - __new_end);
  }

  template <class _Pred>
  size_type __erase_if(_Pred __pred) noexcept(_CCCL_TRAIT(_CUDA_VSTD::is_nothrow_move_assignable, _Tp))
  {
    const pointer __old_end = __unwrapped_end();
    const pointer __new_end = _CUDA_VSTD::remove_if(__unwrapped_begin(), __old_end, _CUDA_VSTD::move(__pred));
    this->__destroy_from(__new_end);
    return static_cast<size_type>(__old_end - __new_end);
  }

  //! @brief Destroys all elements in the \c vector and sets the size to 0
  void clear() noexcept
  {
    this->__destroy_from(begin());
  }

  //! @brief Provides sufficient storage for \p __count elements in the \c vector without creating any new elements
  //! @param __size The intended capacity of the vector.
  //! If `__size <= vec.capacity()` this is a noop
  void reserve(const size_type __size) noexcept
  {
    if (__size <= capacity())
    {
      return;
    }

    uninitialized_buffer<_Tp, _Properties...> __new_buf{resource(), __size};
    _CUDA_VSTD::uninitialized_move(__unwrapped_begin(), __unwrapped_end(), __new_buf.data());
    __buf_ = _CUDA_VSTD::move(__new_buf);
  }

  //! @brief Changes the size of the \c vector to \p __size and value-initializes new elements
  //! @param __size The intended size of the vector.
  //! If `__size < vec.size()` then it destroys all superfluous elements. Otherwise, it value-initializes new elements
  void resize(const size_type __size) noexcept
  {
    if (__size <= __size_)
    {
      this->__destroy_from(begin() + __size);
    }
    else
    {
      if (__size <= capacity())
      {
        this->__uninitialized_value_construct_n(end(), __size - __size_);
      }
      else
      {
        // We rely on begin / end in the uninitialized_move call, but if use the internal
        // __uninitialized_value_construct_n then we have already adopted the size so classic algorithm here, because we
        // have not moved anything over yet
        uninitialized_buffer<_Tp, _Properties...> __new_buf{resource(), __size};
        _CUDA_VSTD::uninitialized_value_construct_n(__new_buf.begin() + __size_, __size - __size_);
        _CUDA_VSTD::uninitialized_move(begin(), end(), __new_buf.begin());
        __buf_  = _CUDA_VSTD::move(__new_buf);
        __size_ = __size;
      }
    }
  }

  //! @brief Changes the size of the \c vector to \p __size and copy-constructs new elements from \p __value
  //! @param __size The intended size of the vector.
  //! @param __value The element to be copied into the vector when growing.
  //! If `__size < vec.size()` then it destroys all superfluous elements. Otherwise, it copy-constructs new elements
  //! from \p __value
  void resize(const size_type __size, const _Tp& __value) noexcept
  {
    if (__size <= __size_)
    {
      this->__destroy_from(begin() + __size);
    }
    else
    {
      if (__size <= capacity())
      {
        this->__uninitialized_fill_n(end(), __size - __size_, __value);
      }
      else
      {
        // We rely on begin / end in the uninitialized_move call, but if use the internal
        // uninitialized_fill_n then we have already adopted the size so classic algorithm here, because we
        // have not moved anything over yet
        uninitialized_buffer<_Tp, _Properties...> __new_buf{resource(), __size};
        _CUDA_VSTD::uninitialized_fill_n(__new_buf.begin() + __size_, __size - __size_, __value);
        _CUDA_VSTD::uninitialized_move(begin(), end(), __new_buf.begin());
        _CUDA_VSTD::__destroy(begin(), end());
        __buf_  = _CUDA_VSTD::move(__new_buf);
        __size_ = __size;
      }
    }
  }

  //! @brief Changes the size of the \c vector to \p __size and leaves new elements uninitialized
  //! @param __size The intended size of the vector.
  //! If `__size < vec.size()` then it destroys all superfluous elements. Otherwise, it provides sufficient storage and
  //! adjusts the size of the vector, but leaves the new elements uninitialized.
  void resize(const size_type __size, uninit_t) noexcept
  {
    if (__size <= __size_)
    {
      this->__destroy_from(begin() + __size);
    }
    else
    {
      if (capacity() < __size)
      {
        uninitialized_buffer<_Tp, _Properties...> __new_buf{resource(), __size};
        _CUDA_VSTD::uninitialized_move(begin(), end(), __new_buf.begin());
        _CUDA_VSTD::__destroy(begin(), end());
        __buf_ = _CUDA_VSTD::move(__new_buf);
      }
      __size_ = __size;
    }
  }

  //! @brief Reallocates the storage, so that `vec.capacity() == vec.size()`
  //! If `vec.size() == vec.capacity()` this is a noop. If `vec.empty()` holds, then no storage is allocated
  void shrink_to_fit() noexcept
  {
    if (__size_ == capacity())
    {
      return;
    }

    uninitialized_buffer<_Tp, _Properties...> __new_buf{resource(), __size_};
    _CUDA_VSTD::uninitialized_move(__unwrapped_begin(), __unwrapped_end(), __new_buf.data());
    __buf_ = _CUDA_VSTD::move(__new_buf);
  }

  //! @brief Swaps the contents of a vector with those of \p __other
  //! @param __other The other vector.
  void swap(vector& __other) noexcept
  {
    uninitialized_buffer<_Tp, _Properties...> __temp(_CUDA_VSTD::move(__other.__buf_));
    __other.__buf_ = _CUDA_VSTD::move(__buf_);
    __buf_         = _CUDA_VSTD::move(__temp);
    _CUDA_VSTD::swap(__size_, __other.__size_);
  }

  //! @brief Swaps the contents of two vectors
  //! @param __lhs One vector.
  //! @param __rhs The other vector.
  friend void swap(vector& __lhs, vector& __rhs) noexcept
  {
    __lhs.swap(__rhs);
  }
  //! @}

#  ifndef DOXYGEN_SHOULD_SKIP_THIS // friend functions are currently broken
  //! @brief Forwards the passed properties
  _LIBCUDACXX_TEMPLATE(class _Property)
  _LIBCUDACXX_REQUIRES((!property_with_value<_Property>) _LIBCUDACXX_AND _CUDA_VSTD::_One_of<_Property, _Properties...>)
  friend void get_property(const vector&, _Property) noexcept {}
#  endif // DOXYGEN_SHOULD_SKIP_THIS
};

// [containers.sequences.inplace.vector.erasure]
template <class _Tp, class... _Properties>
size_t erase(vector<_Tp, _Properties...>& __cont,
             const _Tp& __value) noexcept(_CCCL_TRAIT(_CUDA_VSTD::is_nothrow_move_assignable, _Tp))
{
  return __cont.__erase(__value);
}

template <class _Tp, class _Pred, class... _Properties>
size_t erase_if(vector<_Tp, _Properties...>& __cont,
                _Pred __pred) noexcept(_CCCL_TRAIT(_CUDA_VSTD::is_nothrow_move_assignable, _Tp))
{
  return __cont.__erase_if(__pred);
}

template <class _Tp>
using device_vector = vector<_Tp, _CUDA_VMR::device_accessible>;

} // namespace cuda::experimental

_CCCL_POP_MACROS

#endif // _CCCL_STD_VER >= 2014 && !_CCCL_COMPILER_MSVC_2017 && LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#endif //__CUDAX__CONTAINERS_VECTOR_H
