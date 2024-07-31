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
//! The elements are initialized during construction, which may require a kernel launch.
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
  //! No destructor is run if `_Tp` is trivially destructible.
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
#  if _CCCL_STD_VER >= 2017
      ::new (_CUDA_VSTD::__voidify(*__curr)) _Tp(_CUDA_VRANGES::iter_move(__first));
#  else // ^^^ C++17 ^^^ / vvv C++14 vvv
      ::new (_CUDA_VSTD::__voidify(*__curr)) _Tp(_CUDA_VSTD::move(*__first));
#  endif // _CCCL_STD_VER <= 2014
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
#  if _CCCL_STD_VER >= 2017
      ::new (_CUDA_VSTD::__voidify(*__curr)) _Tp(_CUDA_VRANGES::iter_move(__first));
#  else // ^^^ C++17 ^^^ / vvv C++14 vvv
      ::new (_CUDA_VSTD::__voidify(*__curr)) _Tp(_CUDA_VSTD::move(*__first));
#  endif // _CCCL_STD_VER <= 2014
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
      this->__uninitialized_copy(__other.__unwrapped_begin(), __other.__unwrapped_end(), __unwrapped_begin());
    }
  }

  //! @brief Move-constructs a vector
  //! @param __other The other vector.
  //! The new vector takes ownership of the allocation of \p __other and resets it.
  vector(vector&& __other) noexcept
      : __buf_(_CUDA_VSTD::move(__other.__buf_))
      , __size_(_CUDA_VSTD::exchange(__other.__size_, 0))
  {}

  //! @brief Copy-assigns a vector
  //! @param __other The other vector.
  //! @note Even if the old vector would have enough storage available, we may have to reallocate if the stored memory
  //! resource is not equal to the new one. In that case no memory is allocated if \p __other is empty.
  vector& operator=(const vector& __other)
  {
    // There is sufficient space in the allocation and the resources are compatible
    if (resource() == __other.resource() && capacity() >= __other.__size_)
    {
      if (__size_ >= __other.__size_)
      {
        const auto __res =
          _CUDA_VSTD::copy(__other.__unwrapped_begin(), __other.__unwrapped_end(), __unwrapped_begin());
        this->__destroy_from(__res);
      }
      else
      {
        const auto __res =
          _CUDA_VSTD::copy(__other.__unwrapped_begin(), __other.__unwrapped_begin() + __size_, __unwrapped_begin());
        this->__uninitialized_copy(__other.__unwrapped_begin() + __size_, __other.__unwrapped_end(), __res);
      }
      return *this;
    }

    // We need to reallocate and copy. Note we do not change the size of the current vector until the copy is done
    uninitialized_buffer<_Tp, _Properties...> __new_buf{__other.resource(), __other.__size_};
    _CUDA_VSTD::uninitialized_copy(__other.__unwrapped_begin(), __other.__unwrapped_end(), __new_buf.begin());

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

  //! @brief Assigns an initializer_list to a vector, replacing its content with that of the initializer_list
  //! @param __ilist The initializer_list to be assigned
  vector& operator=(_CUDA_VSTD::initializer_list<_Tp> __ilist)
  {
    const auto __count = __ilist.size();
    if (__count <= __size_)
    {
      const iterator __new_end = _CUDA_VSTD::copy(__ilist.begin(), __ilist.end(), begin());
      this->__destroy_from(__new_end);
      return *this;
    }

    if (capacity() < __count)
    {
      uninitialized_buffer<_Tp, _Properties...> __new_buf{resource(), __count};
      _CUDA_VSTD::uninitialized_copy(__ilist.begin(), __ilist.end(), __new_buf.begin());
      _CUDA_VSTD::__destroy(__unwrapped_begin(), __unwrapped_end());
      __buf_  = _CUDA_VSTD::move(__new_buf);
      __size_ = __count;
    }
    else
    {
      _CUDA_VSTD::copy(__ilist.begin(), __ilist.begin() + __size_, __unwrapped_begin());
      this->__uninitialized_copy(__ilist.begin() + __size_, __ilist.end(), __unwrapped_end());
    }
    return *this;
  }

  //! @brief Destroys the \c vector and deallocates the storage after destroying all elements
  //! @note Does not destroy elements if `is_trivially_destructible_v<_Tp>` holds.
  ~vector() noexcept
  {
    this->__destroy_from(__unwrapped_begin());
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
      this->__uninitialized_value_construct_n(__unwrapped_begin(), __size);
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
      this->__uninitialized_fill_n(__unwrapped_begin(), __size, __value);
    }
  }

  //! @brief Constructs a vector of size \p __size using a memory and leaves all elements uninitialized
  //! @param __mr The memory resource to allocate the vector with.
  //! @param __size The size of the vector.
  //! @warning This constructor does *NOT* initialize any elements. It is the user's responsibility to ensure that the
  //! elements within `[vec.begin(), vec.end())` are properly initialized, e.g with `cuda::std::uninitialized_copy`.
  //! At the destruction of the \c vector all elements in the range `[vec.begin(), vec.end())` will be destroyed.
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
      this->__uninitialized_copy(__ilist.begin(), __ilist.end(), __unwrapped_begin());
    }
  }

  //! @brief Constructs a vector using a memory resource and an input range
  //! @param __mr The memory resource to allocate the vector with.
  //! @param __range The input range to be moved into the vector.
  //! @note If `__range.size() == 0` then no memory is allocated. May allocate multiple times in case of input ranges.
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

#  ifndef DOXYGEN_SHOULD_SKIP_THIS // doxygen conflates the overloads
  //! @brief Constructs a vector using a memory resource and an input range
  //! @param __mr The memory resource to allocate the vector with.
  //! @param __range The input range to be moved into the vector.
  //! @note If `__range.size() == 0` then no memory is allocated.
  _LIBCUDACXX_TEMPLATE(class _Range)
  _LIBCUDACXX_REQUIRES(__compatible_range<_Range> _LIBCUDACXX_AND _CUDA_VRANGES::forward_range<_Range> _LIBCUDACXX_AND
                         _CUDA_VRANGES::sized_range<_Range>)
  vector(__resource_ref __mr, _Range&& __range)
      : __buf_(__mr, static_cast<size_type>(_CUDA_VRANGES::size(__range)))
  {
    if (capacity() > 0)
    {
      this->__uninitialized_move(
        _CUDA_VRANGES::begin(__range), _CUDA_VRANGES::__unwrap_end(__range), __unwrapped_begin());
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
      this->__uninitialized_move(
        _CUDA_VRANGES::begin(__range), _CUDA_VRANGES::__unwrap_end(__range), __unwrapped_begin());
    }
  }
#  endif // DOXYGEN_SHOULD_SKIP_THIS
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

  //! @addtogroup assign
  //! @{
  //! @brief Replaces the content of the vector with `__count` copies of `__value`
  //! @param __count The number of elements to assign.
  //! @param __value The element to be copied.
  //! @note Neither frees not allocates memory if `__first == __last`.
  void assign(const size_type __count, const _Tp& __value)
  {
    if (capacity() < __count)
    {
      uninitialized_buffer<_Tp, _Properties...> __new_buf{__buf_.resource(), __count};
      _CUDA_VSTD::uninitialized_fill_n(__new_buf.data(), __count, __value);
      _CUDA_VSTD::__destroy(__unwrapped_begin(), __unwrapped_end());
      __buf_  = _CUDA_VSTD::move(__new_buf);
      __size_ = __count;
      return;
    }

    const pointer __begin = __unwrapped_begin();
    const pointer __end   = __unwrapped_end();
    if (__count < __size_)
    {
      _CUDA_VSTD::fill(__begin, __begin + __count, __value);
      this->__destroy_from(__begin + __count);
    }
    else
    {
      _CUDA_VSTD::fill(__begin, __end, __value);
      this->__uninitialized_fill_n(__end, __count - __size_, __value);
    }
  }

  //! @brief Replaces the content of the vector with the sequence `[__first, __last)`
  //! @param __first Iterator to the first element of the input sequence.
  //! @param __last Iterator after the last element of the input sequence.
  //! @note Neither frees not allocates memory if `__first == __last`. May allocate multiple times in case of input
  //! iterators.
  _LIBCUDACXX_TEMPLATE(class _Iter)
  _LIBCUDACXX_REQUIRES(_CUDA_VSTD::__is_cpp17_input_iterator<_Iter>::value _LIBCUDACXX_AND(
    !_CUDA_VSTD::__is_cpp17_forward_iterator<_Iter>::value))
  void assign(_Iter __first, _Iter __last)
  {
    pointer __end = __unwrapped_end();
    for (pointer __current = __unwrapped_begin(); __current != __end; ++__current, (void) ++__first)
    {
      if (__first == __last)
      {
        this->__destroy_from(__current);
        return;
      }
      *__current = *__first;
    }

    for (; __first != __last; ++__first)
    {
      emplace_back(*__first);
    }
  }

  //! @brief Replaces the content of the vector with the sequence `[__first, __last)`
  //! @param __first Iterator to the first element of the input sequence.
  //! @param __last Iterator after the last element of the input sequence.
  //! @note Neither frees not allocates memory if `__first == __last`.
  _LIBCUDACXX_TEMPLATE(class _Iter)
  _LIBCUDACXX_REQUIRES(_CUDA_VSTD::__is_cpp17_forward_iterator<_Iter>::value)
  void assign(_Iter __first, _Iter __last)
  {
    const auto __count = static_cast<size_type>(_CUDA_VSTD::distance(__first, __last));
    if (capacity() < __count)
    {
      uninitialized_buffer<_Tp, _Properties...> __new_buf{__buf_.resource(), __count};
      _CUDA_VSTD::uninitialized_copy(__first, __last, __new_buf.data());
      _CUDA_VSTD::__destroy(__unwrapped_begin(), __unwrapped_end());
      __buf_  = _CUDA_VSTD::move(__new_buf);
      __size_ = __count;
      return;
    }

    if (__count < __size_)
    {
      const iterator __new_end = _CUDA_VSTD::copy(__first, __last, __unwrapped_begin());
      this->__destroy_from(__new_end);
    }
    else
    {
      _Iter __middle = _CUDA_VSTD::next(__first, __size_);
      _CUDA_VSTD::copy(__first, __middle, __unwrapped_begin());
      this->__uninitialized_copy(__middle, __last, __unwrapped_end());
    }
  }

  //! @brief Replaces the content of the vector with the initializer_list \p __ilist
  //! @param __ilist The initializer_list to be copied into this vector.
  //! @note Neither frees not allocates memory if `__ilist.size() == 0`.
  void assign(_CUDA_VSTD::initializer_list<_Tp> __ilist)
  {
    const auto __count = static_cast<size_type>(__ilist.size());
    if (capacity() < __count)
    {
      uninitialized_buffer<_Tp, _Properties...> __new_buf{__buf_.resource(), __count};
      _CUDA_VSTD::uninitialized_copy(__ilist.begin(), __ilist.end(), __new_buf.data());
      _CUDA_VSTD::__destroy(__unwrapped_begin(), __unwrapped_end());
      __buf_  = _CUDA_VSTD::move(__new_buf);
      __size_ = __count;
      return;
    }

    if (__count < __size_)
    {
      const iterator __new_end = _CUDA_VSTD::copy(__ilist.begin(), __ilist.end(), begin());
      this->__destroy_from(__new_end);
    }
    else
    {
      _CUDA_VSTD::copy(__ilist.begin(), __ilist.begin() + __size_, __unwrapped_begin());
      this->__uninitialized_copy(__ilist.begin() + __size_, __ilist.end(), __unwrapped_end());
    }
  }

  //! @brief Replaces the content of the vector with the range \p __range
  //! @param __range The range to be copied into this vector.
  //! @note Neither frees not allocates memory if `__range.size() == 0`.
  //! @note May reallocate multiple times in case of input ranges.
  _LIBCUDACXX_TEMPLATE(class _Range)
  _LIBCUDACXX_REQUIRES(__compatible_range<_Range> _LIBCUDACXX_AND(!_CUDA_VRANGES::forward_range<_Range>))
  void assign_range(_Range&& __range)
  {
    auto __first      = _CUDA_VRANGES::begin(__range);
    const auto __last = _CUDA_VRANGES::end(__range);
    pointer __end     = __unwrapped_end();
    for (pointer __current = __unwrapped_begin(); __current != __end; ++__current, (void) ++__first)
    {
      if (__first == __last)
      {
        this->__destroy_from(__current);
        return;
      }
      *__current = *__first;
    }

    for (; __first != __last; ++__first)
    {
      emplace_back(*__first);
    }
  }

#  ifndef DOXYGEN_SHOULD_SKIP_THIS // doxygen conflates the overloads
  //! @brief Replaces the content of the vector with the range \p __range
  //! @param __range The range to be copied into this vector.
  //! @note Neither frees not allocates memory if `__range.size() == 0`.
  _LIBCUDACXX_TEMPLATE(class _Range)
  _LIBCUDACXX_REQUIRES(__compatible_range<_Range> _LIBCUDACXX_AND _CUDA_VRANGES::forward_range<_Range> _LIBCUDACXX_AND
                         _CUDA_VRANGES::sized_range<_Range>)
  void assign_range(_Range&& __range)
  {
    const auto __size = _CUDA_VRANGES::size(__range);
    if (capacity() < __size)
    {
      uninitialized_buffer<_Tp, _Properties...> __new_buf{__buf_.resource(), __size};
      _CUDA_VSTD::uninitialized_copy(_CUDA_VSTD::begin(__range), _CUDA_VRANGES::__unwrap_end(__range), __new_buf.data());
      _CUDA_VSTD::__destroy(__unwrapped_begin(), __unwrapped_end());
      __buf_  = _CUDA_VSTD::move(__new_buf);
      __size_ = __size;
      return;
    }

    const auto __first = _CUDA_VRANGES::begin(__range);
    const auto __last  = _CUDA_VRANGES::__unwrap_end(__range);
    if (static_cast<size_type>(__size) < __size_)
    {
      const iterator __new_end = _CUDA_VSTD::copy(__first, __last, __unwrapped_begin());
      this->__destroy_from(__new_end);
    }
    else
    {
      const auto __middle = _CUDA_VSTD::next(__first, __size_);
      _CUDA_VSTD::copy(__first, __middle, __unwrapped_begin());
      this->__uninitialized_copy(__middle, __last, __unwrapped_end());
    }
  }

  //! @brief Replaces the content of the vector with the range \p __range
  //! @param __range The range to be copied into this vector.
  //! @note Neither frees not allocates memory if `__range.size() == 0`.
  _LIBCUDACXX_TEMPLATE(class _Range)
  _LIBCUDACXX_REQUIRES(__compatible_range<_Range> _LIBCUDACXX_AND _CUDA_VRANGES::forward_range<_Range> _LIBCUDACXX_AND(
    !_CUDA_VRANGES::sized_range<_Range>))
  void assign_range(_Range&& __range)
  {
    const auto __first = _CUDA_VRANGES::begin(__range);
    const auto __last  = _CUDA_VRANGES::__unwrap_end(__range);
    const auto __size  = static_cast<size_type>(_CUDA_VRANGES::distance(__first, __last));
    if (capacity() < __size)
    {
      uninitialized_buffer<_Tp, _Properties...> __new_buf{__buf_.resource(), __size};
      _CUDA_VSTD::uninitialized_copy(__first, __last, __new_buf.data());
      _CUDA_VSTD::__destroy(__unwrapped_begin(), __unwrapped_end());
      __buf_  = _CUDA_VSTD::move(__new_buf);
      __size_ = __size;
      return;
    }

    if (__size < __size_)
    {
      const iterator __new_end = _CUDA_VSTD::copy(__first, __last, __unwrapped_begin());
      this->__destroy_from(__new_end);
    }
    else
    {
      const auto __middle = _CUDA_VSTD::next(__first, __size_);
      _CUDA_VSTD::copy(__first, __middle, __unwrapped_begin());
      this->__uninitialized_copy(__middle, __last, __unwrapped_end());
    }
  }
#  endif // DOXYGEN_SHOULD_SKIP_THIS
  //! @}

  //! @addtogroup modification
  //! @{

  //! @brief Inserts a copy of \p __value at position \p __cpos. Elements after \p __cpos are shifted to the back.
  //! @param __cpos Iterator to the position at which \p __value is inserted.
  //! @param __value The element to be copied into the vector.
  //! @return Iterator to the current position of the new element.
  iterator insert(const_iterator __cpos, const _Tp& __value)
  {
    return emplace(__cpos, __value);
  }

  //! @brief Inserts \p __value at position \p __cpos. Elements after \p __cpos are shifted to the back.
  //! @param __cpos Iterator to the position at which \p __value is inserted.
  //! @param __value The element to be moved into the vector.
  //! @return Iterator to the current position of the new element.
  iterator insert(const_iterator __cpos, _Tp&& __value)
  {
    return emplace(__cpos, _CUDA_VSTD::move(__value));
  }

  //! @brief Inserts \p __count copies of \p __value at position \p __cpos. Elements after \p __cpos are shifted to the
  //! back.
  //! @param __cpos Iterator to the position at which \p __value is inserted.
  //! @param __count The number of elements to be copied into the vector.
  //! @param __value The element to be copied into the vector.
  //! @return Iterator to the current position of the first new element.
  iterator insert(const_iterator __cpos, const size_type __count, const _Tp& __value)
  {
    const size_type __pos = __cpos - cbegin();
    if (__size_ + 1 < __pos)
    {
      _CUDA_VSTD::__throw_out_of_range("vector::insert(const_iterator, size_type, T)");
    }
    if (__count == 0)
    {
      return begin() + __pos;
    }

    if (capacity() - __size_ < __count)
    {
      uninitialized_buffer<_Tp, _Properties...> __new_buf{__buf_.resource(), __size_ + __count};
      _CUDA_VSTD::uninitialized_fill_n(__new_buf.data() + __pos, __count, __value);
      _CUDA_VSTD::uninitialized_move(__unwrapped_begin(), __unwrapped_begin() + __pos, __new_buf.data());
      _CUDA_VSTD::uninitialized_move(
        __unwrapped_begin() + __pos, __unwrapped_begin() + __size_, __new_buf.data() + __pos + __count);
      _CUDA_VSTD::__destroy(__unwrapped_begin(), __unwrapped_end());
      __buf_ = _CUDA_VSTD::move(__new_buf);
      __size_ += __count;
      return begin() + __pos;
    }

    const iterator __res = begin() + __pos;
    const iterator __end = end();
    if (__pos == __size_)
    {
      this->__uninitialized_fill_n(__end, __count, __value);
      return __res;
    }

    const iterator __middle = __res + __count;
    if (__end <= __middle)
    { // all existing elements are pushed into uninitialized storage
      this->__uninitialized_fill(__end, __middle, __value);
      this->__uninitialized_move(__res, __end, __middle);
      _CUDA_VSTD::fill(__pos, __end, __value);
    }
    else
    { // some elements get copied into existing storage
      this->__uninitialized_move(__end - __count, __end, __end);
      _CUDA_VSTD::move_backward(__res, __end - __count, __end);
      _CUDA_VSTD::fill(__res, __middle, __value);
    }
    return __res;
  }

  //! @brief Inserts copies of the sequence `[__first, __last)]` at position \p __cpos. Elements after \p __cpos are
  //! shifted to the back.
  //! @param __cpos Iterator to the position at which the sequence is inserted.
  //! @param __first Iterator to the first element to be copied into the vector.
  //! @param __last Iterator after to the last element to be copied into the vector.
  //! @return Iterator to the current position of the first new element.
  //! @note May allocate multiple time in case of input iterators
  _LIBCUDACXX_TEMPLATE(class _Iter)
  _LIBCUDACXX_REQUIRES(_CUDA_VSTD::__is_cpp17_input_iterator<_Iter>::value _LIBCUDACXX_AND(
    !_CUDA_VSTD::__is_cpp17_forward_iterator<_Iter>::value))
  iterator insert(const_iterator __cpos, _Iter __first, _Iter __last)
  {
    const size_type __pos = __cpos - cbegin();
    if (__size_ + 1 < __pos)
    {
      _CUDA_VSTD::__throw_out_of_range("vector::insert(const_iterator, Iter, Iter)");
    }

    if (__first == __last)
    {
      return begin() + __pos;
    }

    // add all new elements to the back then rotate
    const size_type __old_size = __size_;
    for (; __first != __last; ++__first)
    {
      emplace_back(*__first);
    }

    if (__old_size != __pos)
    {
      _CUDA_VSTD::rotate(__unwrapped_begin() + __pos, __unwrapped_begin() + __old_size, __unwrapped_end());
    }
    return begin() + __pos;
  }

  //! @brief Inserts copies of the sequence `[__first, __last)]` at position \p __cpos. Elements after \p __cpos are
  //! shifted to the back.
  //! @param __cpos Iterator to the position at which the sequence is inserted.
  //! @param __first Iterator to the first element to be copied into the vector.
  //! @param __last Iterator after to the last element to be copied into the vector.
  //! @return Iterator to the current position of the first new element.
  _LIBCUDACXX_TEMPLATE(class _Iter)
  _LIBCUDACXX_REQUIRES(_CUDA_VSTD::__is_cpp17_forward_iterator<_Iter>::value)
  iterator insert(const_iterator __cpos, _Iter __first, _Iter __last)
  {
    const size_type __pos = __cpos - cbegin();
    if (__size_ + 1 < __pos)
    {
      _CUDA_VSTD::__throw_out_of_range("vector::insert(const_iterator, Iter, Iter)");
    }

    if (__first == __last)
    {
      return begin() + __pos;
    }

    const auto __count = _CUDA_VSTD::distance(__first, __last);
    if (capacity() < __count)
    {
      uninitialized_buffer<_Tp, _Properties...> __new_buf{__buf_.resource(), __size_ + __count};
      _CUDA_VSTD::uninitialized_copy(__first, __last, __new_buf.data() + __pos);
      _CUDA_VSTD::uninitialized_move(__unwrapped_begin(), __unwrapped_begin() + __pos, __new_buf.data());
      _CUDA_VSTD::uninitialized_move(__unwrapped_begin() + __pos, __unwrapped_end(), __new_buf.data() + __pos + __count);
      _CUDA_VSTD::__destroy(__unwrapped_begin(), __unwrapped_end());
      __buf_ = _CUDA_VSTD::move(__new_buf);
      __size_ += __count;
      return begin() + __pos;
    }

    const iterator __res = begin() + __pos;
    const iterator __end = end();
    if (__pos == __size_)
    {
      this->__uninitialized_copy(__first, __last, __end);
      return __res;
    }

    const iterator __middle   = __res + __count;
    const size_type __to_copy = __size_ - __pos;
    if (__end <= __middle)
    { // all existing elements are pushed into uninitialized storage
      _Iter __imiddle = _CUDA_VSTD::next(__first, __to_copy);
      this->__uninitialized_copy(__imiddle, __last, __end);
      this->__uninitialized_move(__res, __end, __middle);
      _CUDA_VSTD::copy(__first, __imiddle, __res);
    }
    else
    { // all new elements get copied into existing storage
      this->__uninitialized_move(__end - __count, __end, __end);
      _CUDA_VSTD::move_backward(__res, __end - __count, __end);
      _CUDA_VSTD::copy(__first, __last, __res);
    }

    return __res;
  }

  //! @brief Inserts an initializer_list at position \p __cpos. Elements after \p __cpos are shifted to the back.
  //! @param __cpos Iterator to the position at which the sequence is inserted.
  //! @param __ilist The initializer_list containing the elements to be inserted.
  //! @return Iterator to the current position of the first new element.
  iterator insert(const_iterator __cpos, _CUDA_VSTD::initializer_list<_Tp> __ilist)
  {
    const size_type __pos = __cpos - cbegin();
    if (__size_ + 1 < __pos)
    {
      _CUDA_VSTD::__throw_out_of_range("vector::insert(const_iterator, initializer_list)");
    }

    const auto __count = __ilist.size();
    if (__count == 0)
    {
      return __pos;
    }

    if (capacity() < __count)
    {
      uninitialized_buffer<_Tp, _Properties...> __new_buf{__buf_.resource(), __size_ + __count};
      _CUDA_VSTD::uninitialized_copy(__ilist.begin(), __ilist.end(), __new_buf.data() + __pos);
      _CUDA_VSTD::uninitialized_move(__unwrapped_begin(), __unwrapped_begin() + __pos, __new_buf.data());
      _CUDA_VSTD::uninitialized_move(__unwrapped_begin() + __pos, __unwrapped_end(), __new_buf.data() + __pos + __count);
      _CUDA_VSTD::__destroy(__unwrapped_begin(), __unwrapped_end());
      __buf_ = _CUDA_VSTD::move(__new_buf);
      __size_ += __count;
      return begin() + __pos;
    }

    const iterator __res = begin() + __pos;
    const iterator __end = end();
    if (__pos == __size_)
    {
      this->__uninitialized_copy(__ilist.begin(), __ilist.end(), __end);
      return __res;
    }

    const iterator __middle   = __res + __count;
    const size_type __to_copy = __size_ - __pos;
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

  //! @brief Inserts a sequence \p __range at position \p __cpos. Elements after \p __cpos are shifted to the back.
  //! @param __cpos Iterator to the position at which the sequence is inserted.
  //! @param __range The range containing the elements to be inserted.
  //! @return Iterator to the current position of the first new element.
  //! @note May allocate multiple times in case of input ranges.
  _LIBCUDACXX_TEMPLATE(class _Range)
  _LIBCUDACXX_REQUIRES(__compatible_range<_Range> _LIBCUDACXX_AND(!_CUDA_VRANGES::forward_range<_Range>))
  iterator insert_range(const_iterator __cpos, _Range&& __range)
  {
    const size_type __pos = __cpos - cbegin();
    if (__size_ + 1 < __pos)
    {
      _CUDA_VSTD::__throw_out_of_range("vector::insert(const_iterator, Iter, Iter)");
    }

    auto __first = _CUDA_VRANGES::begin(__range);
    auto __last  = _CUDA_VRANGES::end(__range);
    if (__first == __last)
    {
      return begin() + __pos;
    }

    // add all new elements to the back then rotate
    const size_type __old_size = __size_;
    for (; __first != __last; ++__first)
    {
      emplace_back(*__first);
    }

    if (__old_size != __pos)
    {
      _CUDA_VSTD::rotate(__unwrapped_begin() + __pos, __unwrapped_begin() + __old_size, __unwrapped_end());
    }
    return begin() + __pos;
  }

#  ifndef DOXYGEN_SHOULD_SKIP_THIS // doxygen conflates both overloads
  //! @brief Inserts a sequence \p __range at position \p __cpos. Elements after \p __cpos are shifted to the back.
  //! @param __cpos Iterator to the position at which the sequence is inserted.
  //! @param __range The range containing the elements to be inserted.
  //! @return Iterator to the current position of the first new element.
  _LIBCUDACXX_TEMPLATE(class _Range)
  _LIBCUDACXX_REQUIRES(__compatible_range<_Range> _LIBCUDACXX_AND _CUDA_VRANGES::forward_range<_Range>)
  iterator insert_range(const_iterator __cpos, _Range&& __range)
  {
    auto __first = _CUDA_VRANGES::begin(__range);
    return insert(__cpos, _CUDA_VRANGES::begin(__range), _CUDA_VRANGES::__unwrap_end(__range));
  }
#  endif // DOXYGEN_SHOULD_SKIP_THIS

  //! @brief Appends a sequence \p __range at the end of the vector.
  //! @param __range The range containing the elements to be appended.
  //! @note May allocate multiple times in case of input ranges.
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

#  ifndef DOXYGEN_SHOULD_SKIP_THIS // doxygen conflates both overloads
  //! @brief Appends a sequence \p __range at the end of the vector.
  //! @param __range The range containing the elements to be appended.
  _LIBCUDACXX_TEMPLATE(class _Range)
  _LIBCUDACXX_REQUIRES(__compatible_range<_Range> _LIBCUDACXX_AND _CUDA_VRANGES::forward_range<_Range>)
  void append_range(_Range&& __range)
  {
    insert(end(), _CUDA_VRANGES::begin(__range), _CUDA_VRANGES::__unwrap_end(__range));
  }
#  endif // DOXYGEN_SHOULD_SKIP_THIS

  //! @brief Constructs a new element at position \p __cpos. Elements after \p __cpos are shifted to the back.
  //! @param __cpos Iterator to the position at which the new element is constructed.
  //! @param __args The arguments forwarded to the constructor.
  //! @return Iterator to the current position of the new element.
  template <class... _Args>
  iterator emplace(const_iterator __cpos, _Args&&... __args)
  {
    const size_type __pos = __cpos - cbegin();
    if (__size_ < __pos)
    {
      _CUDA_VSTD::__throw_out_of_range("vector::emplace(const_iterator, Args...)");
    }

    if (__size_ == capacity())
    {
      uninitialized_buffer<_Tp, _Properties...> __new_buf{resource(), __size_ + 1};
      _CUDA_VSTD::__construct_at(__new_buf.data() + __pos, _CUDA_VSTD::forward<_Args>(__args)...);
      _CUDA_VSTD::uninitialized_move(__unwrapped_begin(), __unwrapped_begin() + __pos, __new_buf.data());
      _CUDA_VSTD::uninitialized_move(__unwrapped_begin() + __pos, __unwrapped_end(), __new_buf.data() + __pos + 1);
      _CUDA_VSTD::__destroy(__unwrapped_begin(), __unwrapped_end());
      __buf_ = _CUDA_VSTD::move(__new_buf);
      ++__size_;
      return begin() + __pos;
    }

    if (__pos == __size_)
    {
      emplace_back(_CUDA_VSTD::forward<_Args>(__args)...);
    }
    else
    {
      const pointer __middle = __unwrapped_begin() + __pos;
      const pointer __end    = __unwrapped_end();
      _Tp __temp{_CUDA_VSTD::forward<_Args>(__args)...};
      emplace_back(_CUDA_VSTD::move(*(__end - 1)));
      _CUDA_VSTD::move_backward(__middle, __end - 1, __end);
      *__middle = _CUDA_VSTD::move(__temp);
    }

    return begin() + __pos;
  }

  //! @brief Constructs a new element at the end of the vector.
  //! @param __args The arguments forwarded to the constructor.
  //! @return Reference to the new element.
  template <class... _Args>
  reference emplace_back(_Args&&... __args)
  {
    if (__size_ == capacity())
    {
      uninitialized_buffer<_Tp, _Properties...> __new_buf{resource(), __size_ + 1};
      _CUDA_VSTD::__construct_at(__new_buf.data() + __size_, _CUDA_VSTD::forward<_Args>(__args)...);
      _CUDA_VSTD::uninitialized_move(__unwrapped_begin(), __unwrapped_end(), __new_buf.data());
      _CUDA_VSTD::__destroy(__unwrapped_begin(), __unwrapped_end());
      __buf_ = _CUDA_VSTD::move(__new_buf);
      ++__size_;
      return back();
    }

    auto __final = _CUDA_VSTD::__construct_at(__unwrapped_end(), _CUDA_VSTD::forward<_Args>(__args)...);
    ++__size_;
    return *__final;
  }

  //! @brief Copies a new element to the end of the vector.
  //! @param __value The element to be copied.
  //! @return Reference to the new element.
  reference push_back(const _Tp& __value)
  {
    return emplace_back(__value);
  }

  //! @brief Moves a new element to the end of the vector.
  //! @param __value The element to be copied.
  //! @return Reference to the new element.
  reference push_back(_Tp&& __value)
  {
    return emplace_back(_CUDA_VSTD::move(__value));
  }

  //! @brief Removes the last element of the vector.
  void pop_back() noexcept
  {
    this->__destroy_from(--end());
  }

  //! @brief Removes the element pointed to by \p __cpos. All elements after \p __cpos are moved to the front.
  //! @param __cpos Iterator to the position of the element to be removed.
  //! @return Iterator to the new element at \p __cpos.
  iterator erase(const_iterator __cpos) noexcept(_CCCL_TRAIT(_CUDA_VSTD::is_nothrow_move_assignable, _Tp))
  {
    const size_type __pos = __cpos - cbegin();
    if (__pos == __size_)
    {
      return begin() + __pos;
    }

    if (__size_ == 0 || __size_ < __pos)
    {
      _CUDA_VSTD_NOVERSION::terminate();
    }

    const pointer __middle = __unwrapped_begin() + __pos;
    const pointer __end    = __unwrapped_end();
    _CUDA_VSTD::move(__middle + 1, __end, __middle);
    this->__destroy_from(__end - 1);
    return __middle;
  }

  //! @brief Removes the elements between \p __cfirst and \p __clast. All elements after \p __clast are moved to the
  //! front.
  //! @param __cfirst Iterator to the first element to be removed.
  //! @param __clast Iterator after the last element to be removed.
  //! @return Iterator to the new element at \p __cfirst.
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
  //! @brief Removes all elements that are equal to \p __value
  //! @param __value The element to be removed.
  //! @return The number of elements that have been removed.
  size_type __erase(const _Tp& __value) noexcept(_CCCL_TRAIT(_CUDA_VSTD::is_nothrow_move_assignable, _Tp))
  {
    const pointer __old_end = __unwrapped_end();
    const pointer __new_end = _CUDA_VSTD::remove(__unwrapped_begin(), __old_end, __value);
    this->__destroy_from(__new_end);
    return static_cast<size_type>(__old_end - __new_end);
  }

  //! @brief Removes all elements that satisfy \p __pred
  //! @param __pred The unary predicate selecting elements to be removed.
  //! @return The number of elements that have been removed.
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
    this->__destroy_from(__unwrapped_begin());
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
      this->__destroy_from(__unwrapped_begin() + __size);
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
        _CUDA_VSTD::uninitialized_move(__unwrapped_begin(), __unwrapped_end(), __new_buf.begin());
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
      this->__destroy_from(__unwrapped_begin() + __size);
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
        _CUDA_VSTD::uninitialized_move(__unwrapped_begin(), __unwrapped_end(), __new_buf.begin());
        _CUDA_VSTD::__destroy(__unwrapped_begin(), __unwrapped_end());
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
      this->__destroy_from(__unwrapped_begin() + __size);
    }
    else
    {
      if (capacity() < __size)
      {
        uninitialized_buffer<_Tp, _Properties...> __new_buf{resource(), __size};
        _CUDA_VSTD::uninitialized_move(__unwrapped_begin(), __unwrapped_end(), __new_buf.begin());
        _CUDA_VSTD::__destroy(__unwrapped_begin(), __unwrapped_end());
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

  //! @addtogroup comparison
  //! @{

  //! @brief Compares two vectors for equality
  //! @param __lhs One vector.
  //! @param __rhs The other vector.
  //! @return true, if \p __lhs and \p __rhs contain equal elements have the same size
  _CCCL_NODISCARD_FRIEND constexpr bool
  operator==(const vector& __lhs, const vector& __rhs) noexcept(noexcept(_CUDA_VSTD::equal(
    __lhs.__unwrapped_begin(), __lhs.__unwrapped_end(), __rhs.__unwrapped_begin(), __rhs.__unwrapped_end())))
  {
    return _CUDA_VSTD::equal(
      __lhs.__unwrapped_begin(), __lhs.__unwrapped_end(), __rhs.__unwrapped_begin(), __rhs.__unwrapped_end());
  }
#  if _CCCL_STD_VER <= 2017
  //! @brief Compares two vectors for inequality
  //! @param __lhs One vector.
  //! @param __rhs The other vector.
  //! @return false, if \p __lhs and \p __rhs contain equal elements have the same size
  _CCCL_NODISCARD_FRIEND constexpr bool
  operator!=(const vector& __lhs, const vector& __rhs) noexcept(noexcept(_CUDA_VSTD::equal(
    __lhs.__unwrapped_begin(), __lhs.__unwrapped_end(), __rhs.__unwrapped_begin(), __rhs.__unwrapped_end())))
  {
    return !_CUDA_VSTD::equal(
      __lhs.__unwrapped_begin(), __lhs.__unwrapped_end(), __rhs.__unwrapped_begin(), __rhs.__unwrapped_end());
  }
#  endif // _CCCL_STD_VER <= 2017

#  ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
  _CCCL_NODISCARD_FRIEND constexpr __synth_three_way_result_t<_Tp>
  operator<=>(const vector& __lhs, const vector& __rhs) noexcept(noexcept(_CUDA_VSTD::lexicographical_compare_three_way(
    __lhs.__unwrapped_begin(), __lhs.__unwrapped_end(), __rhs.__unwrapped_begin(), __rhs.__unwrapped_end())))
  {
    return _CUDA_VSTD::lexicographical_compare_three_way(
      __lhs.__unwrapped_begin(), __lhs.__unwrapped_end(), __rhs.__unwrapped_begin(), __rhs.__unwrapped_end());
  }
#  else // ^^^ !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR ^^^ / vvv _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR vvv
  //! @brief lexicographically compares two vectors
  //! @param __lhs One vector.
  //! @param __rhs The other vector.
  //! @return true, if \p __lhs compares lexicographically less than \p __rhs.
  _CCCL_NODISCARD_FRIEND constexpr bool
  operator<(const vector& __lhs, const vector& __rhs) noexcept(noexcept(_CUDA_VSTD::lexicographical_compare(
    __lhs.__unwrapped_begin(), __lhs.__unwrapped_end(), __rhs.__unwrapped_begin(), __rhs.__unwrapped_end())))
  {
    return _CUDA_VSTD::lexicographical_compare(
      __lhs.__unwrapped_begin(), __lhs.__unwrapped_end(), __rhs.__unwrapped_begin(), __rhs.__unwrapped_end());
  }
  //! @brief lexicographically compares two vectors
  //! @param __lhs One vector.
  //! @param __rhs The other vector.
  //! @return true, if \p __rhs compares lexicographically less than \p __hhs.
  _CCCL_NODISCARD_FRIEND constexpr bool
  operator>(const vector& __lhs, const vector& __rhs) noexcept(noexcept(_CUDA_VSTD::lexicographical_compare(
    __lhs.__unwrapped_begin(), __lhs.__unwrapped_end(), __rhs.__unwrapped_begin(), __rhs.__unwrapped_end())))
  {
    return __rhs < __lhs;
  }
  //! @brief lexicographically compares two vectors
  //! @param __lhs One vector.
  //! @param __rhs The other vector.
  //! @return false, if \p __rhs compares lexicographically less than \p __lhs.
  _CCCL_NODISCARD_FRIEND constexpr bool
  operator<=(const vector& __lhs, const vector& __rhs) noexcept(noexcept(_CUDA_VSTD::lexicographical_compare(
    __lhs.__unwrapped_begin(), __lhs.__unwrapped_end(), __rhs.__unwrapped_begin(), __rhs.__unwrapped_end())))
  {
    return !(__rhs < __lhs);
  }
  //! @brief lexicographically compares two vectors
  //! @param __lhs One vector.
  //! @param __rhs The other vector.
  //! @return false, if \p __lhs compares lexicographically less than \p __rhs.
  _CCCL_NODISCARD_FRIEND constexpr bool
  operator>=(const vector& __lhs, const vector& __rhs) noexcept(noexcept(_CUDA_VSTD::lexicographical_compare(
    __lhs.__unwrapped_begin(), __lhs.__unwrapped_end(), __rhs.__unwrapped_begin(), __rhs.__unwrapped_end())))
  {
    return !(__lhs < __rhs);
  }
#  endif // _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

  //! @}

#  ifndef DOXYGEN_SHOULD_SKIP_THIS // friend functions are currently broken
  //! @brief Forwards the passed properties
  _LIBCUDACXX_TEMPLATE(class _Property)
  _LIBCUDACXX_REQUIRES((!property_with_value<_Property>) _LIBCUDACXX_AND _CUDA_VSTD::_One_of<_Property, _Properties...>)
  friend void get_property(const vector&, _Property) noexcept {}
#  endif // DOXYGEN_SHOULD_SKIP_THIS
};

// [containers.sequences.inplace.vector.erasure]
//! @brief Removes all elements that are equal to \p __value from \p __cont.
//! @param __cont The vector storing the elements.
//! @param __value The element to be removed.
//! @return The number of elements that have been removed.
template <class _Tp, class... _Properties>
size_t erase(vector<_Tp, _Properties...>& __cont,
             const _Tp& __value) noexcept(_CCCL_TRAIT(_CUDA_VSTD::is_nothrow_move_assignable, _Tp))
{
  return __cont.__erase(__value);
}

//! @brief Removes all elements that satisfy \p __pred from \p __cont.
//! @param __cont The vector storing the elements.
//! @param __pred The unary predicate selecting elements to be removed.
//! @return The number of elements that have been removed.
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
