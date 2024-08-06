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

#include <cuda/experimental/__container/heterogeneous_iterator.h>
#include <cuda/experimental/__container/uninitialized_buffer.h>
#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__utility/select_execution_space.h>

#if _CCCL_STD_VER >= 2014 && !defined(_CCCL_COMPILER_MSVC_2017) \
  && defined(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

//! @file The \c vector class provides a container of contiguous memory
namespace cuda::experimental
{

//! @rst
//!
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
//! In addition to being type safe, ``vector`` also takes a set of :ref:`properties
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

//! We restrict all accessors of the vector based on the execution space

template <class _Derived, _ExecutionSpace>
struct __vector_access;

template <class _Derived>
struct __vector_access<_Derived, _ExecutionSpace::__host>
{
private:
  _CCCL_NODISCARD _CCCL_HOST constexpr _Derived& __derived() noexcept
  {
    return static_cast<_Derived&>(*this);
  }

  _CCCL_NODISCARD _CCCL_HOST constexpr _Derived const& __derived() const noexcept
  {
    return static_cast<_Derived const&>(*this);
  }

public:
  using reference       = typename _Derived::reference;
  using const_reference = typename _Derived::const_reference;
  using pointer         = typename _Derived::pointer;
  using const_pointer   = typename _Derived::const_pointer;

  //! @addtogroup access
  //! @{
  //! @brief Returns a reference to the \p __n 'th element of the vector
  //! @param __n The index of the element we want to access
  _CCCL_NODISCARD _CCCL_HOST constexpr reference operator[](const size_t __n) noexcept
  {
    return *(__derived().data() + __n);
  }

  //! @brief Returns a reference to the \p __n 'th element of the vector
  //! @param __n The index of the element we want to access
  _CCCL_NODISCARD _CCCL_HOST constexpr const_reference operator[](const size_t __n) const noexcept
  {
    return *(__derived().data() + __n);
  }

  //! @brief Returns a reference to the first element of the vector
  _CCCL_NODISCARD _CCCL_HOST constexpr reference first() noexcept
  {
    return *__derived().data();
  }

  //! @brief Returns a reference to the first element of the vector
  _CCCL_NODISCARD _CCCL_HOST constexpr const_reference first() const noexcept
  {
    return *__derived().data();
  }

  //! @brief Returns a reference to the last element of the vector
  _CCCL_NODISCARD _CCCL_HOST constexpr reference back() noexcept
  {
    return *(__derived().data() + __derived().size() - 1);
  }

  //! @brief Returns a reference to the last element of the vector
  _CCCL_NODISCARD _CCCL_HOST constexpr const_reference back() const noexcept
  {
    return *(__derived().data() + __derived().size() - 1);
  }
  //! @}
};

template <class _Derived>
struct __vector_access<_Derived, _ExecutionSpace::__device>
{
private:
  _CCCL_NODISCARD _CCCL_DEVICE constexpr _Derived& __derived() noexcept
  {
    return static_cast<_Derived&>(*this);
  }

  _CCCL_NODISCARD _CCCL_DEVICE constexpr _Derived const& __derived() const noexcept
  {
    return static_cast<_Derived const&>(*this);
  }

public:
  using reference       = typename _Derived::reference;
  using const_reference = typename _Derived::const_reference;
  using pointer         = typename _Derived::pointer;
  using const_pointer   = typename _Derived::const_pointer;

  //! @addtogroup access
  //! @{
  //! @brief Returns a reference to the \p __n 'th element of the vector
  //! @param __n The index of the element we want to access
  _CCCL_NODISCARD _CCCL_DEVICE constexpr reference operator[](const size_t __n) noexcept
  {
    return *(__derived().data() + __n);
  }

  //! @brief Returns a reference to the \p __n 'th element of the vector
  //! @param __n The index of the element we want to access
  _CCCL_NODISCARD _CCCL_DEVICE constexpr const_reference operator[](const size_t __n) const noexcept
  {
    return *(__derived().data() + __n);
  }

  //! @brief Returns a reference to the first element of the vector
  _CCCL_NODISCARD _CCCL_DEVICE constexpr reference first() noexcept
  {
    return *__derived().data();
  }

  //! @brief Returns a reference to the first element of the vector
  _CCCL_NODISCARD _CCCL_DEVICE constexpr const_reference first() const noexcept
  {
    return *__derived().data();
  }

  //! @brief Returns a reference to the last element of the vector
  _CCCL_NODISCARD _CCCL_DEVICE constexpr reference back() noexcept
  {
    return *(__derived().data() + __derived().size() - 1);
  }

  //! @brief Returns a reference to the last element of the vector
  _CCCL_NODISCARD _CCCL_DEVICE constexpr const_reference back() const noexcept
  {
    return *(__derived().data() + __derived().size() - 1);
  }
  //! @}
};

template <class _Derived>
struct __vector_access<_Derived, _ExecutionSpace::__host_device>
{
private:
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr _Derived& __derived() noexcept
  {
    return static_cast<_Derived&>(*this);
  }

  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr _Derived const& __derived() const noexcept
  {
    return static_cast<_Derived const&>(*this);
  }

public:
  using reference       = typename _Derived::reference;
  using const_reference = typename _Derived::const_reference;
  using pointer         = typename _Derived::pointer;
  using const_pointer   = typename _Derived::const_pointer;

  //! @addtogroup access
  //! @{
  //! @brief Returns a reference to the \p __n 'th element of the vector
  //! @param __n The index of the element we want to access
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr reference operator[](const size_t __n) noexcept
  {
    return *(__derived().data() + __n);
  }

  //! @brief Returns a reference to the \p __n 'th element of the vector
  //! @param __n The index of the element we want to access
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr const_reference operator[](const size_t __n) const noexcept
  {
    return *(__derived().data() + __n);
  }

  //! @brief Returns a reference to the first element of the vector
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr reference first() noexcept
  {
    return *__derived().data();
  }

  //! @brief Returns a reference to the first element of the vector
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr const_reference first() const noexcept
  {
    return *__derived().data();
  }

  //! @brief Returns a reference to the last element of the vector
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr reference back() noexcept
  {
    return *(__derived().data() + __derived().size() - 1);
  }

  //! @brief Returns a reference to the last element of the vector
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr const_reference back() const noexcept
  {
    return *(__derived().data() + __derived().size() - 1);
  }
  //! @}
};

template <class _Tp, class... _Properties>
struct __vector_base
    : public __vector_access<__vector_base<_Tp, _Properties...>, __select_execution_space<_Properties...>>
{
protected:
  uninitialized_buffer<_Tp, _Properties...> __buf_;
  size_t __size_ = 0; // initialized to 0 in case initialization of the elements might throw

  using __base = __vector_access<__vector_base<_Tp, _Properties...>, __select_execution_space<_Properties...>>;

public:
  using value_type      = _Tp;
  using reference       = _Tp&;
  using const_reference = const _Tp&;
  using pointer         = _Tp*;
  using const_pointer   = const _Tp*;

  __vector_base(_CUDA_VMR::resource_ref<_Properties...> __mr, const size_t __size)
      : __base(__mr, __size)
  {}

  //! @brief Copy constructs a \c vector
  //! @param __other The other vector
  //! The new vector has capacity of \p __other.size() which is potentially less than \p __other.capacity()
  __vector_base(const __vector_base& __other)
      : __base(__other.resource(), __other.size())
  {
    if (__other.size() != 0)
    {
      this->__uninitialized_copy(__other.begin(), __other.end(), this->begin());
    }
  }

  //! @brief Move constructs a \c vector
  //! @param __other The other vector
  //! The new vector takes ownership of the allocation of \p __other
  __vector_base(__vector_base&& __other) noexcept
      : __buf_(_CUDA_VSTD::move(__buf_))
      , __size_(_CUDA_VSTD::exchange(__other.__size_, 0))
  {}

  //! @brief Copy assignment
  //! @param __other The other vector
  //! @note Even if the old vector would have enough storage available, we have to reallocate if the stored memory
  //! resource is not equal to the new one.
  __vector_base& operator=(const __vector_base& __other)
  {
    // There is sufficient space in the allocation and the resources are compatible
    if (this->resource() == __other.resource() && this->__size_ >= __other.size())
    {
      if (__size_ >= __other.__size_)
      {
        const auto __res = _CUDA_VSTD::copy(__other.begin(), __other.end(), this->begin());
        _CUDA_VSTD::__destroy(__res, this->end());
      }
      else
      {
        const auto __res = _CUDA_VSTD::copy(__other.begin(), __other.begin() + this->__size_, this->begin());
        this->__uninitialized_copy(__other.begin() + this->__size_, __other.end(), __res);
      }
      return *this;
    }

    // We need to reallocate and copy. Note we do not change the size of the current vector until the copy is done
    uninitialized_buffer<_Tp, _Properties...> __new_buf{__other.resource(), __other.__size_};
    _CUDA_VSTD::uninitialized_copy(__other.begin(), __other.end(), __new_buf.begin());

    // Now that everything is set up bring over the new data
    _CUDA_VSTD::__destroy(this->begin(), this->end());
    _CUDA_VSTD::swap(__buf_, __new_buf);
    this->__size_ = __other.__size_;
    return *this;
  }

  //! @brief Move assignment
  //! @param __other The other vector
  __vector_base& operator=(__vector_base&& __other) noexcept
  {
    if (this == _CUDA_VSTD::addressof(__other))
    {
      return *this;
    }

    _CUDA_VSTD::__destroy(this->begin(), this->end());
    __buf_        = _CUDA_VSTD::move(__other.__buf_);
    this->__size_ = __other.__size_;
    return *this;
  }

  //! @brief Destroys the \c vector deallocating the storage after destroying all elements
  //! @note Does not destroy elements if `is_trivially_destructible_v<_Tp>` holds
  ~__vector_base() noexcept
  {
    _CCCL_IF_CONSTEXPR (!_CCCL_TRAIT(_CUDA_VSTD::is_trivially_destructible, _Tp))
    {
      _CUDA_VSTD::__destroy(this->begin(), this->end());
    }
  }
};

//! @tparam _T the type to be stored in the buffer
//! @tparam _Properties... The properties the allocated memory satisfies
template <class _Tp, class... _Properties>
class vector : public __vector_base<_Tp, _Properties...>
{
private:
  using __base = __vector_base<_Tp, _Properties...>;

public:
  using value_type             = _Tp;
  using reference              = _Tp&;
  using const_reference        = const _Tp&;
  using pointer                = _Tp*;
  using const_pointer          = const _Tp*;
  using iterator               = heterogeneous_iterator<_Tp, false, __select_execution_space<_Properties...>>;
  using const_iterator         = heterogeneous_iterator<_Tp, true, __select_execution_space<_Properties...>>;
  using reverse_iterator       = _CUDA_VSTD::reverse_iterator<iterator>;
  using const_reverse_iterator = _CUDA_VSTD::reverse_iterator<const_iterator>;
  using size_type              = size_t;

  // Delegate to base constructors
  using __base::__base;

  //! @addtogroup construction
  //! @{
  //! @brief Constructs a \c vector using a memory resource
  //! @param __mr The memory resource to allocate memory within the vector.
  //! @note No memory is allocated
  vector(_CUDA_VMR::resource_ref<_Properties...> __mr)
      : __base(__mr, 0)
  {}

  //! @brief Constructs a \c vector of size \p __size using a memory resource and value-initializes all elements
  //! @param __mr The memory resource to allocate the vector with.
  //! @param __size The size of the vector.
  //! @note If `__size == 0` then no memory is allocated
  vector(_CUDA_VMR::resource_ref<_Properties...> __mr, const size_t __size)
      : __base(__mr, __size)
  {
    if (__size != 0)
    {
      this->__uninitialized_default_construct_n(this->begin(), __size);
    }
  }

  //! @brief Constructs a \c vector of size \p __size using a memory resource and copy-constructs all elements from
  //! \p __value
  //! @param __mr The memory resource to allocate the vector with.
  //! @param __size The size of the vector.
  //! @param __value The value all elements are copied from.
  //! @note If `__size == 0` then no memory is allocated
  vector(_CUDA_VMR::resource_ref<_Properties...> __mr, const size_t __size, const _Tp& __value)
      : __base(__mr, __size)
  {
    if (__size != 0)
    {
      this->__uninitialized_fill_n(this->begin(), this->__size_, __value);
    }
  }

  //! @brief Constructs a \c vector of size \p __size using a memory and leaves all elements uninitialized
  //! @param __mr The memory resource to allocate the vector with.
  //! @param __size The size of the vector
  //! @warning This constructor does *NOT* initialize any elements. It is the users responsibility to ensure that the
  //! elements within `[vec.begin(), vec.end())` are properly initialized, e.g with `cuda::std::uninitialized_copy`
  //! At the destruction of the \c vector all elements in the range `[vec.begin(), vec.end())` will be destroyed
  vector(_CUDA_VMR::resource_ref<_Properties...> __mr, const size_t __size, ::cuda::experimental::uninit_t)
      : __base(__mr, __size)
  {}

  //! @brief Constructs a \c vector from a range of input iterators using a memory resource and copy-constructs all
  //! elements from the range ``[__first, __last)``
  //! @param __mr The memory resource to allocate the vector with.
  //! @param __first The start of the input sequence.
  //! @param __last The end of the input sequence.
  //! @note If `__first == __last` then no memory is allocated. Might allocate multiple times
  _LIBCUDACXX_TEMPLATE(class _Iter)
  _LIBCUDACXX_REQUIRES(_CUDA_VSTD::__is_cpp17_input_iterator<_Iter>::value _LIBCUDACXX_AND(
    !_CUDA_VSTD::__is_cpp17_forward_iterator<_Iter>::value))
  vector(_CUDA_VMR::resource_ref<_Properties...> __mr, _Iter __first, _Iter __last)
      : __base(__mr, 0)
  {
    for (; __first != __last; ++__first)
    {
      emplace_back(*__first);
    }
  }

  //! @brief Constructs a \c vector from a range of forward iterators using a memory resource and copy-constructs all
  //! elements from the range ``[__first, __last)``
  //! @param __mr The memory resource to allocate the vector with.
  //! @param __first The start of the input sequence.
  //! @param __last The end of the input sequence.
  //! @note If `__first == __last` then no memory is allocated
  _LIBCUDACXX_TEMPLATE(class _Iter)
  _LIBCUDACXX_REQUIRES(_CUDA_VSTD::__is_cpp17_forward_iterator<_Iter>::value)
  vector(_CUDA_VMR::resource_ref<_Properties...> __mr, _Iter __first, _Iter __last)
      : __base(__mr, static_cast<size_t>(_CUDA_VSTD::distance(__first, __last)))
  {
    if (this->__size_ != 0)
    {
      this->__uninitialized_copy(__first, __last, this->begin());
    }
  }

  //! @brief Constructs a \c vector from an initializer_list \p __ilist using a memory resource and copy-constructs all
  //! elements from \p __ilist
  //! @param __mr The memory resource to allocate the vector with.
  //! @param __ilist The initializer_list being copied into the vector
  //! @note If `__ilist.size() == 0` then no memory is allocated
  vector(_CUDA_VMR::resource_ref<_Properties...> __mr, _CUDA_VSTD::initializer_list<_Tp> __ilist)
      : __base(__mr, __ilist.size())
  {
    if (this->__size_ != 0)
    {
      this->__uninitialized_copy(__ilist.begin(), __ilist.end(), this->begin());
    }
  }
  //! @}

  //! @addtogroup iterators
  //! @{
  //! @brief Returns an iterator to the first element of the vector. If the vector is empty, the returned iterator will
  //! be equal to end().
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr iterator begin() noexcept
  {
    return iterator{this->data()};
  }

  //! @brief Returns an immutable iterator to the first element of the vector. If the vector is empty, the returned
  //! iterator will be equal to end().
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr const_iterator begin() const noexcept
  {
    return const_iterator{this->data()};
  }

  //! @brief Returns an immutable iterator to the first element of the vector. If the vector is empty, the returned
  //! iterator will be equal to end().
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr const_iterator cbegin() const noexcept
  {
    return const_iterator{this->data()};
  }

  //! @brief Returns an iterator to the element following the last element of the vector. This element acts as a
  //! placeholder; attempting to access it results in undefined behavior.
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr iterator end() noexcept
  {
    return iterator{this->data() + this->__size_};
  }

  //! @brief Returns an immutable iterator to the element following the last element of the vector. This element acts as
  //! a placeholder; attempting to access it results in undefined behavior.
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr const_iterator end() const noexcept
  {
    return const_iterator{this->data() + this->__size_};
  }

  //! @brief Returns an immutable iterator to the element following the last element of the vector. This element acts as
  //! a placeholder; attempting to access it results in undefined behavior.
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr const_iterator cend() const noexcept
  {
    return const_iterator{this->data() + this->__size_};
  }

  //! @brief Returns a reverse iterator to the first element of the reversed vector. It corresponds to the last element
  //! of the non-reversed vector. If the vector is empty, the returned iterator is equal to rend().
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr reverse_iterator rbegin() noexcept
  {
    return reverse_iterator{end()};
  }

  //! @brief Returns an immutable reverse iterator to the first element of the reversed vector. It corresponds to the
  //! last element of the non-reversed vector. If the vector is empty, the returned iterator is equal to rend().
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr const_reverse_iterator rbegin() const noexcept
  {
    return const_reverse_iterator{end()};
  }

  //! @brief Returns an immutable reverse iterator to the first element of the reversed vector. It corresponds to the
  //! last element of the non-reversed vector. If the vector is empty, the returned iterator is equal to rend().
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr const_reverse_iterator crbegin() const noexcept
  {
    return const_reverse_iterator{end()};
  }

  //! @brief Returns a reverse iterator to the element following the last element of the reversed vector. It corresponds
  //! to the element preceding the first element of the non-reversed vector. This element acts as a placeholder,
  //! attempting to access it results in undefined behavior.
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr reverse_iterator rend() noexcept
  {
    return reverse_iterator{begin()};
  }

  //! @brief Returns an immutable reverse iterator to the element following the last element of the reversed vector. It
  //! corresponds to the element preceding the first element of the non-reversed vector. This element acts as a
  //! placeholder, attempting to access it results in undefined behavior.
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr const_reverse_iterator rend() const noexcept
  {
    return const_reverse_iterator{begin()};
  }

  //! @brief Returns an immutable reverse iterator to the element following the last element of the reversed vector. It
  //! corresponds to the element preceding the first element of the non-reversed vector. This element acts as a
  //! placeholder, attempting to access it results in undefined behavior.
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr const_reverse_iterator crend() const noexcept
  {
    return const_reverse_iterator{begin()};
  }

  //! @brief Returns a pointer to the first element of the vector. If the vector has not allocated memory the pointer
  //! will be null.
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr pointer data() noexcept
  {
    return this->data();
  }

  //! @brief Returns a pointer to the first element of the vector. If the vector has not allocated memory the pointer
  //! will be null.
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr const_pointer data() const noexcept
  {
    return this->data();
  }
  //! @}

  //! @addtogroup capacity
  //! @{
  //! @brief Returns the current number of elements stored in the vector
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr size_t size() const noexcept
  {
    return this->__size_;
  }

  //! @brief Returns true if the vector is empty
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr bool empty() const noexcept
  {
    return this->__size_ == 0;
  }

  //! @brief Returns the capacity of the current allocation of the vector
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr size_t capacity() const noexcept
  {
    return this->__size_;
  }

  //! @rst
  //! Returns the :ref:`resource_ref <libcudacxx-extended-api-memory-resources-resource-ref>` used to allocate
  //! the buffer
  //! @endrst
  _CCCL_NODISCARD _CCCL_HOST_DEVICE _CUDA_VMR::resource_ref<_Properties...> resource() const noexcept
  {
    return this->__mr_;
  }
  //! @}

  //! @addtogroup modification
  //! @{
  //! @brief Destroys all elements in the \c vector and sets the size to 0
  _CCCL_HOST_DEVICE void clear() noexcept
  {
    _CCCL_IF_CONSTEXPR (!_CCCL_TRAIT(_CUDA_VSTD::is_trivially_destructible, _Tp))
    {
      _CUDA_VSTD::__destroy(this->begin(), this->end());
    }
    this->__size_ = 0;
  }

  //! @brief Changes the size of the \c vector to \p __size and value-initializes new elements
  //! @param __size The intended size of the \c vector
  //! If `__size < vec.size()` then it destroys all superfluous elements. Otherwise, it value-initializes new elements
  void resize(const size_t __count) noexcept
  {
    if (__count < this->__size_)
    {
      _CCCL_IF_CONSTEXPR (!_CCCL_TRAIT(_CUDA_VSTD::is_trivially_destructible, _Tp))
      {
        _CUDA_VSTD::__destroy(this->begin() + __count, this->end());
      }
    }
    else
    {
      if (__count < this->capacity())
      {
        this->__uninitialized_value_construct_n(end(), __count - this->__size_);
      }
      else
      {
        uninitialized_buffer<_Tp, _Properties...> __new_buf{this->resource(), __count};
        this->__uninitialized_value_construct_n(__new_buf.begin() + this->__size_, __count - this->__size_);
        this->__uninitialized_move(this->begin(), this->end(), __new_buf.begin());
        _CUDA_VSTD::swap(this->__buf_, __new_buf);
      }
    }
  }

  //! @brief Changes the size of the \c vector to \p __size and copy-constructs new elements from \p __value
  //! @param __size The intended size of the \c vector
  //! @param __value The element to be copied into the \c vector when growing
  //! If `__size < vec.size()` then it destroys all superfluous elements. Otherwise, it copy-constructs new elements
  //! from \p __value
  void resize(const size_t __count, const _Tp& __value = {}) noexcept
  {
    if (__count < this->__size_)
    {
      _CCCL_IF_CONSTEXPR (!_CCCL_TRAIT(_CUDA_VSTD::is_trivially_destructible, _Tp))
      {
        _CUDA_VSTD::__destroy(this->begin() + __count, this->end());
      }
    }
    else
    {
      if (__count < this->capacity())
      {
        this->__uninitialized_fill_n(end(), __count - this->__size_, __value);
      }
      else
      {
        uninitialized_buffer<_Tp, _Properties...> __new_buf{this->resource(), __count};
        this->__uninitialized_fill_n(__new_buf.begin() + this->__size_, __count - this->__size_, __value);
        this->__uninitialized_move(this->begin(), this->end(), __new_buf.begin());
        _CUDA_VSTD::swap(this->__buf_, __new_buf);
      }
    }
  }

  //! @brief swaps two \c vector
  _CCCL_HOST_DEVICE void swap(vector& __other) noexcept
  {
    _CUDA_VSTD::swap(this->__buf_, __other.__buf_);
    _CUDA_VSTD::swap(this->__size_, __other._size);
  }
  //! @}

#  ifndef DOXYGEN_SHOULD_SKIP_THIS // friend functions are currently broken
  //! @brief Forwards the passed properties
  _LIBCUDACXX_TEMPLATE(class _Property)
  _LIBCUDACXX_REQUIRES((!property_with_value<_Property>) _LIBCUDACXX_AND _CUDA_VSTD::_One_of<_Property, _Properties...>)
  friend constexpr void get_property(const vector&, _Property) noexcept {}
#  endif // DOXYGEN_SHOULD_SKIP_THIS
};

template <class _Tp>
using device_vector = vector<_Tp, _CUDA_VMR::device_accessible>;

} // namespace cuda::experimental

#endif // _CCCL_STD_VER >= 2014 && !_CCCL_COMPILER_MSVC_2017 && LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#endif //__CUDAX__CONTAINERS_VECTOR_H
