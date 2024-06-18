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
#include <cuda/experimental/__container/heterogeneous_iterator.h>
#include <cuda/experimental/__container/uninitialized_buffer.h>
#include <cuda/std/__concepts/_One_of.h>
#include <cuda/std/__memory/align.h>
#include <cuda/std/span>
#include <cuda/stream_ref>

#if _CCCL_STD_VER >= 2014 && !defined(_CCCL_COMPILER_MSVC_2017) \
  && defined(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

//! @file The \c vector class provides a typed vector allocated from a given memory resource.
namespace cuda::experimental
{

struct uninitialized_construction_t
{};

_LIBCUDACXX_CPO_ACCESSIBILITY uninitialized_construction_t uninitialized_construction{};

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
//! at compile time. However, we can only forward stateless properties. If a user wants to use a stateful one, then they
//! need to implement :ref:`get_property(const vector&, Property)
//! <libcudacxx-extended-api-memory-resources-properties>`.
//!
//! .. warning::
//!
//!    ``vector`` stores a reference to the provided memory :ref:`memory resource
//!    <libcudacxx-extended-api-memory-resources-resource>`. It is the users resposibility to ensure the lifetime of the
//!    resource exceeds the lifetime of the vector.
//!
//! @endrst
//! @tparam T the type to be stored in the buffer
//! @tparam Properties... The properties the allocated memory satisfies
template <class _Tp, class... _Properties>
class vector
{
private:
  uninitialized_buffer<T, _Properties...> __buf_;
  size_t __size_ = 0;

public:
  using value_type             = _Tp;
  using reference              = _Tp&;
  using const_reference        = const _Tp&;
  using pointer                = _Tp*;
  using const_pointer          = const _Tp*;
  using iterator               = heterogeneous_iterator<_Tp, false, __select_execution_space<_Properties...>>;
  using const_iterator         = heterogeneous_iterator<_Tp, true, __select_execution_space<_Properties...>>;
  using reverse_iterator       = reverse_iterator<iterator>;
  using const_reverse_iterator = reverse_iterator<const_iterator>;
  using size_type              = size_t;

  //! @addtogroup construction
  //! @{
  //! @brief Constructs a \c vector from a memory resource
  //! @param mr The memory resource to allocate the vector with.
  //! No storage is allocated
  vector(_CUDA_VMR::resource_ref<_Properties...> __mr)
      : __buf_(__mr, 0)
      , __size_(0)
  {}

  //! @brief Constructs a \c vector of size \p size from a memory resource and value-initializes all elements
  //! @param mr The memory resource to allocate the vector with.
  //! @param size The size of the vector.
  vector(_CUDA_VMR::resource_ref<_Properties...> __mr, const size_t __size)
      : __buf_(__mr, __size)
      , __size_(__size)
  {
    _CUDA_VSTD::uninitialized_default_construct_n(__buf_.begin(), __size_);
  }

  //! @brief Constructs a \c vector of size \p size from a memory resource and copy constructs its elements from
  //! \p value
  //! @param mr The memory resource to allocate the vector with.
  //! @param size The size of the vector.
  //! @param value The value all elements are copied from.
  vector(_CUDA_VMR::resource_ref<_Properties...> __mr, const size_t __size, const _Tp& __value)
      : __buf_(__mr, __size)
      , __size_(__size)
  {
    _CUDA_VSTD::uninitialized_fill_n(__buf_.begin(), __size_, __value);
  }

  //! @brief Copy constructs a \c vector
  //! @param other The other vector
  //! The new vector has capacity of \p other.size() which is potentially less than \p other.capacity()
  vector(const vector& __other)
      : __buf_(__other.resource(), __other.size())
      , __size_(__other.size())
  {
    _CUDA_VSTD::uninitialized_copy(__other.begin(), __other.end(), begin());
  }

  //! @brief Move constructs a \c vector
  //! @param other The other vector
  //! The new vector takes ownership of the allocation of \p other
  vector(vector&& __other) noexcept
      : __buf_(_CUDA_VSTD::move(__buf_))
      , __size_(__other.__size_)
  {
    __other.__size_ = 0;
  }

  //! @brief Constructs a \c vector from a \c uninitialized_buffer
  //! @param mr The memory resource to allocate the vector with.
  //! @param size The size of the vector
  //! @param tag Tag type signaling that the user intents to manually initialize the allocated memory
  //! @warning This constructor does *NOT* initialize any elements. It is the users responsibility to ensure that the
  //! elements within `[buf.begin(), buf:begin() + size)` are properly initialized, e.g with `std::uninitialized_copy`
  vector(_CUDA_VMR::resource_ref<_Properties...> __mr, const size_t __size, uninitialized_construction_t __tag)
      : __buf_(__mr, __size)
      , __size_(__size)
  {
    _LIBCUDACXX_ASSERT(__size <= __buf.size(),
                       "vector(buf, size): size is larger than the capacity of the passed buffer");
  }

  //! @brief Copy assignment
  //! @param other The other vector
  //! @note Even if the old vector would have enough storage available, we have to reallocate if the stored memory
  //! resource is not equal to the new one.
  vector& operator=(const vector& __other)
  {
    // There is sufficient space in the allocation and the resources are compatible
    if (resource() == __other.resource() && __buf_.size() == __other.__buf_.size())
    {
      if (__size_ >= __other.__size_)
      {
        const auto __res = _CUDA_VSTD::copy(__other.begin(), __other.end(), begin());
        _CUDA_VSTD::__destroy(__res, end());
      }
      else
      {
        const auto __res = _CUDA_VSTD::copy(__other.begin(), __other.begin() + __size_, begin());
        _CUDA_VSTD::uninitialized_copy(__other.begin() + __size_, __other.end(), __res);
      }
      return *this;
    }

    // We need to reallocate and copy
    const uninitialized_buffer<T, _Properties...> __new_buf_{__other.resource(), __other.__size_};
    _CUDA_VSTD::uninitialized_copy(__other.begin(), __other.end(), __new_buf_.begin());

    // Now that everything is set up bring over the new data
    _CUDA_VSTD::__destroy(begin(), end());
    __buf_ = _CUDA_VSTD::move(__new_buf_);
    return *this;
  }

  //! @brief Move assignment
  //! @param other The other vector
  vector& operator=(vector&& __other) noexcept
  {
    _CUDA_VSTD::__destroy(begin(), end());
    __buf_  = _CUDA_VSTD::move(__other.__buf_);
    __size_ = __other.__size_;
    return *this;
  }

  //! @brief Destroys the \c vector deallocating the storage after destroying all elements
  ~vector() noexcept
  {
    _CCCL_IF_CONSTEXPR (!_CCCL_TRAIT(is_trivially_destructible, _Tp))
    {
      _CUDA_VSTD::__destroy(begin(), end());
    }
  }
  //! @}

  //! @addtogroup iterators
  //! @{
  //! @brief Returns pointer to the start of the vector
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr iterator begin() noexcept
  {
    return __buf_.data();
  }

  //! @brief Returns pointer to the start of the vector
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr const_iterator begin() const noexcept
  {
    return __buf_.data();
  }

  //! @brief Returns pointer to the start of the vector
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr const_iterator cbegin() const noexcept
  {
    return __buf_.data();
  }

  //! @brief Returns pointer to the end of the vector
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr iterator end() noexcept
  {
    return __buf_.data() + __size_;
  }

  //! @brief Returns pointer to the start of the vector
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr const_iterator end() const noexcept
  {
    return __buf_.data() + __size_;
  }

  //! @brief Returns pointer to the start of the vector
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr const_iterator cend() const noexcept
  {
    return __buf_.data() + __size_;
  }

  //! @brief Returns pointer to the start of the vector
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr reverse_iterator rbegin() noexcept
  {
    return reverse_iterator{end()};
  }

  //! @brief Returns pointer to the start of the vector
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr const_reverse_iterator rbegin() const noexcept
  {
    return const_reverse_iterator{end()};
  }

  //! @brief Returns pointer to the start of the vector
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr const_reverse_iterator crbegin() const noexcept
  {
    return const_reverse_iterator{end()};
  }

  //! @brief Returns pointer to the end of the vector
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr reverse_iterator rend() noexcept
  {
    return reverse_iterator{begin()};
  }

  //! @brief Returns pointer to the start of the vector
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr const_reverse_iterator rend() const noexcept
  {
    return const_reverse_iterator{begin()};
  }

  //! @brief Returns pointer to the start of the vector
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr const_reverse_iterator crend() const noexcept
  {
    return const_reverse_iterator{begin()};
  }

  //! @brief Returns pointer to the start of the vector
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr pointer data() noexcept
  {
    return __buf_.data();
  }

  //! @brief Returns pointer to the start of the vector
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr const_pointer data() const noexcept
  {
    return __buf_.data();
  }
  //! @}

  //! @addtogroup capacity
  //! @{
  //! @brief Returns the size of the vector
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr size_t size() const noexcept
  {
    return __size_;
  }

  //! @brief Returns true if the vector is empty
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr bool empty() const noexcept
  {
    return __size_ == 0;
  }

  //! @brief Returns the capacity of the vector
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr size_t capacity() const noexcept
  {
    return __buf_.size();
  }

  //! @brief Returns the memory resource used to allocate the storage
  _CCCL_NODISCARD _CCCL_HOST_DEVICE _CUDA_VMR::resource_ref<_Properties...> resource() const noexcept
  {
    return __mr_;
  }
  //! @}

  //! @addtogroup access
  //! @{
  //! @brief Returns a reference to the \p n'th element of the vector
  //! @param n The index of the element we want to access
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr reference operator[](const size_t __n) noexcept
  {
    return *(__buf_.data() + __n);
  }

  //! @brief Returns a reference to the \p n'th element of the vector
  //! @param n The index of the element we want to access
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr const_reference operator[](const size_t __n) const noexcept
  {
    return *(__buf_.data() + __n);
  }

  //! @brief Returns a reference to the first element of the vector
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr reference first() noexcept
  {
    return *__buf_.data();
  }

  //! @brief Returns a reference to the first element of the vector
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr const_reference first() const noexcept
  {
    return *__buf_.data();
  }

  //! @brief Returns a reference to the last element of the vector
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr reference back() noexcept
  {
    return *(__buf_.data() + __size_ - 1);
  }

  //! @brief Returns a reference to the last element of the vector
  _CCCL_NODISCARD _CCCL_HOST_DEVICE constexpr const_reference back() const noexcept
  {
    return *(__buf_.data() + __size_ - 1);
  }
  //! @}

  //! @addtogroup modification
  //! @{
  _CCCL_HOST_DEVICE void clear() noexcept
  {
    _CCCL_IF_CONSTEXPR (!_CCCL_TRAIT(is_trivially_destructible, _Tp))
    {
      _CUDA_VSTD::__destroy(begin(), end());
    }
    __size_ = 0;
  }

  _CCCL_HOST_DEVICE void resize(const size_t __count, const _Tp& __value = {}) noexcept
  {
    if (__count < __size_)
    {
      _CCCL_IF_CONSTEXPR (!_CCCL_TRAIT(is_trivially_destructible, _Tp))
      {
        _CUDA_VSTD::__destroy(begin() + __count, end());
      }
    }
    else
    {
      if (__count < __buf_.size())
      {
        _CUDA_VSTD::uninitialized_fill_n(end(), __count - __size_, __value);
      }
      else
      {
        uninitialized_buffer<_Tp, _Properties...> __new_buf{__buf_.resource(), __count};
        _CUDA_VSTD::uninitialized_fill_n(__new_buf.begin() + __size_, __count - __size_, __value);
        _CUDA_VSTD::uninitialized_move(begin(), end(), __new_buf.begin());
        __buf_ = _CUDA_VSTD::move(__new_buf);
      }
    }
    __size_ = __count;
    return;
  }

  _CCCL_HOST_DEVICE void swap(vector& __other) noexcept
  {
    __buf_.swap(__other.__buf_);
    _CUDA_VSTD::swap(__size_, __other._size);
  }
  //! @}

#  ifndef DOXYGEN_SHOULD_SKIP_THIS // friend functions are currently brocken
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
