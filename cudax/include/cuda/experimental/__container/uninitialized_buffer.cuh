//===----------------------------------------------------------------------===//
//
// Part of the CUDA Toolkit, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__CONTAINERS_UNINITIALIZED_BUFFER_H
#define __CUDAX__CONTAINERS_UNINITIALIZED_BUFFER_H

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
#include <cuda/std/__memory/align.h>
#include <cuda/std/__new/launder.h>
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/span>

#include <cuda/experimental/__memory_resource/any_resource.cuh>

#if _CCCL_STD_VER >= 2014 && !defined(_CCCL_COMPILER_MSVC_2017) \
  && defined(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

//! @file
//! The \c uninitialized_buffer class provides a typed buffer allocated from a given memory resource.
namespace cuda::experimental
{

//! @rst
//! .. _cudax-containers-uninitialized-buffer:
//!
//! Uninitialized type-safe memory storage
//! ---------------------------------------
//!
//! ``uninitialized_buffer`` provides a typed buffer allocated from a given :ref:`memory resource
//! <libcudacxx-extended-api-memory-resources-resource>`. It handles alignment and release of the allocation.
//! The memory is uninitialized, so that a user needs to ensure elements are properly constructed.
//!
//! In addition to being type-safe, ``uninitialized_buffer`` also takes a set of :ref:`properties
//! <libcudacxx-extended-api-memory-resources-properties>` to ensure that e.g. execution space constraints are checked
//! at compile time. However, we can only forward stateless properties. If a user wants to use a stateful one, then they
//! need to implement :ref:`get_property(const device_buffer&, Property)
//! <libcudacxx-extended-api-memory-resources-properties>`.
//!
//! @endrst
//! @tparam _Tp the type to be stored in the buffer
//! @tparam _Properties... The properties the allocated memory satisfies
template <class _Tp, class... _Properties>
class uninitialized_buffer
{
private:
  static_assert(_CUDA_VMR::__contains_execution_space_property<_Properties...>,
                "The properties of cuda::experimental::uninitialized_buffer must contain at least one execution space "
                "property!");

  template <class, class...>
  friend class uninitialized_buffer;

  //! @brief Helper to check whether a different buffer still statisfies all properties of this one
  template <class... _OtherProperties>
  static constexpr bool __properties_match =
    !_CCCL_TRAIT(_CUDA_VSTD::is_same,
                 _CUDA_VSTD::__make_type_set<_Properties...>,
                 _CUDA_VSTD::__make_type_set<_OtherProperties...>)
    && _CUDA_VSTD::__type_set_contains<_CUDA_VSTD::__make_type_set<_OtherProperties...>, _Properties...>;

  using __resource = ::cuda::experimental::mr::any_resource<_Properties...>;
  __resource __mr_;
  size_t __count_ = 0;
  void* __buf_    = nullptr;

  //! @brief Determines the allocation size given the alignment and size of `T`
  _CCCL_NODISCARD _CCCL_HOST_DEVICE static constexpr size_t __get_allocation_size(const size_t __count) noexcept
  {
    constexpr size_t __alignment = alignof(_Tp);
    return (__count * sizeof(_Tp) + (__alignment - 1)) & ~(__alignment - 1);
  }

  //! @brief Determines the properly aligned start of the buffer given the alignment and size of  `T`
  _CCCL_NODISCARD _CCCL_HOST_DEVICE _Tp* __get_data() const noexcept
  {
    constexpr size_t __alignment = alignof(_Tp);
    size_t __space               = __get_allocation_size(__count_);
    void* __ptr                  = __buf_;
    return _CUDA_VSTD::launder(
      reinterpret_cast<_Tp*>(_CUDA_VSTD::align(__alignment, __count_ * sizeof(_Tp), __ptr, __space)));
  }

  //! @brief Causes the buffer to be treated as a span when passed to cudax::launch.
  //! @pre The buffer must have the cuda::mr::device_accessible property.
  _CCCL_NODISCARD_FRIEND _CUDA_VSTD::span<_Tp>
  __cudax_launch_transform(::cuda::stream_ref, uninitialized_buffer& __self) noexcept
  {
    static_assert(_CUDA_VSTD::__is_included_in<_CUDA_VMR::device_accessible, _Properties...>,
                  "The buffer must be device accessible to be passed to `launch`");
    return {__self.__get_data(), __self.size()};
  }

  //! @brief Causes the buffer to be treated as a span when passed to cudax::launch
  //! @pre The buffer must have the cuda::mr::device_accessible property.
  _CCCL_NODISCARD_FRIEND _CUDA_VSTD::span<const _Tp>
  __cudax_launch_transform(::cuda::stream_ref, const uninitialized_buffer& __self) noexcept
  {
    static_assert(_CUDA_VSTD::__is_included_in<_CUDA_VMR::device_accessible, _Properties...>,
                  "The buffer must be device accessible to be passed to `launch`");
    return {__self.__get_data(), __self.size()};
  }

public:
  using value_type = _Tp;
  using reference  = _Tp&;
  using pointer    = _Tp*;
  using size_type  = size_t;

  //! @brief Constructs a \c uninitialized_buffer and allocates sufficient storage for \p __count elements through
  //! \p __mr
  //! @param __mr The memory resource to allocate the buffer with.
  //! @param __count The desired size of the buffer.
  //! @note Depending on the alignment requirements of `T` the size of the underlying allocation might be larger
  //! than `count * sizeof(T)`.
  //! @note Only allocates memory when \p __count > 0
  uninitialized_buffer(__resource __mr, const size_t __count)
      : __mr_(_CUDA_VSTD::move(__mr))
      , __count_(__count)
      , __buf_(__count_ == 0 ? nullptr : __mr_.allocate(__get_allocation_size(__count_)))
  {}

  uninitialized_buffer(const uninitialized_buffer&)            = delete;
  uninitialized_buffer& operator=(const uninitialized_buffer&) = delete;

  //! @brief Move-constructs a \c uninitialized_buffer from \p __other
  //! @param __other Another \c uninitialized_buffer
  //! Takes ownership of the allocation in \p __other and resets it
  uninitialized_buffer(uninitialized_buffer&& __other) noexcept
      : __mr_(_CUDA_VSTD::move(__other.__mr_))
      , __count_(_CUDA_VSTD::exchange(__other.__count_, 0))
      , __buf_(_CUDA_VSTD::exchange(__other.__buf_, nullptr))
  {}

  //! @brief Move-constructs a \c uninitialized_buffer from another \c uninitialized_buffer with matching properties
  //! @param __other Another \c uninitialized_buffer
  //! Takes ownership of the allocation in \p __other and resets it
  _LIBCUDACXX_TEMPLATE(class... _OtherProperties)
  _LIBCUDACXX_REQUIRES(__properties_match<_OtherProperties...>)
  uninitialized_buffer(uninitialized_buffer<_Tp, _OtherProperties...>&& __other) noexcept
      : __mr_(_CUDA_VSTD::move(__other.__mr_))
      , __count_(_CUDA_VSTD::exchange(__other.__count_, 0))
      , __buf_(_CUDA_VSTD::exchange(__other.__buf_, nullptr))
  {}

  //! @brief Move-assings a \c uninitialized_buffer from \p __other
  //! @param __other Another \c uninitialized_buffer
  //! Deallocates the current allocation and then takes ownership of the allocation in \p __other and resets it
  uninitialized_buffer& operator=(uninitialized_buffer&& __other) noexcept
  {
    if (this == _CUDA_VSTD::addressof(__other))
    {
      return *this;
    }

    if (__buf_)
    {
      __mr_.deallocate(__buf_, __get_allocation_size(__count_));
    }

    __mr_    = _CUDA_VSTD::move(__other.__mr_);
    __count_ = _CUDA_VSTD::exchange(__other.__count_, 0);
    __buf_   = _CUDA_VSTD::exchange(__other.__buf_, nullptr);
    return *this;
  }

  //! @brief Destroys an \c uninitialized_buffer deallocates the buffer
  //! @warning The destructor does not destroy any objects that may or may not reside within the buffer. It is the
  //! user's responsibility to ensure that all objects within the buffer have been properly destroyed.
  ~uninitialized_buffer()
  {
    if (__buf_)
    {
      __mr_.deallocate(__buf_, __get_allocation_size(__count_));
    }
  }

  //! @brief Returns an aligned pointer to the first element in the allocation
  _CCCL_NODISCARD pointer begin() const noexcept
  {
    return __get_data();
  }

  //! @brief Returns an aligned pointer to the element after the last element in the allocation
  _CCCL_NODISCARD pointer end() const noexcept
  {
    return __get_data() + __count_;
  }

  //! @brief Returns an aligned pointer to the first element in the allocation
  _CCCL_NODISCARD pointer data() const noexcept
  {
    return __get_data();
  }

  //! @brief Returns the size of the allocation
  _CCCL_NODISCARD constexpr size_type size() const noexcept
  {
    return __count_;
  }

  //! @rst
  //! Returns a \c const reference to the :ref:`any_resource <cudax-memory-resource-any-resource>`
  //! that holds the memory resource used to allocate the buffer
  //! @endrst
  _CCCL_NODISCARD const __resource& get_resource() const noexcept
  {
    return __mr_;
  }

  //! @brief Swaps the contents with those of another \c uninitialized_buffer.
  //! @param __other The other \c uninitialized_buffer.
  constexpr void swap(uninitialized_buffer& __other) noexcept
  {
    __mr_.swap(__other.__mr_);
    _CUDA_VSTD::swap(__count_, __other.__count_);
    _CUDA_VSTD::swap(__buf_, __other.__buf_);
  }

#  ifndef DOXYGEN_SHOULD_SKIP_THIS // friend functions are currently broken
  //! @brief Forwards the passed Properties
  _LIBCUDACXX_TEMPLATE(class _Property)
  _LIBCUDACXX_REQUIRES(
    (!property_with_value<_Property>) _LIBCUDACXX_AND _CUDA_VSTD::__is_included_in<_Property, _Properties...>)
  friend constexpr void get_property(const uninitialized_buffer&, _Property) noexcept {}
#  endif // DOXYGEN_SHOULD_SKIP_THIS

  //! @brief Internal method to swap elements allocation and size with another \c uninitialized_buffer.
  //! @param __other The other \c uninitialized_buffer.
  //! This is mainly used when growing containers, where we want to retain the resource but replace allocation.
  _CCCL_HOST_DEVICE constexpr void __swap_allocations(uninitialized_buffer& __other) noexcept
  {
    _CUDA_VSTD::swap(__count_, __other.__count_);
    _CUDA_VSTD::swap(__buf_, __other.__buf_);
  }
};

template <class _Tp>
using uninitialized_device_buffer = uninitialized_buffer<_Tp, _CUDA_VMR::device_accessible>;

} // namespace cuda::experimental

#endif // _CCCL_STD_VER >= 2014 && !_CCCL_COMPILER_MSVC_2017 && LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#endif //__CUDAX__CONTAINERS_UNINITIALIZED_BUFFER_H
