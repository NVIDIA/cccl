//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__CONTAINERS_UNINITIALIZED_ASYNC_BUFFER_H
#define __CUDAX__CONTAINERS_UNINITIALIZED_ASYNC_BUFFER_H

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
#include <cuda/std/__type_traits/type_set.h>
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/span>
#include <cuda/stream_ref>

#include <cuda/experimental/__memory_resource/any_resource.cuh>
#include <cuda/experimental/__memory_resource/properties.cuh>

#if _CCCL_STD_VER >= 2014 && !_CCCL_COMPILER(MSVC2017) && defined(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

//! @file
//! The \c uninitialized_async_buffer class provides a typed buffer allocated in stream-order from a given memory
//! resource.
namespace cuda::experimental
{

//! @rst
//! .. _cudax-containers-uninitialized-async-buffer:
//!
//! Uninitialized stream-ordered type-safe memory storage
//! ------------------------------------------------------
//!
//! ``uninitialized_async_buffer`` provides a typed buffer allocated in stream order from a given :ref:`async memory
//! resource <libcudacxx-extended-api-memory-resources-resource>`. It handles alignment and release of the allocation.
//! The memory is uninitialized, so that a user needs to ensure elements are properly constructed.
//!
//! In addition to being type safe, ``uninitialized_async_buffer`` also takes a set of :ref:`properties
//! <libcudacxx-extended-api-memory-resources-properties>` to ensure that e.g. execution space constraints are checked
//! at compile time. However, only stateless properties can be forwarded. To use a stateful property,
//! implement :ref:`get_property(const uninitialized_async_buffer&, Property)
//! <libcudacxx-extended-api-memory-resources-properties>`.
//!
//! .. warning::
//!
//!    ``uninitialized_async_buffer`` uses `stream-ordered allocation
//!    <https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/>`__. It is the user's
//!    resposibility to ensure the lifetime of both the provided async resource and the stream exceed the lifetime of
//!    the buffer.
//!
//! @endrst
//! @tparam _T the type to be stored in the buffer
//! @tparam _Properties... The properties the allocated memory satisfies
template <class _Tp, class... _Properties>
class uninitialized_async_buffer
{
private:
  static_assert(_CUDA_VMR::__contains_execution_space_property<_Properties...>,
                "The properties of cuda::experimental::uninitialized_async_buffer must contain at least one "
                "execution space property!");

  using __async_resource = ::cuda::experimental::any_async_resource<_Properties...>;

  __async_resource __mr_;
  ::cuda::stream_ref __stream_ = {};
  size_t __count_              = 0;
  void* __buf_                 = nullptr;

  template <class, class...>
  friend class uninitialized_async_buffer;

  //! @brief Helper to check whether a different buffer still statisfies all properties of this one
  template <class... _OtherProperties>
  static constexpr bool __properties_match =
    !_CCCL_TRAIT(_CUDA_VSTD::is_same,
                 _CUDA_VSTD::__make_type_set<_Properties...>,
                 _CUDA_VSTD::__make_type_set<_OtherProperties...>)
    && _CUDA_VSTD::__type_set_contains_v<_CUDA_VSTD::__make_type_set<_OtherProperties...>, _Properties...>;

  //! @brief Determines the allocation size given the alignment and size of `T`
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI static constexpr size_t __get_allocation_size(const size_t __count) noexcept
  {
    constexpr size_t __alignment = alignof(_Tp);
    return (__count * sizeof(_Tp) + (__alignment - 1)) & ~(__alignment - 1);
  }

  //! @brief Determines the properly aligned start of the buffer given the alignment and size of `T`
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI constexpr _Tp* __get_data() const noexcept
  {
    constexpr size_t __alignment = alignof(_Tp);
    size_t __space               = __get_allocation_size(__count_);
    void* __ptr                  = __buf_;
    return _CUDA_VSTD::launder(
      reinterpret_cast<_Tp*>(_CUDA_VSTD::align(__alignment, __count_ * sizeof(_Tp), __ptr, __space)));
  }

  //! @brief Causes the buffer to be treated as a span when passed to cudax::launch.
  //! @pre The buffer must have the cuda::mr::device_accessible property.
  template <class _Tp2 = _Tp>
  _CCCL_NODISCARD_FRIEND _CCCL_HIDE_FROM_ABI auto
  __cudax_launch_transform(::cuda::stream_ref, uninitialized_async_buffer& __self) noexcept
    _CCCL_TRAILING_REQUIRES(_CUDA_VSTD::span<_Tp>)(
      _CUDA_VSTD::same_as<_Tp, _Tp2>&& _CUDA_VSTD::__is_included_in_v<mr::device_accessible, _Properties...>)
  {
    // TODO add auto synchronization
    return {__self.__get_data(), __self.size()};
  }

  //! @brief Causes the buffer to be treated as a span when passed to cudax::launch
  //! @pre The buffer must have the cuda::mr::device_accessible property.
  template <class _Tp2 = _Tp>
  _CCCL_NODISCARD_FRIEND _CCCL_HIDE_FROM_ABI auto
  __cudax_launch_transform(::cuda::stream_ref, const uninitialized_async_buffer& __self) noexcept
    _CCCL_TRAILING_REQUIRES(_CUDA_VSTD::span<const _Tp>)(
      _CUDA_VSTD::same_as<_Tp, _Tp2>&& _CUDA_VSTD::__is_included_in_v<mr::device_accessible, _Properties...>)
  {
    // TODO add auto synchronization
    return {__self.__get_data(), __self.size()};
  }

public:
  using value_type = _Tp;
  using reference  = _Tp&;
  using pointer    = _Tp*;
  using size_type  = size_t;

  //! @brief Constructs an \c uninitialized_async_buffer, allocating sufficient storage for \p __count elements through
  //! \p __mr
  //! @param __mr The async memory resource to allocate the buffer with.
  //! @param __stream The CUDA stream used for stream-ordered allocation.
  //! @param __count The desired size of the buffer.
  //! @note Depending on the alignment requirements of `T` the size of the underlying allocation might be larger
  //! than `count * sizeof(T)`. Only allocates memory when \p __count > 0
  _CCCL_HIDE_FROM_ABI
  uninitialized_async_buffer(__async_resource __mr, const ::cuda::stream_ref __stream, const size_t __count)
      : __mr_(_CUDA_VSTD::move(__mr))
      , __stream_(__stream)
      , __count_(__count)
      , __buf_(__count_ == 0 ? nullptr : __mr_.allocate_async(__get_allocation_size(__count_), __stream_))
  {}

  _CCCL_HIDE_FROM_ABI uninitialized_async_buffer(const uninitialized_async_buffer&)            = delete;
  _CCCL_HIDE_FROM_ABI uninitialized_async_buffer& operator=(const uninitialized_async_buffer&) = delete;

  //! @brief Move-constructs a \c uninitialized_async_buffer from \p __other
  //! @param __other Another \c uninitialized_async_buffer
  //! Takes ownership of the allocation in \p __other and resets it
  _CCCL_HIDE_FROM_ABI uninitialized_async_buffer(uninitialized_async_buffer&& __other) noexcept
      : __mr_(_CUDA_VSTD::move(__other.__mr_))
      , __stream_(_CUDA_VSTD::exchange(__other.__stream_, {}))
      , __count_(_CUDA_VSTD::exchange(__other.__count_, 0))
      , __buf_(_CUDA_VSTD::exchange(__other.__buf_, nullptr))
  {}

  //! @brief Move-constructs a \c uninitialized_async_buffer from \p __other
  //! @param __other Another \c uninitialized_async_buffer with matching properties
  //! Takes ownership of the allocation in \p __other and resets it
  _CCCL_TEMPLATE(class... _OtherProperties)
  _CCCL_REQUIRES(__properties_match<_OtherProperties...>)
  _CCCL_HIDE_FROM_ABI uninitialized_async_buffer(uninitialized_async_buffer<_Tp, _OtherProperties...>&& __other) noexcept
      : __mr_(_CUDA_VSTD::move(__other.__mr_))
      , __stream_(_CUDA_VSTD::exchange(__other.__stream_, {}))
      , __count_(_CUDA_VSTD::exchange(__other.__count_, 0))
      , __buf_(_CUDA_VSTD::exchange(__other.__buf_, nullptr))
  {}

  //! @brief Move-assings a \c uninitialized_async_buffer from \p __other
  //! @param __other Another \c uninitialized_async_buffer
  //! Deallocates the current allocation and then takes ownership of the allocation in \p __other and resets it
  _CCCL_HIDE_FROM_ABI uninitialized_async_buffer& operator=(uninitialized_async_buffer&& __other) noexcept
  {
    if (this == _CUDA_VSTD::addressof(__other))
    {
      return *this;
    }

    if (__buf_)
    {
      __mr_.deallocate_async(__buf_, __get_allocation_size(__count_), __stream_);
    }
    __mr_     = __other.__mr_;
    __stream_ = _CUDA_VSTD::exchange(__other.__stream_, {});
    __count_  = _CUDA_VSTD::exchange(__other.__count_, 0);
    __buf_    = _CUDA_VSTD::exchange(__other.__buf_, nullptr);
    return *this;
  }

  //! @brief Destroys an \c uninitialized_async_buffer and deallocates the buffer in stream order on the stream that was
  //! used to create the buffer.
  //! @warning The destructor does not destroy any objects that may or may not reside within the buffer. It is the
  //! user's responsibility to ensure that all objects within the buffer have been properly destroyed.
  _CCCL_HIDE_FROM_ABI ~uninitialized_async_buffer()
  {
    if (__buf_)
    {
      __mr_.deallocate_async(__buf_, __get_allocation_size(__count_), __stream_);
    }
  }

  //! @brief Returns an aligned pointer to the first element in the buffer
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI constexpr pointer begin() const noexcept
  {
    return __get_data();
  }

  //! @brief Returns an aligned pointer to the element following the last element of the buffer.
  //! This element acts as a placeholder; attempting to access it results in undefined behavior.
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI constexpr pointer end() const noexcept
  {
    return __get_data() + __count_;
  }

  //! @brief Returns an aligned pointer to the first element in the buffer
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI constexpr pointer data() const noexcept
  {
    return __get_data();
  }

  //! @brief Returns the size of the buffer
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI constexpr size_type size() const noexcept
  {
    return __count_;
  }

  //! @brief Returns the size of the buffer in bytes
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI constexpr size_type size_bytes() const noexcept
  {
    return __count_ * sizeof(_Tp);
  }

  //! @rst
  //! Returns a \c const reference to the :ref:`any_async_resource <cudax-memory-resource-any-async-resource>`
  //! that holds the memory resource used to allocate the buffer
  //! @endrst
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI const __async_resource& get_resource() const noexcept
  {
    return __mr_;
  }

  //! @brief Returns the stored stream
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI constexpr ::cuda::stream_ref get_stream() const noexcept
  {
    return __stream_;
  }

  //! @brief Replaces the stored stream
  //! @param __new_stream the new stream
  //! @note Always synchronizes with the old stream
  _CCCL_HIDE_FROM_ABI constexpr void change_stream(::cuda::stream_ref __new_stream)
  {
    if (__new_stream != __stream_)
    {
      __stream_.wait();
    }
    __stream_ = __new_stream;
  }

#  ifndef _CCCL_DOXYGEN_INVOKED // friend functions are currently broken
  //! @brief Forwards the passed properties
  _CCCL_TEMPLATE(class _Property)
  _CCCL_REQUIRES((!property_with_value<_Property>) _CCCL_AND _CUDA_VSTD::__is_included_in_v<_Property, _Properties...>)
  _CCCL_HIDE_FROM_ABI friend constexpr void get_property(const uninitialized_async_buffer&, _Property) noexcept {}
#  endif // _CCCL_DOXYGEN_INVOKED

  //! @brief Internal method to grow the allocation to a new size \p __count.
  //! @param __count The new size of the allocation.
  //! @return An \c uninitialized_async_buffer that holds the previous allocation
  //! @warning This buffer must outlive the returned buffer
  _CCCL_HIDE_FROM_ABI uninitialized_async_buffer __replace_allocation(const size_t __count)
  {
    // Create a new buffer with a reference to the stored memory resource and swap allocation information
    uninitialized_async_buffer __ret{_CUDA_VMR::async_resource_ref<_Properties...>{__mr_}, __stream_, __count};
    _CUDA_VSTD::swap(__count_, __ret.__count_);
    _CUDA_VSTD::swap(__buf_, __ret.__buf_);
    return __ret;
  }
};

template <class _Tp>
using uninitialized_async_device_buffer = uninitialized_async_buffer<_Tp, mr::device_accessible>;

} // namespace cuda::experimental

#endif // _CCCL_STD_VER >= 2014 && !_CCCL_COMPILER(MSVC2017) && LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#endif //__CUDAX__CONTAINERS_UNINITIALIZED_ASYNC_BUFFER_H
