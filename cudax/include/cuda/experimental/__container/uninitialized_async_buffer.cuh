//===----------------------------------------------------------------------===//
//
// Part of the CUDA Toolkit, under the Apache License v2.0 with LLVM Exceptions.
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
#include <cuda/std/__concepts/_One_of.h>
#include <cuda/std/__memory/align.h>
#include <cuda/std/__new/launder.h>
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/span>
#include <cuda/stream_ref>

#include <cuda/experimental/__memory_resource/any_resource.cuh>

#if _CCCL_STD_VER >= 2014 && !defined(_CCCL_COMPILER_MSVC_2017) \
  && defined(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

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
  using __async_resource = ::cuda::experimental::mr::any_async_resource<_Properties...>;
  __async_resource __mr_;
  ::cuda::stream_ref __stream_ = {};
  size_t __count_              = 0;
  void* __buf_                 = nullptr;

  //! @brief Determines the allocation size given the alignment and size of `T`
  _CCCL_NODISCARD static constexpr size_t __get_allocation_size(const size_t __count) noexcept
  {
    constexpr size_t __alignment = alignof(_Tp);
    return (__count * sizeof(_Tp) + (__alignment - 1)) & ~(__alignment - 1);
  }

  //! @brief Determines the properly aligned start of the buffer given the alignment and size of `T`
  _CCCL_NODISCARD constexpr _Tp* __get_data() const noexcept
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
  __cudax_launch_transform(::cuda::stream_ref, uninitialized_async_buffer& __self) noexcept
  {
    static_assert(_CUDA_VSTD::_One_of<_CUDA_VMR::device_accessible, _Properties...>,
                  "The buffer must be device accessible to be passed to `launch`");
    return {__self.__get_data(), __self.size()};
  }

  //! @brief Causes the buffer to be treated as a span when passed to cudax::launch
  //! @pre The buffer must have the cuda::mr::device_accessible property.
  _CCCL_NODISCARD_FRIEND _CUDA_VSTD::span<const _Tp>
  __cudax_launch_transform(::cuda::stream_ref, const uninitialized_async_buffer& __self) noexcept
  {
    static_assert(_CUDA_VSTD::_One_of<_CUDA_VMR::device_accessible, _Properties...>,
                  "The buffer must be device accessible to be passed to `launch`");
    return {__self.__get_data(), __self.size()};
  }

public:
  using value_type = _Tp;
  using reference  = _Tp&;
  using pointer    = _Tp*;
  using size_type  = size_t;

  //! @brief Constructs an \c uninitialized_async_buffer, allocating sufficient storage for \p __count elements using
  //! \p __mr
  //! @param __mr The async memory resource to allocate the buffer with.
  //! @param __stream The CUDA stream used for stream-ordered allocation.
  //! @param __count The desired size of the buffer.
  //! @note Depending on the alignment requirements of `T` the size of the underlying allocation might be larger
  //! than `count * sizeof(T)`. Only allocates memory when \p __count > 0
  uninitialized_async_buffer(__async_resource __mr, const ::cuda::stream_ref __stream, const size_t __count)
      : __mr_(_CUDA_VSTD::move(__mr))
      , __stream_(__stream)
      , __count_(__count)
      , __buf_(__count_ == 0 ? nullptr : __mr_.allocate_async(__get_allocation_size(__count_), __stream_))
  {}

  uninitialized_async_buffer(const uninitialized_async_buffer&)            = delete;
  uninitialized_async_buffer& operator=(const uninitialized_async_buffer&) = delete;

  //! @brief Move construction
  //! @param __other Another \c uninitialized_async_buffer
  uninitialized_async_buffer(uninitialized_async_buffer&& __other) noexcept
      : __mr_(_CUDA_VSTD::move(__other.__mr_))
      , __stream_(_CUDA_VSTD::exchange(__other.__stream_, {}))
      , __count_(_CUDA_VSTD::exchange(__other.__count_, 0))
      , __buf_(_CUDA_VSTD::exchange(__other.__buf_, nullptr))
  {}

  //! @brief Move assignment
  //! @param __other Another \c uninitialized_async_buffer
  uninitialized_async_buffer& operator=(uninitialized_async_buffer&& __other) noexcept
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
  ~uninitialized_async_buffer()
  {
    if (__buf_)
    {
      __mr_.deallocate_async(__buf_, __get_allocation_size(__count_), __stream_);
    }
  }

  //! @brief Returns an aligned pointer to the buffer
  _CCCL_NODISCARD constexpr pointer begin() const noexcept
  {
    return __get_data();
  }

  //! @brief Returns an aligned pointer to the element following the last element of the buffer.
  //! This element acts as a placeholder; attempting to access it results in undefined behavior.
  _CCCL_NODISCARD constexpr pointer end() const noexcept
  {
    return __get_data() + __count_;
  }

  //! @brief Returns an aligned pointer to the buffer
  _CCCL_NODISCARD constexpr pointer data() const noexcept
  {
    return __get_data();
  }

  //! @brief Returns the size of the buffer
  _CCCL_NODISCARD constexpr size_t size() const noexcept
  {
    return __count_;
  }

  //! @rst
  //! Returns a \c const reference to the :ref:`any_async_resource <cudax-memory-resource-any-async-resource>`
  //! that holds the memory resource used to allocate the buffer
  //! @endrst
  _CCCL_NODISCARD const __async_resource& get_resource() const noexcept
  {
    return __mr_;
  }

  //! @brief Returns the stored stream
  _CCCL_NODISCARD constexpr ::cuda::stream_ref get_stream() const noexcept
  {
    return __stream_;
  }

  //! @brief Replaces the stored stream
  //! @param __new_stream the new stream
  //! @note Always synchronizes with the old stream
  constexpr void change_stream(::cuda::stream_ref __new_stream)
  {
    if (__new_stream != __stream_)
    {
      __stream_.wait();
    }
    __stream_ = __new_stream;
  }

  //! @brief Swaps the contents with those of another \c uninitialized_async_buffer
  //! @param __other The other \c uninitialized_async_buffer.
  constexpr void swap(uninitialized_async_buffer& __other) noexcept
  {
    _CUDA_VSTD::swap(__mr_, __other.__mr_);
    _CUDA_VSTD::swap(__count_, __other.__count_);
    _CUDA_VSTD::swap(__buf_, __other.__buf_);
  }

#  ifndef DOXYGEN_SHOULD_SKIP_THIS // friend functions are currently broken
  //! @brief Forwards the passed properties
  _LIBCUDACXX_TEMPLATE(class _Property)
  _LIBCUDACXX_REQUIRES((!property_with_value<_Property>) _LIBCUDACXX_AND _CUDA_VSTD::_One_of<_Property, _Properties...>)
  friend constexpr void get_property(const uninitialized_async_buffer&, _Property) noexcept {}
#  endif // DOXYGEN_SHOULD_SKIP_THIS
};

template <class _Tp>
using uninitialized_async_device_buffer = uninitialized_async_buffer<_Tp, _CUDA_VMR::device_accessible>;

} // namespace cuda::experimental

#endif // _CCCL_STD_VER >= 2014 && !_CCCL_COMPILER_MSVC_2017 && LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#endif //__CUDAX__CONTAINERS_UNINITIALIZED_ASYNC_BUFFER_H
