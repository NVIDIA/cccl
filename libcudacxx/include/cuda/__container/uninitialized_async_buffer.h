//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#if _CCCL_HAS_CTK()

#  include <cuda/__memory_resource/any_resource.h>
#  include <cuda/__memory_resource/properties.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__memory/addressof.h>
#  include <cuda/std/__memory/align.h>
#  include <cuda/std/__new/launder.h>
#  include <cuda/std/__type_traits/type_set.h>
#  include <cuda/std/__utility/exchange.h>
#  include <cuda/std/__utility/move.h>
#  include <cuda/std/__utility/swap.h>
#  include <cuda/std/span>

#  include <cuda/std/__cccl/prologue.h>

//! @file
//! The \c __uninitialized_async_buffer class provides a typed buffer allocated
//! in stream-order from a given memory resource.
_CCCL_BEGIN_NAMESPACE_CUDA

//! @rst
//! .. _cudax-containers-uninitialized-async-buffer:
//!
//! Uninitialized stream-ordered type-safe memory storage
//! ------------------------------------------------------
//!
//! ``__uninitialized_async_buffer`` provides a typed buffer allocated in stream
//! order from a given :ref:`async memory resource
//! <libcudacxx-extended-api-memory-resources-resource>`. It handles alignment
//! and release of the allocation. The memory is uninitialized, so that a user
//! needs to ensure elements are properly constructed.
//!
//! In addition to being type safe, ``__uninitialized_async_buffer`` also takes
//! a set of :ref:`properties
//! <libcudacxx-extended-api-memory-resources-properties>` to ensure that e.g.
//! execution space constraints are checked at compile time. However, only
//! stateless properties can be forwarded. To use a stateful property, implement
//! :ref:`get_property(const __uninitialized_async_buffer&, Property)
//! <libcudacxx-extended-api-memory-resources-properties>`.
//!
//! .. warning::
//!
//!    ``__uninitialized_async_buffer`` uses `stream-ordered allocation
//!    <https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/>`__.
//!    It is the user's responsibility to ensure the lifetime of both the
//!    provided async resource and the stream exceed the lifetime of the buffer.
//!
//! @endrst
//! @tparam _Tp the type to be stored in the buffer
//! @tparam _Properties... The properties the allocated memory satisfies
template <class _Tp, class... _Properties>
class __uninitialized_async_buffer
{
private:
  static_assert(::cuda::mr::__contains_execution_space_property<_Properties...>,
                "The properties of cuda::__uninitialized_async_buffer must contain at least one execution space "
                "property!");

  using __async_resource = ::cuda::mr::any_resource<_Properties...>;

  __async_resource __mr_;
  ::cuda::stream_ref __stream_ = {::cudaStream_t{}};
  size_t __count_              = 0;
  void* __buf_                 = nullptr;

  template <class, class...>
  friend class __uninitialized_async_buffer;

  //! @brief Helper to check whether a different buffer still satisfies all
  //! properties of this one
  template <class... _OtherProperties>
  static constexpr bool __properties_match =
    !::cuda::std::is_same_v<::cuda::std::__make_type_set<_Properties...>,
                            ::cuda::std::__make_type_set<_OtherProperties...>>
    && ::cuda::std::__type_set_contains_v<::cuda::std::__make_type_set<_OtherProperties...>, _Properties...>;

  //! @brief Determines the allocation size given the alignment and size of `T`
  [[nodiscard]] _CCCL_HIDE_FROM_ABI static constexpr size_t __get_allocation_size(const size_t __count) noexcept
  {
    constexpr size_t __alignment = alignof(_Tp);
    return (__count * sizeof(_Tp) + (__alignment - 1)) & ~(__alignment - 1);
  }

  //! @brief Determines the properly aligned start of the buffer given the
  //! alignment and size of `T`
  [[nodiscard]] _CCCL_HIDE_FROM_ABI _Tp* __get_data() const noexcept
  {
    constexpr size_t __alignment = alignof(_Tp);
    size_t __space               = __get_allocation_size(__count_);
    void* __ptr                  = __buf_;
    return ::cuda::std::launder(
      static_cast<_Tp*>(::cuda::std::align(__alignment, __count_ * sizeof(_Tp), __ptr, __space)));
  }

  //! @brief Causes the buffer to be treated as a span when passed to
  //! cuda::launch.
  //! @pre The buffer must have the cuda::mr::device_accessible property.
  template <class _Tp2 = _Tp>
  [[nodiscard]] _CCCL_HIDE_FROM_ABI friend auto
  transform_launch_argument(::cuda::stream_ref, __uninitialized_async_buffer& __self) noexcept
    _CCCL_TRAILING_REQUIRES(::cuda::std::span<_Tp>)(
      ::cuda::std::same_as<_Tp, _Tp2>&& ::cuda::std::__is_included_in_v<::cuda::mr::device_accessible, _Properties...>)
  {
    return {__self.__get_data(), __self.size()};
  }

  //! @brief Causes the buffer to be treated as a span when passed to
  //! cuda::launch
  //! @pre The buffer must have the cuda::mr::device_accessible property.
  template <class _Tp2 = _Tp>
  [[nodiscard]] _CCCL_HIDE_FROM_ABI friend auto
  transform_launch_argument(::cuda::stream_ref, const __uninitialized_async_buffer& __self) noexcept
    _CCCL_TRAILING_REQUIRES(::cuda::std::span<const _Tp>)(
      ::cuda::std::same_as<_Tp, _Tp2>&& ::cuda::std::__is_included_in_v<::cuda::mr::device_accessible, _Properties...>)
  {
    return {__self.__get_data(), __self.size()};
  }

#  ifndef _CCCL_DOXYGEN_INVOKED
  // This is needed to ensure that we do not do a deep copy in
  // __replace_allocation
  struct __fake_resource_ref
  {
    __async_resource* __resource_;

    void* allocate_sync(std::size_t __size, std::size_t __alignment)
    {
      return __resource_->allocate_sync(__size, __alignment);
    }

    void deallocate_sync(void* __ptr, std::size_t __size, std::size_t __alignment) noexcept
    {
      __resource_->deallocate_sync(__ptr, __size, __alignment);
    }

    void* allocate(::cuda::stream_ref __stream, std::size_t __size, std::size_t __alignment)
    {
      return __resource_->allocate(__stream, __size, __alignment);
    }

    void deallocate(::cuda::stream_ref __stream, void* __ptr, std::size_t __size, std::size_t __alignment) noexcept
    {
      __resource_->deallocate(__stream, __ptr, __size, __alignment);
    }

    friend bool operator==(const __fake_resource_ref& __lhs, const __fake_resource_ref& __rhs) noexcept
    {
      return *__lhs.__resource_ == *__rhs.__resource_;
    }
    friend bool operator!=(const __fake_resource_ref& __lhs, const __fake_resource_ref& __rhs) noexcept
    {
      return *__lhs.__resource_ != *__rhs.__resource_;
    }

    //! @brief Forwards the passed properties
    _CCCL_TEMPLATE(class _Property)
    _CCCL_REQUIRES(::cuda::std::__is_included_in_v<_Property, _Properties...>)
    _CCCL_HIDE_FROM_ABI friend constexpr void get_property(const __fake_resource_ref&, _Property) noexcept {}
  };
#  endif // _CCCL_DOXYGEN_INVOKED

public:
  using value_type      = _Tp;
  using reference       = _Tp&;
  using const_reference = const _Tp&;
  using pointer         = _Tp*;
  using const_pointer   = const _Tp*;
  using size_type       = size_t;

  //! @brief Constructs an \c __uninitialized_async_buffer, allocating
  //! sufficient storage for \p __count elements through
  //! \p __mr
  //! @param __mr The async memory resource to allocate the buffer with.
  //! @param __stream The CUDA stream used for stream-ordered allocation.
  //! @param __count The desired size of the buffer.
  //! @note Depending on the alignment requirements of `T` the size of the
  //! underlying allocation might be larger than `count * sizeof(T)`. Only
  //! allocates memory when \p __count > 0
  _CCCL_HIDE_FROM_ABI
  __uninitialized_async_buffer(__async_resource __mr, const ::cuda::stream_ref __stream, const size_t __count)
      : __mr_(::cuda::std::move(__mr))
      , __stream_(__stream)
      , __count_(__count)
      , __buf_(__count_ == 0 ? nullptr : __mr_.allocate(__stream_, __get_allocation_size(__count_)))
  {}

  _CCCL_HIDE_FROM_ABI __uninitialized_async_buffer(const __uninitialized_async_buffer&)            = delete;
  _CCCL_HIDE_FROM_ABI __uninitialized_async_buffer& operator=(const __uninitialized_async_buffer&) = delete;

  //! @brief Move-constructs a \c __uninitialized_async_buffer from \p __other
  //! @param __other Another \c __uninitialized_async_buffer
  //! Takes ownership of the allocation in \p __other and resets it
  _CCCL_HIDE_FROM_ABI __uninitialized_async_buffer(__uninitialized_async_buffer&& __other) noexcept
      : __mr_(::cuda::std::move(__other.__mr_))
      , __stream_(::cuda::std::exchange(__other.__stream_, ::cuda::stream_ref{::cudaStream_t{}}))
      , __count_(::cuda::std::exchange(__other.__count_, 0))
      , __buf_(::cuda::std::exchange(__other.__buf_, nullptr))
  {}

  //! @brief Move-constructs a \c __uninitialized_async_buffer from \p __other
  //! @param __other Another \c __uninitialized_async_buffer with matching
  //! properties Takes ownership of the allocation in \p __other and resets it
  _CCCL_TEMPLATE(class... _OtherProperties)
  _CCCL_REQUIRES(__properties_match<_OtherProperties...>)
  _CCCL_HIDE_FROM_ABI
  __uninitialized_async_buffer(__uninitialized_async_buffer<_Tp, _OtherProperties...>&& __other) noexcept
      : __mr_(::cuda::std::move(__other.__mr_))
      , __stream_(::cuda::std::exchange(__other.__stream_, ::cuda::stream_ref{::cudaStream_t{}}))
      , __count_(::cuda::std::exchange(__other.__count_, 0))
      , __buf_(::cuda::std::exchange(__other.__buf_, nullptr))
  {}

  //! @brief Move-assigns a \c __uninitialized_async_buffer from \p __other
  //! @param __other Another \c __uninitialized_async_buffer
  //! Deallocates the current allocation and then takes ownership of the
  //! allocation in \p __other and resets it
  _CCCL_HIDE_FROM_ABI __uninitialized_async_buffer& operator=(__uninitialized_async_buffer&& __other) noexcept
  {
    if (this == ::cuda::std::addressof(__other))
    {
      return *this;
    }

    if (__buf_)
    {
      __mr_.deallocate(__stream_, __buf_, __get_allocation_size(__count_));
    }
    __mr_     = ::cuda::std::move(__other.__mr_);
    __stream_ = ::cuda::std::exchange(__other.__stream_, ::cuda::stream_ref{::cudaStream_t{}});
    __count_  = ::cuda::std::exchange(__other.__count_, 0);
    __buf_    = ::cuda::std::exchange(__other.__buf_, nullptr);
    return *this;
  }

  //! @brief Destroys an \c __uninitialized_async_buffer, deallocates the buffer
  //! in stream order on the stream that is stored in the buffer and destroys
  //! the memory resource.
  //! @param __stream The stream to deallocate the buffer on.
  //! @warning destroy does not destroy any objects that may or may not reside
  //! within the buffer. It is the user's responsibility to ensure that all
  //! objects within the buffer have been properly destroyed.
  _CCCL_HIDE_FROM_ABI void destroy(::cuda::stream_ref __stream)
  {
    if (__buf_)
    {
      __mr_.deallocate(__stream, __buf_, __get_allocation_size(__count_));
      __buf_   = nullptr;
      __count_ = 0;
    }
    // TODO should we make sure we move the mr only once by moving it to the if
    // above? It won't work for 0 count buffers, so we would probably need a
    // separate bool to track it
    auto __tmp_mr = ::cuda::std::move(__mr_);
  }

  //! @brief Destroys an \c __uninitialized_async_buffer, deallocates the buffer
  //! in stream order on the stream that is stored in the buffer and destroys
  //! the memory resource.
  //! @warning destroy does not destroy any objects that may or may not reside
  //! within the buffer. It is the user's responsibility to ensure that all
  //! objects within the buffer have been properly destroyed.
  _CCCL_HIDE_FROM_ABI void destroy()
  {
    destroy(__stream_);
  }

  //! @brief Destroys an \c __uninitialized_async_buffer and deallocates the
  //! buffer in stream order on the stream that was used to create the buffer.
  //! @warning The destructor does not destroy any objects that may or may not
  //! reside within the buffer. It is the user's responsibility to ensure that
  //! all objects within the buffer have been properly destroyed.
  _CCCL_HIDE_FROM_ABI ~__uninitialized_async_buffer()
  {
    destroy();
  }

  //! @brief Returns an aligned pointer to the first element in the buffer
  [[nodiscard]] _CCCL_HIDE_FROM_ABI constexpr pointer begin() noexcept
  {
    return __get_data();
  }

  //! @overload
  [[nodiscard]] _CCCL_HIDE_FROM_ABI constexpr const_pointer begin() const noexcept
  {
    return __get_data();
  }

  //! @brief Returns an aligned pointer to the element following the last
  //! element of the buffer. This element acts as a placeholder; attempting to
  //! access it results in undefined behavior.
  [[nodiscard]] _CCCL_HIDE_FROM_ABI constexpr pointer end() noexcept
  {
    return __get_data() + __count_;
  }

  //! @overload
  [[nodiscard]] _CCCL_HIDE_FROM_ABI constexpr const_pointer end() const noexcept
  {
    return __get_data() + __count_;
  }

  //! @brief Returns an aligned pointer to the first element in the buffer
  [[nodiscard]] _CCCL_HIDE_FROM_ABI constexpr pointer data() noexcept
  {
    return __get_data();
  }

  //! @overload
  [[nodiscard]] _CCCL_HIDE_FROM_ABI constexpr const_pointer data() const noexcept
  {
    return __get_data();
  }

  //! @brief Returns the size of the buffer
  [[nodiscard]] _CCCL_HIDE_FROM_ABI constexpr size_type size() const noexcept
  {
    return __count_;
  }

  //! @brief Returns the size of the buffer in bytes
  [[nodiscard]] _CCCL_HIDE_FROM_ABI constexpr size_type size_bytes() const noexcept
  {
    return __count_ * sizeof(_Tp);
  }

  //! @rst
  //! Returns a \c const reference to the :ref:`any_resource
  //! <cuda-memory-resource-any-async-resource>` that holds the memory resource
  //! used to allocate the buffer
  //! @endrst
  [[nodiscard]] _CCCL_HIDE_FROM_ABI const __async_resource& memory_resource() const noexcept
  {
    return __mr_;
  }

  //! @brief Returns the stored stream
  //! @note Stream used to allocate the buffer is initially stored in the
  //! buffer, but can be changed with `set_stream`
  [[nodiscard]] _CCCL_HIDE_FROM_ABI constexpr ::cuda::stream_ref stream() const noexcept
  {
    return __stream_;
  }

  //! @brief Replaces the stored stream
  //! @param __new_stream the new stream
  //! @note Always synchronizes with the old stream
  _CCCL_HIDE_FROM_ABI constexpr void set_stream(::cuda::stream_ref __new_stream)
  {
    if (__new_stream != __stream_)
    {
      __stream_.sync();
    }
    __stream_ = __new_stream;
  }

  //! @brief Replaces the stored stream
  //! @param __new_stream the new stream
  //! @warning This does not synchronize between \p __new_stream and the current
  //! stream. It is the user's responsibility to ensure proper stream order
  //! going forward
  _CCCL_HIDE_FROM_ABI constexpr void set_stream_unsynchronized(::cuda::stream_ref __new_stream) noexcept
  {
    __stream_ = __new_stream;
  }

  //! @brief Forwards the passed properties
  _CCCL_TEMPLATE(class _Property)
  _CCCL_REQUIRES((!property_with_value<_Property>) _CCCL_AND ::cuda::std::__is_included_in_v<_Property, _Properties...>)
  _CCCL_HIDE_FROM_ABI friend constexpr void get_property(const __uninitialized_async_buffer&, _Property) noexcept {}

  //! @brief Internal method to grow the allocation to a new size \p __count.
  //! @param __count The new size of the allocation.
  //! @return An \c __uninitialized_async_buffer that holds the previous
  //! allocation
  //! @warning This buffer must outlive the returned buffer
  _CCCL_HIDE_FROM_ABI __uninitialized_async_buffer __replace_allocation(const size_t __count)
  {
    // Create a new buffer with a reference to the stored memory resource and
    // swap allocation information
    __uninitialized_async_buffer __ret{__fake_resource_ref{::cuda::std::addressof(__mr_)}, __stream_, __count};
    ::cuda::std::swap(__count_, __ret.__count_);
    ::cuda::std::swap(__buf_, __ret.__buf_);
    return __ret;
  }
};

template <class _Tp>
using uninitialized_async_device_buffer = __uninitialized_async_buffer<_Tp, ::cuda::mr::device_accessible>;
_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif //__CUDAX__CONTAINERS_UNINITIALIZED_ASYNC_BUFFER_H
