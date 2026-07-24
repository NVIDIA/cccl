//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___CONTAINER_RESIZABLE_BUFFER_H
#define _CUDA___CONTAINER_RESIZABLE_BUFFER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/__container/buffer.h>
#  include <cuda/std/__host_stdlib/stdexcept>
#  include <cuda/std/__memory/addressof.h>
#  include <cuda/std/__utility/exchange.h>
#  include <cuda/std/__utility/move.h>
#  include <cuda/std/__utility/swap.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA
_CCCL_BEGIN_NAMESPACE_ABI_VER4_BUMP

//! @brief Internal buffer adapter that tracks allocation capacity separately
//! from logical size.
//!
//! `cuda::__resizable_buffer` preserves the `cuda::buffer` interface through
//! public inheritance. The inherited `size()` remains the logical number of
//! elements, while `capacity()` is the number of elements that can be addressed
//! without reallocating.
template <class _Tp, class... _Properties>
class __resizable_buffer : public buffer<_Tp, _Properties...>
{
  using __base_t = buffer<_Tp, _Properties...>;

public:
  using size_type = typename __base_t::size_type;

private:
  size_type __capacity_ = this->size();

  _CCCL_HOST_API void __replace_allocation(::cuda::stream_ref __stream, size_type __new_capacity)
  {
    const auto __old_size     = this->size();
    const auto __old_capacity = __capacity_;
    _CCCL_ASSERT(__old_size <= __new_capacity, "cuda::__resizable_buffer cannot reallocate below size");

    auto __old_buffer = __base_t::__replace_allocation(__stream, __new_capacity);
    __capacity_       = __new_capacity;
    __base_t::__set_size_unsynchronized(__old_size);
    __old_buffer.__set_size_unsynchronized(__old_capacity);

    if (__old_size != 0)
    {
      ::cuda::__copy_cross_buffers(__stream, *this, __old_buffer, __old_size);
      // Free on the copy stream so the stream-ordered deallocation happens
      // after the read from the old allocation.
      __old_buffer.destroy(__stream);
    }
    else
    {
      __old_buffer.destroy();
    }
  }

  _CCCL_HOST_API void __replace_allocation_discard(::cuda::stream_ref __stream, size_type __new_capacity)
  {
    const auto __old_capacity = __capacity_;

    auto __old_buffer = __base_t::__replace_allocation(__stream, __new_capacity);
    __capacity_       = __new_capacity;
    __old_buffer.__set_size_unsynchronized(__old_capacity);
    __old_buffer.destroy();
  }

public:
  using __base_t::__base_t;

  //! @brief Constructs a resizable buffer by taking over an existing
  //! `cuda::buffer` allocation.
  //!
  //! The initial capacity is the source buffer size.
  _CCCL_HOST_API explicit __resizable_buffer(__base_t&& __buffer) noexcept
      : __base_t(::cuda::std::move(__buffer))
  {}

  __resizable_buffer(const __resizable_buffer&)            = delete;
  __resizable_buffer& operator=(const __resizable_buffer&) = delete;

  _CCCL_HOST_API __resizable_buffer(__resizable_buffer&& __other) noexcept
      : __base_t(::cuda::std::move(__other))
      , __capacity_(::cuda::std::exchange(__other.__capacity_, 0))
  {}

  _CCCL_HOST_API __resizable_buffer& operator=(__resizable_buffer&& __other) noexcept
  {
    if (this != ::cuda::std::addressof(__other))
    {
      __base_t::__destroy_with_capacity(this->stream(), __capacity_);
      __base_t::operator=(::cuda::std::move(__other));
      __capacity_ = ::cuda::std::exchange(__other.__capacity_, 0);
    }
    return *this;
  }

  _CCCL_HOST_API ~__resizable_buffer()
  {
    destroy();
  }

  //! @brief Returns the number of elements that fit in the current allocation
  //! without reallocating.
  [[nodiscard]] _CCCL_HOST_API size_type capacity() const noexcept
  {
    return __capacity_;
  }

  //! @brief Returns `capacity() * sizeof(value_type)`.
  [[nodiscard]] _CCCL_HOST_API size_type capacity_bytes() const noexcept
  {
    return __capacity_ * sizeof(_Tp);
  }

  //! @brief Changes the logical size without initializing any new elements.
  //!
  //! This overload can only shrink. Growing requires `cuda::no_init` to make
  //! the lack of initialization explicit.
  _CCCL_HOST_API void resize(size_type __new_size)
  {
    if (__new_size > this->size())
    {
      _CCCL_THROW(::std::invalid_argument, "cuda::__resizable_buffer::resize requires cuda::no_init to grow");
    }
    __base_t::__set_size_unsynchronized(__new_size);
  }

  //! @brief Changes the logical size without reallocating.
  //!
  //! This overload can shrink or grow within `capacity()`. Growing beyond
  //! capacity requires an explicit stream.
  _CCCL_HOST_API void resize(size_type __new_size, ::cuda::no_init_t)
  {
    if (__new_size > __capacity_)
    {
      _CCCL_THROW(::std::invalid_argument,
                  "cuda::__resizable_buffer::resize requires an explicit stream to grow beyond capacity");
    }
    __base_t::__set_size_unsynchronized(__new_size);
  }

  //! @brief Changes the logical size, reallocating on \p __stream if needed.
  //!
  //! If reallocation is needed, existing logical elements are copied to the new
  //! allocation and newly exposed elements are left uninitialized.
  _CCCL_HOST_API void resize(::cuda::stream_ref __stream, size_type __new_size, ::cuda::no_init_t)
  {
    if (__new_size > __capacity_)
    {
      __replace_allocation(__stream, __new_size);
    }
    __base_t::__set_size_unsynchronized(__new_size);
  }

  //! @brief Changes the logical size, reallocating on \p __stream if needed,
  //! without preserving existing contents.
  //!
  //! All elements in the resulting logical range are left uninitialized.
  _CCCL_HOST_API void resize_discard(::cuda::stream_ref __stream, size_type __new_size, ::cuda::no_init_t)
  {
    if (__new_size > __capacity_)
    {
      __replace_allocation_discard(__stream, __new_size);
    }
    __base_t::__set_size_unsynchronized(__new_size);
  }

  //! @brief Swaps the allocation, logical size, stream, memory resource, and
  //! capacity with \p __other.
  _CCCL_HOST_API void swap(__resizable_buffer& __other) noexcept
  {
    __base_t::swap(__other);
    ::cuda::std::swap(__capacity_, __other.__capacity_);
  }

  _CCCL_HOST_API friend void swap(__resizable_buffer& __lhs, __resizable_buffer& __rhs) noexcept
  {
    __lhs.swap(__rhs);
  }

  //! @brief Destroys the allocation using capacity rather than logical size.
  _CCCL_HOST_API void destroy(::cuda::stream_ref __stream)
  {
    __base_t::__destroy_with_capacity(__stream, __capacity_);
    __capacity_ = 0;
  }

  _CCCL_HOST_API void destroy()
  {
    destroy(this->stream());
  }
};

_CCCL_END_NAMESPACE_ABI_VER4_BUMP
_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___CONTAINER_RESIZABLE_BUFFER_H
