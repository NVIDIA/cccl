//===----------------------------------------------------------------------===//
//
// Part of the CUDA Toolkit, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___CONTAINER_SIMPLE_VECTOR
#define _CUDA___CONTAINER_SIMPLE_VECTOR

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/no_init.h>
#include <cuda/std/__host_stdlib/new>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__type_traits/is_trivially_destructible.h>
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Contiguous storage for @c _Tp that allocates uninitialized, correctly aligned memory.
//! The user is responsible for constructing every element when using the @c no_init constructor.
template <class _Tp>
class __simple_vector
{
private:
  size_t __size_;
  _Tp* __begin_;

  struct alignas(alignof(_Tp)) __fake_payload
  {
    unsigned char __data[sizeof(_Tp)];
  };

  _CCCL_HOST_DEVICE_API void __destroy() noexcept
  {
    if (__begin_ != nullptr)
    {
      if constexpr (!::cuda::std::is_trivially_destructible_v<_Tp>)
      {
        ::cuda::std::__destroy(__begin_, __begin_ + __size_);
      }
      // Need to go through __fake_payload to avoid calling the destructor twice
      ::delete[] reinterpret_cast<__fake_payload*>(__begin_);
    }
  }

public:
  using size_type      = size_t;
  using value_type     = _Tp;
  using iterator       = _Tp*;
  using const_iterator = const _Tp*;

  _CCCL_HIDE_FROM_ABI __simple_vector()                                  = delete;
  _CCCL_HIDE_FROM_ABI __simple_vector(const __simple_vector&)            = delete;
  _CCCL_HIDE_FROM_ABI __simple_vector& operator=(const __simple_vector&) = delete;

  _CCCL_HOST_DEVICE_API __simple_vector(__simple_vector&& __other) noexcept
      : __size_(::cuda::std::exchange(__other.__size_, 0ull))
      , __begin_(::cuda::std::exchange(__other.__begin_, nullptr))
  {}
  _CCCL_HOST_DEVICE_API __simple_vector& operator=(__simple_vector&& __other) noexcept
  {
    if (this != ::cuda::std::addressof(__other))
    {
      __destroy();
      __size_  = ::cuda::std::exchange(__other.__size_, 0ull);
      __begin_ = ::cuda::std::exchange(__other.__begin_, nullptr);
    }
    return *this;
  }

  //! @brief Allocates storage for @p __size elements without constructing them.
  //! @param __size The number of elements to allocate storage for.
  //! @note We can only do this because the user is requires to construct the elements in place
  _CCCL_HOST_DEVICE_API explicit __simple_vector(size_type __size, no_init_t)
      : __size_{__size}
      , __begin_{__size_ == 0 ? nullptr : reinterpret_cast<_Tp*>(::new __fake_payload[__size])}
  {}

  _CCCL_HOST_DEVICE_API ~__simple_vector() noexcept
  {
    __destroy();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr size_type size() const noexcept
  {
    return __size_;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr bool empty() const noexcept
  {
    return __size_ == 0;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API inline _Tp* data() noexcept
  {
    return __begin_;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API inline const _Tp* data() const noexcept
  {
    return __begin_;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API inline iterator begin() noexcept
  {
    return __begin_;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API inline const_iterator begin() const noexcept
  {
    return __begin_;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API inline iterator end() noexcept
  {
    return __begin_ + __size_;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API inline const_iterator end() const noexcept
  {
    return __begin_ + __size_;
  }
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___CONTAINER_SIMPLE_VECTOR
