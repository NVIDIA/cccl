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
#include <cuda/std/__new/allocate.h>
#include <cuda/std/__new/launder.h>
#include <cuda/std/__type_traits/is_trivially_destructible.h>
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Contiguous storage for @c _Tp that allocates uninitialized, correctly aligned memory.
//! The user is responsible for constructing every element when using the @c no_init constructor.
//! This is a cheaper drop-in replacement for @c std::vector when only RAII storage is needed.
template <class _Tp>
class __simple_vector
{
private:
  size_t __size_;
  _Tp* __begin_;

  _CCCL_HOST_DEVICE_API static _Tp* __create(size_t __size)
  {
    if (__size == 0 || __size > static_cast<size_t>(-1) / sizeof(_Tp))
    {
      return nullptr;
    }
    return reinterpret_cast<_Tp*>(::cuda::std::__cccl_allocate(__size * sizeof(_Tp), alignof(_Tp)));
  }

  _CCCL_HOST_DEVICE_API void __destroy() noexcept
  {
    if (__begin_ != nullptr)
    {
      if constexpr (!::cuda::std::is_trivially_destructible_v<_Tp>)
      {
        ::cuda::std::__reverse_destroy(__begin_, __begin_ + __size_);
      }
      ::cuda::std::__cccl_deallocate(__begin_, __size_ * sizeof(_Tp), alignof(_Tp));
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

  //! @brief Move-constructs from another vector, transferring ownership of its storage.
  //! @param[in] __other The vector to move from. After the move it is empty and may only be assigned to or destroyed.
  _CCCL_HOST_DEVICE_API __simple_vector(__simple_vector&& __other) noexcept
      : __size_(::cuda::std::exchange(__other.__size_, 0ull))
      , __begin_(::cuda::std::exchange(__other.__begin_, nullptr))
  {}

  //! @brief Move-assigns from another vector, destroying this vector's elements and taking ownership of the other
  //! vector's storage.
  //! @param[in] __other The vector to move from. After the move it is empty and may only be assigned to or destroyed.
  //! @return A reference to this vector.
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
  //! @param[in] __size The number of elements to allocate storage for.
  //! @param[in] __no_init Tag that selects uninitialized storage.
  //! @note The user is required to construct every element in place before it is used or destroyed.
  _CCCL_HOST_DEVICE_API explicit __simple_vector(size_type __size, [[maybe_unused]] no_init_t __no_init)
      : __size_{__size}
      , __begin_{__create(__size_)}
  {}

  //! @brief Destroys the constructed elements and deallocates the storage.
  _CCCL_HOST_DEVICE_API ~__simple_vector() noexcept
  {
    __destroy();
  }

  //! @brief Returns the number of elements.
  //! @return The number of elements.
  [[nodiscard]] _CCCL_HOST_DEVICE_API inline constexpr size_type size() const noexcept
  {
    return __size_;
  }

  //! @brief Returns whether the vector holds no elements.
  //! @return @c true if @c size() is zero, otherwise @c false.
  [[nodiscard]] _CCCL_HOST_DEVICE_API inline constexpr bool empty() const noexcept
  {
    return __size_ == 0;
  }

  //! @brief Returns a pointer to the first element.
  //! @return A pointer to the first element, or @c nullptr if the vector is empty.
  [[nodiscard]] _CCCL_HOST_DEVICE_API inline constexpr _Tp* data() noexcept
  {
    return __begin_;
  }

  //! @overload
  [[nodiscard]] _CCCL_HOST_DEVICE_API inline constexpr const _Tp* data() const noexcept
  {
    return __begin_;
  }

  //! @brief Returns an iterator to the first element. If the vector is empty, the returned iterator equals @c end().
  //! @return An iterator to the first element, or @c nullptr if the vector is empty.
  [[nodiscard]] _CCCL_HOST_DEVICE_API inline constexpr iterator begin() noexcept
  {
    return __begin_;
  }

  //! @overload
  [[nodiscard]] _CCCL_HOST_DEVICE_API inline constexpr const_iterator begin() const noexcept
  {
    return __begin_;
  }

  //! @brief Returns an iterator to the element following the last element. This element acts as a placeholder;
  //! attempting to access it results in undefined behavior.
  //! @return An iterator past the last element, or @c nullptr if the vector is empty.
  [[nodiscard]] _CCCL_HOST_DEVICE_API inline constexpr iterator end() noexcept
  {
    return __size_ == 0 ? nullptr : __begin_ + __size_;
  }

  //! @overload
  [[nodiscard]] _CCCL_HOST_DEVICE_API inline constexpr const_iterator end() const noexcept
  {
    return __size_ == 0 ? nullptr : __begin_ + __size_;
  }
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___CONTAINER_SIMPLE_VECTOR
