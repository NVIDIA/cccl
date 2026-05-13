//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_RESOURCE_SHARED_BLOCK_PTR_H
#define _CUDA___MEMORY_RESOURCE_SHARED_BLOCK_PTR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__new_>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/atomic>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_MR

//! @brief A ref-counted control block holding a payload and an atomic reference count.
template <class _Payload>
struct __shared_control_block
{
  template <class... _Args>
  _CCCL_HOST_API explicit __shared_control_block(_Args&&... __args)
      : __payload(::cuda::std::forward<_Args>(__args)...)
  {}

  _CCCL_NO_UNIQUE_ADDRESS _Payload __payload;
  ::cuda::std::atomic<int> __ref_count{1};
};

//! @brief A ref-counted smart pointer to a ``__shared_control_block<_Payload>``.
//!
//! Manages shared ownership of a heap-allocated control block. When the last
//! ``__shared_block_ptr`` referencing a block is destroyed, the block (and its
//! payload) are deleted.
//!
//! Used as the common ref-counting mechanism for ``shared_resource`` and the
//! shared memory pool types.
template <class _Payload>
class __shared_block_ptr
{
  using __block_t = __shared_control_block<_Payload>;

  __block_t* __block_ = nullptr;

public:
  //! @brief Constructs a null ``__shared_block_ptr`` with no control block.
  _CCCL_HIDE_FROM_ABI __shared_block_ptr() = default;

  //! @brief Constructs a new control block, forwarding arguments to the payload.
  _CCCL_TEMPLATE(class _Arg, class... _Rest)
  _CCCL_REQUIRES((!::cuda::std::is_same_v<::cuda::std::remove_cvref_t<_Arg>, __shared_block_ptr>) )
  _CCCL_HOST_API explicit __shared_block_ptr(_Arg&& __arg, _Rest&&... __rest)
      : __block_(new __block_t(::cuda::std::forward<_Arg>(__arg), ::cuda::std::forward<_Rest>(__rest)...))
  {}

  _CCCL_HOST_API __shared_block_ptr(const __shared_block_ptr& __other) noexcept
      : __block_(__other.__block_)
  {
    if (__block_)
    {
      __block_->__ref_count.fetch_add(1, ::cuda::std::memory_order_relaxed);
    }
  }

  _CCCL_HOST_API __shared_block_ptr(__shared_block_ptr&& __other) noexcept
      : __block_(::cuda::std::exchange(__other.__block_, nullptr))
  {}

  _CCCL_HOST_API __shared_block_ptr& operator=(const __shared_block_ptr& __other) noexcept
  {
    __shared_block_ptr(__other).swap(*this);
    return *this;
  }

  _CCCL_HOST_API __shared_block_ptr& operator=(__shared_block_ptr&& __other) noexcept
  {
    __shared_block_ptr(::cuda::std::move(__other)).swap(*this);
    return *this;
  }

  _CCCL_HOST_API ~__shared_block_ptr()
  {
    if (__block_ && __block_->__ref_count.fetch_sub(1, ::cuda::std::memory_order_release) == 1)
    {
      ::cuda::std::atomic_thread_fence(::cuda::std::memory_order_acquire);
      delete __block_;
    }
  }

  _CCCL_HOST_API void swap(__shared_block_ptr& __other) noexcept
  {
    ::cuda::std::swap(__block_, __other.__block_);
  }

  _CCCL_HOST_API friend void swap(__shared_block_ptr& __lhs, __shared_block_ptr& __rhs) noexcept
  {
    __lhs.swap(__rhs);
  }

  [[nodiscard]] _CCCL_HOST_API _Payload& __payload() noexcept
  {
    _CCCL_ASSERT(__block_, "dereferencing a null __shared_block_ptr");
    return __block_->__payload;
  }

  [[nodiscard]] _CCCL_HOST_API const _Payload& __payload() const noexcept
  {
    _CCCL_ASSERT(__block_, "dereferencing a null __shared_block_ptr");
    return __block_->__payload;
  }

  [[nodiscard]] _CCCL_HOST_API explicit operator bool() const noexcept
  {
    return __block_ != nullptr;
  }

  [[nodiscard]] _CCCL_HOST_API friend bool
  operator==(const __shared_block_ptr& __lhs, const __shared_block_ptr& __rhs) noexcept
  {
    return __lhs.__block_ == __rhs.__block_;
  }

#if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_HOST_API friend bool
  operator!=(const __shared_block_ptr& __lhs, const __shared_block_ptr& __rhs) noexcept
  {
    return __lhs.__block_ != __rhs.__block_;
  }
#endif // _CCCL_STD_VER <= 2017
};

_CCCL_END_NAMESPACE_CUDA_MR

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY_RESOURCE_SHARED_BLOCK_PTR_H
