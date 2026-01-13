//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXPERIMENTAL_UTILITY_SHARED_PTR
#define __CUDAX_EXPERIMENTAL_UTILITY_SHARED_PTR

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__memory/allocator_traits.h>
#include <cuda/std/__new/device_new.h>
#include <cuda/std/__type_traits/is_class.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/atomic>

// #include <cuda/experimental/__execution/lazy.cuh>
#include <cuda/experimental/__utility/manual_lifetime.cuh>
#include <cuda/experimental/__utility/scope_exit.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

// This file contains a simplified implementation of a shared_ptr-like type
// for use in CUDA C++ code. It supports basic shared ownership semantics,
// but not advanced features like weak_ptr or aliasing constructors.
namespace cuda::experimental
{
template <class _Ty>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __shared_ptr;

template <class _Ty, class... _Args>
_CCCL_API __shared_ptr<_Ty> __make_shared(_Args&&... __args)
{
  return __shared_ptr<_Ty>::__make_shared(static_cast<_Args&&>(__args)...);
}

template <class _Ty, class _Alloc, class... _Args>
_CCCL_API __shared_ptr<_Ty> __allocate_shared(const _Alloc& __alloc, _Args&&... __args)
{
  return __shared_ptr<_Ty>::__allocate_shared(__alloc, static_cast<_Args&&>(__args)...);
}

namespace __detail
{
struct __shared_ptr_base
{
  struct __control_block
  {
    using __destroy_vfn_t           = void(__control_block*, void*) noexcept;
    __destroy_vfn_t* __destroy_vfn_ = nullptr;

    ::cuda::std::atomic<size_t> __ref_count_{1};
    ::cuda::std::atomic<size_t> __reserved_{0};
  };

  __control_block* __cb_ptr_ = nullptr;
};

template <class _Alloc, class _Value>
using __rebind_alloc_t = typename ::cuda::std::allocator_traits<_Alloc>::template rebind_alloc<_Value>;
} // namespace __detail

template <class _Ty>
struct _CCCL_TYPE_VISIBILITY_DEFAULT _CCCL_DECLSPEC_EMPTY_BASES __shared_ptr //
    : private __detail::__shared_ptr_base
{
  using element_type = _Ty;

  _CCCL_HIDE_FROM_ABI __shared_ptr() noexcept = default;

  _CCCL_API __shared_ptr(::cuda::std::nullptr_t) noexcept {}

  // move constructor
  _CCCL_API __shared_ptr(__shared_ptr&& __other) noexcept
      : __detail::__shared_ptr_base{::cuda::std::exchange(__other.__cb_ptr_, nullptr)}
      , __val_ptr_{::cuda::std::exchange(__other.__val_ptr_, nullptr)}
  {}

  // copy constructor
  _CCCL_API __shared_ptr(const __shared_ptr& __other) noexcept
      : __detail::__shared_ptr_base{__other.__cb_ptr_}
      , __val_ptr_{__other.__val_ptr_}
  {
    if (__cb_ptr_)
    {
      __cb_ptr_->__ref_count_.fetch_add(1, ::cuda::std::memory_order_acq_rel);
    }
  }

  // converting move constructor
  _CCCL_TEMPLATE(class _Other)
  _CCCL_REQUIRES(::cuda::std::convertible_to<_Other*, _Ty*>)
  _CCCL_API __shared_ptr(__shared_ptr<_Other>&& __other) noexcept
      : __detail::__shared_ptr_base{::cuda::std::exchange(__other.__cb_ptr_, nullptr)}
      , __val_ptr_{::cuda::std::exchange(__other.__val_ptr_, nullptr)}
  {}

  // converting copy constructor
  _CCCL_TEMPLATE(class _Other)
  _CCCL_REQUIRES(::cuda::std::convertible_to<_Other*, _Ty*>)
  _CCCL_API __shared_ptr(const __shared_ptr<_Other>& __other) noexcept
      : __detail::__shared_ptr_base{__other.__cb_ptr_}
      , __val_ptr_{__other.__val_ptr_}
  {
    if (__cb_ptr_)
    {
      __cb_ptr_->__ref_count_.fetch_add(1, ::cuda::std::memory_order_acq_rel);
    }
  }

  _CCCL_API __shared_ptr& operator=(__shared_ptr&& __other) noexcept
  {
    __shared_ptr(_CCCL_MOVE(__other)).swap(*this);
    return *this;
  }

  _CCCL_API __shared_ptr& operator=(const __shared_ptr& __other) noexcept
  {
    __shared_ptr(__other).swap(*this);
    return *this;
  }

  _CCCL_TEMPLATE(class _Other)
  _CCCL_REQUIRES(::cuda::std::convertible_to<_Other*, _Ty*>)
  _CCCL_API __shared_ptr& operator=(__shared_ptr<_Other>&& __other) noexcept
  {
    __shared_ptr(_CCCL_MOVE(__other)).swap(*this);
    return *this;
  }

  _CCCL_TEMPLATE(class _Other)
  _CCCL_REQUIRES(::cuda::std::convertible_to<_Other*, _Ty*>)
  _CCCL_API __shared_ptr& operator=(const __shared_ptr<_Other>& __other) noexcept
  {
    __shared_ptr(__other).swap(*this);
    return *this;
  }

  _CCCL_API ~__shared_ptr()
  {
    reset();
  }

  _CCCL_API void swap(__shared_ptr& __other) noexcept
  {
    ::cuda::std::swap(__cb_ptr_, __other.__cb_ptr_);
    ::cuda::std::swap(__val_ptr_, __other.__val_ptr_);
  }

  _CCCL_API friend void swap(__shared_ptr& __lhs, __shared_ptr& __rhs) noexcept
  {
    __lhs.swap(__rhs);
  }

  [[nodiscard]] _CCCL_API _Ty* operator->() const noexcept
  {
    return __val_ptr_;
  }

  [[nodiscard]] _CCCL_API _Ty& operator*() const noexcept
  {
    return *__val_ptr_;
  }

  [[nodiscard]] _CCCL_API _Ty* get() const noexcept
  {
    return __val_ptr_;
  }

  _CCCL_API void reset() noexcept
  {
    if (__cb_ptr_)
    {
      if (__cb_ptr_->__ref_count_.fetch_sub(1, ::cuda::std::memory_order_acq_rel) == 1)
      {
        __cb_ptr_->__destroy_vfn_(__cb_ptr_, __val_ptr_);
      }
      __cb_ptr_  = nullptr;
      __val_ptr_ = nullptr;
    }
  }

  [[nodiscard]] _CCCL_API size_t use_count() const noexcept
  {
    return __cb_ptr_ ? __cb_ptr_->__ref_count_.load(::cuda::std::memory_order_acquire) : 0;
  }

  [[nodiscard]] _CCCL_API explicit operator bool() const noexcept
  {
    return __cb_ptr_ != nullptr;
  }

  [[nodiscard]] _CCCL_API bool operator!() const noexcept
  {
    return __cb_ptr_ == nullptr;
  }

  [[nodiscard]] _CCCL_API bool operator==(const __shared_ptr& __other) const noexcept
  {
    return __cb_ptr_ == __other.__cb_ptr_;
  }

  [[nodiscard]] _CCCL_API bool operator!=(const __shared_ptr& __other) const noexcept
  {
    return !(*this == __other);
  }

private:
  template <class>
  friend struct __shared_ptr;

  template <class _Ty2, class... _Args>
  _CCCL_API friend __shared_ptr<_Ty2> __make_shared(_Args&&...);

  template <class _Ty2, class _Alloc, class... _Args>
  _CCCL_API friend __shared_ptr<_Ty2> __allocate_shared(const _Alloc&, _Args&&...);

  _CCCL_API explicit __shared_ptr(__shared_ptr_base::__control_block* __cb_ptr, _Ty* __val_ptr) noexcept
      : __detail::__shared_ptr_base{__cb_ptr}
      , __val_ptr_{__val_ptr}
  {}

  template <class _Deleter>
  struct __deleter_wrapper
  {
    _CCCL_API __deleter_wrapper(_Deleter __deleter) noexcept
        : __deleter_{static_cast<_Deleter&&>(__deleter)}
    {}

    _CCCL_API void operator()(_Ty* __ptr) noexcept
    {
      __deleter_(__ptr);
    }

    _Deleter __deleter_;
  };

  struct _CCCL_DECLSPEC_EMPTY_BASES __control_block : __shared_ptr_base::__control_block
  {
    using __destroy_vfn_t = void(__control_block*) noexcept;

    template <class... _Args>
    _CCCL_API explicit __control_block(_Args&&... __args) noexcept(
      ::cuda::std::is_nothrow_constructible_v<_Ty, _Args...>)
        : __shared_ptr_base::__control_block{&__control_block::__destroy}
        , __value_(static_cast<_Args&&>(__args)...)
    {}

    _CCCL_API static void __destroy(__shared_ptr_base::__control_block* __cp_ptr, void*) noexcept
    {
      delete static_cast<__control_block*>(__cp_ptr);
    }

    _Ty __value_;

  protected:
    // control blocks must be destroyed using the __destroy_vfn_ member.
    _CCCL_HIDE_FROM_ABI ~__control_block() = default;
  };

  template <class _Deleter, bool _IsClass = ::cuda::std::is_class_v<_Deleter>>
  struct _CCCL_DECLSPEC_EMPTY_BASES __control_block_with_deleter
      : __control_block
      , _Deleter
  {
    static_assert(::cuda::std::is_nothrow_move_constructible_v<_Deleter>, "Deleter must be nothrow movable");

    template <class... _Args>
    _CCCL_API explicit __control_block_with_deleter(_Deleter __deleter, _Args&&... __args) noexcept(
      ::cuda::std::is_nothrow_constructible_v<_Ty, _Args...>)
        : __control_block(static_cast<_Args&&>(__args)...)
        , _Deleter{static_cast<_Deleter&&>(__deleter)}
    {
      this->__destroy_vfn_ = &__control_block_with_deleter::__destroy;
    }

    _CCCL_API static void __destroy(__shared_ptr_base::__control_block* __cb_ptr, void* __val_ptr) noexcept
    {
      _Deleter& __deleter = *static_cast<__control_block_with_deleter*>(__cb_ptr);
      __deleter(static_cast<_Ty*>(__val_ptr));
    }
  };

  template <class _Deleter>
  struct __control_block_with_deleter<_Deleter, false> : __control_block_with_deleter<__deleter_wrapper<_Deleter>>
  {
    using __control_block_with_deleter<__deleter_wrapper<_Deleter>>::__control_block_with_deleter;
  };

  template <class _Alloc>
  struct _CCCL_DECLSPEC_EMPTY_BASES __allocator_deleter : _Alloc
  {
    using __control_block_t = __control_block_with_deleter<__allocator_deleter>;
    static_assert(::cuda::std::is_nothrow_copy_constructible_v<_Alloc>, "Allocator must be nothrow copyable");

    _CCCL_API explicit __allocator_deleter(const _Alloc& __alloc) noexcept
        : _Alloc{__alloc}
    {}

    _CCCL_API void operator()([[maybe_unused]] _Ty* __ptr) noexcept
    {
      __control_block_t* __cb_ptr = static_cast<__control_block_t*>(this);
      _CCCL_ASSERT(::cuda::std::addressof(__cb_ptr->__value_) == __ptr, "Pointer mismatch in allocator deleter");

      using __cb_alloc_t = __detail::__rebind_alloc_t<_Alloc, __control_block_t>;
      __cb_alloc_t __cb_alloc{static_cast<_Alloc&>(*__cb_ptr)};

      // Run the destructor for the control block and deallocate it:
      ::cuda::std::allocator_traits<__cb_alloc_t>::destroy(__cb_alloc, __cb_ptr);
      ::cuda::std::allocator_traits<__cb_alloc_t>::deallocate(__cb_alloc, __cb_ptr, 1);
    }
  };

  template <class... _Args>
  _CCCL_API static __shared_ptr __make_shared(_Args&&... __args)
  {
    auto* __cb_ptr = ::new __control_block(static_cast<_Args&&>(__args)...);
    return __shared_ptr{__cb_ptr, ::cuda::std::addressof(__cb_ptr->__value_)};
  }

  template <class _Alloc, class... _Args>
  _CCCL_API static __shared_ptr __allocate_shared(const _Alloc& __alloc, _Args&&... __args)
  {
    using __control_block_t = __control_block_with_deleter<__allocator_deleter<_Alloc>>;
    using __cb_alloc_t      = __detail::__rebind_alloc_t<_Alloc, __control_block_t>;
    using __traits_t        = ::cuda::std::allocator_traits<__cb_alloc_t>;
    __cb_alloc_t __cb_alloc{__alloc};

    // allocate memory for control block
    auto* __cb_ptr = __traits_t::allocate(__cb_alloc, 1);

    // use scope_exit to deallocate if construction throws
    scope_exit __on_exit([__cb_ptr, &__cb_alloc]() noexcept {
      __traits_t::deallocate(__cb_alloc, __cb_ptr, 1);
    });
    __traits_t::construct(__cb_alloc, __cb_ptr, __allocator_deleter<_Alloc>{__alloc}, static_cast<_Args&&>(__args)...);
    __on_exit.release();

    return __shared_ptr{__cb_ptr, ::cuda::std::addressof(__cb_ptr->__value_)};
  }

  _Ty* __val_ptr_ = nullptr;
};
} // namespace cuda::experimental

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXPERIMENTAL_UTILITY_SHARED_PTR
