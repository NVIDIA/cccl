//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_STOP_TOKEN
#define __CUDAX_EXECUTION_STOP_TOKEN

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__thread/threading_support.h>
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/atomic>

#include <cuda/experimental/__execution/thread.cuh>
#include <cuda/experimental/__execution/utility.cuh>

#if _CCCL_HAS_INCLUDE(<stop_token>) && __cpp_lib_jthread >= 201911
#  include <stop_token>
#endif

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
// [stoptoken.inplace], class inplace_stop_token
class _CCCL_TYPE_VISIBILITY_DEFAULT inplace_stop_token;

// [stopsource.inplace], class inplace_stop_source
class _CCCL_TYPE_VISIBILITY_DEFAULT inplace_stop_source;

// [stopcallback.inplace], class template inplace_stop_callback
template <class _Callback>
class inplace_stop_callback;

namespace __stok
{
struct __inplace_stop_callback_base
{
  _CCCL_API constexpr void __execute() noexcept
  {
    this->__execute_fn_(this);
  }

protected:
  using __execute_fn_t _CCCL_NODEBUG_ALIAS = void(__inplace_stop_callback_base*) noexcept;

  _CCCL_API constexpr explicit __inplace_stop_callback_base(
    const inplace_stop_source* __source, __execute_fn_t* __execute) noexcept
      : __source_(__source)
      , __execute_fn_(__execute)
  {}

  _CCCL_API constexpr void __register_callback() noexcept;

  friend inplace_stop_source;

  const inplace_stop_source* __source_;
  __execute_fn_t* __execute_fn_;
  __inplace_stop_callback_base* __next_      = nullptr;
  __inplace_stop_callback_base** __prev_ptr_ = nullptr;
  bool* __removed_during_callback_           = nullptr;
  ::cuda::std::atomic<bool> __callback_completed_{false};
};

struct __spin_wait
{
  _CCCL_HIDE_FROM_ABI __spin_wait() noexcept = default;

  _CCCL_API void __wait() noexcept
  {
    if (__count_ == 0)
    {
      execution::__this_thread_yield();
    }
    else
    {
      --__count_;
      ::cuda::std::__cccl_thread_yield_processor();
    }
  }

private:
  static constexpr uint32_t __yield_threshold = 20;
  uint32_t __count_                           = __yield_threshold;
};

template <template <class> class>
struct __check_type_alias_exists;
} // namespace __stok

// [stoptoken.never], class never_stop_token
struct _CCCL_TYPE_VISIBILITY_DEFAULT never_stop_token
{
private:
  struct __callback_type
  {
    _CCCL_API constexpr explicit __callback_type(never_stop_token, ::cuda::std::__ignore_t) noexcept {}
  };

public:
  template <class>
  using callback_type _CCCL_NODEBUG_ALIAS = __callback_type;

  _CCCL_API static constexpr auto stop_requested() noexcept -> bool
  {
    return false;
  }

  _CCCL_API static constexpr auto stop_possible() noexcept -> bool
  {
    return false;
  }

  _CCCL_API friend constexpr auto operator==(const never_stop_token&, const never_stop_token&) noexcept -> bool
  {
    return true;
  }

  _CCCL_API friend constexpr auto operator!=(const never_stop_token&, const never_stop_token&) noexcept -> bool
  {
    return false;
  }
};

template <class _Callback>
class _CCCL_TYPE_VISIBILITY_DEFAULT inplace_stop_callback;

// [stopsource.inplace], class inplace_stop_source
class _CCCL_TYPE_VISIBILITY_DEFAULT inplace_stop_source
{
public:
  _CCCL_HIDE_FROM_ABI inplace_stop_source() noexcept = default;
  _CCCL_API ~inplace_stop_source();
  inplace_stop_source(inplace_stop_source&&) = delete;

  _CCCL_API constexpr auto get_token() const noexcept -> inplace_stop_token;

  _CCCL_API auto request_stop() noexcept -> bool;

  _CCCL_API auto stop_requested() const noexcept -> bool
  {
    return (__state_.load(::cuda::std::memory_order_acquire) & __stop_requested_flag) != 0;
  }

private:
  friend inplace_stop_token;
  friend __stok::__inplace_stop_callback_base;
  template <class>
  friend class inplace_stop_callback;

  _CCCL_API auto __lock() const noexcept -> uint8_t;
  _CCCL_API void __unlock(uint8_t) const noexcept;

  _CCCL_API auto __try_lock_unless_stop_requested(bool) const noexcept -> bool;

  _CCCL_API auto __try_add_callback(__stok::__inplace_stop_callback_base*) const noexcept -> bool;

  _CCCL_API void __remove_callback(__stok::__inplace_stop_callback_base*) const noexcept;

  static constexpr uint8_t __stop_requested_flag = 1;
  static constexpr uint8_t __locked_flag         = 2;

  mutable ::cuda::std::atomic<uint8_t> __state_{0};
  mutable __stok::__inplace_stop_callback_base* __callbacks_ = nullptr;
  execution::__thread_id __notifying_thread_;
};

// [stoptoken.inplace], class inplace_stop_token
class _CCCL_TYPE_VISIBILITY_DEFAULT inplace_stop_token
{
public:
  template <class _Fun>
  using callback_type _CCCL_NODEBUG_ALIAS = inplace_stop_callback<_Fun>;

  _CCCL_HIDE_FROM_ABI inplace_stop_token() = default;

  _CCCL_HIDE_FROM_ABI inplace_stop_token(const inplace_stop_token& __other) noexcept = default;

  _CCCL_API constexpr inplace_stop_token(inplace_stop_token&& __other) noexcept
      : __source_(execution::__exchange(__other.__source_, {}))
  {}

  _CCCL_HIDE_FROM_ABI constexpr auto operator=(const inplace_stop_token& __other) noexcept
    -> inplace_stop_token& = default;

  _CCCL_API constexpr auto operator=(inplace_stop_token&& __other) noexcept -> inplace_stop_token&
  {
    __source_ = execution::__exchange(__other.__source_, nullptr);
    return *this;
  }

  [[nodiscard]] _CCCL_API constexpr auto stop_requested() const noexcept -> bool
  {
    return __source_ != nullptr && __source_->stop_requested();
  }

  [[nodiscard]] _CCCL_API constexpr auto stop_possible() const noexcept -> bool
  {
    return __source_ != nullptr;
  }

  _CCCL_API constexpr void swap(inplace_stop_token& __other) noexcept
  {
    execution::__swap(__source_, __other.__source_);
  }

  _CCCL_API friend constexpr auto operator==(const inplace_stop_token& __a, const inplace_stop_token& __b) noexcept
    -> bool
  {
    return __a.__source_ == __b.__source_;
  }

  _CCCL_API friend constexpr auto operator!=(const inplace_stop_token& __a, const inplace_stop_token& __b) noexcept
    -> bool
  {
    return __a.__source_ != __b.__source_;
  }

private:
  friend inplace_stop_source;
  template <class>
  friend class inplace_stop_callback;

  _CCCL_API constexpr explicit inplace_stop_token(const inplace_stop_source* __source) noexcept
      : __source_(__source)
  {}

  const inplace_stop_source* __source_ = nullptr;
};

_CCCL_API constexpr auto inplace_stop_source::get_token() const noexcept -> inplace_stop_token
{
  return inplace_stop_token{this};
}

// [stopcallback.inplace], class template inplace_stop_callback
template <class _Fun>
class _CCCL_TYPE_VISIBILITY_DEFAULT inplace_stop_callback : __stok::__inplace_stop_callback_base
{
public:
  template <class _Fun2>
  _CCCL_API constexpr explicit inplace_stop_callback(inplace_stop_token __token,
                                                     _Fun2&& __fun) noexcept(__nothrow_constructible<_Fun, _Fun2>)
      : __stok::__inplace_stop_callback_base(__token.__source_, &inplace_stop_callback::__execute_impl)
      , __fun(static_cast<_Fun2&&>(__fun))
  {
    __register_callback();
  }

  _CCCL_API ~inplace_stop_callback()
  {
    if (__source_ != nullptr)
    {
      __source_->__remove_callback(this);
    }
  }

private:
  _CCCL_API static constexpr void __execute_impl(__stok::__inplace_stop_callback_base* __cb) noexcept
  {
    static_cast<_Fun&&>(static_cast<inplace_stop_callback*>(__cb)->__fun)();
  }

  _Fun __fun;
};

namespace __stok
{
_CCCL_API constexpr void __inplace_stop_callback_base::__register_callback() noexcept
{
  if (__source_ != nullptr)
  {
    if (!__source_->__try_add_callback(this))
    {
      __source_ = nullptr;
      // _Callback not registered because stop_requested() was true.
      // Execute inline here.
      __execute();
    }
  }
}
} // namespace __stok

_CCCL_API inline inplace_stop_source::~inplace_stop_source()
{
  _CCCL_ASSERT((__state_.load(::cuda::std::memory_order_relaxed) & __locked_flag) == 0, "");
  _CCCL_ASSERT(__callbacks_ == nullptr, "");
}

_CCCL_API inline auto inplace_stop_source::request_stop() noexcept -> bool
{
  if (!__try_lock_unless_stop_requested(true))
  {
    return true;
  }

  __notifying_thread_ = execution::__this_thread_id();

  // We are responsible for executing callbacks.
  while (__callbacks_ != nullptr)
  {
    auto* __callbk        = __callbacks_;
    __callbk->__prev_ptr_ = nullptr;
    __callbacks_          = __callbk->__next_;
    if (__callbacks_ != nullptr)
    {
      __callbacks_->__prev_ptr_ = &__callbacks_;
    }

    __state_.store(__stop_requested_flag, ::cuda::std::memory_order_release);

    bool __removed_during_callback_      = false;
    __callbk->__removed_during_callback_ = &__removed_during_callback_;

    __callbk->__execute();

    if (!__removed_during_callback_)
    {
      __callbk->__removed_during_callback_ = nullptr;
      __callbk->__callback_completed_.store(true, ::cuda::std::memory_order_release);
    }

    __lock();
  }

  __state_.store(__stop_requested_flag, ::cuda::std::memory_order_release);
  return false;
}

_CCCL_API inline auto inplace_stop_source::__lock() const noexcept -> uint8_t
{
  __stok::__spin_wait __spin;
  auto __old_state = __state_.load(::cuda::std::memory_order_relaxed);
  do
  {
    while ((__old_state & __locked_flag) != 0)
    {
      __spin.__wait();
      __old_state = __state_.load(::cuda::std::memory_order_relaxed);
    }
  } while (!__state_.compare_exchange_weak(
    __old_state, __old_state | __locked_flag, ::cuda::std::memory_order_acquire, ::cuda::std::memory_order_relaxed));

  return __old_state;
}

_CCCL_API inline void inplace_stop_source::__unlock(uint8_t __old_state) const noexcept
{
  (void) __state_.store(__old_state, ::cuda::std::memory_order_release);
}

_CCCL_API inline auto inplace_stop_source::__try_lock_unless_stop_requested(bool __set_stop_requested) const noexcept
  -> bool
{
  __stok::__spin_wait __spin;
  auto __old_state = __state_.load(::cuda::std::memory_order_relaxed);
  do
  {
    while (true)
    {
      if ((__old_state & __stop_requested_flag) != 0)
      {
        // Stop already requested.
        return false;
      }
      else if (__old_state == 0)
      {
        break;
      }
      else
      {
        __spin.__wait();
        __old_state = __state_.load(::cuda::std::memory_order_relaxed);
      }
    }
  } while (!__state_.compare_exchange_weak(
    __old_state,
    __set_stop_requested ? (__locked_flag | __stop_requested_flag) : __locked_flag,
    ::cuda::std::memory_order_acq_rel,
    ::cuda::std::memory_order_relaxed));

  // Lock acquired successfully
  return true;
}

_CCCL_API inline auto
inplace_stop_source::__try_add_callback(__stok::__inplace_stop_callback_base* __callbk) const noexcept -> bool
{
  if (!__try_lock_unless_stop_requested(false))
  {
    return false;
  }

  __callbk->__next_     = __callbacks_;
  __callbk->__prev_ptr_ = &__callbacks_;
  if (__callbacks_ != nullptr)
  {
    __callbacks_->__prev_ptr_ = &__callbk->__next_;
  }
  __callbacks_ = __callbk;

  __unlock(0);

  return true;
}

_CCCL_API inline void
inplace_stop_source::__remove_callback(__stok::__inplace_stop_callback_base* __callbk) const noexcept
{
  auto __old_state = __lock();

  if (__callbk->__prev_ptr_ != nullptr)
  {
    // _Callback has not been executed yet.
    // Remove from the list.
    *__callbk->__prev_ptr_ = __callbk->__next_;
    if (__callbk->__next_ != nullptr)
    {
      __callbk->__next_->__prev_ptr_ = __callbk->__prev_ptr_;
    }
    __unlock(__old_state);
  }
  else
  {
    auto __notifying_thread_ = this->__notifying_thread_;
    __unlock(__old_state);

    // _Callback has either already been executed or is
    // currently executing on another thread.
    if (execution::__this_thread_id() == __notifying_thread_)
    {
      if (__callbk->__removed_during_callback_ != nullptr)
      {
        *__callbk->__removed_during_callback_ = true;
      }
    }
    else
    {
      // Concurrently executing on another thread.
      // Wait until the other thread finishes executing the callback.
      __stok::__spin_wait __spin;
      while (!__callbk->__callback_completed_.load(::cuda::std::memory_order_acquire))
      {
        __spin.__wait();
      }
    }
  }
}

struct __on_stop_request
{
  inplace_stop_source& __source_;

  _CCCL_API void operator()() const noexcept
  {
    __source_.request_stop();
  }
};

template <class _Token, class _Callback>
using stop_callback_for_t _CCCL_NODEBUG_ALIAS = typename _Token::template callback_type<_Callback>;
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_STOP_TOKEN
