//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/atomic>

#include "config.cuh"
#include "thread.cuh"
#include "utility.cuh"

#if __has_include(<stop_token>) && __cpp_lib_jthread >= 201911
#  include <stop_token>
#endif

// This must be the last #include
#include "prologue.cuh"

// warning #20012-D: __device__ annotation is ignored on a
// function("inplace_stop_source") that is explicitly defaulted on its first
// declaration
_CCCL_NV_DIAG_SUPPRESS(20012)

namespace cuda::experimental::__async
{
// [stoptoken.inplace], class inplace_stop_token
class inplace_stop_token;

// [stopsource.inplace], class inplace_stop_source
class inplace_stop_source;

// [stopcallback.inplace], class template inplace_stop_callback
template <class _Callback>
class inplace_stop_callback;

namespace _stok
{
struct _inplace_stop_callback_base
{
  _CCCL_HOST_DEVICE void _execute() noexcept
  {
    this->_execute_fn(this);
  }

protected:
  using _execute_fn_t = void(_inplace_stop_callback_base*) noexcept;

  _CCCL_HOST_DEVICE explicit _inplace_stop_callback_base( //
    const inplace_stop_source* _source, //
    _execute_fn_t* _execute) noexcept
      : _source(_source)
      , _execute_fn(_execute)
  {}

  _CCCL_HOST_DEVICE void _register_callback() noexcept;

  friend inplace_stop_source;

  const inplace_stop_source* _source;
  _execute_fn_t* _execute_fn;
  _inplace_stop_callback_base* _next      = nullptr;
  _inplace_stop_callback_base** _prev_ptr = nullptr;
  bool* _removed_during_callback          = nullptr;
  _CUDA_VSTD::atomic<bool> _callback_completed{false};
};

struct _spin_wait
{
  _spin_wait() noexcept = default;

  _CCCL_HOST_DEVICE void _wait() noexcept
  {
    if (_count == 0)
    {
      __async::_this_thread_yield();
    }
    else
    {
      --_count;
      _CUDAX_PAUSE();
    }
  }

private:
  static constexpr uint32_t _yield_threshold = 20;
  uint32_t _count                            = _yield_threshold;
};

template <template <class> class>
struct _check_type_alias_exists;
} // namespace _stok

// [stoptoken.never], class never_stop_token
struct never_stop_token
{
private:
  struct _callback_type
  {
    _CCCL_HOST_DEVICE explicit _callback_type(never_stop_token, _ignore) noexcept {}
  };

public:
  template <class>
  using callback_type = _callback_type;

  _CCCL_HOST_DEVICE static constexpr auto stop_requested() noexcept -> bool
  {
    return false;
  }

  _CCCL_HOST_DEVICE static constexpr auto stop_possible() noexcept -> bool
  {
    return false;
  }

  _CCCL_HOST_DEVICE friend constexpr bool operator==(const never_stop_token&, const never_stop_token&) noexcept
  {
    return true;
  }

  _CCCL_HOST_DEVICE friend constexpr bool operator!=(const never_stop_token&, const never_stop_token&) noexcept
  {
    return false;
  }
};

template <class _Callback>
class inplace_stop_callback;

// [stopsource.inplace], class inplace_stop_source
class inplace_stop_source
{
public:
  _CCCL_HOST_DEVICE inplace_stop_source() noexcept = default;
  _CCCL_HOST_DEVICE ~inplace_stop_source();
  _CUDAX_IMMOVABLE(inplace_stop_source);

  _CCCL_HOST_DEVICE auto get_token() const noexcept -> inplace_stop_token;

  _CCCL_HOST_DEVICE auto request_stop() noexcept -> bool;

  _CCCL_HOST_DEVICE auto stop_requested() const noexcept -> bool
  {
    return (_state.load(_CUDA_VSTD::memory_order_acquire) & _stop_requested_flag) != 0;
  }

private:
  friend inplace_stop_token;
  friend _stok::_inplace_stop_callback_base;
  template <class>
  friend class inplace_stop_callback;

  _CCCL_HOST_DEVICE auto _lock() const noexcept -> uint8_t;
  _CCCL_HOST_DEVICE void _unlock(uint8_t) const noexcept;

  _CCCL_HOST_DEVICE auto _try_lock_unless_stop_requested(bool) const noexcept -> bool;

  _CCCL_HOST_DEVICE auto _try_add_callback(_stok::_inplace_stop_callback_base*) const noexcept -> bool;

  _CCCL_HOST_DEVICE void _remove_callback(_stok::_inplace_stop_callback_base*) const noexcept;

  static constexpr uint8_t _stop_requested_flag = 1;
  static constexpr uint8_t _locked_flag         = 2;

  mutable _CUDA_VSTD::atomic<uint8_t> _state{0};
  mutable _stok::_inplace_stop_callback_base* _callbacks = nullptr;
  __async::_thread_id _notifying_thread;
};

// [stoptoken.inplace], class inplace_stop_token
class inplace_stop_token
{
public:
  template <class _Fun>
  using callback_type = inplace_stop_callback<_Fun>;

  _CCCL_HOST_DEVICE inplace_stop_token() noexcept
      : _source(nullptr)
  {}

  inplace_stop_token(const inplace_stop_token& _other) noexcept = default;

  _CCCL_HOST_DEVICE inplace_stop_token(inplace_stop_token&& _other) noexcept
      : _source(__async::_exchange(_other._source, {}))
  {}

  auto operator=(const inplace_stop_token& _other) noexcept -> inplace_stop_token& = default;

  _CCCL_HOST_DEVICE auto operator=(inplace_stop_token&& _other) noexcept -> inplace_stop_token&
  {
    _source = __async::_exchange(_other._source, nullptr);
    return *this;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE auto stop_requested() const noexcept -> bool
  {
    return _source != nullptr && _source->stop_requested();
  }

  [[nodiscard]] _CCCL_HOST_DEVICE auto stop_possible() const noexcept -> bool
  {
    return _source != nullptr;
  }

  _CCCL_HOST_DEVICE void swap(inplace_stop_token& _other) noexcept
  {
    __async::_swap(_source, _other._source);
  }

  _CCCL_HOST_DEVICE friend bool operator==(const inplace_stop_token& _a, const inplace_stop_token& _b) noexcept
  {
    return _a._source == _b._source;
  }

  _CCCL_HOST_DEVICE friend bool operator!=(const inplace_stop_token& _a, const inplace_stop_token& _b) noexcept
  {
    return _a._source != _b._source;
  }

private:
  friend inplace_stop_source;
  template <class>
  friend class inplace_stop_callback;

  _CCCL_HOST_DEVICE explicit inplace_stop_token(const inplace_stop_source* _source) noexcept
      : _source(_source)
  {}

  const inplace_stop_source* _source;
};

_CCCL_HOST_DEVICE inline auto inplace_stop_source::get_token() const noexcept -> inplace_stop_token
{
  return inplace_stop_token{this};
}

// [stopcallback.inplace], class template inplace_stop_callback
template <class _Fun>
class inplace_stop_callback : _stok::_inplace_stop_callback_base
{
public:
  template <class _Fun2>
  _CCCL_HOST_DEVICE explicit inplace_stop_callback(inplace_stop_token _token, _Fun2&& _fun) noexcept(
    _CUDA_VSTD::is_nothrow_constructible_v<_Fun, _Fun2>)
      : _stok::_inplace_stop_callback_base(_token._source, &inplace_stop_callback::_execute_impl)
      , _fun(static_cast<_Fun2&&>(_fun))
  {
    _register_callback();
  }

  _CCCL_HOST_DEVICE ~inplace_stop_callback()
  {
    if (_source != nullptr)
    {
      _source->_remove_callback(this);
    }
  }

private:
  _CCCL_HOST_DEVICE static void _execute_impl(_stok::_inplace_stop_callback_base* cb) noexcept
  {
    _CUDA_VSTD::move(static_cast<inplace_stop_callback*>(cb)->_fun)();
  }

  _CCCL_NO_UNIQUE_ADDRESS _Fun _fun;
};

namespace _stok
{
_CCCL_HOST_DEVICE inline void _inplace_stop_callback_base::_register_callback() noexcept
{
  if (_source != nullptr)
  {
    if (!_source->_try_add_callback(this))
    {
      _source = nullptr;
      // Callback not registered because stop_requested() was true.
      // Execute inline here.
      _execute();
    }
  }
}
} // namespace _stok

_CCCL_HOST_DEVICE inline inplace_stop_source::~inplace_stop_source()
{
  _LIBCUDACXX_ASSERT((_state.load(_CUDA_VSTD::memory_order_relaxed) & _locked_flag) == 0, "");
  _LIBCUDACXX_ASSERT(_callbacks == nullptr, "");
}

_CCCL_HOST_DEVICE inline auto inplace_stop_source::request_stop() noexcept -> bool
{
  if (!_try_lock_unless_stop_requested(true))
  {
    return true;
  }

  _notifying_thread = __async::_this_thread_id();

  // We are responsible for executing callbacks.
  while (_callbacks != nullptr)
  {
    auto* _callbk      = _callbacks;
    _callbk->_prev_ptr = nullptr;
    _callbacks         = _callbk->_next;
    if (_callbacks != nullptr)
    {
      _callbacks->_prev_ptr = &_callbacks;
    }

    _state.store(_stop_requested_flag, _CUDA_VSTD::memory_order_release);

    bool _removed_during_callback     = false;
    _callbk->_removed_during_callback = &_removed_during_callback;

    _callbk->_execute();

    if (!_removed_during_callback)
    {
      _callbk->_removed_during_callback = nullptr;
      _callbk->_callback_completed.store(true, _CUDA_VSTD::memory_order_release);
    }

    _lock();
  }

  _state.store(_stop_requested_flag, _CUDA_VSTD::memory_order_release);
  return false;
}

_CCCL_HOST_DEVICE inline auto inplace_stop_source::_lock() const noexcept -> uint8_t
{
  _stok::_spin_wait _spin;
  auto _old_state = _state.load(_CUDA_VSTD::memory_order_relaxed);
  do
  {
    while ((_old_state & _locked_flag) != 0)
    {
      _spin._wait();
      _old_state = _state.load(_CUDA_VSTD::memory_order_relaxed);
    }
  } while (!_state.compare_exchange_weak(
    _old_state, _old_state | _locked_flag, _CUDA_VSTD::memory_order_acquire, _CUDA_VSTD::memory_order_relaxed));

  return _old_state;
}

_CCCL_HOST_DEVICE inline void inplace_stop_source::_unlock(uint8_t _old_state) const noexcept
{
  (void) _state.store(_old_state, _CUDA_VSTD::memory_order_release);
}

_CCCL_HOST_DEVICE inline auto
inplace_stop_source::_try_lock_unless_stop_requested(bool _set_stop_requested) const noexcept -> bool
{
  _stok::_spin_wait _spin;
  auto _old_state = _state.load(_CUDA_VSTD::memory_order_relaxed);
  do
  {
    while (true)
    {
      if ((_old_state & _stop_requested_flag) != 0)
      {
        // Stop already requested.
        return false;
      }
      else if (_old_state == 0)
      {
        break;
      }
      else
      {
        _spin._wait();
        _old_state = _state.load(_CUDA_VSTD::memory_order_relaxed);
      }
    }
  } while (!_state.compare_exchange_weak(
    _old_state,
    _set_stop_requested ? (_locked_flag | _stop_requested_flag) : _locked_flag,
    _CUDA_VSTD::memory_order_acq_rel,
    _CUDA_VSTD::memory_order_relaxed));

  // Lock acquired successfully
  return true;
}

_CCCL_HOST_DEVICE inline auto
inplace_stop_source::_try_add_callback(_stok::_inplace_stop_callback_base* _callbk) const noexcept -> bool
{
  if (!_try_lock_unless_stop_requested(false))
  {
    return false;
  }

  _callbk->_next     = _callbacks;
  _callbk->_prev_ptr = &_callbacks;
  if (_callbacks != nullptr)
  {
    _callbacks->_prev_ptr = &_callbk->_next;
  }
  _callbacks = _callbk;

  _unlock(0);

  return true;
}

_CCCL_HOST_DEVICE inline void
inplace_stop_source::_remove_callback(_stok::_inplace_stop_callback_base* _callbk) const noexcept
{
  auto _old_state = _lock();

  if (_callbk->_prev_ptr != nullptr)
  {
    // Callback has not been executed yet.
    // Remove from the list.
    *_callbk->_prev_ptr = _callbk->_next;
    if (_callbk->_next != nullptr)
    {
      _callbk->_next->_prev_ptr = _callbk->_prev_ptr;
    }
    _unlock(_old_state);
  }
  else
  {
    auto _notifying_thread = this->_notifying_thread;
    _unlock(_old_state);

    // Callback has either already been executed or is
    // currently executing on another thread.
    if (__async::_this_thread_id() == _notifying_thread)
    {
      if (_callbk->_removed_during_callback != nullptr)
      {
        *_callbk->_removed_during_callback = true;
      }
    }
    else
    {
      // Concurrently executing on another thread.
      // Wait until the other thread finishes executing the callback.
      _stok::_spin_wait _spin;
      while (!_callbk->_callback_completed.load(_CUDA_VSTD::memory_order_acquire))
      {
        _spin._wait();
      }
    }
  }
}

struct _on_stop_request
{
  inplace_stop_source& _source;

  _CCCL_HOST_DEVICE void operator()() const noexcept
  {
    _source.request_stop();
  }
};

template <class Token, class Callback>
using stop_callback_for_t = typename Token::template callback_type<Callback>;
} // namespace cuda::experimental::__async

_CCCL_NV_DIAG_DEFAULT(20012)

#include "epilogue.cuh"
