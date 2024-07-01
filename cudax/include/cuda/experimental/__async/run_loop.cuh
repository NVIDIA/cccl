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

#include "config.cuh"

// libcu++ does not have <cuda/std/mutex> or <cuda/std/condition_variable>
#if !defined(__CUDA_ARCH__)

#  include <condition_variable>
#  include <mutex>

#  include "completion_signatures.cuh"
#  include "env.cuh"
#  include "exception.cuh"
#  include "queries.cuh"
#  include "utility.cuh"

// Must be the last include
#  include "prologue.cuh"

namespace cuda::experimental::__async
{
class run_loop;

struct _task : _immovable
{
  _task* _next = this;

  union
  {
    _task* _tail;
    void (*_execute_fn)(_task*) noexcept;
  };

  _CCCL_HOST_DEVICE void _execute() noexcept
  {
    (*_execute_fn)(this);
  }
};

template <class Rcvr>
struct _operation : _task
{
  run_loop* _loop;
  _CCCL_NO_UNIQUE_ADDRESS Rcvr _rcvr;

  using completion_signatures = //
    __async::completion_signatures<set_value_t(), set_error_t(::std::exception_ptr), set_stopped_t()>;

  _CCCL_HOST_DEVICE static void _execute_impl(_task* _p) noexcept
  {
    auto& _rcvr = static_cast<_operation*>(_p)->_rcvr;
    _CUDAX_TRY( //
      ({ //
        if (get_stop_token(get_env(_rcvr)).stop_requested())
        {
          set_stopped(static_cast<Rcvr&&>(_rcvr));
        }
        else
        {
          set_value(static_cast<Rcvr&&>(_rcvr));
        }
      }),
      _CUDAX_CATCH(...)( //
        { //
          set_error(static_cast<Rcvr&&>(_rcvr), ::std::current_exception());
        }))
  }

  _CCCL_HOST_DEVICE explicit _operation(_task* _tail) noexcept
      : _task{{}, this, _tail}
  {}

  _CCCL_HOST_DEVICE _operation(_task* _next, run_loop* _loop, Rcvr _rcvr)
      : _task{{}, _next}
      , _loop{_loop}
      , _rcvr{static_cast<Rcvr&&>(_rcvr)}
  {
    _execute_fn = &_execute_impl;
  }

  _CCCL_HOST_DEVICE void start() & noexcept;
};

class run_loop
{
  template <class... Ts>
  using _completion_signatures = completion_signatures<Ts...>;

  template <class>
  friend struct _operation;

public:
  run_loop() noexcept
  {
    _head._next = _head._tail = &_head;
  }

  class _scheduler
  {
    struct _schedule_task
    {
      using _t             = _schedule_task;
      using _id            = _schedule_task;
      using sender_concept = sender_t;

      template <class Rcvr>
      _CCCL_HOST_DEVICE auto connect(Rcvr rcvr) const noexcept -> _operation<Rcvr>
      {
        return {&_loop->_head, _loop, static_cast<Rcvr&&>(rcvr)};
      }

    private:
      friend _scheduler;

      struct _env
      {
        run_loop* _loop;

        template <class Tag>
        _CCCL_HOST_DEVICE auto query(get_completion_scheduler_t<Tag>) const noexcept -> _scheduler
        {
          return _loop->get_scheduler();
        }
      };

      _CCCL_HOST_DEVICE auto get_env() const noexcept -> _env
      {
        return _env{_loop};
      }

      _CCCL_HOST_DEVICE explicit _schedule_task(run_loop* _loop) noexcept
          : _loop(_loop)
      {}

      run_loop* const _loop;
    };

    friend run_loop;

    _CCCL_HOST_DEVICE explicit _scheduler(run_loop* _loop) noexcept
        : _loop(_loop)
    {}

    _CCCL_HOST_DEVICE auto query(get_forward_progress_guarantee_t) const noexcept -> forward_progress_guarantee
    {
      return forward_progress_guarantee::parallel;
    }

    run_loop* _loop;

  public:
    using scheduler_concept = scheduler_t;

    [[nodiscard]] _CCCL_HOST_DEVICE auto schedule() const noexcept -> _schedule_task
    {
      return _schedule_task{_loop};
    }

    _CCCL_HOST_DEVICE friend bool operator==(const _scheduler& _a, const _scheduler& _b) noexcept
    {
      return _a._loop == _b._loop;
    }

    _CCCL_HOST_DEVICE friend bool operator!=(const _scheduler& _a, const _scheduler& _b) noexcept
    {
      return _a._loop != _b._loop;
    }
  };

  _CCCL_HOST_DEVICE auto get_scheduler() noexcept -> _scheduler
  {
    return _scheduler{this};
  }

  _CCCL_HOST_DEVICE void run();

  _CCCL_HOST_DEVICE void finish();

private:
  _CCCL_HOST_DEVICE void _push_back(_task* _task);
  _CCCL_HOST_DEVICE auto _pop_front() -> _task*;

  ::std::mutex _mutex{};
  ::std::condition_variable _cv{};
  _task _head{};
  bool _stop = false;
};

template <class Rcvr>
_CCCL_HOST_DEVICE inline void _operation<Rcvr>::start() & noexcept {
  _CUDAX_TRY( //
    ({ //
      _loop->_push_back(this); //
    }), //
    _CUDAX_CATCH(...)( //
      { //
        set_error(static_cast<Rcvr&&>(_rcvr), ::std::current_exception()); //
      })) //
}

_CCCL_HOST_DEVICE inline void run_loop::run()
{
  for (_task* _task; (_task = _pop_front()) != &_head;)
  {
    _task->_execute();
  }
}

_CCCL_HOST_DEVICE inline void run_loop::finish()
{
  ::std::unique_lock _lock{_mutex};
  _stop = true;
  _cv.notify_all();
}

_CCCL_HOST_DEVICE inline void run_loop::_push_back(_task* _task)
{
  ::std::unique_lock _lock{_mutex};
  _task->_next = &_head;
  _head._tail = _head._tail->_next = _task;
  _cv.notify_one();
}

_CCCL_HOST_DEVICE inline auto run_loop::_pop_front() -> _task*
{
  ::std::unique_lock _lock{_mutex};
  _cv.wait(_lock, [this] {
    return _head._next != &_head || _stop;
  });
  if (_head._tail == _head._next)
  {
    _head._tail = &_head;
  }
  return __async::_exchange(_head._next, _head._next->_next);
}
} // namespace cuda::experimental::__async

#  include "epilogue.cuh"

#endif // !defined(__CUDA_ARCH__)
