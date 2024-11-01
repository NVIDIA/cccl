//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_RUN_LOOP
#define __CUDAX_ASYNC_DETAIL_RUN_LOOP

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__detail/config.cuh>

// libcu++ does not have <cuda/std/mutex> or <cuda/std/condition_variable>
#if !defined(__CUDA_ARCH__)

#  include <cuda/experimental/__async/completion_signatures.cuh>
#  include <cuda/experimental/__async/env.cuh>
#  include <cuda/experimental/__async/exception.cuh>
#  include <cuda/experimental/__async/queries.cuh>
#  include <cuda/experimental/__async/utility.cuh>

#  include <condition_variable>
#  include <mutex>

#  include <cuda/experimental/__async/prologue.cuh>

namespace cuda::experimental::__async
{
class run_loop;

struct __task : __immovable
{
  using __execute_fn_t = void(__task*) noexcept;

  __task() = default;

  _CUDAX_API explicit __task(__task* __next, __task* __tail) noexcept
      : __next_{__next}
      , __tail_{__tail}
  {}

  _CUDAX_API explicit __task(__task* __next, __execute_fn_t* __execute) noexcept
      : __next_{__next}
      , __execute_fn_{__execute}
  {}

  __task* __next_ = this;

  union
  {
    __task* __tail_ = nullptr;
    __execute_fn_t* __execute_fn_;
  };

  _CUDAX_API void __execute() noexcept
  {
    (*__execute_fn_)(this);
  }
};

template <class _Rcvr>
struct __operation : __task
{
  run_loop* __loop_;
  _CCCL_NO_UNIQUE_ADDRESS _Rcvr __rcvr_;

  using completion_signatures = //
    __async::completion_signatures<set_value_t(), set_error_t(::std::exception_ptr), set_stopped_t()>;

  _CUDAX_API static void __execute_impl(__task* __p) noexcept
  {
    auto& __rcvr = static_cast<__operation*>(__p)->__rcvr_;
    _CUDAX_TRY( //
      ({ //
        if (get_stop_token(get_env(__rcvr)).stop_requested())
        {
          set_stopped(static_cast<_Rcvr&&>(__rcvr));
        }
        else
        {
          set_value(static_cast<_Rcvr&&>(__rcvr));
        }
      }),
      _CUDAX_CATCH(...)( //
        { //
          set_error(static_cast<_Rcvr&&>(__rcvr), ::std::current_exception());
        }))
  }

  _CUDAX_API explicit __operation(__task* __tail_) noexcept
      : __task{this, __tail_}
  {}

  _CUDAX_API __operation(__task* __next_, run_loop* __loop, _Rcvr __rcvr)
      : __task{__next_, &__execute_impl}
      , __loop_{__loop}
      , __rcvr_{static_cast<_Rcvr&&>(__rcvr)}
  {}

  _CUDAX_API void start() & noexcept;
};

class run_loop
{
  template <class... _Ts>
  using __completion_signatures = completion_signatures<_Ts...>;

  template <class>
  friend struct __operation;

public:
  run_loop() noexcept
  {
    __head.__next_ = __head.__tail_ = &__head;
  }

  class __scheduler
  {
    struct __schedule_task
    {
      using __t            = __schedule_task;
      using __id           = __schedule_task;
      using sender_concept = sender_t;

      template <class _Rcvr>
      _CUDAX_API auto connect(_Rcvr __rcvr) const noexcept -> __operation<_Rcvr>
      {
        return {&__loop_->__head, __loop_, static_cast<_Rcvr&&>(__rcvr)};
      }

    private:
      friend __scheduler;

      struct __env
      {
        run_loop* __loop_;

        template <class _Tag>
        _CUDAX_API auto query(get_completion_scheduler_t<_Tag>) const noexcept -> __scheduler
        {
          return __loop_->get_scheduler();
        }
      };

      _CUDAX_API auto get_env() const noexcept -> __env
      {
        return __env{__loop_};
      }

      _CUDAX_API explicit __schedule_task(run_loop* __loop) noexcept
          : __loop_(__loop)
      {}

      run_loop* const __loop_;
    };

    friend run_loop;

    _CUDAX_API explicit __scheduler(run_loop* __loop) noexcept
        : __loop_(__loop)
    {}

    _CUDAX_API auto query(get_forward_progress_guarantee_t) const noexcept -> forward_progress_guarantee
    {
      return forward_progress_guarantee::parallel;
    }

    run_loop* __loop_;

  public:
    using scheduler_concept = scheduler_t;

    [[nodiscard]] _CUDAX_API auto schedule() const noexcept -> __schedule_task
    {
      return __schedule_task{__loop_};
    }

    _CUDAX_API friend bool operator==(const __scheduler& __a, const __scheduler& __b) noexcept
    {
      return __a.__loop_ == __b.__loop_;
    }

    _CUDAX_API friend bool operator!=(const __scheduler& __a, const __scheduler& __b) noexcept
    {
      return __a.__loop_ != __b.__loop_;
    }
  };

  _CUDAX_API auto get_scheduler() noexcept -> __scheduler
  {
    return __scheduler{this};
  }

  _CUDAX_API void run();

  _CUDAX_API void finish();

private:
  _CUDAX_API void __push_back(__task* __tsk);
  _CUDAX_API auto __pop_front() -> __task*;

  ::std::mutex __mutex{};
  ::std::condition_variable __cv{};
  __task __head{};
  bool __stop = false;
};

template <class _Rcvr>
_CUDAX_API inline void __operation<_Rcvr>::start() & noexcept {
  _CUDAX_TRY( //
    ({ //
      __loop_->__push_back(this); //
    }), //
    _CUDAX_CATCH(...)( //
      { //
        set_error(static_cast<_Rcvr&&>(__rcvr_), ::std::current_exception()); //
      })) //
}

_CUDAX_API inline void run_loop::run()
{
  for (__task* __tsk = __pop_front(); __tsk != &__head; __tsk = __pop_front())
  {
    __tsk->__execute();
  }
}

_CUDAX_API inline void run_loop::finish()
{
  ::std::unique_lock __lock{__mutex};
  __stop = true;
  __cv.notify_all();
}

_CUDAX_API inline void run_loop::__push_back(__task* __tsk)
{
  ::std::unique_lock __lock{__mutex};
  __tsk->__next_ = &__head;
  __head.__tail_ = __head.__tail_->__next_ = __tsk;
  __cv.notify_one();
}

_CUDAX_API inline auto run_loop::__pop_front() -> __task*
{
  ::std::unique_lock __lock{__mutex};
  __cv.wait(__lock, [this] {
    return __head.__next_ != &__head || __stop;
  });
  if (__head.__tail_ == __head.__next_)
  {
    __head.__tail_ = &__head;
  }
  return __async::__exchange(__head.__next_, __head.__next_->__next_);
}
} // namespace cuda::experimental::__async

#  include <cuda/experimental/__async/epilogue.cuh>

#endif // !defined(__CUDA_ARCH__)

#endif
