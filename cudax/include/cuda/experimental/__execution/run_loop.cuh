//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda/experimental/__execution/atomic_intrusive_queue.cuh>
#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/exception.cuh>
#include <cuda/experimental/__execution/queries.cuh>
#include <cuda/experimental/__execution/utility.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
class _CCCL_TYPE_VISIBILITY_DEFAULT run_loop : __immovable
{
public:
  _CCCL_HIDE_FROM_ABI run_loop() = default;

  _CCCL_API void run() noexcept
  {
    // execute work items until the __finishing_ flag is set:
    while (!__finishing_.load(_CUDA_VSTD::memory_order_acquire))
    {
      __queue_.wait_for_item();
      __execute_all();
    }
    // drain the queue, taking care to execute any tasks that get added while
    // executing the remaining tasks:
    while (__execute_all())
      ;
  }

  _CCCL_API void finish() noexcept
  {
    if (!__finishing_.exchange(true, _CUDA_VSTD::memory_order_acq_rel))
    {
      // push an empty work item to the queue to wake up any waiting threads
      __queue_.push(&__finish_task);
    }
  }

private:
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __task : __immovable
  {
    using __execute_fn_t _CCCL_NODEBUG_ALIAS = void(__task*) noexcept;

    _CCCL_HIDE_FROM_ABI __task() = default;
    _CCCL_TRIVIAL_API explicit __task(__execute_fn_t* __execute_fn) noexcept
        : __execute_fn_(__execute_fn)
    {}

    _CCCL_API void __execute() noexcept
    {
      (*__execute_fn_)(this);
    }

    __execute_fn_t* __execute_fn_ = nullptr;
    __task* __next_               = nullptr;
  };

  template <class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t : __task
  {
    __atomic_intrusive_queue<&__task::__next_>* __queue_;
    _CCCL_NO_UNIQUE_ADDRESS _Rcvr __rcvr_;

    _CCCL_API static void __execute_impl(__task* __p) noexcept
    {
      auto& __rcvr = static_cast<__opstate_t*>(__p)->__rcvr_;
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
        _CUDAX_CATCH(...) //
        ({ //
          set_error(static_cast<_Rcvr&&>(__rcvr), ::std::current_exception());
        }) //
      )
    }

    _CCCL_API __opstate_t(__atomic_intrusive_queue<&__task::__next_>* __queue, _Rcvr __rcvr)
        : __task{&__execute_impl}
        , __queue_{__queue}
        , __rcvr_{static_cast<_Rcvr&&>(__rcvr)}
    {}

    _CCCL_API void start() & noexcept
    {
      __queue_->push(this);
    }
  };

public:
  class _CCCL_TYPE_VISIBILITY_DEFAULT scheduler
  {
    struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t
    {
      using sender_concept _CCCL_NODEBUG_ALIAS = sender_t;

      template <class _Rcvr>
      _CCCL_API auto connect(_Rcvr __rcvr) const noexcept -> __opstate_t<_Rcvr>
      {
        return {&__loop_->__queue_, static_cast<_Rcvr&&>(__rcvr)};
      }

      template <class _Self>
      _CCCL_API static constexpr auto get_completion_signatures() noexcept
      {
#if _CCCL_HAS_EXCEPTIONS()
        return completion_signatures<set_value_t(), set_error_t(::std::exception_ptr), set_stopped_t()>();
#else // ^^^ _CCCL_HAS_EXCEPTIONS() ^^^ / vvv !_CCCL_HAS_EXCEPTIONS() vvv
        return completion_signatures<set_value_t(), set_stopped_t()>();
#endif // !_CCCL_HAS_EXCEPTIONS()
      }

    private:
      friend scheduler;

      struct _CCCL_TYPE_VISIBILITY_DEFAULT __env_t
      {
        run_loop* __loop_;

        _CCCL_API auto query(get_completion_scheduler_t<set_value_t>) const noexcept -> scheduler
        {
          return __loop_->get_scheduler();
        }

        _CCCL_API auto query(get_completion_scheduler_t<set_stopped_t>) const noexcept -> scheduler
        {
          return __loop_->get_scheduler();
        }
      };

      _CCCL_API auto get_env() const noexcept -> __env_t
      {
        return __env_t{__loop_};
      }

    private:
      friend class scheduler;
      _CCCL_API explicit __sndr_t(run_loop* __loop) noexcept
          : __loop_(__loop)
      {}

      run_loop* const __loop_;
    };

    friend run_loop;

    _CCCL_API explicit scheduler(run_loop* __loop) noexcept
        : __loop_(__loop)
    {}

    _CCCL_API auto query(get_forward_progress_guarantee_t) const noexcept -> forward_progress_guarantee
    {
      return forward_progress_guarantee::parallel;
    }

    run_loop* __loop_;

  public:
    using scheduler_concept _CCCL_NODEBUG_ALIAS = scheduler_t;

    [[nodiscard]] _CCCL_API auto schedule() const noexcept -> __sndr_t
    {
      return __sndr_t{__loop_};
    }

    [[nodiscard]] _CCCL_API friend bool operator==(const scheduler& __a, const scheduler& __b) noexcept
    {
      return __a.__loop_ == __b.__loop_;
    }

    [[nodiscard]] _CCCL_API friend bool operator!=(const scheduler& __a, const scheduler& __b) noexcept
    {
      return __a.__loop_ != __b.__loop_;
    }
  };

  [[nodiscard]] _CCCL_API auto get_scheduler() noexcept -> scheduler
  {
    return scheduler{this};
  }

private:
  // Returns true if any tasks were executed.
  _CCCL_API bool __execute_all() noexcept
  {
    // Wait until the queue has tasks to execute and then dequeue all of them.
    auto __queue = __queue_.pop_all();
    if (__queue.empty())
    {
      return false;
    }
    // Execute all the tasks in the queue.
    for (auto __task : __queue)
    {
      __task->__execute();
    }
    __queue.clear();
    return true;
  }

  _CCCL_API static void __noop_(__task*) noexcept {}

  _CUDA_VSTD::atomic<bool> __finishing_{false};
  __atomic_intrusive_queue<&__task::__next_> __queue_{};
  __task __finish_task{&__noop_};
};

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_ASYNC_DETAIL_RUN_LOOP
