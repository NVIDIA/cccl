//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_RUN_LOOP
#define __CUDAX_EXECUTION_RUN_LOOP

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/immovable.h>

#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/atomic_intrusive_queue.cuh>
#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/exception.cuh>
#include <cuda/experimental/__execution/fwd.cuh>
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
    while (!__finishing_.load(::cuda::std::memory_order_acquire))
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
    if (!__finishing_.exchange(true, ::cuda::std::memory_order_acq_rel))
    {
      // push an empty work item to the queue to wake up the consuming thread
      // and let it finish:
      __queue_.push(&__noop_task);
    }
  }

  _CUDAX_SEMI_PRIVATE :
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __task : __immovable
  {
    using __execute_fn_t _CCCL_NODEBUG_ALIAS = void(__task*) noexcept;

    _CCCL_HIDE_FROM_ABI __task() = default;
    _CCCL_NODEBUG_API explicit __task(__execute_fn_t* __execute_fn) noexcept
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
    _Rcvr __rcvr_;

    _CCCL_API static void __execute_impl(__task* __p) noexcept
    {
      static_assert(noexcept(get_stop_token(declval<env_of_t<_Rcvr>>()).stop_requested()));
      auto& __rcvr = static_cast<__opstate_t*>(__p)->__rcvr_;

      if (get_stop_token(get_env(__rcvr)).stop_requested())
      {
        set_stopped(static_cast<_Rcvr&&>(__rcvr));
      }
      else
      {
        set_value(static_cast<_Rcvr&&>(__rcvr));
      }
    }

    _CCCL_API constexpr explicit __opstate_t(__atomic_intrusive_queue<&__task::__next_>* __queue, _Rcvr __rcvr)
        : __task{&__execute_impl}
        , __queue_{__queue}
        , __rcvr_{static_cast<_Rcvr&&>(__rcvr)}
    {}

    _CCCL_API constexpr void start() noexcept
    {
      __queue_->push(this);
    }
  };

public:
  class _CCCL_TYPE_VISIBILITY_DEFAULT scheduler
  {
    friend run_loop;

    _CCCL_API constexpr explicit scheduler(run_loop* __loop) noexcept
        : __loop_(__loop)
    {}

    run_loop* __loop_;

  public:
    using scheduler_concept = scheduler_t;

    struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t
    {
      using sender_concept = sender_t;

      template <class _Rcvr>
      [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) const noexcept -> __opstate_t<_Rcvr>
      {
        return __opstate_t<_Rcvr>{&__loop_->__queue_, static_cast<_Rcvr&&>(__rcvr)};
      }

      template <class _Self>
      [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures() noexcept
      {
        return completion_signatures<set_value_t(), set_stopped_t()>{};
      }

    private:
      friend scheduler;

      struct _CCCL_TYPE_VISIBILITY_DEFAULT __attrs_t
      {
        run_loop* __loop_;

        [[nodiscard]] _CCCL_API constexpr auto query(get_completion_scheduler_t<set_value_t>) const noexcept
          -> scheduler
        {
          return __loop_->get_scheduler();
        }

        [[nodiscard]] _CCCL_API constexpr auto query(get_completion_scheduler_t<set_stopped_t>) const noexcept
          -> scheduler
        {
          return __loop_->get_scheduler();
        }

        [[nodiscard]] _CCCL_API constexpr auto query(get_completion_behavior_t) const noexcept
        {
          return completion_behavior::asynchronous;
        }
      };

    public:
      _CCCL_API constexpr auto get_env() const noexcept -> __attrs_t
      {
        return __attrs_t{__loop_};
      }

    private:
      friend class scheduler;
      _CCCL_API constexpr explicit __sndr_t(run_loop* __loop) noexcept
          : __loop_(__loop)
      {}

      run_loop* const __loop_;
    };

    [[nodiscard]] _CCCL_API constexpr auto schedule() const noexcept -> __sndr_t
    {
      return __sndr_t{__loop_};
    }

    [[nodiscard]] _CCCL_API constexpr auto query(get_forward_progress_guarantee_t) const noexcept
      -> forward_progress_guarantee
    {
      return forward_progress_guarantee::parallel;
    }

    [[nodiscard]] _CCCL_API constexpr auto query(get_completion_scheduler_t<set_value_t>) const noexcept -> scheduler
    {
      return *this;
    }

    [[nodiscard]] _CCCL_API constexpr auto query(get_completion_scheduler_t<set_stopped_t>) const noexcept -> scheduler
    {
      return *this;
    }

    [[nodiscard]] _CCCL_API constexpr auto query(get_completion_behavior_t) const noexcept
    {
      return completion_behavior::asynchronous;
    }

    [[nodiscard]] _CCCL_API friend constexpr bool operator==(const scheduler& __a, const scheduler& __b) noexcept
    {
      return __a.__loop_ == __b.__loop_;
    }

    [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const scheduler& __a, const scheduler& __b) noexcept
    {
      return __a.__loop_ != __b.__loop_;
    }
  };

  [[nodiscard]] _CCCL_API constexpr auto get_scheduler() noexcept -> scheduler
  {
    return scheduler{this};
  }

private:
  // Returns true if any tasks were executed.
  _CCCL_API bool __execute_all() noexcept
  {
    // Dequeue all tasks at once. This returns an __intrusive_queue.
    auto __queue = __queue_.pop_all();

    // Execute all the tasks in the queue.
    auto __it = __queue.begin();
    if (__it == __queue.end())
    {
      return false; // No tasks to execute.
    }

    do
    {
      // Take care to increment the iterator before executing the task,
      // because __execute() may invalidate the current node.
      auto __prev = __it++;
      (*__prev)->__execute();
    } while (__it != __queue.end());

    __queue.clear();
    return true;
  }

  _CCCL_API static void __noop_(__task*) noexcept {}

  ::cuda::std::atomic<bool> __finishing_{false};
  __atomic_intrusive_queue<&__task::__next_> __queue_{};
  __task __noop_task{&__noop_};
};

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_RUN_LOOP
