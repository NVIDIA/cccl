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

#include <cuda/experimental/execution.cuh>

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>

#include "testing.cuh" // IWYU pragma: keep

#if _CCCL_HOST_COMPILATION()

namespace
{
//! Scheduler that will send impulses on user's request.
//! One can obtain senders from this, connect them to receivers and start the operation states.
//! Until the scheduler is told to start the next operation, the actions in the operation states are
//! not executed. This is similar to a task scheduler, but it's single threaded. It has basic
//! thread-safety to allow it to be run with `sync_wait` (which makes us not control when the
//! operation_state object is created and started).
struct impulse_scheduler
{
private:
  //! Command type that can store the action of firing up a sender
  using cmd_t     = std::function<void()>;
  using cmd_vec_t = std::vector<cmd_t>;

  struct data_t : std::enable_shared_from_this<data_t>
  {
    explicit data_t(int id)
        : id_(id)
    {}

    int id_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<std::function<void()>> all_commands_;
  };

  //! That data_t shared between the operation state and the actual scheduler
  //! Shared pointer to allow the scheduler to be copied (not the best semantics, but it will do)
  std::shared_ptr<data_t> data_{};

  template <class Rcvr>
  struct opstate_t : cudax_async::__immovable
  {
    using operation_state_concept = cudax_async::operation_state_t;

    data_t* data_;
    Rcvr rcvr_;

    opstate_t(data_t* data, Rcvr&& rcvr)
        : data_(data)
        , rcvr_(static_cast<Rcvr&&>(rcvr))
    {}

    void start() & noexcept
    {
      // Enqueue another command to the list of all commands
      // The scheduler will start this, whenever start_next() is called
      std::unique_lock lock{data_->mutex_};
      data_->all_commands_.emplace_back([this]() {
        if (cudax_async::get_stop_token(cudax_async::get_env(rcvr_)).stop_requested())
        {
          cudax_async::set_stopped(static_cast<Rcvr&&>(rcvr_));
        }
        else
        {
          cudax_async::set_value(static_cast<Rcvr&&>(rcvr_));
        }
      });
      data_->cv_.notify_all();
    }
  };

  struct env_t
  {
    data_t* data_;

    impulse_scheduler query(cudax_async::get_completion_scheduler_t<cudax_async::set_value_t>) const noexcept
    {
      return impulse_scheduler{data_};
    }

    impulse_scheduler query(cudax_async::get_completion_scheduler_t<cudax_async::set_stopped_t>) const noexcept
    {
      return impulse_scheduler{data_};
    }
  };

  struct sndr_t
  {
    using sender_concept = cudax_async::sender_t;

    data_t* data_;

    template <class Self, class... Env>
    _CCCL_HOST_DEVICE static constexpr auto get_completion_signatures()
    {
      return cudax_async::completion_signatures<cudax_async::set_value_t(), cudax_async::set_stopped_t()>();
    }

    template <class Rcvr>
    opstate_t<Rcvr> connect(Rcvr rcvr)
    {
      return {data_, static_cast<Rcvr&&>(rcvr)};
    }

    auto get_env() const noexcept
    {
      return env_t{data_};
    }
  };

  explicit impulse_scheduler(data_t* data)
      : data_(data->shared_from_this())
  {}

public:
  using scheduler_concept = cudax_async::scheduler_t;

  impulse_scheduler()
      : data_(std::make_shared<data_t>(0))
  {}

  explicit impulse_scheduler(int id)
      : data_(std::make_shared<data_t>(id))
  {}

  ~impulse_scheduler() = default;

  //! Actually start the command from the last started operation_state
  //! Returns immediately if no command registered (i.e., no operation state started)
  bool try_start_next()
  {
    // Wait for a command that we can execute
    std::unique_lock lock{data_->mutex_};

    // If there are no commands in the queue, return false
    if (data_->all_commands_.empty())
    {
      return false;
    }

    // Pop one command from the queue
    auto cmd = std::move(data_->all_commands_.front());
    data_->all_commands_.erase(data_->all_commands_.begin());
    // Exit the lock before executing the command
    lock.unlock();
    // Execute the command, i.e., send an impulse to the connected sender
    cmd();
    // Return true to signal that we started a command
    return true;
  }

  //! Actually start the command from the last started operation_state
  //! Blocks if no command registered (i.e., no operation state started)
  void start_next()
  {
    // Wait for a command that we can execute
    std::unique_lock lock{data_->mutex_};
    while (data_->all_commands_.empty())
    {
      data_->cv_.wait(lock);
    }

    // Pop one command from the queue
    auto cmd = std::move(data_->all_commands_.front());
    data_->all_commands_.erase(data_->all_commands_.begin());
    // Exit the lock before executing the command
    lock.unlock();
    // Execute the command, i.e., send an impulse to the connected sender
    cmd();
  }

  sndr_t schedule() const noexcept
  {
    return sndr_t{data_.get()};
  }

  friend bool operator==(impulse_scheduler, impulse_scheduler) noexcept
  {
    return true;
  }

  friend bool operator!=(impulse_scheduler, impulse_scheduler) noexcept
  {
    return false;
  }
};
} // namespace

#endif // _CCCL_HOST_COMPILATION()
