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

#include <cuda/__utility/immovable.h>

#include <cuda/experimental/execution.cuh>

// IWYU pragma: begin_keep
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
// IWYU pragma: end_keep

#include "testing.cuh" // IWYU pragma: keep

namespace ex = cuda::experimental::execution;

#if _CCCL_HOST_COMPILATION()

namespace
{
namespace _impulse
{
struct _attrs_t
{
  constexpr auto query(ex::get_completion_behavior_t) const noexcept
  {
    return ex::completion_behavior::asynchronous;
  }
};
} // namespace _impulse

//! Scheduler that will send impulses on user's request.
//! One can obtain senders from this, connect them to receivers and start the operation states.
//! Until the scheduler is told to start the next operation, the actions in the operation states are
//! not executed. This is similar to a task scheduler, but it's single threaded. It has basic
//! thread-safety to allow it to be run with `sync_wait` (which makes us not control when the
//! operation_state object is created and started).
struct impulse_scheduler : _impulse::_attrs_t
{
private:
  //! Command type that can store the action of firing up a sender
  using _cmd_t     = std::function<void()>;
  using _cmd_vec_t = std::vector<_cmd_t>;

  struct _data_t : std::enable_shared_from_this<_data_t>
  {
    explicit _data_t(int id)
        : id_(id)
    {}

    int id_;
    std::mutex mutex_{};
    std::condition_variable cv_{};
    std::vector<std::function<void()>> all_commands_{};
  };

  //! That data_t shared between the operation state and the actual scheduler
  //! Shared pointer to allow the scheduler to be copied (not the best semantics, but it will do)
  std::shared_ptr<_data_t> _data_{};

  template <class Rcvr>
  struct _opstate_t : cuda::__immovable
  {
    using operation_state_concept = ex::operation_state_t;

    _data_t* _data_;
    Rcvr _rcvr_;

    explicit _opstate_t(_data_t* data, Rcvr&& rcvr)
        : _data_(data)
        , _rcvr_(static_cast<Rcvr&&>(rcvr))
    {}

    void start() noexcept
    {
      // Enqueue another command to the list of all commands
      // The scheduler will start this, whenever start_next() is called
      std::unique_lock lock{_data_->mutex_};
      _data_->all_commands_.emplace_back([this]() {
        if (ex::get_stop_token(ex::get_env(_rcvr_)).stop_requested())
        {
          ex::set_stopped(static_cast<Rcvr&&>(_rcvr_));
        }
        else
        {
          ex::set_value(static_cast<Rcvr&&>(_rcvr_));
        }
      });
      _data_->cv_.notify_all();
    }
  };

  struct _sndr_t
  {
    using sender_concept = ex::sender_t;

    _data_t* _data_;

    template <class Self>
    _CCCL_HOST_DEVICE static constexpr auto get_completion_signatures()
    {
      return ex::completion_signatures<ex::set_value_t(), ex::set_stopped_t()>();
    }

    template <class Rcvr>
    auto connect(Rcvr rcvr) -> _opstate_t<Rcvr>
    {
      return _opstate_t<Rcvr>{_data_, static_cast<Rcvr&&>(rcvr)};
    }

    auto get_env() const noexcept
    {
      return _impulse::_attrs_t{};
    }
  };

  explicit impulse_scheduler(_data_t* data)
      : _data_(data->shared_from_this())
  {}

public:
  using scheduler_concept = ex::scheduler_t;

  impulse_scheduler()
      : _data_(std::make_shared<_data_t>(0))
  {}

  explicit impulse_scheduler(int id)
      : _data_(std::make_shared<_data_t>(id))
  {}

  ~impulse_scheduler() = default;

  //! Actually start the command from the last started operation_state
  //! Returns immediately if no command registered (i.e., no operation state started)
  bool try_start_next()
  {
    // Wait for a command that we can execute
    std::unique_lock lock{_data_->mutex_};

    // If there are no commands in the queue, return false
    if (_data_->all_commands_.empty())
    {
      return false;
    }

    // Pop one command from the queue
    auto cmd = std::move(_data_->all_commands_.front());
    _data_->all_commands_.erase(_data_->all_commands_.begin());
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
    std::unique_lock lock{_data_->mutex_};
    while (_data_->all_commands_.empty())
    {
      _data_->cv_.wait(lock);
    }

    // Pop one command from the queue
    auto cmd = std::move(_data_->all_commands_.front());
    _data_->all_commands_.erase(_data_->all_commands_.begin());
    // Exit the lock before executing the command
    lock.unlock();
    // Execute the command, i.e., send an impulse to the connected sender
    cmd();
  }

  _sndr_t schedule() const noexcept
  {
    return _sndr_t{_data_.get()};
  }

  friend bool operator==(const impulse_scheduler& a, const impulse_scheduler& b) noexcept
  {
    return a._data_ == b._data_;
  }

  friend bool operator!=(const impulse_scheduler& a, const impulse_scheduler& b) noexcept
  {
    return a._data_ != b._data_;
  }
};
} // namespace

#endif // _CCCL_HOST_COMPILATION()
