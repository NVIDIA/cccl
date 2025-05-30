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

#include "testing.cuh" // IWYU pragma: keep

namespace
{
//! Scheduler that returns a sender that always completes with stopped.
struct stopped_scheduler
{
private:
  struct env_t
  {
    _CCCL_HOST_DEVICE auto query(cudax_async::get_completion_scheduler_t<cudax_async::set_value_t>) const noexcept
    {
      return stopped_scheduler{};
    }

    _CCCL_HOST_DEVICE auto query(cudax_async::get_completion_scheduler_t<cudax_async::set_stopped_t>) const noexcept
    {
      return stopped_scheduler{};
    }
  };

  template <class Rcvr>
  struct opstate_t : cudax::__immovable
  {
    using operation_state_concept = cudax_async::operation_state_t;

    Rcvr _rcvr;

    _CCCL_HOST_DEVICE void start() noexcept
    {
      cudax_async::set_stopped(static_cast<Rcvr&&>(_rcvr));
    }
  };

  struct sndr_t
  {
    using sender_concept = cudax_async::sender_t;

    template <class Self, class... Env>
    _CCCL_HOST_DEVICE static constexpr auto get_completion_signatures()
    {
      return cudax_async::completion_signatures<cudax_async::set_value_t(), cudax_async::set_stopped_t()>();
    }

    template <class Rcvr>
    _CCCL_HOST_DEVICE opstate_t<Rcvr> connect(Rcvr rcvr) const
    {
      return {{}, static_cast<Rcvr&&>(rcvr)};
    }

    _CCCL_HOST_DEVICE env_t get_env() const noexcept
    {
      return {};
    }
  };

  _CCCL_HOST_DEVICE friend bool operator==(stopped_scheduler, stopped_scheduler) noexcept
  {
    return true;
  }

  _CCCL_HOST_DEVICE friend bool operator!=(stopped_scheduler, stopped_scheduler) noexcept
  {
    return false;
  }

public:
  using scheduler_concept = cudax_async::scheduler_t;

  stopped_scheduler() = default;

  _CCCL_HOST_DEVICE sndr_t schedule() const noexcept
  {
    return {};
  }
};
} // namespace
