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
//! Scheduler that returns a sender that always completes inline
//! (successfully).
struct inline_scheduler
{
private:
  struct _env_t
  {
    _CCCL_HOST_DEVICE auto query(cudax_async::get_completion_scheduler_t<cudax_async::set_value_t>) const noexcept
    {
      return inline_scheduler{};
    }
  };

  template <class Rcvr>
  struct _opstate_t : cudax::__immovable
  {
    using operation_state_concept = cudax_async::operation_state_t;

    Rcvr _rcvr;

    _CCCL_HOST_DEVICE void start() noexcept
    {
      cudax_async::set_value(static_cast<Rcvr&&>(_rcvr));
    }
  };

public:
  using scheduler_concept = cudax_async::scheduler_t;

  struct _sndr_t
  {
    using sender_concept = cudax_async::sender_t;

    template <class Self, class... Env>
    _CCCL_HOST_DEVICE static constexpr auto get_completion_signatures()
    {
      return cudax_async::completion_signatures<cudax_async::set_value_t()>();
    }

    template <class Rcvr>
    _CCCL_HOST_DEVICE _opstate_t<Rcvr> connect(Rcvr rcvr) const
    {
      return {{}, static_cast<Rcvr&&>(rcvr)};
    }

    _CCCL_HOST_DEVICE _env_t get_env() const noexcept
    {
      return {};
    }
  };

  inline_scheduler() = default;

  _CCCL_HOST_DEVICE _sndr_t schedule() const noexcept
  {
    return {};
  }

  _CCCL_HOST_DEVICE friend bool operator==(inline_scheduler, inline_scheduler) noexcept
  {
    return true;
  }

  _CCCL_HOST_DEVICE friend bool operator!=(inline_scheduler, inline_scheduler) noexcept
  {
    return false;
  }
};
} // namespace
