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

#include "testing.cuh" // IWYU pragma: keep

namespace
{
struct _error_scheduler_attrs_t
{
  template <class _Env>
  _CCCL_HOST_DEVICE auto
  query(cudax_async::get_completion_scheduler_t<cudax_async::set_value_t>, const _Env& env) const noexcept
    -> decltype(cudax_async::get_completion_scheduler<cudax_async::set_value_t>(env, env))
  {
    return cudax_async::get_completion_scheduler<cudax_async::set_value_t>(env, env);
  }

  template <class _Env>
  _CCCL_HOST_DEVICE auto
  query(cudax_async::get_completion_scheduler_t<cudax_async::set_error_t>, const _Env& env) const noexcept
    -> decltype(cudax_async::get_completion_scheduler<cudax_async::set_error_t>(env, env))
  {
    return cudax_async::get_completion_scheduler<cudax_async::set_error_t>(env, env);
  }

  template <class _Env>
  _CCCL_HOST_DEVICE auto
  query(cudax_async::get_completion_domain_t<cudax_async::set_value_t>, const _Env& env) const noexcept
    -> decltype(cudax_async::get_completion_domain<cudax_async::set_value_t>(env, env))
  {
    return cudax_async::get_completion_domain<cudax_async::set_value_t>(env, env);
  }

  template <class _Env>
  _CCCL_HOST_DEVICE auto
  query(cudax_async::get_completion_domain_t<cudax_async::set_error_t>, const _Env& env) const noexcept
    -> decltype(cudax_async::get_completion_domain<cudax_async::set_error_t>(env, env))
  {
    return cudax_async::get_completion_domain<cudax_async::set_error_t>(env, env);
  }

  _CCCL_HOST_DEVICE static constexpr auto query(cudax_async::get_completion_behavior_t) noexcept
  {
    return cudax_async::completion_behavior::inline_completion;
  }
};

//! Scheduler that returns a sender that always completes with error.
template <class Error>
struct error_scheduler : _error_scheduler_attrs_t
{
private:
  template <class Rcvr>
  struct _opstate_t : cuda::__immovable
  {
    using operation_state_concept = cudax_async::operation_state_t;

    Rcvr _rcvr;
    Error _err;

    _CCCL_HOST_DEVICE void start() noexcept
    {
      cudax_async::set_error(static_cast<Rcvr&&>(_rcvr), static_cast<Error&&>(_err));
    }
  };

  struct _sndr_t
  {
    using sender_concept = cudax_async::sender_t;

    template <class Self>
    _CCCL_HOST_DEVICE static constexpr auto get_completion_signatures()
    {
      return cudax_async::completion_signatures< //
        cudax_async::set_value_t(), //
        cudax_async::set_error_t(Error)>();
    }

    template <class Rcvr>
    _CCCL_HOST_DEVICE auto connect(Rcvr rcvr) const -> _opstate_t<Rcvr>
    {
      return {{}, static_cast<Rcvr&&>(rcvr), _err};
    }

    _CCCL_HOST_DEVICE auto get_env() const noexcept -> _error_scheduler_attrs_t
    {
      return {};
    }

    Error _err;
  };

  Error _err{};

public:
  using scheduler_concept = cudax_async::scheduler_t;

  _CCCL_HIDE_FROM_ABI error_scheduler() = default;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE explicit constexpr error_scheduler(Error err)
      : _err(static_cast<Error&&>(err))
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE auto schedule() const noexcept -> _sndr_t
  {
    return {_err};
  }

  _CCCL_HOST_DEVICE friend constexpr bool operator==(const error_scheduler&, const error_scheduler&) noexcept
  {
    return true;
  }

  _CCCL_HOST_DEVICE friend constexpr bool operator!=(const error_scheduler&, const error_scheduler&) noexcept
  {
    return false;
  }
};
} // namespace
