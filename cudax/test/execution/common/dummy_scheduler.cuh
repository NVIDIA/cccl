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

namespace ex = cuda::experimental::execution;

namespace
{
namespace _dummy
{
template <class Domain>
struct _attrs_t
{
  _CCCL_HOST_DEVICE constexpr auto query(ex::get_completion_scheduler_t<ex::set_value_t>) const noexcept;

  _CCCL_HOST_DEVICE constexpr auto query(ex::get_completion_domain_t<ex::set_value_t>) const noexcept
  {
    return Domain{};
  }
};

template <class Rcvr>
struct _opstate_t : cuda::__immovable
{
  using operation_state_concept = ex::operation_state_t;

  _CCCL_HOST_DEVICE constexpr void start() noexcept
  {
    ex::set_value(static_cast<Rcvr&&>(_rcvr));
  }

  Rcvr _rcvr;
};

template <class Domain>
struct _sndr_t
{
  using sender_concept = ex::sender_t;

  template <class Self>
  _CCCL_HOST_DEVICE static constexpr auto get_completion_signatures() noexcept
  {
    return ex::completion_signatures<ex::set_value_t()>();
  }

  template <class Rcvr>
  _CCCL_HOST_DEVICE constexpr auto connect(Rcvr rcvr) const noexcept -> _opstate_t<Rcvr>
  {
    return {{}, static_cast<Rcvr&&>(rcvr)};
  }

  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto get_env() const noexcept
  {
    return _attrs_t<Domain>{};
  }
};
} // namespace _dummy

//! Scheduler that returns a sender that always completes inline (successfully).
template <class Domain = ex::default_domain>
struct dummy_scheduler : _dummy::_attrs_t<Domain>
{
  using scheduler_concept = ex::scheduler_t;

  _CCCL_HOST_DEVICE static constexpr auto schedule() noexcept -> _dummy::_sndr_t<Domain>
  {
    return {};
  }

  _CCCL_HOST_DEVICE friend constexpr bool operator==(dummy_scheduler, dummy_scheduler) noexcept
  {
    return true;
  }

  _CCCL_HOST_DEVICE friend constexpr bool operator!=(dummy_scheduler, dummy_scheduler) noexcept
  {
    return false;
  }
};

namespace _dummy
{
template <class Domain>
_CCCL_HOST_DEVICE constexpr auto _attrs_t<Domain>::query(ex::get_completion_scheduler_t<ex::set_value_t>) const noexcept
{
  return dummy_scheduler<Domain>{};
}
} // namespace _dummy

} // namespace
