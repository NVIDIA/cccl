//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/execution.cuh>

#include <system_error>

#include "common/checked_receiver.cuh"
#include "common/inline_scheduler.cuh"

namespace ex = cuda::experimental::execution;

C2H_TEST("simple test of just sender factory", "[just]")
{
  auto sndr  = ex::just(42);
  using Sndr = decltype(sndr);
  STATIC_REQUIRE(ex::__is_sender<Sndr>);
  STATIC_REQUIRE(ex::get_completion_behavior<Sndr>() == ex::completion_behavior::inline_completion);
  CHECK(
    ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(sndr), ex::prop{ex::get_scheduler, inline_scheduler{}})
    == inline_scheduler{});
  auto op = ex::connect(sndr, checked_value_receiver{42});
  ex::start(op);
}

C2H_TEST("simple test of just_error sender factory", "[just]")
{
  auto ec    = ::std::errc::invalid_argument;
  auto sndr  = ex::just_error(ec);
  using Sndr = decltype(sndr);
  STATIC_REQUIRE(ex::__is_sender<Sndr>);
  STATIC_REQUIRE(ex::get_completion_behavior<Sndr>() == ex::completion_behavior::inline_completion);
  CHECK(
    ex::get_completion_scheduler<ex::set_error_t>(ex::get_env(sndr), ex::prop{ex::get_scheduler, inline_scheduler{}})
    == inline_scheduler{});
  auto op = ex::connect(sndr, checked_error_receiver{ec});
  ex::start(op);
}

C2H_TEST("simple test of just_stopped sender factory", "[just]")
{
  auto sndr  = ex::just_stopped();
  using Sndr = decltype(sndr);
  STATIC_REQUIRE(ex::__is_sender<Sndr>);
  STATIC_REQUIRE(ex::get_completion_behavior<Sndr>() == ex::completion_behavior::inline_completion);
  CHECK(
    ex::get_completion_scheduler<ex::set_stopped_t>(ex::get_env(sndr), ex::prop{ex::get_scheduler, inline_scheduler{}})
    == inline_scheduler{});
  auto op = ex::connect(sndr, checked_stopped_receiver{});
  ex::start(op);
}
