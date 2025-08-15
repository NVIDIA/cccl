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

namespace ex = cuda::experimental::execution;

C2H_TEST("simple test of just sender factory", "[just]")
{
  auto sndr  = ex::just(42);
  using Sndr = decltype(sndr);
  static_assert(ex::__is_sender<Sndr>, "just(42) must be a sender");
  static_assert(ex::get_completion_behavior<Sndr>() == ex::completion_behavior::inline_completion);
  auto op = ex::connect(sndr, checked_value_receiver{42});
  ex::start(op);
}

C2H_TEST("simple test of just_error sender factory", "[just]")
{
  auto ec    = ::std::errc::invalid_argument;
  auto sndr  = ex::just_error(ec);
  using Sndr = decltype(sndr);
  static_assert(ex::__is_sender<Sndr>, "just_error(...) must be a sender");
  static_assert(ex::get_completion_behavior<Sndr>() == ex::completion_behavior::inline_completion);
  auto op = ex::connect(sndr, checked_error_receiver{ec});
  ex::start(op);
}

C2H_TEST("simple test of just_stopped sender factory", "[just]")
{
  auto sndr  = ex::just_stopped();
  using Sndr = decltype(sndr);
  static_assert(ex::__is_sender<Sndr>, "just_stopped() must be a sender");
  static_assert(ex::get_completion_behavior<Sndr>() == ex::completion_behavior::inline_completion);
  auto op = ex::connect(sndr, checked_stopped_receiver{});
  ex::start(op);
}
