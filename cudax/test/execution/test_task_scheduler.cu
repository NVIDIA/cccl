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

#include "common/checked_receiver.cuh" // IWYU pragma: keep
#include "common/dummy_scheduler.cuh" // IWYU pragma: keep
#include "common/error_scheduler.cuh" // IWYU pragma: keep
#include "common/stopped_scheduler.cuh" // IWYU pragma: keep
#include "common/utility.cuh" // IWYU pragma: keep

namespace ex = cuda::experimental::execution;

namespace
{
C2H_TEST("task_scheduler default constructor", "[scheduler][task_scheduler]")
{
  ex::task_scheduler sched;
  STATIC_CHECK(ex::scheduler<decltype(sched)>);
  CHECK(sched.has_value() == false);
}

C2H_TEST("task_scheduler", "[scheduler][task_scheduler]")
{
  ex::task_scheduler sched = dummy_scheduler{};
  CHECK(sched.has_value() == true);
  auto sndr = sched.schedule();
  STATIC_CHECK(ex::sender<decltype(sndr)>);
  CHECK(sndr.has_value() == true);
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{});
  ex::start(op);
  // The receiver checks that it's called
}

C2H_TEST("task_scheduler", "[scheduler][task_scheduler]")
{
  ex::task_scheduler sched = dummy_scheduler{};
  CHECK(sched.has_value() == true);
  auto sndr  = sched.schedule() | ex::bulk(ex::par_unseq, 100, [](int) {});
  using Sndr = decltype(sndr);
  using Env  = ex::env<>;
  auto op    = ex::connect(cuda::std::move(sndr), checked_value_receiver{});
  ex::start(op);
  // The receiver checks that it's called
}
} // namespace
