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
C2H_TEST("simple task_scheduler test", "[scheduler][task_scheduler]")
{
  ex::task_scheduler sched{dummy_scheduler{}};
  STATIC_CHECK(ex::scheduler<decltype(sched)>);
  auto sndr = sched.schedule();
  STATIC_CHECK(ex::sender<decltype(sndr)>);
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{});
  ex::start(op);
  // The receiver checks that it's called
}

C2H_TEST("task_scheduler starts work on the correct execution context", "[scheduler][task_scheduler]")
{
  ex::thread_context ctx;
  ex::task_scheduler sched{ctx.get_scheduler()};
  auto sndr  = ex::starts_on(sched, ex::just() | ex::then([] {
                                     return ::std::this_thread::get_id();
                                   }));
  auto [tid] = ex::sync_wait(cuda::std::move(sndr)).value();
  CHECK(tid == ctx.get_id());
}

#if !_CCCL_HOST_COMPILATION()
static __device__ bool g_called = false;
#else
static bool g_called = false;
#endif

template <class Sndr>
struct protect : private Sndr
{
  using sender_concept = ex::sender_t;
  _CCCL_API explicit protect(Sndr sndr)
      : Sndr{cuda::std::move(sndr)}
  {}
  using Sndr::connect;
  using Sndr::get_completion_signatures;
  using Sndr::get_env;
};

struct test_domain
{
  _CCCL_TEMPLATE(class Sndr, class Env)
  _CCCL_REQUIRES(ex::sender_for<Sndr, ex::bulk_chunked_t>)
  _CCCL_API auto transform_sender(ex::set_value_t, Sndr sndr, const Env&) const
  {
    return ex::then(protect{cuda::std::move(sndr)}, []() noexcept {
      g_called = true;
    });
  }
};

C2H_TEST("bulk_unchunked dispatches correctly through task_scheduler", "[scheduler][task_scheduler]")
{
  ex::task_scheduler sched{dummy_scheduler<test_domain>{}};
  auto sndr  = ex::on(sched, ex::just(-1) | ex::bulk_chunked(ex::par_unseq, 100, [](int, int, int&) {}));
  g_called   = false;
  auto [val] = ex::sync_wait(cuda::std::move(sndr)).value();
  CHECK(val == -1);
  CHECK(g_called);
}

C2H_TEST("bulk dispatches correctly through task_scheduler", "[scheduler][task_scheduler]")
{
  ex::task_scheduler sched{dummy_scheduler<test_domain>{}};
  auto sndr  = ex::on(sched, ex::just(-1) | ex::bulk(ex::par_unseq, 100, [](int, int&) {}));
  g_called   = false;
  auto [val] = ex::sync_wait(cuda::std::move(sndr)).value();
  CHECK(val == -1);
  CHECK(g_called);
}
} // namespace
