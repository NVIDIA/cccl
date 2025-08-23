//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/execution.cuh>

#include "../common/testing.cuh" // IWYU pragma: keep
#include "common/checked_receiver.cuh"
#include "common/error_scheduler.cuh"
#include "common/inline_scheduler.cuh"
#include "common/stopped_scheduler.cuh"
#include "common/utility.cuh"

#if _CCCL_HOST_COMPILATION()
#  include "common/impulse_scheduler.cuh"
#endif

namespace
{
C2H_TEST("starts_on simple example", "[adaptors][starts_on]")
{
  auto snd = cudax_async::starts_on(inline_scheduler<>{}, cudax_async::just(42));
  auto op  = cudax_async::connect(std::move(snd), checked_value_receiver{42});
  cudax_async::start(op);
  // The receiver checks if we receive the right value
}

C2H_TEST("starts_on can be piped", "[adaptors][starts_on]")
{
  // Use starts_on with the inline scheduler and pipe with then
  auto snd = cudax_async::starts_on(inline_scheduler<>{}, cudax_async::just(42)) //
           | cudax_async::then([](int val) {
               return val * 2;
             });
  auto op = cudax_async::connect(std::move(snd), checked_value_receiver{84});
  cudax_async::start(op);
  // The receiver checks if we receive the transformed value
}

#if _CCCL_HOST_COMPILATION()

C2H_TEST("starts_on with impulse scheduler", "[adaptors][starts_on]")
{
  bool sender_executed = false;

  impulse_scheduler sched;
  auto snd = cudax_async::starts_on(sched, cudax_async::just() | cudax_async::then([&]() {
                                             sender_executed = true;
                                             return 13;
                                           }));

  auto op = cudax_async::connect(std::move(snd), checked_value_receiver{13});
  cudax_async::start(op);

  // At this point, the scheduler should have been started but the sender not yet executed
  CUDAX_REQUIRE(!sender_executed);

  // Tell the scheduler to start executing one task
  sched.start_next();

  // Now the sender should be executed
  CUDAX_REQUIRE(sender_executed);
}

C2H_TEST("starts_on execution order", "[adaptors][starts_on]")
{
  int counter = 0;
  impulse_scheduler sched;

  auto snd = cudax_async::starts_on(sched, cudax_async::just() | cudax_async::then([&]() {
                                             return ++counter;
                                           }));

  auto op = cudax_async::connect(std::move(snd), checked_value_receiver{1});
  cudax_async::start(op);

  // Counter should still be 0 since scheduler hasn't executed yet
  CUDAX_CHECK(counter == 0);

  // Tell the scheduler to start executing
  sched.start_next();

  // Now the sender should have executed and incremented counter
  CUDAX_CHECK(counter == 1);
}

C2H_TEST("starts_on with thread context", "[adaptors][starts_on]")
{
  cudax_async::thread_context thread;
  bool executed = false;

  auto snd = cudax_async::starts_on(thread.get_scheduler(), cudax_async::just() | cudax_async::then([&]() {
                                                              executed = true;
                                                              return 123;
                                                            }));

  auto op = cudax_async::connect(std::move(snd), checked_value_receiver{123});
  cudax_async::start(op);

  thread.join();

  // The work should have been executed on the thread
  CUDAX_REQUIRE(executed);
}

#endif

C2H_TEST("starts_on can be called with rvalue ref scheduler", "[adaptors][starts_on]")
{
  auto snd = cudax_async::starts_on(inline_scheduler<>{}, cudax_async::just(42));
  auto op  = cudax_async::connect(std::move(snd), checked_value_receiver{42});
  cudax_async::start(op);
}

C2H_TEST("starts_on can be called with const ref scheduler", "[adaptors][starts_on]")
{
  const inline_scheduler<> sched;
  auto snd = cudax_async::starts_on(sched, cudax_async::just(42));
  auto op  = cudax_async::connect(std::move(snd), checked_value_receiver{42});
  cudax_async::start(op);
}

C2H_TEST("starts_on can be called with ref scheduler", "[adaptors][starts_on]")
{
  inline_scheduler<> sched;
  auto snd = cudax_async::starts_on(sched, cudax_async::just(42));
  auto op  = cudax_async::connect(std::move(snd), checked_value_receiver{42});
  cudax_async::start(op);
}

C2H_TEST("starts_on forwards scheduler errors", "[adaptors][starts_on]")
{
  auto ec = error_code{std::errc::invalid_argument};
  error_scheduler<error_code> sched{ec};
  auto snd = cudax_async::starts_on(sched, cudax_async::just(42));
  auto op  = cudax_async::connect(std::move(snd), checked_error_receiver{ec});
  cudax_async::start(op);
  // The receiver checks if we receive the error from the scheduler
}

C2H_TEST("starts_on forwards scheduler errors of other types", "[adaptors][starts_on]")
{
  error_scheduler<string> sched{string{"scheduler error"}};
  auto snd = cudax_async::starts_on(sched, cudax_async::just(42));
  auto op  = cudax_async::connect(std::move(snd), checked_error_receiver{string{"scheduler error"}});
  cudax_async::start(op);
}

C2H_TEST("starts_on forwards scheduler stopped signal", "[adaptors][starts_on]")
{
  stopped_scheduler sched{};
  auto snd = cudax_async::starts_on(sched, cudax_async::just(42));
  auto op  = cudax_async::connect(std::move(snd), checked_stopped_receiver{});
  cudax_async::start(op);
}

C2H_TEST("starts_on forwards sender errors", "[adaptors][starts_on]")
{
  auto ec  = error_code{std::errc::operation_not_permitted};
  auto snd = cudax_async::starts_on(inline_scheduler<>{}, cudax_async::just_error(ec));
  auto op  = cudax_async::connect(std::move(snd), checked_error_receiver{ec});
  cudax_async::start(op);
}

C2H_TEST("starts_on forwards sender stopped signal", "[adaptors][starts_on]")
{
  auto snd = cudax_async::starts_on(inline_scheduler<>{}, cudax_async::just_stopped());
  auto op  = cudax_async::connect(std::move(snd), checked_stopped_receiver{});
  cudax_async::start(op);
}

C2H_TEST("starts_on preserves multiple values", "[adaptors][starts_on]")
{
  auto snd = cudax_async::starts_on(inline_scheduler<>{}, cudax_async::just(1, 2.5, string{"hello"}));
  auto op  = cudax_async::connect(std::move(snd), checked_value_receiver{1, 2.5, string{"hello"}});
  cudax_async::start(op);
}

C2H_TEST("starts_on has the values_type corresponding to the child sender", "[adaptors][starts_on]")
{
  inline_scheduler<> sched{};

  check_value_types<types<int>>(cudax_async::starts_on(sched, cudax_async::just(1)));
  check_value_types<types<int, double>>(cudax_async::starts_on(sched, cudax_async::just(3, 0.14)));
  check_value_types<types<int, double, string>>(
    cudax_async::starts_on(sched, cudax_async::just(3, 0.14, string{"pi"})));
}

C2H_TEST("starts_on includes error_types from both scheduler and sender", "[adaptors][starts_on]")
{
  inline_scheduler<> sched1{};
  error_scheduler<std::error_code> sched2{std::make_error_code(std::errc::invalid_argument)};
  error_scheduler<int> sched3{43};

  // Inline scheduler has no errors, sender has no errors
  check_error_types<>(cudax_async::starts_on(sched1, cudax_async::just(1)));

  // Error scheduler has std::error_code, sender has no errors
  check_error_types<std::error_code>(cudax_async::starts_on(sched2, cudax_async::just(2)));

  // Error scheduler has int, sender has no errors
  check_error_types<int>(cudax_async::starts_on(sched3, cudax_async::just(3)));
}

C2H_TEST("starts_on sends_stopped includes both scheduler and sender", "[adaptors][starts_on]")
{
  inline_scheduler<> sched1{};
  error_scheduler<error_code> sched2{error_code{std::errc::invalid_argument}};
  stopped_scheduler sched3{};

  // Neither scheduler nor sender sends stopped
  check_sends_stopped<false>(cudax_async::starts_on(sched1, cudax_async::just(1)));

  // Scheduler can send stopped (through error_scheduler), sender doesn't
  check_sends_stopped<true>(cudax_async::starts_on(sched2, cudax_async::just(2)));

  // Scheduler sends stopped, sender doesn't
  check_sends_stopped<true>(cudax_async::starts_on(sched3, cudax_async::just(3)));
}

C2H_TEST("starts_on works with const sender", "[adaptors][starts_on]")
{
  const auto base_sender = cudax_async::just(42);
  auto snd               = cudax_async::starts_on(inline_scheduler<>{}, base_sender);
  auto op                = cudax_async::connect(std::move(snd), checked_value_receiver{42});
  cudax_async::start(op);
}
} // namespace
