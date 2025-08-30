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
#include "common/dummy_scheduler.cuh"
#include "common/error_scheduler.cuh"
#include "common/stopped_scheduler.cuh"
#include "common/utility.cuh"

#if _CCCL_HOST_COMPILATION()
#  include "common/impulse_scheduler.cuh"
#endif

#if _CCCL_COMPILER(GCC, <, 12)
// suppress buggy warning on older gcc versions
_CCCL_DIAG_SUPPRESS_GCC("-Wmissing-field-initializers")
#endif

namespace ex = cuda::experimental::execution;

namespace
{
C2H_TEST("starts_on simple example", "[adaptors][starts_on]")
{
  auto snd = ex::starts_on(dummy_scheduler{}, ex::just(42));
  auto op  = ex::connect(std::move(snd), checked_value_receiver{42});
  ex::start(op);
  // The receiver checks if we receive the right value
}

C2H_TEST("starts_on can be piped", "[adaptors][starts_on]")
{
  // Use starts_on with the inline scheduler and pipe with then
  auto snd = ex::starts_on(dummy_scheduler{}, ex::just(42)) //
           | ex::then([](int val) {
               return val * 2;
             });
  auto op = ex::connect(std::move(snd), checked_value_receiver{84});
  ex::start(op);
  // The receiver checks if we receive the transformed value
}

#if _CCCL_HOST_COMPILATION()

C2H_TEST("starts_on with impulse scheduler", "[adaptors][starts_on]")
{
  bool sender_executed = false;

  impulse_scheduler sched;
  auto snd = ex::starts_on(sched, ex::just() | ex::then([&]() {
                                    sender_executed = true;
                                    return 13;
                                  }));

  auto op = ex::connect(std::move(snd), checked_value_receiver{13});
  ex::start(op);

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

  auto snd = ex::starts_on(sched, ex::just() | ex::then([&]() {
                                    return ++counter;
                                  }));

  auto op = ex::connect(std::move(snd), checked_value_receiver{1});
  ex::start(op);

  // Counter should still be 0 since scheduler hasn't executed yet
  CUDAX_CHECK(counter == 0);

  // Tell the scheduler to start executing
  sched.start_next();

  // Now the sender should have executed and incremented counter
  CUDAX_CHECK(counter == 1);
}

C2H_TEST("starts_on with thread context", "[adaptors][starts_on]")
{
  ex::thread_context thread;
  bool executed = false;

  auto snd = ex::starts_on(thread.get_scheduler(), ex::just() | ex::then([&]() {
                                                     executed = true;
                                                     return 123;
                                                   }));

  auto op = ex::connect(std::move(snd), checked_value_receiver{123});
  ex::start(op);

  thread.join();

  // The work should have been executed on the thread
  CUDAX_REQUIRE(executed);
}

#endif

C2H_TEST("starts_on can be called with rvalue ref scheduler", "[adaptors][starts_on]")
{
  auto snd = ex::starts_on(dummy_scheduler{}, ex::just(42));
  auto op  = ex::connect(std::move(snd), checked_value_receiver{42});
  ex::start(op);
}

C2H_TEST("starts_on can be called with const ref scheduler", "[adaptors][starts_on]")
{
  const dummy_scheduler<> sched;
  auto snd = ex::starts_on(sched, ex::just(42));
  auto op  = ex::connect(std::move(snd), checked_value_receiver{42});
  ex::start(op);
}

C2H_TEST("starts_on can be called with ref scheduler", "[adaptors][starts_on]")
{
  dummy_scheduler<> sched;
  auto snd = ex::starts_on(sched, ex::just(42));
  auto op  = ex::connect(std::move(snd), checked_value_receiver{42});
  ex::start(op);
}

C2H_TEST("starts_on forwards scheduler errors", "[adaptors][starts_on]")
{
  auto ec = error_code{std::errc::invalid_argument};
  error_scheduler<error_code> sched{ec};
  auto snd = ex::starts_on(sched, ex::just(42));
  auto op  = ex::connect(std::move(snd), checked_error_receiver{ec});
  ex::start(op);
  // The receiver checks if we receive the error from the scheduler
}

C2H_TEST("starts_on forwards scheduler errors of other types", "[adaptors][starts_on]")
{
  error_scheduler<string> sched{string{"scheduler error"}};
  auto snd = ex::starts_on(sched, ex::just(42));
  auto op  = ex::connect(std::move(snd), checked_error_receiver{string{"scheduler error"}});
  ex::start(op);
}

C2H_TEST("starts_on forwards scheduler stopped signal", "[adaptors][starts_on]")
{
  stopped_scheduler sched{};
  auto snd = ex::starts_on(sched, ex::just(42));
  auto op  = ex::connect(std::move(snd), checked_stopped_receiver{});
  ex::start(op);
}

C2H_TEST("starts_on forwards sender errors", "[adaptors][starts_on]")
{
  auto ec  = error_code{std::errc::operation_not_permitted};
  auto snd = ex::starts_on(dummy_scheduler{}, ex::just_error(ec));
  auto op  = ex::connect(std::move(snd), checked_error_receiver{ec});
  ex::start(op);
}

C2H_TEST("starts_on forwards sender stopped signal", "[adaptors][starts_on]")
{
  auto snd = ex::starts_on(dummy_scheduler{}, ex::just_stopped());
  auto op  = ex::connect(std::move(snd), checked_stopped_receiver{});
  ex::start(op);
}

C2H_TEST("starts_on preserves multiple values", "[adaptors][starts_on]")
{
  auto snd = ex::starts_on(dummy_scheduler{}, ex::just(1, 2.5, string{"hello"}));
  auto op  = ex::connect(std::move(snd), checked_value_receiver{1, 2.5, string{"hello"}});
  ex::start(op);
}

C2H_TEST("starts_on has the values_type corresponding to the child sender", "[adaptors][starts_on]")
{
  dummy_scheduler<> sched{};

  check_value_types<types<int>>(ex::starts_on(sched, ex::just(1)));
  check_value_types<types<int, double>>(ex::starts_on(sched, ex::just(3, 0.14)));
  check_value_types<types<int, double, string>>(ex::starts_on(sched, ex::just(3, 0.14, string{"pi"})));
}

C2H_TEST("starts_on includes error_types from both scheduler and sender", "[adaptors][starts_on]")
{
  dummy_scheduler<> sched1{};
  error_scheduler<std::error_code> sched2{std::make_error_code(std::errc::invalid_argument)};
  error_scheduler<int> sched3{43};

  // Inline scheduler has no errors, sender has no errors
  check_error_types<>(ex::starts_on(sched1, ex::just(1)));

  // Error scheduler has std::error_code, sender has no errors
  check_error_types<std::error_code>(ex::starts_on(sched2, ex::just(2)));

  // Error scheduler has int, sender has no errors
  check_error_types<int>(ex::starts_on(sched3, ex::just(3)));
}

C2H_TEST("starts_on sends_stopped includes both scheduler and sender", "[adaptors][starts_on]")
{
  dummy_scheduler<> sched1{};
  error_scheduler<error_code> sched2{error_code{std::errc::invalid_argument}};
  stopped_scheduler sched3{};

  // Neither scheduler nor sender sends stopped
  check_sends_stopped<false>(ex::starts_on(sched1, ex::just(1)));

  // Scheduler does not send stopped but the sender does
  check_sends_stopped<true>(ex::starts_on(sched2, ex::just_stopped()));

  // Scheduler sends stopped, sender doesn't
  check_sends_stopped<true>(ex::starts_on(sched3, ex::just(3)));
}

C2H_TEST("starts_on works with const sender", "[adaptors][starts_on]")
{
  const auto base_sender = ex::just(42);
  auto snd               = ex::starts_on(dummy_scheduler{}, base_sender);
  auto op                = ex::connect(std::move(snd), checked_value_receiver{42});
  ex::start(op);
}

struct test_domain
{};

C2H_TEST("starts_on has the right completion scheduler", "[adaptors][starts_on]")
{
  SECTION("thread scheduler with a sender that completes inline")
  {
    ex::thread_context thread;
    auto sch = thread.get_scheduler();
    auto snd = ex::starts_on(sch, ex::just());
    CHECK(ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(snd)) == sch);
  }

  SECTION("thread scheduler with a sender that completes on another thread")
  {
    ex::thread_context thread1, thread2;
    auto sch1 = thread1.get_scheduler(), sch2 = thread2.get_scheduler();
    auto snd = ex::starts_on(sch1, ex::starts_on(sch2, ex::just()));
    CHECK(ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(snd)) == sch2);
  }

  SECTION("inline scheduler with inline sender completion with a given starting scheduler")
  {
    ex::thread_context thread;
    auto sch = thread.get_scheduler();
    auto env = ex::prop{ex::get_scheduler, sch};
    auto snd = ex::starts_on(ex::inline_scheduler{}, ex::just());
    STATIC_REQUIRE(!cudax::__callable<ex::get_completion_scheduler_t<ex::set_value_t>, ex::env_of_t<decltype(snd)>>);
    CHECK(ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(snd), env) == sch);
  }

  SECTION("inline scheduler with inline sender completion with an inline starting scheduler")
  {
    constexpr auto sch                  = ex::inline_scheduler{};
    [[maybe_unused]] constexpr auto env = ex::prop{ex::get_scheduler, sch};
    [[maybe_unused]] constexpr auto snd = ex::starts_on(sch, ex::just());
    using snd_t                         = decltype(snd);
    STATIC_REQUIRE(!cudax::__callable<ex::get_completion_scheduler_t<ex::set_value_t>, ex::env_of_t<snd_t>>);
    STATIC_REQUIRE(ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(snd), env) == sch);
  }

  SECTION("inline scheduler but sender knows where it completes")
  {
    ex::thread_context thread;
    auto sch = thread.get_scheduler();
    auto snd = ex::starts_on(dummy_scheduler{}, ex::schedule(sch));
    CHECK(ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(snd)) == sch);
  }
}
} // anonymous namespace
