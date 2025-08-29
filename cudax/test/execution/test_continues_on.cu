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

//
#include "common/checked_receiver.cuh"
#include "common/dummy_scheduler.cuh"
#include "common/error_scheduler.cuh"
#include "common/impulse_scheduler.cuh" // IWYU pragma: keep
#include "common/stopped_scheduler.cuh"
#include "common/utility.cuh"
#include "testing.cuh" // IWYU pragma: keep

namespace ex = cuda::experimental::execution;

namespace
{
C2H_TEST("continues_on simple example", "[adaptors][continues_on]")
{
  auto snd = ex::continues_on(ex::just(13), ex::inline_scheduler{});
  auto op  = ex::connect(std::move(snd), checked_value_receiver{13});

  static_assert(ex::get_completion_behavior<decltype(snd)>() == ex::completion_behavior::inline_completion);
  ex::start(op);
  // The receiver checks if we receive the right value
}

#if _CCCL_HOST_COMPILATION()

C2H_TEST("continues_on can be piped", "[adaptors][continues_on]")
{
  // Just continues_on a value to the impulse scheduler
  bool called{false};
  auto sched = impulse_scheduler{};
  auto snd   = ex::just(13) //
           | ex::continues_on(sched) //
           | ex::then([&](int val) {
               called = true;
               return val;
             });
  static_assert(ex::get_completion_behavior<decltype(snd)>() == ex::completion_behavior::asynchronous);
  // Start the operation
  auto op = ex::connect(std::move(snd), checked_value_receiver{13});
  ex::start(op);

  // The value will be available when the scheduler will execute the next operation
  CUDAX_REQUIRE(!called);
  sched.start_next();
  CUDAX_REQUIRE(called);
}

C2H_TEST("continues_on calls the receiver when the scheduler dictates", "[adaptors][continues_on]")
{
  bool called{false};
  impulse_scheduler sched;
  auto snd = ex::then(ex::continues_on(ex::just(13), sched), [&](int val) {
    called = true;
    return val;
  });
  auto op  = ex::connect(snd, checked_value_receiver{13});
  ex::start(op);
  // Up until this point, the scheduler didn't start any task; no effect expected
  CUDAX_CHECK(!called);

  // Tell the scheduler to start executing one task
  sched.start_next();
  CUDAX_CHECK(called);
}

C2H_TEST("continues_on calls the given sender when the scheduler dictates", "[adaptors][continues_on]")
{
  int counter{0};
  auto snd_base = ex::just() //
                | ex::then([&]() -> int {
                    ++counter;
                    return 19;
                  });

  impulse_scheduler sched;
  auto snd = ex::then(ex::continues_on(std::move(snd_base), sched), [&](int val) {
    ++counter;
    return val;
  });
  auto op  = ex::connect(std::move(snd), checked_value_receiver{19});
  ex::start(op);
  // The sender is started, even if the scheduler hasn't yet triggered
  CUDAX_CHECK(counter == 1);
  // ... but didn't send the value to the receiver yet

  // Tell the scheduler to start executing one task
  sched.start_next();

  // Now the base sender is called, and a value is sent to the receiver
  CUDAX_CHECK(counter == 2);
}

C2H_TEST("continues_on works when changing threads", "[adaptors][continues_on]")
{
  ex::thread_context thread;
  bool called{false};

  {
    // lunch some work on the thread pool
    auto snd = ex::continues_on(ex::just(), thread.get_scheduler()) //
             | ex::then([&] {
                 called = true;
               });
    ex::start_detached(std::move(snd));
  }

  thread.join();

  // the work should be executed
  CUDAX_REQUIRE(called);
}

#endif // _CCCL_HOST_COMPILATION()

C2H_TEST("continues_on can be called with rvalue ref scheduler", "[adaptors][continues_on]")
{
  auto snd = ex::continues_on(ex::just(13), dummy_scheduler<>{});
  auto op  = ex::connect(std::move(snd), checked_value_receiver{13});
  ex::start(op);
  // The receiver checks if we receive the right value
}

C2H_TEST("continues_on can be called with const ref scheduler", "[adaptors][continues_on]")
{
  const dummy_scheduler<> sched;
  auto snd = ex::continues_on(ex::just(13), sched);
  auto op  = ex::connect(std::move(snd), checked_value_receiver{13});
  ex::start(op);
  // The receiver checks if we receive the right value
}

C2H_TEST("continues_on can be called with ref scheduler", "[adaptors][continues_on]")
{
  dummy_scheduler<> sched;
  auto snd = ex::continues_on(ex::just(13), sched);
  auto op  = ex::connect(std::move(snd), checked_value_receiver{13});
  ex::start(op);
  // The receiver checks if we receive the right value
}

C2H_TEST("continues_on forwards set_error calls", "[adaptors][continues_on]")
{
  auto ec = error_code{std::errc::invalid_argument};
  error_scheduler<error_code> sched{ec};
  auto snd = ex::continues_on(ex::just(13), sched);
  auto op  = ex::connect(std::move(snd), checked_error_receiver{ec});
  ex::start(op);
  // The receiver checks if we receive an error
}

C2H_TEST("continues_on forwards set_error calls of other types", "[adaptors][continues_on]")
{
  error_scheduler<string> sched{string{"error"}};
  auto snd = ex::continues_on(ex::just(13), sched);
  auto op  = ex::connect(std::move(snd), checked_error_receiver{string{"error"}});
  ex::start(op);
  // The receiver checks if we receive an error
}

C2H_TEST("continues_on forwards set_stopped calls", "[adaptors][continues_on]")
{
  stopped_scheduler sched{};
  auto snd = ex::continues_on(ex::just(13), sched);
  auto op  = ex::connect(std::move(snd), checked_stopped_receiver{});
  ex::start(op);
  // The receiver checks if we receive the stopped signal
}

C2H_TEST("continues_on has the values_type corresponding to the given values", "[adaptors][continues_on]")
{
  dummy_scheduler<> sched{};

  check_value_types<types<int>>(ex::continues_on(ex::just(1), sched));
  check_value_types<types<int, double>>(ex::continues_on(ex::just(3, 0.14), sched));
  check_value_types<types<int, double, string>>(ex::continues_on(ex::just(3, 0.14, string{"pi"}), sched));
}

C2H_TEST("continues_on keeps error_types from scheduler's sender", "[adaptors][continues_on]")
{
  dummy_scheduler<> sched1{};
  error_scheduler<std::error_code> sched2{std::make_error_code(std::errc::invalid_argument)};
  error_scheduler<int> sched3{43};

  check_error_types<>(ex::continues_on(ex::just(1), sched1));
  check_error_types<std::error_code>(ex::continues_on(ex::just(2), sched2));
  check_error_types<int>(ex::continues_on(ex::just(3), sched3));
}

C2H_TEST("continues_on sends an exception_ptr if value types are potentially throwing when copied",
         "[adaptors][continues_on]")
{
  dummy_scheduler<> sched{};

#if _CCCL_HOST_COMPILATION()
  check_error_types<std::exception_ptr>(ex::continues_on(ex::just(potentially_throwing{}), sched));
#else
  // No exceptions in device code:
  check_error_types<>(ex::continues_on(ex::just(potentially_throwing{}), sched));
#endif
}

C2H_TEST("continues_on keeps sends_stopped from scheduler's sender", "[adaptors][continues_on]")
{
  dummy_scheduler<> sched1{};
  stopped_scheduler sched2{};

  check_sends_stopped<false>(ex::continues_on(ex::just(1), sched1));
  check_sends_stopped<true>(ex::continues_on(ex::just_stopped(), sched1));
  check_sends_stopped<true>(ex::continues_on(ex::just(3), sched2));
}
} // namespace
