//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__async/sender.cuh>

//
#include "common/checked_receiver.cuh"
#include "common/error_scheduler.cuh"
#include "common/impulse_scheduler.cuh"
#include "common/inline_scheduler.cuh"
#include "common/stopped_scheduler.cuh"
#include "common/utility.cuh"
#include "testing.cuh"

namespace
{
C2H_TEST("continue_on simple example", "[adaptors][continue_on]")
{
  auto snd = cudax_async::continue_on(cudax_async::just(13), inline_scheduler{});
  auto op  = cudax_async::connect(std::move(snd), checked_value_receiver{13});
  cudax_async::start(op);
  // The receiver checks if we receive the right value
}

#if !defined(__CUDA_ARCH__)

C2H_TEST("continue_on can be piped", "[adaptors][continue_on]")
{
  // Just continue_on a value to the impulse scheduler
  bool called{false};
  auto sched = impulse_scheduler{};
  auto snd   = cudax_async::just(13) //
           | cudax_async::continue_on(sched) //
           | cudax_async::then([&](int val) {
               called = true;
               return val;
             });
  // Start the operation
  auto op = cudax_async::connect(std::move(snd), checked_value_receiver{13});
  cudax_async::start(op);

  // The value will be available when the scheduler will execute the next operation
  CUDAX_REQUIRE(!called);
  sched.start_next();
  CUDAX_REQUIRE(called);
}

C2H_TEST("continue_on calls the receiver when the scheduler dictates", "[adaptors][continue_on]")
{
  bool called{false};
  impulse_scheduler sched;
  auto snd = cudax_async::then(cudax_async::continue_on(cudax_async::just(13), sched), [&](int val) {
    called = true;
    return val;
  });
  auto op  = cudax_async::connect(snd, checked_value_receiver{13});
  cudax_async::start(op);
  // Up until this point, the scheduler didn't start any task; no effect expected
  CUDAX_CHECK(!called);

  // Tell the scheduler to start executing one task
  sched.start_next();
  CUDAX_CHECK(called);
}

C2H_TEST("continue_on calls the given sender when the scheduler dictates", "[adaptors][continue_on]")
{
  int counter{0};
  auto snd_base = cudax_async::just() //
                | cudax_async::then([&]() -> int {
                    ++counter;
                    return 19;
                  });

  impulse_scheduler sched;
  auto snd = cudax_async::then(cudax_async::continue_on(std::move(snd_base), sched), [&](int val) {
    ++counter;
    return val;
  });
  auto op  = cudax_async::connect(std::move(snd), checked_value_receiver{19});
  cudax_async::start(op);
  // The sender is started, even if the scheduler hasn't yet triggered
  CUDAX_CHECK(counter == 1);
  // ... but didn't send the value to the receiver yet

  // Tell the scheduler to start executing one task
  sched.start_next();

  // Now the base sender is called, and a value is sent to the receiver
  CUDAX_CHECK(counter == 2);
}

C2H_TEST("continue_on works when changing threads", "[adaptors][continue_on]")
{
  cudax_async::thread_context thread;
  bool called{false};

  {
    // lunch some work on the thread pool
    auto snd = cudax_async::continue_on(cudax_async::just(), thread.get_scheduler()) //
             | cudax_async::then([&] {
                 called = true;
               });
    cudax_async::start_detached(std::move(snd));
  }

  thread.join();

  // the work should be executed
  CUDAX_REQUIRE(called);
}

#endif

C2H_TEST("continue_on can be called with rvalue ref scheduler", "[adaptors][continue_on]")
{
  auto snd = cudax_async::continue_on(cudax_async::just(13), inline_scheduler{});
  auto op  = cudax_async::connect(std::move(snd), checked_value_receiver{13});
  cudax_async::start(op);
  // The receiver checks if we receive the right value
}

C2H_TEST("continue_on can be called with const ref scheduler", "[adaptors][continue_on]")
{
  const inline_scheduler sched;
  auto snd = cudax_async::continue_on(cudax_async::just(13), sched);
  auto op  = cudax_async::connect(std::move(snd), checked_value_receiver{13});
  cudax_async::start(op);
  // The receiver checks if we receive the right value
}

C2H_TEST("continue_on can be called with ref scheduler", "[adaptors][continue_on]")
{
  inline_scheduler sched;
  auto snd = cudax_async::continue_on(cudax_async::just(13), sched);
  auto op  = cudax_async::connect(std::move(snd), checked_value_receiver{13});
  cudax_async::start(op);
  // The receiver checks if we receive the right value
}

C2H_TEST("continue_on forwards set_error calls", "[adaptors][continue_on]")
{
  auto ec = error_code{std::errc::invalid_argument};
  error_scheduler<error_code> sched{ec};
  auto snd = cudax_async::continue_on(cudax_async::just(13), sched);
  auto op  = cudax_async::connect(std::move(snd), checked_error_receiver{ec});
  cudax_async::start(op);
  // The receiver checks if we receive an error
}

C2H_TEST("continue_on forwards set_error calls of other types", "[adaptors][continue_on]")
{
  error_scheduler<string> sched{string{"error"}};
  auto snd = cudax_async::continue_on(cudax_async::just(13), sched);
  auto op  = cudax_async::connect(std::move(snd), checked_error_receiver{string{"error"}});
  cudax_async::start(op);
  // The receiver checks if we receive an error
}

C2H_TEST("continue_on forwards set_stopped calls", "[adaptors][continue_on]")
{
  stopped_scheduler sched{};
  auto snd = cudax_async::continue_on(cudax_async::just(13), sched);
  auto op  = cudax_async::connect(std::move(snd), checked_stopped_receiver{});
  cudax_async::start(op);
  // The receiver checks if we receive the stopped signal
}

C2H_TEST("continue_on has the values_type corresponding to the given values", "[adaptors][continue_on]")
{
  inline_scheduler sched{};

  check_value_types<types<int>>(cudax_async::continue_on(cudax_async::just(1), sched));
  check_value_types<types<int, double>>(cudax_async::continue_on(cudax_async::just(3, 0.14), sched));
  check_value_types<types<int, double, string>>(
    cudax_async::continue_on(cudax_async::just(3, 0.14, string{"pi"}), sched));
}

C2H_TEST("continue_on keeps error_types from scheduler's sender", "[adaptors][continue_on]")
{
  inline_scheduler sched1{};
  error_scheduler<std::error_code> sched2{std::make_error_code(std::errc::invalid_argument)};
  error_scheduler<int> sched3{43};

  check_error_types<>(cudax_async::continue_on(cudax_async::just(1), sched1));
  check_error_types<std::error_code>(cudax_async::continue_on(cudax_async::just(2), sched2));
  check_error_types<int>(cudax_async::continue_on(cudax_async::just(3), sched3));
}

C2H_TEST("continue_on sends an exception_ptr if value types are potentially throwing when copied",
         "[adaptors][continue_on]")
{
  inline_scheduler sched{};

#if !defined(__CUDA_ARCH__)
  check_error_types<std::exception_ptr>(cudax_async::continue_on(cudax_async::just(potentially_throwing{}), sched));
#else
  // No exceptions in device code:
  check_error_types<>(cudax_async::continue_on(cudax_async::just(potentially_throwing{}), sched));
#endif
}

C2H_TEST("continue_on keeps sends_stopped from scheduler's sender", "[adaptors][continue_on]")
{
  inline_scheduler sched1{};
  error_scheduler<error_code> sched2{error_code{std::errc::invalid_argument}};
  stopped_scheduler sched3{};

  check_sends_stopped<false>(cudax_async::continue_on(cudax_async::just(1), sched1));
  check_sends_stopped<true>(cudax_async::continue_on(cudax_async::just(2), sched2));
  check_sends_stopped<true>(cudax_async::continue_on(cudax_async::just(3), sched3));
}
} // namespace
