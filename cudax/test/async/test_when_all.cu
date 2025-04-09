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

#include "common/checked_receiver.cuh"
#include "common/error_scheduler.cuh"
#include "common/impulse_scheduler.cuh"
#include "common/stopped_scheduler.cuh"
#include "common/utility.cuh"
#include "testing.cuh"

namespace
{
C2H_TEST("when_all simple example", "[when_all]")
{
  auto snd  = cudax_async::when_all(cudax_async::just(3), cudax_async::just(0.1415));
  auto snd1 = std::move(snd) | cudax_async::then([](int x, double y) {
                return x + y;
              });
  auto op   = cudax_async::connect(std::move(snd1), checked_value_receiver{3.1415});
  cudax_async::start(op);
}

C2H_TEST("when_all returning two values can be waited on", "[when_all]")
{
  auto snd = cudax_async::when_all(cudax_async::just(2), cudax_async::just(3));
  check_values(std::move(snd), 2, 3);
}

C2H_TEST("when_all with 5 senders", "[when_all]")
{
  auto snd = cudax_async::when_all(
    cudax_async::just(2), cudax_async::just(3), cudax_async::just(5), cudax_async::just(7), cudax_async::just(11));
  check_values(std::move(snd), 2, 3, 5, 7, 11);
}

C2H_TEST("when_all with just one sender", "[when_all]")
{
  auto snd = cudax_async::when_all(cudax_async::just(2));
  check_values(std::move(snd), 2);
}

C2H_TEST("when_all with move-only types", "[when_all]")
{
  auto snd = cudax_async::when_all(cudax_async::just(movable(2)));
  check_values(std::move(snd), movable(2));
}

C2H_TEST("when_all with no senders", "[when_all]")
{
  auto snd = cudax_async::when_all();
  check_values(std::move(snd));
}

C2H_TEST("when_all when one sender sends void", "[when_all]")
{
  auto snd = cudax_async::when_all(cudax_async::just(2), cudax_async::just());
  check_values(std::move(snd), 2);
}

#if !defined(__CUDA_ARCH__)

C2H_TEST("when_all completes when children complete", "[when_all]")
{
  impulse_scheduler sched;
  bool called{false};
  auto snd = cudax_async::when_all(cudax_async::just(11) | cudax_async::continue_on(sched),
                                   cudax_async::just(13) | cudax_async::continue_on(sched),
                                   cudax_async::just(17) | cudax_async::continue_on(sched))
           | cudax_async::then([&](int a, int b, int c) {
               called = true;
               return a + b + c;
             });
  auto op = cudax_async::connect(std::move(snd), checked_value_receiver{41});
  cudax_async::start(op);
  // The when_all scheduler will complete only after 3 impulses
  CUDAX_CHECK_FALSE(called);
  sched.start_next();
  CUDAX_CHECK_FALSE(called);
  sched.start_next();
  CUDAX_CHECK_FALSE(called);
  sched.start_next();
  CUDAX_CHECK(called);
}

#endif

C2H_TEST("when_all can be used with just_*", "[when_all]")
{
  auto snd = cudax_async::when_all(cudax_async::just(2), cudax_async::just_error(42), cudax_async::just_stopped());
  auto op  = cudax_async::connect(std::move(snd), checked_error_receiver{42});
  cudax_async::start(op);
}

C2H_TEST("when_all terminates with error if one child terminates with error", "[when_all]")
{
  error_scheduler sched{42};
  auto snd = cudax_async::when_all(
    cudax_async::just(2), cudax_async::just(5) | cudax_async::continue_on(sched), cudax_async::just(7));
  auto op = cudax_async::connect(std::move(snd), checked_error_receiver{42});
  cudax_async::start(op);
}

C2H_TEST("when_all terminates with stopped if one child is cancelled", "[when_all]")
{
  stopped_scheduler sched;
  auto snd = cudax_async::when_all(
    cudax_async::just(2), cudax_async::just(5) | cudax_async::continue_on(sched), cudax_async::just(7));
  auto op = cudax_async::connect(std::move(snd), checked_stopped_receiver{});
  cudax_async::start(op);
}

#if !defined(__CUDA_ARCH__)

C2H_TEST("when_all cancels remaining children if error is detected", "[when_all]")
{
  impulse_scheduler sched;
  error_scheduler err_sched{42};
  bool called1{false};
  bool called3{false};
  bool cancelled{false};
  auto snd = cudax_async::when_all(
    cudax_async::start_on(sched, cudax_async::just()) | cudax_async::then([&] {
      called1 = true;
    }),
    cudax_async::start_on(sched, cudax_async::just(5) | cudax_async::continue_on(err_sched)),
    cudax_async::start_on(sched, cudax_async::just()) | cudax_async::then([&] {
      called3 = true;
    }) | cudax_async::let_stopped([&] {
      cancelled = true;
      return cudax_async::just();
    }));
  auto op = cudax_async::connect(std::move(snd), checked_error_receiver{42});
  cudax_async::start(op);
  // The first child will complete; the third one will be cancelled
  CUDAX_CHECK_FALSE(called1);
  CUDAX_CHECK_FALSE(called3);
  sched.start_next(); // start the first child
  CUDAX_CHECK(called1);
  sched.start_next(); // start the second child; this will generate an error
  CUDAX_CHECK_FALSE(called3);
  sched.start_next(); // start the third child
  CUDAX_CHECK_FALSE(called3);
  CUDAX_CHECK(cancelled);
}

C2H_TEST("when_all cancels remaining children if cancel is detected", "[when_all]")
{
  stopped_scheduler stopped_sched;
  impulse_scheduler sched;
  bool called1{false};
  bool called3{false};
  bool cancelled{false};
  auto snd = cudax_async::when_all(
    cudax_async::start_on(sched, cudax_async::just()) | cudax_async::then([&] {
      called1 = true;
    }),
    cudax_async::start_on(sched, cudax_async::just(5) | cudax_async::continue_on(stopped_sched)),
    cudax_async::start_on(sched, cudax_async::just()) | cudax_async::then([&] {
      called3 = true;
    }) | cudax_async::let_stopped([&] {
      cancelled = true;
      return cudax_async::just();
    }));
  auto op = cudax_async::connect(std::move(snd), checked_stopped_receiver{});
  cudax_async::start(op);
  // The first child will complete; the third one will be cancelled
  CUDAX_CHECK_FALSE(called1);
  CUDAX_CHECK_FALSE(called3);
  sched.start_next(); // start the first child
  CUDAX_CHECK(called1);
  sched.start_next(); // start the second child; this will call set_stopped
  CUDAX_CHECK_FALSE(called3);
  sched.start_next(); // start the third child
  CUDAX_CHECK_FALSE(called3);
  CUDAX_CHECK(cancelled);
}

#endif

template <class... Ts>
struct just_ref
{
  using sender_concept = cudax_async::sender_t;

  template <class Self, class... Env>
  _CCCL_HOST_DEVICE static constexpr auto get_completion_signatures()
  {
    return cudax_async::completion_signatures<cudax_async::set_value_t(Ts & ...)>{};
  }

  _CCCL_HOST_DEVICE just_ref connect(cudax_async::__ignore) const
  {
    return {};
  }
};

C2H_TEST("when_all has the values_type based on the children, decayed and as rvalue "
         "references",
         "[when_all]")
{
  check_value_types<types<int>>(cudax_async::when_all(cudax_async::just(13)));
  check_value_types<types<double>>(cudax_async::when_all(cudax_async::just(3.14)));
  check_value_types<types<int, double>>(cudax_async::when_all(cudax_async::just(3, 0.14)));

  check_value_types<types<>>(cudax_async::when_all(cudax_async::just()));

  check_value_types<types<int, double>>(cudax_async::when_all(cudax_async::just(3), cudax_async::just(0.14)));
  check_value_types<types<int, double, int, double>>(
    cudax_async::when_all(cudax_async::just(3), cudax_async::just(0.14), cudax_async::just(1, 0.4142)));

  // if one child returns void, then the value is simply missing
  check_value_types<types<int, double>>(
    cudax_async::when_all(cudax_async::just(3), cudax_async::just(), cudax_async::just(0.14)));

  // if one child has no value completion, the when_all has no value
  // completion
  check_value_types<>(
    cudax_async::when_all(cudax_async::just(3), cudax_async::just_stopped(), cudax_async::just(0.14)));

  // if children send references, they get decayed
  check_value_types<types<int, double>>(cudax_async::when_all(just_ref<int>(), just_ref<double>()));
}

C2H_TEST("when_all has the error_types based on the children", "[when_all]")
{
  check_error_types<int>(cudax_async::when_all(cudax_async::just_error(13)));

  check_error_types<double>(cudax_async::when_all(cudax_async::just_error(3.14)));

  check_error_types<>(cudax_async::when_all(cudax_async::just()));

  check_error_types<int, double>(cudax_async::when_all(cudax_async::just_error(3), cudax_async::just_error(0.14)));

  check_error_types<int, double, string>(cudax_async::when_all(
    cudax_async::just_error(3), cudax_async::just_error(0.14), cudax_async::just_error(string{"err"})));

  check_error_types<error_code>(cudax_async::when_all(
    cudax_async::just(13),
    cudax_async::just_error(error_code{std::errc::invalid_argument}),
    cudax_async::just_stopped()));

#if !defined(__CUDA_ARCH__)
  // if the child sends something with a potentially throwing decay-copy,
  // the when_all has an exception_ptr error completion.
  check_error_types<std::exception_ptr>(cudax_async::when_all(just_ref<potentially_throwing>()));
#else
  // in device code, there are no exceptions:
  check_error_types<>(cudax_async::when_all(just_ref<potentially_throwing>()));
#endif
}

C2H_TEST("when_all has the sends_stopped == true", "[when_all]")
{
  check_sends_stopped<true>(cudax_async::when_all(cudax_async::just(13)));
  check_sends_stopped<true>(cudax_async::when_all(cudax_async::just_error(-1)));
  check_sends_stopped<true>(cudax_async::when_all(cudax_async::just_stopped()));

  check_sends_stopped<true>(cudax_async::when_all(cudax_async::just(3), cudax_async::just(0.14)));
  check_sends_stopped<true>(
    cudax_async::when_all(cudax_async::just(3), cudax_async::just_error(-1), cudax_async::just_stopped()));
}
} // namespace
