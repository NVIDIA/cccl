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

#include "common/checked_receiver.cuh"
#include "common/dummy_scheduler.cuh"
#include "common/error_scheduler.cuh"
#include "common/stopped_scheduler.cuh"
#include "common/utility.cuh"

namespace ex = cuda::experimental::execution;

namespace
{
constexpr struct get_frob_t : cuda::std::execution::__basic_query<get_frob_t>
{
  _CCCL_HOST_DEVICE static constexpr bool query(ex::forwarding_query_t) noexcept
  {
    return true;
  }
} get_frob;

C2H_TEST("then returns a sender", "[adaptors][then]")
{
  auto snd = ex::then(ex::just(), [] {});
  static_assert(ex::sender<decltype(snd)>);
  (void) snd;
}

C2H_TEST("then with environment returns a sender", "[adaptors][then]")
{
  auto snd = ex::then(ex::just(), [] {});
  static_assert(ex::sender_in<decltype(snd), ex::env<>>);
  (void) snd;
}

C2H_TEST("then simple example", "[adaptors][then]")
{
  bool called{false};
  auto snd = ex::then(ex::just(), [&] {
    called = true;
  });
  auto op  = ex::connect(std::move(snd), checked_value_receiver{});
  ex::start(op);
  // The receiver checks that it's called
  // we also check that the function was invoked
  CHECK(called);
}

C2H_TEST("then can be piped", "[adaptors][then]")
{
  auto snd = ex::just() | ex::then([] {});
  STATIC_REQUIRE(ex::sender<decltype(snd)>);
  (void) snd;
}

C2H_TEST("then returning void can be waited on", "[adaptors][then]")
{
  auto snd = ex::just() | ex::then([] {});
  STATIC_REQUIRE(ex::sender<decltype(snd)>);
  ex::sync_wait(std::move(snd));
}

C2H_TEST("then can be used to transform the value", "[adaptors][then]")
{
  auto snd = ex::just(13) | ex::then([](int x) -> int {
               return 2 * x + 1;
             });
  wait_for_value(std::move(snd), 27);
}

C2H_TEST("then can be used to change the value type", "[adaptors][then]")
{
  auto snd = ex::just(3) | ex::then([](int x) -> double {
               return x + 0.1415;
             });
  wait_for_value(std::move(snd), 3.1415); // NOLINT(modernize-use-std-numbers)
}

C2H_TEST("then can be used with multiple parameters", "[adaptors][then]")
{
  auto snd = ex::just(3, 0.1415) | ex::then([](int x, double y) -> double {
               return x + y;
             });
  wait_for_value(std::move(snd), 3.1415); // NOLINT(modernize-use-std-numbers)
}

#if _CCCL_HAS_EXCEPTIONS() && _CCCL_HOST_COMPILATION()
C2H_TEST("then can throw, and set_error will be called", "[adaptors][then]")
{
  auto snd = ex::just(13) | ex::then([](int) -> int {
               throw std::logic_error{"err"};
             });
  auto op  = ex::connect(std::move(snd), checked_error_receiver{std::logic_error{"err"}});
  ex::start(op);
}
#endif // _CCCL_HAS_EXCEPTIONS() && _CCCL_HOST_COMPILATION()

C2H_TEST("then can be used with just_error", "[adaptors][then]")
{
  auto snd = ex::just_error(string{"err"}) | ex::then([]() -> int {
               return 17;
             });
  STATIC_REQUIRE(ex::sender<decltype(snd)>);
  auto op = ex::connect(std::move(snd), checked_error_receiver{string{"err"}});
  ex::start(op);
}

C2H_TEST("then can be used with just_stopped", "[adaptors][then]")
{
  auto snd = ex::just_stopped() | ex::then([]() -> int {
               return 17;
             });
  STATIC_REQUIRE(ex::sender<decltype(snd)>);
  auto op = ex::connect(std::move(snd), checked_stopped_receiver{});
  ex::start(op);
}

C2H_TEST("then function is not called on error", "[adaptors][then]")
{
  bool called{false};
  error_scheduler sched{-1};
  auto snd = ex::just(13) | ex::continues_on(sched) | ex::then([&](int x) -> int {
               called = true;
               return x + 5;
             });
  STATIC_REQUIRE(ex::sender<decltype(snd)>);
  auto op = ex::connect(std::move(snd), checked_error_receiver{-1});
  ex::start(op);
  CHECK_FALSE(called);
}

C2H_TEST("then function is not called when cancelled", "[adaptors][then]")
{
  bool called{false};
  stopped_scheduler sched;
  auto snd = ex::just(13) | ex::continues_on(sched) | ex::then([&](int x) -> int {
               called = true;
               return x + 5;
             });
  STATIC_REQUIRE(ex::sender<decltype(snd)>);
  auto op = ex::connect(std::move(snd), checked_stopped_receiver{});
  ex::start(op);
  CHECK_FALSE(called);
}

C2H_TEST("then advertises completion schedulers", "[adaptors][then]")
{
  dummy_scheduler sched{};

  SECTION("for value channel")
  {
    auto snd = ex::schedule(sched) | ex::then([] {});
    STATIC_REQUIRE(ex::sender<decltype(snd)>);
    REQUIRE(ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(snd)) == sched);
  }
}

C2H_TEST("then forwards env", "[adaptors][then]")
{
  SECTION("returns env by value")
  {
    auto snd = ex::just(0) | ex::write_attrs(ex::prop{get_frob, 100}) | ex::then([](int) {});
    CHECK(get_frob(ex::get_env(snd)) == 100);
  }

  SECTION("returns env by reference")
  {
    auto snd = ex::just(0) | ex::write_attrs(ex::prop{get_frob, 100}) | ex::then([](int) {});
    CHECK(get_frob(ex::get_env(snd)) == 100);
  }
}

C2H_TEST("then has the values_type corresponding to the given values", "[adaptors][then]")
{
  check_value_types<types<int>>(ex::just() | ex::then([] {
                                  return 7;
                                }));
  check_value_types<types<double>>(ex::just() | ex::then([] {
                                     return 3.14;
                                   }));
  check_value_types<types<string>>(ex::just() | ex::then([] {
                                     return string{"hello"};
                                   }));
}

C2H_TEST("then keeps error_types from input sender", "[adaptors][then]")
{
  dummy_scheduler sched1{};
  error_scheduler sched2{error_code{std::errc::invalid_argument}};
  error_scheduler sched3{43};

  check_error_types(ex::just() | ex::continues_on(sched1) | ex::then([]() noexcept {}));
  check_error_types<error_code>(ex::just() | ex::continues_on(sched2) | ex::then([]() noexcept {}));
#if _CCCL_HAS_EXCEPTIONS() && _CCCL_HOST_COMPILATION()
  check_error_types<std::exception_ptr, int>(ex::just() | ex::continues_on(sched3) | ex::then([] {}));
#else
  check_error_types<int>(ex::just() | ex::continues_on(sched3) | ex::then([] {}));
#endif
}

C2H_TEST("then keeps sends_stopped from input sender", "[adaptors][then]")
{
  dummy_scheduler sched1{};
  error_scheduler sched2{error_code{std::errc::invalid_argument}};
  stopped_scheduler sched3{};

  check_sends_stopped<false>(ex::just() | ex::continues_on(sched1) | ex::then([] {}));
  check_sends_stopped<false>(ex::just() | ex::continues_on(sched2) | ex::then([] {}));
  check_sends_stopped<true>(ex::just() | ex::continues_on(sched3) | ex::then([] {}));
}

C2H_TEST("then can return by reference", "[adaptors][then]")
{
  string str("hello"), *pstr = &str;
  auto snd = ex::just() | ex::then([pstr]() noexcept -> decltype(auto) {
               return *pstr;
             });
  check_value_types<types<string&>>(snd);
  check_error_types<>(snd);
  check_sends_stopped<false>(snd);
}

#if _CCCL_HAS_EXCEPTIONS() && _CCCL_HOST_COMPILATION()

struct throws_on_copy
{
  throws_on_copy()                 = default;
  throws_on_copy(throws_on_copy&&) = default;
  throws_on_copy(const throws_on_copy&)
  {
    throw std::runtime_error{"copy"};
  }
};

C2H_TEST("sync_wait can handle when then() returns a throws-on-copy type by reference", "[adaptors][then][sync_wait]")
{
  ex::thread_context worker{};
  throws_on_copy local, *plocal = &local;
  auto snd = ex::schedule(worker.get_scheduler()) | ex::then([pstr = plocal]() noexcept -> decltype(auto) {
               return *pstr;
             });
  check_value_types<types<throws_on_copy&>>(snd);
  check_error_types<>(snd);
  check_sends_stopped<true>(snd);
  CHECK_THROWS_AS(ex::sync_wait(std::move(snd)), std::runtime_error);
  worker.join();
}

#endif

// Return a different sender when we invoke this custom defined then implementation
struct then_test_domain
{
  _CCCL_TEMPLATE(class Sender, class... Env)
  _CCCL_REQUIRES(cuda::std::same_as<ex::tag_of_t<Sender>, ex::then_t>)
  static auto transform_sender(Sender&&, Env&&...)
  {
    return ex::just(string{"ciao"});
  }
};

C2H_TEST("then can be customized early", "[adaptors][then]")
{
  // The customization will return a different value
  dummy_scheduler<then_test_domain> sched;
  auto snd = ex::just(string{"hello"}) | ex::continues_on(sched) | ex::then([](string x) {
               return x + ", world";
             });
  wait_for_value(std::move(snd), string{"ciao"});
}

C2H_TEST("then can be customized late", "[adaptors][then]")
{
  // The customization will return a different value
  dummy_scheduler<then_test_domain> sched;
  auto snd = ex::just(string{"hello"})
           | ex::on(sched, ex::then([](string x) {
                      return x + ", world";
                    }))
           | ex::write_env(ex::prop{ex::get_scheduler, dummy_scheduler()});
  wait_for_value(std::move(snd), string{"ciao"});
}
} // namespace
