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

// IWYU pragma: keep
#include "common/checked_receiver.cuh"
#include "common/dummy_scheduler.cuh" // IWYU pragma: keep
#include "common/error_scheduler.cuh" // IWYU pragma: keep
#include "common/impulse_scheduler.cuh" // IWYU pragma: keep
#include "common/stopped_scheduler.cuh" // IWYU pragma: keep
#include "common/utility.cuh"
#include "testing.cuh" // IWYU pragma: keep

namespace ex = cuda::experimental::execution;

namespace
{
// Return a different sender when we invoke this custom defined let_value implementation
struct let_value_test_domain
{
  _CCCL_TEMPLATE(class Sender)
  _CCCL_REQUIRES(::cuda::std::same_as<ex::tag_of_t<Sender>, ex::let_value_t>)
  static auto transform_sender(Sender&&)
  {
    return ex::just(std::string{"hallo"});
  }
};

C2H_TEST("let_value returns a sender", "[adaptors][let_value]")
{
  auto sndr  = ex::let_value(ex::just(), [] {
    return ex::just();
  });
  using Sndr = decltype(sndr);
  static_assert(ex::sender<Sndr>);
  static_assert(ex::get_completion_behavior<Sndr>() == ex::completion_behavior::inline_completion);
  (void) sndr;
}

C2H_TEST("let_value with environment returns a sender", "[adaptors][let_value]")
{
  auto sndr  = ex::let_value(ex::just(), [] {
    return ex::just();
  });
  using Sndr = decltype(sndr);
  static_assert(ex::sender_in<Sndr, ex::env<>>);
  static_assert(ex::get_completion_behavior<Sndr>() == ex::completion_behavior::inline_completion);
  (void) sndr;
}

C2H_TEST("let_value simple example", "[adaptors][let_value]")
{
  bool called{false};
  auto sndr = ex::let_value(ex::just(), [&] {
    called = true;
    return ex::just();
  });
  auto op   = ex::connect(std::move(sndr), checked_value_receiver{});
  ex::start(op);
  // The receiver checks that it's called
  // we also check that the function was invoked
  CHECK(called);
}

C2H_TEST("let_value can be piped", "[adaptors][let_value]")
{
  auto sndr = ex::just() | ex::let_value([] {
                return ex::just();
              });
  (void) sndr;
}

C2H_TEST("let_value returning void can we waited on", "[adaptors][let_value]")
{
  auto sndr = ex::just() | ex::let_value([] {
                return ex::just();
              });
  ex::sync_wait(std::move(sndr));
}

C2H_TEST("let_value can be used to produce values", "[adaptors][let_value]")
{
  auto sndr = ex::just() | ex::let_value([] {
                return ex::just(13);
              });
  wait_for_value(std::move(sndr), 13);
}

C2H_TEST("let_value can be used to transform values", "[adaptors][let_value]")
{
  auto sndr = ex::just(13) | ex::let_value([](int& x) {
                return ex::just(x + 4);
              });
  wait_for_value(std::move(sndr), 17);
}

C2H_TEST("let_value can be used with multiple parameters", "[adaptors][let_value]")
{
  auto sndr = ex::just(3, 0.1415) | ex::let_value([](int& x, double y) {
                return ex::just(x + y);
              });
  wait_for_value(std::move(sndr), 3.1415); // NOLINT(modernize-use-std-numbers)
}

C2H_TEST("let_value can be used to change the sender", "[adaptors][let_value]")
{
  auto sndr = ex::just(13) | ex::let_value([](int& x) {
                return ex::just_error(x + 4);
              });
  auto op   = ex::connect(std::move(sndr), checked_error_receiver{13 + 4});
  ex::start(op);
}

#if _CCCL_HOST_COMPILATION()

auto is_prime(int x) -> bool
{
  if (x > 2 && (x % 2 == 0))
  {
    return false;
  }
  int d = 3;
  while (d * d < x)
  {
    if (x % d == 0)
    {
      return false;
    }
    d += 2;
  }
  return true;
}

C2H_TEST("let_value can be used for composition", "[adaptors][let_value]")
{
  bool called1{false};
  bool called2{false};
  bool called3{false};
  auto f1 = [&](int& x) {
    called1 = true;
    return ex::just(2 * x);
  };
  auto f2 = [&](int& x) {
    called2 = true;
    return ex::just(x + 3);
  };
  auto f3 = [&](int& x) {
    called3 = true;
    if (!is_prime(x))
    {
      throw std::logic_error("not prime");
    }
    return ex::just(x);
  };
  auto sndr = ex::just(13) //
            | ex::let_value(f1) //
            | ex::let_value(f2) //
            | ex::let_value(f3) //
    ;
  wait_for_value(std::move(sndr), 29);
  CHECK(called1);
  CHECK(called2);
  CHECK(called3);
}

C2H_TEST("let_value can throw, and set_error will be called", "[adaptors][let_value]")
{
  auto sndr = ex::just(13) //
            | ex::let_value([](int&) -> decltype(ex::just(0)) {
                throw std::logic_error{"err"};
              });
  auto op = ex::connect(std::move(sndr), checked_error_receiver{std::logic_error{"err"}});
  ex::start(op);
}

C2H_TEST("let_value can be used with just_error", "[adaptors][let_value]")
{
  auto sndr = ex::just_error(std::string{"err"}) //
            | ex::let_value([]() {
                return ex::just(17);
              });
  auto op = ex::connect(std::move(sndr), checked_error_receiver{std::string{"err"}});
  ex::start(op);
}

C2H_TEST("let_value can be used with just_stopped", "[adaptors][let_value]")
{
  auto sndr = ex::just_stopped() | ex::let_value([]() {
                return ex::just(17);
              });
  auto op   = ex::connect(std::move(sndr), checked_stopped_receiver{});
  ex::start(op);
}

C2H_TEST("let_value function is not called on error", "[adaptors][let_value]")
{
  bool called{false};
  error_scheduler<int> sched{-1};
  auto sndr = ex::just(13) //
            | ex::continues_on(sched) //
            | ex::let_value([&](int& x) {
                called = true;
                return ex::just(x + 5);
              });
  auto op = ex::connect(std::move(sndr), checked_error_receiver{-1});
  ex::start(op);
  CHECK_FALSE(called);
}

C2H_TEST("let_value function is not called when cancelled", "[adaptors][let_value]")
{
  bool called{false};
  stopped_scheduler sched;
  auto sndr = ex::just(13) //
            | ex::continues_on(sched) //
            | ex::let_value([&](int& x) {
                called = true;
                return ex::just(x + 5);
              });
  auto op = ex::connect(std::move(sndr), checked_stopped_receiver{});
  ex::start(op);
  CHECK_FALSE(called);
}

C2H_TEST("let_value exposes a parameter that is destructed when the main operation is destructed",
         "[adaptors][let_value]")
{
  // Type that sets into a received boolean when the dtor is called
  struct my_type
  {
    bool* p_called_{nullptr};

    explicit my_type(bool* p_called)
        : p_called_(p_called)
    {}

    my_type(my_type&& rhs)
        : p_called_(rhs.p_called_)
    {
      rhs.p_called_ = nullptr;
    }

    auto operator=(my_type&& rhs) -> my_type&
    {
      if (p_called_)
      {
        *p_called_ = true;
      }
      p_called_     = rhs.p_called_;
      rhs.p_called_ = nullptr;
      return *this;
    }

    ~my_type()
    {
      if (p_called_)
      {
        *p_called_ = true;
      }
    }
  };

  bool param_destructed{false};
  bool fun_called{false};
  impulse_scheduler sched;

  auto sndr = ex::just(my_type(&param_destructed)) //
            | ex::let_value([&](const my_type&) {
                CHECK_FALSE(param_destructed);
                fun_called = true;
                return ex::just(13) | ex::continues_on(sched);
              });

  {
    int res{0};
    auto op = ex::connect(std::move(sndr), proxy_value_receiver{res});
    ex::start(op);
    // The function is called immediately after starting the operation
    CHECK(fun_called);
    // As the returned sender didn't complete yet, the parameter must still be alive
    CHECK_FALSE(param_destructed);
    CHECK(res == 0);

    // Now, tell the scheduler to execute the final operation
    sched.start_next();

    // The parameter is going to be destructed when the op is destructed; it should be valid now
    CHECK_FALSE(param_destructed);
    CHECK(res == 13);
  }

  // At this point everything can be destructed
  CHECK(param_destructed);
}

C2H_TEST("let_value works when changing threads", "[adaptors][let_value]")
{
  ex::thread_context worker;
  cuda::std::atomic<bool> called{false};
  {
    // lunch some work on the worker thread
    auto sndr =
      ex::just(7) //
      | ex::continues_on(worker.get_scheduler()) //
      | ex::let_value([](int& x) {
          return ex::just(x * 2 - 1);
        }) //
      | ex::then([&](int x) {
          CHECK(x == 13);
          called.store(true);
        });

    using Sndr = decltype(sndr);

    static_assert(ex::get_completion_behavior<Sndr>() == ex::completion_behavior::asynchronous);
    ex::start_detached(std::move(sndr));
  }
  worker.join();
  // the work should be executed
  REQUIRE(called);
}

C2H_TEST("let_value has the values_type corresponding to the given values", "[adaptors][let_value]")
{
  check_value_types<types<int>>(ex::just() | ex::let_value([] {
                                  return ex::just(7);
                                }));
  check_value_types<types<double>>(ex::just() | ex::let_value([] {
                                     return ex::just(3.14);
                                   }));
  check_value_types<types<movable>>(ex::just() | ex::let_value([] {
                                      return ex::just(movable{0});
                                    }));
}

C2H_TEST("let_value keeps error_types from input sender", "[adaptors][let_value]")
{
  dummy_scheduler sched1{};
  error_scheduler sched2{::std::exception_ptr{}};
  error_scheduler<int> sched3{43};

  check_error_types<std::exception_ptr>( //
    ex::just() | ex::continues_on(sched1) | ex::let_value([] {
      return ex::just();
    }));
  check_error_types<std::exception_ptr>( //
    ex::just() | ex::continues_on(sched2) | ex::let_value([] {
      return ex::just();
    }));
  check_error_types<int, std::exception_ptr>( //
    ex::just() | ex::continues_on(sched3) | ex::let_value([] {
      return ex::just();
    }));

  // NOT YET SUPPORTED
  // check_error_types<>( //
  //   ex::just() | ex::continues_on(sched1) | ex::let_value([]_CCCL_HOST_DEVICE() noexcept {
  //     return ex::just();
  //   }));
  // check_error_types<std::exception_ptr>( //
  //   ex::just() | ex::continues_on(sched2) | ex::let_value([]_CCCL_HOST_DEVICE() noexcept {
  //     return ex::just();
  //   }));
  // check_error_types<int>( //
  //   ex::just() | ex::continues_on(sched3) | ex::let_value([]_CCCL_HOST_DEVICE() noexcept {
  //     return ex::just();
  //   }));
}

C2H_TEST("let_value keeps sends_stopped from input sender", "[adaptors][let_value]")
{
  dummy_scheduler sched1{};
  stopped_scheduler sched2{};

  check_sends_stopped<false>( //
    ex::just() | ex::continues_on(sched1) | ex::let_value([] {
      return ex::just();
    }));
  check_sends_stopped<true>( //
    ex::just() | ex::continues_on(sched2) | ex::let_value([] {
      return ex::just();
    }));
}

C2H_TEST("let_value can be customized", "[adaptors][let_value]")
{
  auto attrs = ex::prop{ex::get_completion_domain<ex::set_value_t>, let_value_test_domain{}};

  // The customization will return a different value
  auto sndr = ex::just(std::string{"hello"}) //
            | ex::write_attrs(attrs) //
            | ex::let_value([](std::string& x) {
                return ex::just(x + ", world");
              });
  wait_for_value(std::move(sndr), std::string{"hallo"});
}

C2H_TEST("let_value can nest", "[adaptors][let_value]")
{
  auto work = ex::just(2) //
            | ex::let_value([](int x) { //
                return ex::just() //
                     | ex::let_value([=] { //
                         return ex::just(x);
                       });
              });
  wait_for_value(std::move(work), 2);
}

constexpr struct test_query_t : ex::forwarding_query_t
{
  template <class Env>
  _CCCL_API constexpr auto operator()(const Env& env) const noexcept -> decltype(env.query(*this))
  {
    return env.query(*this);
  }
} test_query{};

C2H_TEST("let_value works when the function returns a dependent sender", "[adaptors][let_value]")
{
  auto sndr     = ex::write_env(ex::just() | ex::let_value([] {
                              return ex::read_env(test_query);
                            }),
                            ex::prop{test_query, 42});
  auto [result] = ex::sync_wait(std::move(sndr)).value();
  CUDAX_CHECK(result == 42);
}

// NOT YET SUPPORTED
// struct bad_receiver
// {
//   using receiver_concept = ex::receiver_t;

//   bad_receiver(bool& completed) noexcept
//       : completed_{completed}
//   {}

//   bad_receiver(bad_receiver&& other) noexcept(false) // BAD!
//       : completed_(other.completed_)
//   {}

//   void set_value() noexcept
//   {
//     completed_ = true;
//   }

//   bool& completed_;
// };

// C2H_TEST("let_value does not add std::exception_ptr even if the receiver is bad", "[adaptors][let_value]")
// {
//   auto sndr = ex::let_value(ex::just(), []_CCCL_HOST_DEVICE() noexcept {
//     return ex::just();
//   });
//   check_error_types<>(sndr);
//   bool completed{false};
//   auto op = ex::connect(std::move(sndr), bad_receiver{completed}); // should compile
//   ex::start(op);
//   CHECK(completed);
// }

#endif // _CCCL_HOST_COMPILATION()

#if !_CCCL_CUDA_COMPILER(NVCC)
// This example causes nvcc to segfault
struct let_value_test_domain2
{};

C2H_TEST("let_value predecessor's domain is accessible via the receiver connected to the secondary sender",
         "[adaptors][let_value]")
{
  auto attrs  = ex::prop{ex::get_completion_domain<ex::set_value_t>, let_value_test_domain2{}};
  using Sndr2 = decltype(ex::read_env(ex::get_domain));

  auto sndr = ex::just() //
            | ex::write_attrs(attrs) //
            | ex::let_value([]() noexcept -> Sndr2 {
                return ex::read_env(ex::get_domain);
              });
  auto [result] = ex::sync_wait(std::move(sndr)).value();
  static_assert(::cuda::std::is_same_v<decltype(result), let_value_test_domain2>);
  (void) result;
}
#endif // !_CCCL_CUDA_COMPILER(NVCC)

C2H_TEST("let_value has the correct completion domain", "[adaptors][let_value]")
{
  auto attrs = ex::prop{ex::get_completion_domain<ex::set_value_t>, let_value_test_domain{}};
  auto sndr  = ex::just() | ex::let_value([=] {
                return ex::write_attrs(ex::just(), attrs);
              });
  auto dom   = ex::get_completion_domain<ex::set_value_t>(ex::get_env(sndr));
  static_assert(::cuda::std::is_same_v<decltype(dom), let_value_test_domain>);
}

} // namespace
