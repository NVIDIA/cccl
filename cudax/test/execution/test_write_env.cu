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

#include "common/checked_receiver.cuh"
#include "common/utility.cuh"
#include "testing.cuh"

namespace ex = cuda::experimental::execution;

namespace
{
// Custom domain for testing
struct my_domain
{};

// Simple test query
struct query1_t : ex::forwarding_query_t
{
  template <class Env>
  _CCCL_API auto operator()(const Env& env) const noexcept -> decltype(env.query(*this))
  {
    return env.query(*this);
  }
};

inline constexpr auto query1 = query1_t{};
} // namespace

C2H_TEST("write_env basic functionality", "[write_env][adaptors]")
{
  // Test that write_env wraps a sender and adds environment information
  auto env  = ex::prop{query1, 42};
  auto sndr = ex::write_env(ex::just(42), env);

  auto op = ex::connect(std::move(sndr), checked_value_receiver{42});
  ex::start(op);
  // the receiver will check the result
}

C2H_TEST("write_env pipe syntax", "[write_env][adaptors]")
{
  // Test that write_env supports pipe syntax (sndr | write_env(env))
  auto env  = ex::prop{query1, 42};
  auto sndr = ex::just(42) | ex::write_env(env);

  auto op = ex::connect(std::move(sndr), checked_value_receiver{42});
  ex::start(op);
  // the receiver will check the result
}

#if _CCCL_HOST_COMPILATION()
C2H_TEST("write_env updates the receiver's environment", "[write_env][adaptors]")
{
  auto sndr = ex::just() | ex::let_value([]() {
                return ex::read_env(query1);
              })
            | ex::write_env(ex::prop{query1, 42});

  auto [result] = ex::sync_wait(std::move(sndr)).value();
  CUDAX_CHECK(result == 42);
}

struct fake_allocator
{};

C2H_TEST("write_env does not hide the receiver's environment", "[write_env][adaptors]")
{
  auto sndr = ex::just() | ex::let_value([]() {
                return ex::read_env(ex::get_allocator);
              })
            | ex::write_env(ex::prop{query1, 42});

  auto env      = ex::prop{ex::get_allocator, fake_allocator{}};
  auto [result] = ex::sync_wait(std::move(sndr), env).value();
  STATIC_REQUIRE(std::is_same_v<decltype(result), fake_allocator>);
}

C2H_TEST("write_env prefers the passed env over the receiver's env", "[write_env][adaptors]")
{
  auto sndr = ex::just() | ex::let_value([]() {
                return ex::read_env(query1);
              })
            | ex::write_env(ex::prop{query1, 42});

  auto env      = ex::prop{query1, 100};
  auto [result] = ex::sync_wait(std::move(sndr), env).value();
  CUDAX_CHECK(result == 42);
}
#endif // _CCCL_HOST_COMPILATION()
