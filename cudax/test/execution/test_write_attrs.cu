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
#include "testing.cuh" // IWYU pragma: keep

namespace
{
struct my_domain
{};

C2H_TEST("basic test of write_attrs", "[write_attrs]")
{
  auto sndr = cudax_async::just(42) | cudax_async::write_attrs(cudax_async::prop{cudax_async::get_domain, my_domain{}});
  [[maybe_unused]] auto domain = cudax_async::get_domain(cudax_async::get_env(sndr));
  STATIC_REQUIRE(std::is_same_v<decltype(domain), my_domain>);

  // Check that the sender can be connected and started
  auto op = cudax_async::connect(std::move(sndr), checked_value_receiver{42});
  cudax_async::start(op);
}
} // namespace
