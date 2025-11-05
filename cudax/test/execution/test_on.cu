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

#include "testing.cuh"

namespace ex = cudax::execution;

__host__ __device__ bool _on_device() noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, //
                    ({ return false; }),
                    ({ return true; }));
}

auto const main_thread_id = ::std::this_thread::get_id();

void simple_start_on_thread_test()
{
  ex::thread_context ctx;
  auto sch  = ctx.get_scheduler();
  auto sndr = ex::on(sch, ex::just() | ex::then([] {
                            CUDAX_CHECK(::std::this_thread::get_id() != main_thread_id);
                          }))
            | ex::then([]() -> int {
                CUDAX_CHECK(::std::this_thread::get_id() == main_thread_id);
                return 42;
              });
  auto [result] = ex::sync_wait(std::move(sndr)).value();
  CUDAX_CHECK(result == 42);
}

void simple_continue_on_thread_test()
{
  ex::thread_context ctx;
  auto sch  = ctx.get_scheduler();
  auto sndr = ex::just() | ex::on(sch, ex::then([] {
                                    CUDAX_CHECK(::std::this_thread::get_id() != main_thread_id);
                                  }))
            | ex::then([]() -> int {
                CUDAX_CHECK(::std::this_thread::get_id() == main_thread_id);
                return 42;
              });
  auto [result] = ex::sync_wait(std::move(sndr)).value();
  CUDAX_CHECK(result == 42);
}

void simple_start_on_stream_test()
{
  cudax::stream str{cuda::device_ref(0)};
  auto sch  = cudax::stream_ref{str};
  auto sndr = ex::on(sch, ex::just(42) | ex::then([] __host__ __device__(int i) noexcept -> int {
                            return _on_device() ? i : -i;
                          }))
            | ex::then([] __host__ __device__(int i) noexcept -> int {
                return _on_device() ? -1 : i;
              });
  auto [result] = ex::sync_wait(std::move(sndr)).value();
  CUDAX_CHECK(result == 42);
}

void simple_continue_on_stream_test()
{
  cudax::stream str{cuda::device_ref(0)};
  auto sch  = cudax::stream_ref{str};
  auto sndr = ex::just(42) | ex::on(sch, ex::then([] __host__ __device__(int i) noexcept -> int {
                                      return _on_device() ? i : -i;
                                    }))
            | ex::then([] __host__ __device__(int i) noexcept -> int {
                return _on_device() ? -1 : i;
              });
  auto [result] = ex::sync_wait(std::move(sndr)).value();
  CUDAX_CHECK(result == 42);
}

void test_continues_on_updates_env()
{
  ex::thread_context ctx;
  auto sch  = ctx.get_scheduler();
  auto sndr = ex::just() | ex::on(sch, ex::let_value([] {
                                    return ex::read_env(ex::get_scheduler);
                                  }))
            | ex::then([](auto sch2) -> int {
                STATIC_REQUIRE(cuda::std::same_as<decltype(sch2), decltype(sch)>);
                return 42;
              });
  auto [result] = ex::sync_wait(std::move(sndr)).value();
  CUDAX_CHECK(result == 42);
}

namespace
{
C2H_TEST("simple on(sch, sndr) thread test", "[on]")
{
  simple_start_on_thread_test();
}

C2H_TEST("simple on(sndr, sch, closure) thread test", "[on]")
{
  simple_continue_on_thread_test();
}

C2H_TEST("simple on(sch, sndr) stream test", "[on][stream]")
{
  simple_start_on_stream_test();
}

C2H_TEST("simple on(sndr, sch, closure) stream test", "[on][stream]")
{
  simple_continue_on_stream_test();
}

C2H_TEST("test that on(sndr, sch, closure) updates the env for closure", "[on][stream]")
{
  test_continues_on_updates_env();
}
} // namespace
