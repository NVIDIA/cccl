//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/launch.cuh>
#include <cuda/experimental/stream.cuh>

#include <catch2/catch.hpp>
#include <utility.cuh>

constexpr auto one_thread_dims = cudax::make_hierarchy(cudax::block_dims<1>(), cudax::grid_dims<1>());

TEST_CASE("Can create a stream and launch work into it", "[stream]")
{
  cudax::stream str;
  ::test::managed<int> i(0);
  cudax::launch(str, one_thread_dims, ::test::assign_42{}, i.get());
  str.wait();
  CUDAX_REQUIRE(*i == 42);
}

TEST_CASE("From native handle", "[stream]")
{
  cudaStream_t handle;
  CUDART(cudaStreamCreate(&handle));
  {
    auto stream = cudax::stream::from_native_handle(handle);

    ::test::managed<int> i(0);
    cudax::launch(stream, one_thread_dims, ::test::assign_42{}, i.get());
    stream.wait();
    CUDAX_REQUIRE(*i == 42);
    (void) stream.release();
  }
  CUDART(cudaStreamDestroy(handle));
}

TEST_CASE("Can add dependency into a stream", "[stream]")
{
  cudax::stream waiter, waitee;
  CUDAX_REQUIRE(waiter != waitee);

  auto verify_dependency = [&](const auto& insert_dependency) {
    ::test::managed<int> i(0);
    ::cuda::atomic_ref atomic_i(*i);

    cudax::launch(waitee, one_thread_dims, ::test::spin_until_80{}, i.get());
    cudax::launch(waitee, one_thread_dims, ::test::assign_42{}, i.get());
    insert_dependency();
    cudax::launch(waiter, one_thread_dims, ::test::verify_42{}, i.get());
    CUDAX_REQUIRE(atomic_i.load() != 42);
    CUDAX_REQUIRE(!waiter.ready());
    atomic_i.store(80);
    waiter.wait();
    waitee.wait();
  };

  SECTION("Stream wait declared event")
  {
    verify_dependency([&]() {
      cudax::event ev(waitee);
      waiter.wait(ev);
    });
  }

  SECTION("Stream wait returned event")
  {
    verify_dependency([&]() {
      auto ev = waitee.record_event();
      waiter.wait(ev);
    });
  }

  SECTION("Stream wait returned timed event")
  {
    verify_dependency([&]() {
      auto ev = waitee.record_timed_event();
      waiter.wait(ev);
    });
  }

  SECTION("Stream wait stream")
  {
    verify_dependency([&]() {
      waiter.wait(waitee);
    });
  }
}

TEST_CASE("Stream priority", "[stream]")
{
  cudax::stream stream_default_prio;
  CUDAX_REQUIRE(stream_default_prio.priority() == cudax::stream::default_priority);

  auto priority = cudax::stream::default_priority - 1;
  cudax::stream stream(0, priority);
  CUDAX_REQUIRE(stream.priority() == priority);
}
