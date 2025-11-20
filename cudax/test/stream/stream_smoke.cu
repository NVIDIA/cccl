//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/launch.cuh>
#include <cuda/experimental/stream.cuh>

#include <testing.cuh>
#include <utility.cuh>

C2H_CCCLRT_TEST("Can create a stream and launch work into it", "[stream]")
{
  cudax::stream str{cuda::device_ref{0}};
  ::test::pinned<int> i(0);
  cudax::launch(str, ::test::one_thread_dims, ::test::assign_42{}, i.get());
  str.sync();
  CUDAX_REQUIRE(*i == 42);
}

C2H_CCCLRT_TEST("From native handle", "[stream]")
{
  cuda::__ensure_current_context guard(cuda::device_ref{0});
  cudaStream_t handle;
  CUDART(cudaStreamCreate(&handle));
  {
    auto stream = cudax::stream::from_native_handle(handle);

    ::test::pinned<int> i(0);
    cudax::launch(stream, ::test::one_thread_dims, ::test::assign_42{}, i.get());
    stream.sync();
    CUDAX_REQUIRE(*i == 42);
    (void) stream.release();
  }
  CUDART(cudaStreamDestroy(handle));
}

template <typename StreamType>
void add_dependency_test(const StreamType& waiter, const StreamType& waitee)
{
  CUDAX_REQUIRE(waiter != waitee);

  auto verify_dependency = [&](const auto& insert_dependency) {
    ::test::pinned<int> i(0);
    ::cuda::atomic_ref atomic_i(*i);

    cudax::launch(waitee, ::test::one_thread_dims, ::test::spin_until_80{}, i.get());
    cudax::launch(waitee, ::test::one_thread_dims, ::test::assign_42{}, i.get());
    insert_dependency();
    cudax::launch(waiter, ::test::one_thread_dims, ::test::verify_42{}, i.get());
    CUDAX_REQUIRE(atomic_i.load() != 42);
    CUDAX_REQUIRE(!waiter.is_done());
    atomic_i.store(80);
    waiter.sync();
    waitee.sync();
  };

  SECTION("Stream wait declared event")
  {
    verify_dependency([&]() {
      cuda::event ev(waitee);
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

C2H_CCCLRT_TEST("Can add dependency into a stream", "[stream]")
{
  cudax::stream waiter{cuda::device_ref{0}}, waitee{cuda::device_ref{0}};

  add_dependency_test<cudax::stream>(waiter, waitee);
  add_dependency_test<cudax::stream_ref>(waiter, waitee);
}

C2H_CCCLRT_TEST("Stream priority", "[stream]")
{
  cudax::stream stream_default_prio{cuda::device_ref{0}};
  CUDAX_REQUIRE(stream_default_prio.priority() == cudax::stream::default_priority);

  auto priority = cudax::stream::default_priority - 1;
  cudax::stream stream{cuda::device_ref{0}, priority};
  CUDAX_REQUIRE(stream.priority() == priority);
}

C2H_CCCLRT_TEST("Stream get device", "[stream]")
{
  cudax::stream dev0_stream(cuda::device_ref{0});
  CUDAX_REQUIRE(dev0_stream.device() == 0);

  cudax::__ensure_current_device guard(cuda::device_ref{*std::prev(cuda::devices.end())});
  cudaStream_t stream_handle;
  CUDART(cudaStreamCreate(&stream_handle));
  auto stream_cudart = cudax::stream::from_native_handle(stream_handle);
  CUDAX_REQUIRE(stream_cudart.device() == *std::prev(cuda::devices.end()));
  auto stream_ref_cudart = cudax::stream_ref(stream_handle);
  CUDAX_REQUIRE(stream_ref_cudart.device() == *std::prev(cuda::devices.end()));

  INFO("Can create a side stream using logical device");
  {
    if (test::cuda_driver_version() >= 12050)
    {
      auto ldev = dev0_stream.logical_device();
      CUDAX_REQUIRE(ldev.kind() == cudax::logical_device::kinds::device);
      cudax::stream side_stream(ldev);
      CUDAX_REQUIRE(side_stream.device() == dev0_stream.device());
    }
  }
}

C2H_CCCLRT_TEST("Stream ID", "[stream]")
{
  STATIC_REQUIRE(cuda::std::is_same_v<unsigned long long, cuda::std::underlying_type_t<cuda::stream_id>>);
  STATIC_REQUIRE(cuda::std::is_same_v<cuda::stream_id, decltype(cuda::std::declval<cudax::stream_ref>().id())>);

  cudax::stream stream1{cuda::device_ref{0}};
  cudax::stream stream2{cuda::device_ref{0}};

  // Test that id() returns a valid ID
  auto id1 = stream1.id();
  auto id2 = stream2.id();

  // Test that different streams have different IDs
  CUDAX_REQUIRE(id1 != id2);

  // Test that the same stream returns the same ID when called multiple times
  CUDAX_REQUIRE(stream1.id() == id1);
  CUDAX_REQUIRE(stream2.id() == id2);

  {
    // Test that stream_ref also supports id()
    // NULL stream needs a device to be set
    cudax::__ensure_current_device guard(cuda::device_ref{0});
    cudax::stream_ref ref1(cudaStream_t{});
    cudax::stream_ref ref2(stream1);

    CUDAX_REQUIRE(ref1.id() != ref2.id());
    CUDAX_REQUIRE(ref2.id() == id1);
  }
}
