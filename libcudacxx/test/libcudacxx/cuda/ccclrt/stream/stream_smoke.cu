//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/devices>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <testing.cuh>

C2H_CCCLRT_TEST("Can create a stream and launch work into it", "[stream]")
{
  cuda::stream str{cuda::device_ref{0}};
  ::test::pinned<int> i(0);
  ::test::launch_kernel_single_thread(str, ::test::assign_42{}, i.get());
  str.sync();
  CCCLRT_REQUIRE(*i == 42);
}

C2H_CCCLRT_TEST("From native handle", "[stream]")
{
  cuda::__ensure_current_context guard(cuda::device_ref{0});
  cudaStream_t handle;
  CUDART(cudaStreamCreate(&handle));
  {
    auto stream = cuda::stream::from_native_handle(handle);

    ::test::pinned<int> i(0);
    ::test::launch_kernel_single_thread(stream, ::test::assign_42{}, i.get());
    stream.sync();
    CCCLRT_REQUIRE(*i == 42);
    (void) stream.release();
  }
  CUDART(cudaStreamDestroy(handle));
}

template <typename StreamType>
void add_dependency_test(const StreamType& waiter, const StreamType& waitee)
{
  CCCLRT_REQUIRE(waiter != waitee);

  auto verify_dependency = [&](const auto& insert_dependency) {
    ::test::pinned<int> i(0);
    ::cuda::atomic_ref atomic_i(*i);

    ::test::launch_kernel_single_thread(waitee, ::test::spin_until_80{}, i.get());
    ::test::launch_kernel_single_thread(waitee, ::test::assign_42{}, i.get());
    insert_dependency();
    ::test::launch_kernel_single_thread(waiter, ::test::verify_42{}, i.get());
    CCCLRT_REQUIRE(atomic_i.load() != 42);
    CCCLRT_REQUIRE(!waiter.is_done());
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
  cuda::stream waiter{cuda::device_ref{0}}, waitee{cuda::device_ref{0}};

  add_dependency_test<cuda::stream>(waiter, waitee);
  add_dependency_test<cuda::stream_ref>(waiter, waitee);
}

C2H_CCCLRT_TEST("Stream priority", "[stream]")
{
  cuda::stream stream_default_prio{cuda::device_ref{0}};
  CCCLRT_REQUIRE(stream_default_prio.priority() == cuda::stream::default_priority);

  auto priority = cuda::stream::default_priority - 1;
  cuda::stream stream{cuda::device_ref{0}, priority};
  CCCLRT_REQUIRE(stream.priority() == priority);
}

C2H_CCCLRT_TEST("Stream get device", "[stream]")
{
  cuda::stream dev0_stream(cuda::device_ref{0});
  CCCLRT_REQUIRE(dev0_stream.device() == 0);

  cuda::__ensure_current_context guard(cuda::device_ref{*std::prev(cuda::devices.end())});
  cudaStream_t stream_handle;
  CUDART(cudaStreamCreate(&stream_handle));
  auto stream_cudart = cuda::stream::from_native_handle(stream_handle);
  CCCLRT_REQUIRE(stream_cudart.device() == *std::prev(cuda::devices.end()));
  auto stream_ref_cudart = cuda::stream_ref(stream_handle);
  CCCLRT_REQUIRE(stream_ref_cudart.device() == *std::prev(cuda::devices.end()));
}

C2H_CCCLRT_TEST("Stream ID", "[stream]")
{
  STATIC_REQUIRE(cuda::std::is_same_v<unsigned long long, cuda::std::underlying_type_t<cuda::stream_id>>);
  STATIC_REQUIRE(cuda::std::is_same_v<cuda::stream_id, decltype(cuda::std::declval<cuda::stream_ref>().id())>);

  cuda::stream stream1{cuda::device_ref{0}};
  cuda::stream stream2{cuda::device_ref{0}};

  // Test that id() returns a valid ID
  auto id1 = stream1.id();
  auto id2 = stream2.id();

  // Test that different streams have different IDs
  CCCLRT_REQUIRE(id1 != id2);

  // Test that the same stream returns the same ID when called multiple times
  CCCLRT_REQUIRE(stream1.id() == id1);
  CCCLRT_REQUIRE(stream2.id() == id2);

  {
    // Test that stream_ref also supports id()
    // NULL stream needs a device to be set
    cuda::__ensure_current_context guard(cuda::device_ref{0});
    cuda::stream_ref ref1(::cudaStream_t{});
    cuda::stream_ref ref2(stream1);

    CCCLRT_REQUIRE(ref1.id() != ref2.id());
    CCCLRT_REQUIRE(ref2.id() == id1);
  }
}

C2H_CCCLRT_TEST("Invalid stream", "[stream]")
{
  // 1. Test the signature
  STATIC_REQUIRE(cuda::std::is_same_v<const cuda::invalid_stream_t, decltype(cuda::invalid_stream)>);

  // 2. Test explicit construction of stream_ref from invalid_stream
  STATIC_REQUIRE(cuda::std::is_constructible_v<cuda::stream_ref, cuda::invalid_stream_t>);
  STATIC_REQUIRE(!cuda::std::is_convertible_v<cuda::invalid_stream_t, cuda::stream_ref>);
  {
    cuda::stream_ref stream{cuda::invalid_stream};
    CCCLRT_REQUIRE(stream.get() == (cudaStream_t) (~0ull));
  }

  // 3. Test stream_ref comparisons
  {
    cuda::stream_ref valid_stream{(cudaStream_t) (123ull)};
    cuda::stream_ref invalid_stream{cuda::invalid_stream};

    CCCLRT_REQUIRE(!(valid_stream == cuda::invalid_stream));
    CCCLRT_REQUIRE(invalid_stream == cuda::invalid_stream);
    CCCLRT_REQUIRE(!(cuda::invalid_stream == valid_stream));
    CCCLRT_REQUIRE(cuda::invalid_stream == invalid_stream);

    CCCLRT_REQUIRE(valid_stream != cuda::invalid_stream);
    CCCLRT_REQUIRE(!(invalid_stream != cuda::invalid_stream));
    CCCLRT_REQUIRE(cuda::invalid_stream != valid_stream);
    CCCLRT_REQUIRE(!(cuda::invalid_stream != invalid_stream));
  }
}
