//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/event.cuh>
#include <cuda/experimental/stream.cuh>

#include <catch2/catch.hpp>
#include <utility.cuh>

namespace
{
namespace test
{
cudax::event_ref fn_takes_event_ref(cudax::event_ref ref)
{
  return ref;
}
} // namespace test
} // namespace

static_assert(!_CUDA_VSTD::is_default_constructible_v<cudax::event_ref>);
static_assert(!_CUDA_VSTD::is_default_constructible_v<cudax::event>);
static_assert(!_CUDA_VSTD::is_default_constructible_v<cudax::timed_event>);

TEST_CASE("can construct an event_ref from a cudaEvent_t", "[event]")
{
  ::cudaEvent_t ev;
  CUDAX_REQUIRE(::cudaEventCreate(&ev) == ::cudaSuccess);
  cudax::event_ref ref(ev);
  CUDAX_REQUIRE(ref.get() == ev);
  CUDAX_REQUIRE(!!ref);
  // test implicit converstion from cudaEvent_t:
  cudax::event_ref ref2 = ::test::fn_takes_event_ref(ev);
  CUDAX_REQUIRE(ref2.get() == ev);
  CUDAX_REQUIRE(::cudaEventDestroy(ev) == ::cudaSuccess);
  // test an empty event_ref:
  cudax::event_ref ref3(::cudaEvent_t{});
  CUDAX_REQUIRE(ref3.get() == ::cudaEvent_t{});
  CUDAX_REQUIRE(!ref3);
}

TEST_CASE("can copy construct an event_ref and compare for equality", "[event]")
{
  ::cudaEvent_t ev;
  CUDAX_REQUIRE(::cudaEventCreate(&ev) == ::cudaSuccess);
  const cudax::event_ref ref(ev);
  const cudax::event_ref ref2 = ref;
  CUDAX_REQUIRE(ref2 == ref);
  CUDAX_REQUIRE(!(ref != ref2));
  CUDAX_REQUIRE((ref ? true : false)); // test contextual convertibility to bool
  CUDAX_REQUIRE(!!ref);
  CUDAX_REQUIRE(::cudaEvent_t{} != ref);
  CUDAX_REQUIRE(::cudaEventDestroy(ev) == ::cudaSuccess);
  // copy from empty event_ref:
  const cudax::event_ref ref3(::cudaEvent_t{});
  const cudax::event_ref ref4 = ref3;
  CUDAX_REQUIRE(ref4 == ref3);
  CUDAX_REQUIRE(!(ref3 != ref4));
  CUDAX_REQUIRE(!ref4);
}

TEST_CASE("can use event_ref to record and wait on an event", "[event]")
{
  ::cudaEvent_t ev;
  CUDAX_REQUIRE(::cudaEventCreate(&ev) == ::cudaSuccess);
  const cudax::event_ref ref(ev);

  test::managed<int> i(0);
  cudax::stream stream;
  cudax::launch(stream, ::test::one_thread_dims, ::test::assign_42{}, i.get());
  ref.record(stream);
  ref.wait();
  CUDAX_REQUIRE(ref.is_done());
  CUDAX_REQUIRE(*i == 42);

  stream.wait();
  CUDAX_REQUIRE(::cudaEventDestroy(ev) == ::cudaSuccess);
}

TEST_CASE("can construct an event with a stream_ref", "[event]")
{
  cudax::stream stream;
  cudax::event ev(static_cast<cuda::stream_ref>(stream));
  CUDAX_REQUIRE(ev.get() != ::cudaEvent_t{});
}

TEST_CASE("can wait on an event", "[event]")
{
  cudax::stream stream;
  ::test::managed<int> i(0);
  cudax::launch(stream, ::test::one_thread_dims, ::test::assign_42{}, i.get());
  cudax::event ev(stream);
  ev.wait();
  CUDAX_REQUIRE(ev.is_done());
  CUDAX_REQUIRE(*i == 42);
  stream.wait();
}

TEST_CASE("can take the difference of two timed_event objects", "[event]")
{
  cudax::stream stream;
  ::test::managed<int> i(0);
  cudax::timed_event start(stream);
  cudax::launch(stream, ::test::one_thread_dims, ::test::assign_42{}, i.get());
  cudax::timed_event end(stream);
  end.wait();
  CUDAX_REQUIRE(end.is_done());
  CUDAX_REQUIRE(*i == 42);
  auto elapsed = end - start;
  CUDAX_REQUIRE(elapsed.count() >= 0);
  STATIC_REQUIRE(_CUDA_VSTD::is_same_v<decltype(elapsed), _CUDA_VSTD::chrono::nanoseconds>);
  stream.wait();
}

TEST_CASE("can observe the event in not ready state", "[event]")
{
  ::test::managed<int> i(0);
  ::cuda::atomic_ref atomic_i(*i);

  cudax::stream stream;

  cudax::launch(stream, ::test::one_thread_dims, ::test::spin_until_80{}, i.get());
  cudax::event ev(stream);
  CUDAX_REQUIRE(!ev.is_done());
  atomic_i.store(80);
  ev.wait();
  CUDAX_REQUIRE(ev.is_done());
}
