//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/stream>

#include <testing.cuh>
#include <utility.cuh>

namespace
{
namespace test
{
cuda::event_ref fn_takes_event_ref(cuda::event_ref ref)
{
  return ref;
}
} // namespace test
} // namespace

static_assert(!::cuda::std::is_default_constructible_v<cuda::event_ref>);
static_assert(!::cuda::std::is_default_constructible_v<cuda::event>);
static_assert(!::cuda::std::is_default_constructible_v<cuda::timed_event>);

C2H_CCCLRT_TEST("can construct an event_ref from a cudaEvent_t", "[event]")
{
  cuda::__ensure_current_context guard(cuda::device_ref{0});
  ::cudaEvent_t ev;
  CCCLRT_REQUIRE(::cudaEventCreate(&ev) == ::cudaSuccess);
  cuda::event_ref ref(ev);
  CCCLRT_REQUIRE(ref.get() == ev);
  CCCLRT_REQUIRE(!!ref);
  // test implicit conversion from cudaEvent_t:
  cuda::event_ref ref2 = ::test::fn_takes_event_ref(ev);
  CCCLRT_REQUIRE(ref2.get() == ev);
  CCCLRT_REQUIRE(::cudaEventDestroy(ev) == ::cudaSuccess);
  // test an empty event_ref:
  cuda::event_ref ref3(::cudaEvent_t{});
  CCCLRT_REQUIRE(ref3.get() == ::cudaEvent_t{});
  CCCLRT_REQUIRE(!ref3);
}

C2H_CCCLRT_TEST("can copy construct an event_ref and compare for equality", "[event]")
{
  cuda::__ensure_current_context guard(cuda::device_ref{0});
  ::cudaEvent_t ev;
  CCCLRT_REQUIRE(::cudaEventCreate(&ev) == ::cudaSuccess);
  const cuda::event_ref ref(ev);
  const cuda::event_ref ref2 = ref;
  CCCLRT_REQUIRE(ref2 == ref);
  CCCLRT_REQUIRE(!(ref != ref2));
  CCCLRT_REQUIRE((ref ? true : false)); // test contextual convertibility to bool
  CCCLRT_REQUIRE(!!ref);
  CCCLRT_REQUIRE(::cudaEvent_t{} != ref);
  CCCLRT_REQUIRE(::cudaEventDestroy(ev) == ::cudaSuccess);
  // copy from empty event_ref:
  const cuda::event_ref ref3(::cudaEvent_t{});
  const cuda::event_ref ref4 = ref3;
  CCCLRT_REQUIRE(ref4 == ref3);
  CCCLRT_REQUIRE(!(ref3 != ref4));
  CCCLRT_REQUIRE(!ref4);
}

C2H_CCCLRT_TEST("can use event_ref to record and wait on an event", "[event]")
{
  cuda::__ensure_current_context guard(cuda::device_ref{0});
  ::cudaEvent_t ev;
  CCCLRT_REQUIRE(::cudaEventCreate(&ev) == ::cudaSuccess);
  const cuda::event_ref ref(ev);

  test::pinned<int> i(0);
  cuda::stream stream{cuda::device_ref{0}};
  ::test::launch_kernel_single_thread(stream, ::test::assign_42{}, i.get());
  ref.record(stream);
  ref.sync();
  CCCLRT_REQUIRE(ref.is_done());
  CCCLRT_REQUIRE(*i == 42);

  stream.sync();
  CCCLRT_REQUIRE(::cudaEventDestroy(ev) == ::cudaSuccess);
}

C2H_CCCLRT_TEST("can construct an event with a stream_ref", "[event]")
{
  cuda::stream stream{cuda::device_ref{0}};
  cuda::event ev(static_cast<cuda::stream_ref>(stream));
  CCCLRT_REQUIRE(ev.get() != ::cudaEvent_t{});
}

C2H_CCCLRT_TEST("can construct an event with a device_ref", "[event]")
{
  cuda::device_ref device{0};
  cuda::event ev(device);
  CCCLRT_REQUIRE(ev.get() != ::cudaEvent_t{});
  cuda::stream stream{device};
  ev.record(stream);
  ev.sync();
  CCCLRT_REQUIRE(ev.is_done());
}

C2H_CCCLRT_TEST("can wait on an event", "[event]")
{
  cuda::stream stream{cuda::device_ref{0}};
  ::test::pinned<int> i(0);
  ::test::launch_kernel_single_thread(stream, ::test::assign_42{}, i.get());
  cuda::event ev(stream);
  ev.sync();
  CCCLRT_REQUIRE(ev.is_done());
  CCCLRT_REQUIRE(*i == 42);
  stream.sync();
}

C2H_CCCLRT_TEST("can take the difference of two timed_event objects", "[event]")
{
  cuda::stream stream{cuda::device_ref{0}};
  ::test::pinned<int> i(0);
  cuda::timed_event start(stream);
  ::test::launch_kernel_single_thread(stream, ::test::assign_42{}, i.get());
  cuda::timed_event end(stream);
  end.sync();
  CCCLRT_REQUIRE(end.is_done());
  CCCLRT_REQUIRE(*i == 42);
  auto elapsed = end - start;
  CCCLRT_REQUIRE(elapsed.count() >= 0);
  STATIC_REQUIRE(::cuda::std::is_same_v<decltype(elapsed), ::cuda::std::chrono::nanoseconds>);
  stream.sync();
}

C2H_CCCLRT_TEST("can observe the event in not ready state", "[event]")
{
  ::test::pinned<int> i(0);
  ::cuda::atomic_ref atomic_i(*i);

  cuda::stream stream{cuda::device_ref{0}};

  ::test::launch_kernel_single_thread(stream, ::test::spin_until_80{}, i.get());
  cuda::event ev(stream);
  CCCLRT_REQUIRE(!ev.is_done());
  atomic_i.store(80);
  ev.sync();
  CCCLRT_REQUIRE(ev.is_done());
}
