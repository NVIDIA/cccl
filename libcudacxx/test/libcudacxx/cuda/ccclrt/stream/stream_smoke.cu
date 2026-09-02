//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/devices>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <functional>

#include <testing.cuh>

#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#if TEST_HAS_EXCEPTIONS()
namespace
{
#  define CCCLRT_DEFAULT_STREAM_INVALID_CONTEXT_MATCHER()                                            \
    Catch::Matchers::MessageMatches(                                                                 \
      Catch::Matchers::ContainsSubstring("The NULL/default stream requires a current CUDA context.") \
      && Catch::Matchers::ContainsSubstring("Set the current device or make a CUDA context current"))

#  define CCCLRT_REQUIRE_THROWS_DEFAULT_STREAM_INVALID_CONTEXT(EXPR) \
    REQUIRE_THROWS_MATCHES(EXPR, cuda::cuda_error, CCCLRT_DEFAULT_STREAM_INVALID_CONTEXT_MATCHER())

void check_default_stream_invalid_context_message(::CUstream native_stream)
{
  cuda::stream_ref stream{native_stream};

  CCCLRT_REQUIRE_THROWS_DEFAULT_STREAM_INVALID_CONTEXT(stream.sync());
  CCCLRT_REQUIRE_THROWS_DEFAULT_STREAM_INVALID_CONTEXT((void) stream.is_done());
}
} // namespace
#  undef CCCLRT_REQUIRE_THROWS_DEFAULT_STREAM_INVALID_CONTEXT
#  undef CCCLRT_DEFAULT_STREAM_INVALID_CONTEXT_MATCHER
#endif // TEST_HAS_EXCEPTIONS()

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

C2H_CCCLRT_TEST("Stream construction uses the explicit device", "[stream][multi_gpu]")
{
  if (cuda::devices.size() < 2)
  {
    return;
  }

  cuda::device_ref current_device{0};
  cuda::device_ref explicit_device{1};

  auto stream = [&]() {
    cuda::__ensure_current_context guard(current_device);
    return cuda::stream{explicit_device};
  }();

  CCCLRT_REQUIRE(stream.device() == explicit_device);
}

C2H_CCCLRT_TEST("Stream dependency uses the explicit stream device", "[stream][multi_gpu]")
{
  if (cuda::devices.size() < 2)
  {
    return;
  }

  cuda::device_ref current_device{0};
  cuda::device_ref explicit_device{1};

  cuda::stream waiter{explicit_device};
  cuda::stream waitee{explicit_device};

  ::test::pinned<int> value(0);
  ::cuda::atomic_ref atomic_value(*value);

  ::test::launch_kernel_single_thread(waitee, ::test::spin_until_80{}, value.get());
  ::test::launch_kernel_single_thread(waitee, ::test::assign_42{}, value.get());

  {
    cuda::__ensure_current_context guard(current_device);
    waiter.wait(waitee);
  }

  ::test::launch_kernel_single_thread(waiter, ::test::verify_42{}, value.get());
  CCCLRT_REQUIRE(atomic_value.load() != 42);
  CCCLRT_REQUIRE(!waiter.is_done());

  atomic_value.store(80);
  waiter.sync();
  waitee.sync();
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
#if _CCCL_COMPILER(NVHPC, <, 25, 11)
  CCCLRT_REQUIRE(cuda::std::to_underlying(id1) != cuda::std::to_underlying(id2));
#else // ^^^ _CCCL_COMPILER(NVHPC, <, 25, 11) ^^^ / vvv !_CCCL_COMPILER(NVHPC, <, 25, 11) vvv
  CCCLRT_REQUIRE(id1 != id2);
#endif // ^^^ !_CCCL_COMPILER(NVHPC, <, 25, 11) ^^^

  // Test that the same stream returns the same ID when called multiple times
#if _CCCL_COMPILER(NVHPC, <, 25, 11)
  CCCLRT_REQUIRE(cuda::std::to_underlying(stream1.id()) == cuda::std::to_underlying(id1));
  CCCLRT_REQUIRE(cuda::std::to_underlying(stream2.id()) == cuda::std::to_underlying(id2));
#else // ^^^ _CCCL_COMPILER(NVHPC, <, 25, 11) ^^^ / vvv !_CCCL_COMPILER(NVHPC, <, 25, 11) vvv
  CCCLRT_REQUIRE(stream1.id() == id1);
  CCCLRT_REQUIRE(stream2.id() == id2);
#endif // ^^^ !_CCCL_COMPILER(NVHPC, <, 25, 11) ^^^

  {
    // Test that stream_ref also supports id()
    // NULL stream needs a device to be set
    cuda::__ensure_current_context guard(cuda::device_ref{0});
    cuda::stream_ref ref1(::cudaStream_t{});
    cuda::stream_ref ref2(stream1);

#if _CCCL_COMPILER(NVHPC, <, 25, 11)
    CCCLRT_REQUIRE(cuda::std::to_underlying(ref1.id()) != cuda::std::to_underlying(ref2.id()));
    CCCLRT_REQUIRE(cuda::std::to_underlying(ref2.id()) == cuda::std::to_underlying(id1));
#else // ^^^ _CCCL_COMPILER(NVHPC, <, 25, 11) ^^^ / vvv !_CCCL_COMPILER(NVHPC, <, 25, 11) vvv
    CCCLRT_REQUIRE(ref1.id() != ref2.id());
    CCCLRT_REQUIRE(ref2.id() == id1);
#endif // ^^^ !_CCCL_COMPILER(NVHPC, <, 25, 11) ^^^
  }
}

#if _CCCL_HAS_HOST_STD_LIB()
C2H_CCCLRT_TEST("Stream hash", "[stream]")
{
  STATIC_REQUIRE(
    cuda::std::is_same_v<decltype(std::hash<cuda::stream>{}(cuda::std::declval<cuda::stream&>())), size_t>);
  STATIC_REQUIRE(cuda::std::is_default_constructible_v<std::hash<cuda::stream>>);
  STATIC_REQUIRE(cuda::std::is_copy_constructible_v<std::hash<cuda::stream>>);

  cuda::stream stream{cuda::device_ref{0}};

  // The hash is stable across calls.
  CCCLRT_REQUIRE(std::hash<cuda::stream>{}(stream) == std::hash<cuda::stream>{}(stream));

  // A stream and a stream_ref that refer to the same stream are equal, so they
  // must hash equally.
  cuda::stream_ref ref{stream};
  CCCLRT_REQUIRE(ref == stream);
  CCCLRT_REQUIRE(std::hash<cuda::stream>{}(stream) == std::hash<cuda::stream_ref>{}(ref));

  // A moved-to stream owns the same underlying stream, so the hash follows it.
  auto hash_before = std::hash<cuda::stream>{}(stream);
  cuda::stream moved{cuda::std::move(stream)};
  CCCLRT_REQUIRE(std::hash<cuda::stream>{}(moved) == hash_before);
}
#endif // _CCCL_HAS_HOST_STD_LIB()

C2H_CCCLRT_TEST("Default stream diagnostics mention the missing current context", "[stream]")
{
#if TEST_HAS_EXCEPTIONS()
  test::empty_driver_stack();
  CCCLRT_REQUIRE(::cuda::__driver::__ctxGetCurrent() == nullptr);

  check_default_stream_invalid_context_message(nullptr);
  check_default_stream_invalid_context_message(CU_STREAM_LEGACY);
  check_default_stream_invalid_context_message(CU_STREAM_PER_THREAD);
#endif // TEST_HAS_EXCEPTIONS()
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
    CCCLRT_REQUIRE(stream.get() == (cudaStream_t) (~0ull)); // NOLINT(performance-no-int-to-ptr)
  }

  // 3. Test stream_ref comparisons
  {
    cuda::stream_ref valid_stream{(cudaStream_t) (123ull)}; // NOLINT(performance-no-int-to-ptr)
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
