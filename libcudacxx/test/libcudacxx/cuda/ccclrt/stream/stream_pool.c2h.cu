//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// `__logical_device` is not reachable from `<cuda/devices>`, so the internal headers are included
// directly here.
#include <cuda/__device/logical_device.h>
#include <cuda/__device/logical_device_ref.h>
#include <cuda/__driver/driver_api.h>
#include <cuda/devices>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <set>
#include <thread>
#include <vector>

#include <testing.cuh>

namespace
{
template <class Fn>
Fn* driver_fn(const char* name)
{
  return reinterpret_cast<Fn*>(::cuda::__driver::__get_driver_entry_point(name));
}
::CUresult begin_capture(::CUstream stream, ::CUstreamCaptureMode mode)
{
  static auto fn = driver_fn<decltype(::cuStreamBeginCapture)>("cuStreamBeginCapture");
  return fn(stream, mode);
}
::CUresult end_capture(::CUstream stream, ::CUgraph* graph)
{
  static auto fn = driver_fn<decltype(::cuStreamEndCapture)>("cuStreamEndCapture");
  return fn(stream, graph);
}
void destroy_graph(::CUgraph graph)
{
  static auto fn = driver_fn<decltype(::cuGraphDestroy)>("cuGraphDestroy");
  REQUIRE(fn(graph) == ::CUDA_SUCCESS);
}

struct capture_mode_case
{
  const char* name;
  ::CUstreamCaptureMode mode;
};

constexpr capture_mode_case capture_modes[] = {
  {"global", ::CU_STREAM_CAPTURE_MODE_GLOBAL},
  {"thread-local", ::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL},
  {"relaxed", ::CU_STREAM_CAPTURE_MODE_RELAXED},
};
} // namespace

C2H_CCCLRT_TEST("Stream pool type properties", "[stream][stream_pool]")
{
  STATIC_REQUIRE(!cuda::std::is_copy_constructible_v<cuda::stream_pool>);
  STATIC_REQUIRE(!cuda::std::is_copy_assignable_v<cuda::stream_pool>);
  STATIC_REQUIRE(cuda::std::is_nothrow_move_constructible_v<cuda::stream_pool>);
  STATIC_REQUIRE(cuda::std::is_nothrow_move_assignable_v<cuda::stream_pool>);

  // A pool must never be created by accident from a device in an argument list.
  STATIC_REQUIRE(!cuda::std::is_convertible_v<cuda::device_ref, cuda::stream_pool>);
  STATIC_REQUIRE(cuda::std::is_constructible_v<cuda::stream_pool, cuda::device_ref>);
  STATIC_REQUIRE(cuda::std::is_constructible_v<cuda::stream_pool, cuda::device_ref, cuda::std::size_t>);
  STATIC_REQUIRE(cuda::std::is_constructible_v<cuda::stream_pool, cuda::device_ref, cuda::std::size_t, int>);
  STATIC_REQUIRE(cuda::std::is_constructible_v<cuda::stream_pool, cuda::__logical_device_ref>);
  STATIC_REQUIRE(cuda::std::is_constructible_v<cuda::stream_pool, const cuda::__logical_device&>);

  STATIC_REQUIRE(
    cuda::std::is_same_v<decltype(cuda::std::declval<const cuda::stream_pool&>().get_stream()), cuda::stream_ref>);
  STATIC_REQUIRE(cuda::stream_pool::default_size == 16);
}

C2H_CCCLRT_TEST("Stream pool on a device", "[stream][stream_pool]")
{
  const auto device = cuda::devices[0];

  SECTION("The default pool has default_size slots and reports its device and priority")
  {
    const cuda::stream_pool pool{device};
    REQUIRE(pool.size() == cuda::stream_pool::default_size);
    REQUIRE(pool.device() == device);
    REQUIRE(pool.priority() == cuda::stream::default_priority);
    REQUIRE(pool.__logical_device() == cuda::__logical_device_ref{device});
    REQUIRE(pool.__logical_device().kind() == cuda::__logical_device_ref::kinds::device);
  }

  SECTION("Construction creates no stream")
  {
    // No stream exists yet, so the driver context stack was not touched by a stream creation.
    const cuda::stream_pool pool{device, 4};
    REQUIRE(pool.size() == 4);
    REQUIRE(::test::count_driver_stack() == 0);
  }

  SECTION("Round-robin hands out every slot once before repeating")
  {
    const cuda::stream_pool pool{device, 3};

    const cuda::stream_ref s0 = pool.get_stream();
    const cuda::stream_ref s1 = pool.get_stream();
    const cuda::stream_ref s2 = pool.get_stream();

    REQUIRE(s0 != s1);
    REQUIRE(s1 != s2);
    REQUIRE(s0 != s2);

    REQUIRE(pool.get_stream() == s0);
    REQUIRE(pool.get_stream() == s1);
    REQUIRE(pool.get_stream() == s2);
    REQUIRE(pool.get_stream() == s0);
  }

  SECTION("Indexed access wraps around and does not advance the round-robin position")
  {
    const cuda::stream_pool pool{device, 3};

    const cuda::stream_ref s1 = pool.get_stream(1);
    REQUIRE(pool.get_stream(4) == s1);
    REQUIRE(pool.get_stream(7) == s1);

    // Round-robin still starts at slot 0 and reaches slot 1 second.
    const cuda::stream_ref first = pool.get_stream();
    REQUIRE(first != s1);
    REQUIRE(pool.get_stream() == s1);
    REQUIRE(pool.get_stream(0) == first);
  }

  SECTION("A pool of one stream always returns that stream")
  {
    const cuda::stream_pool pool{device, 1};
    const cuda::stream_ref only = pool.get_stream();
    REQUIRE(pool.get_stream() == only);
    REQUIRE(pool.get_stream(0) == only);
    REQUIRE(pool.get_stream(17) == only);
  }

  SECTION("Streams live on the requested device and run work")
  {
    const cuda::stream_pool pool{device, 2};

    for (cuda::std::size_t i = 0; i < 2 * pool.size(); ++i)
    {
      const cuda::stream_ref str = pool.get_stream();
      REQUIRE(str.device() == device);
      REQUIRE(str.__logical_device() == cuda::__logical_device_ref{device});

      ::test::pinned<int> value(0);
      ::test::launch_kernel_single_thread(str, ::test::assign_42{}, value.get());
      str.sync();
      REQUIRE(*value == 42);
    }
  }

  SECTION("Streams are non-blocking")
  {
    const cuda::stream_pool pool{device, 2};
    const cuda::stream_ref str = pool.get_stream();

    unsigned int flags{};
    {
      const cuda::__ensure_current_context guard(device);
      CUDART(cudaStreamGetFlags(str.get(), &flags));
    }
    REQUIRE((flags & cudaStreamNonBlocking) != 0);
  }

  SECTION("The requested priority reaches every stream")
  {
    // The driver clamps a priority to the supported range, so the requested value is only checked
    // on a device that supports more than one priority.
    int least_priority{};
    int greatest_priority{};
    {
      const cuda::__ensure_current_context guard(device);
      CUDART(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
    }

    if (least_priority != greatest_priority)
    {
      const auto priority = cuda::stream::default_priority - 1;
      const cuda::stream_pool pool{device, 2, priority};
      REQUIRE(pool.priority() == priority);
      REQUIRE(pool.get_stream().priority() == priority);
      REQUIRE(pool.get_stream().priority() == priority);
    }
    else
    {
      SUCCEED("The device supports a single stream priority");
    }
  }

  SECTION("A pool on a second device creates streams there")
  {
    if (cuda::devices.size() > 1)
    {
      const auto second = cuda::devices[1];
      const cuda::stream_pool pool{second, 2};
      REQUIRE(pool.device() == second);

      const cuda::stream_ref str = pool.get_stream();
      REQUIRE(str.device() == second);

      ::test::pinned<int> value(0);
      ::test::launch_kernel_single_thread(str, ::test::assign_42{}, value.get());
      str.sync();
      REQUIRE(*value == 42);
    }
  }

  SECTION("streams() lists only the created streams, in slot order")
  {
    const cuda::stream_pool pool{device, 4};
    REQUIRE(pool.streams().empty());

    const cuda::stream_ref s2 = pool.get_stream(2);
    REQUIRE(pool.streams().size() == 1);
    REQUIRE(pool.streams()[0] == s2);

    const cuda::stream_ref s0 = pool.get_stream(0);
    const auto created        = pool.streams();
    REQUIRE(created.size() == 2);
    REQUIRE(created[0] == s0);
    REQUIRE(created[1] == s2);

    // A snapshot does not see streams created later.
    (void) pool.get_stream(1);
    REQUIRE(created.size() == 2);
    REQUIRE(pool.streams().size() == 3);

    // Once every slot was handed out, all of them are listed.
    (void) pool.get_stream(3);
    REQUIRE(pool.streams().size() == pool.size());
  }

  SECTION("streams() can be used to wait for all outstanding work")
  {
    const cuda::stream_pool pool{device, 3};
    ::test::pinned<int> value0(0);
    ::test::pinned<int> value1(0);
    ::test::pinned<int> value2(0);
    ::test::launch_kernel_single_thread(pool.get_stream(), ::test::assign_42{}, value0.get());
    ::test::launch_kernel_single_thread(pool.get_stream(), ::test::assign_42{}, value1.get());
    ::test::launch_kernel_single_thread(pool.get_stream(), ::test::assign_42{}, value2.get());

    for (const cuda::stream_ref str : pool.streams())
    {
      str.sync();
    }
    REQUIRE(*value0 == 42);
    REQUIRE(*value1 == 42);
    REQUIRE(*value2 == 42);
  }

  SECTION("Getting a stream leaves the driver context stack unchanged")
  {
    const cuda::stream_pool pool{device, 2};
    const cuda::__ensure_current_context guard(device);

    const auto before = ::test::count_driver_stack();
    (void) pool.get_stream();
    (void) pool.get_stream(1);
    REQUIRE(::test::count_driver_stack() == before);
  }
}

C2H_CCCLRT_TEST("Stream pool move semantics", "[stream][stream_pool]")
{
  const auto device = cuda::devices[0];

  SECTION("Move construction keeps the streams and the round-robin position")
  {
    cuda::stream_pool source{device, 3};
    const cuda::stream_ref s0 = source.get_stream();
    const cuda::stream_ref s1 = source.get_stream();

    const cuda::stream_pool destination{cuda::std::move(source)};
    REQUIRE(destination.size() == 3);
    REQUIRE(destination.device() == device);
    REQUIRE(destination.get_stream(0) == s0);
    REQUIRE(destination.get_stream(1) == s1);

    // The next round-robin pick is slot 2, then back to slot 0.
    const cuda::stream_ref s2 = destination.get_stream();
    REQUIRE(s2 != s0);
    REQUIRE(s2 != s1);
    REQUIRE(destination.get_stream() == s0);

    // The moved-from pool is empty.
    REQUIRE(source.size() == 0); // NOLINT(bugprone-use-after-move)
  }

  SECTION("Move assignment replaces the streams of the target")
  {
    cuda::stream_pool source{device, 2};
    cuda::stream_pool target{device, 5};
    const cuda::stream_ref s0 = source.get_stream();
    (void) target.get_stream();

    target = cuda::std::move(source);
    REQUIRE(target.size() == 2);
    REQUIRE(target.get_stream(0) == s0);
    REQUIRE(source.size() == 0); // NOLINT(bugprone-use-after-move)
  }

  SECTION("Stream references stay valid across a move")
  {
    cuda::stream_pool source{device, 2};
    const cuda::stream_ref str = source.get_stream();

    const cuda::stream_pool destination{cuda::std::move(source)};

    ::test::pinned<int> value(0);
    ::test::launch_kernel_single_thread(str, ::test::assign_42{}, value.get());
    str.sync();
    REQUIRE(*value == 42);
    REQUIRE(destination.get_stream(0) == str);
  }

  SECTION("Self move assignment is a no-op")
  {
    cuda::stream_pool pool{device, 2};
    const cuda::stream_ref s0 = pool.get_stream();

    cuda::stream_pool& alias = pool;
    pool                     = cuda::std::move(alias);
    REQUIRE(pool.size() == 2);
    REQUIRE(pool.get_stream(0) == s0);
  }
}

C2H_CCCLRT_TEST("Stream pool is usable from several threads", "[stream][stream_pool]")
{
  const auto device = cuda::devices[0];
  const cuda::stream_pool pool{device, 4};

  constexpr int num_threads      = 8;
  constexpr int picks_per_thread = 64;
  std::vector<std::vector<cudaStream_t>> picks(num_threads);
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (int t = 0; t < num_threads; ++t)
  {
    threads.emplace_back([&pool, &picks, t] {
      picks[t].reserve(picks_per_thread);
      for (int i = 0; i < picks_per_thread; ++i)
      {
        picks[t].push_back(pool.get_stream().get());
      }
    });
  }
  for (auto& thread : threads)
  {
    thread.join();
  }

  // Every handle handed out is one of the pool's slots, and every slot was created exactly once.
  std::set<cudaStream_t> slots;
  for (cuda::std::size_t i = 0; i < pool.size(); ++i)
  {
    slots.insert(pool.get_stream(i).get());
  }
  REQUIRE(slots.size() == pool.size());

  for (const auto& thread_picks : picks)
  {
    for (const auto handle : thread_picks)
    {
      REQUIRE(slots.count(handle) == 1);
    }
  }
}

C2H_CCCLRT_TEST("Stream pool fills a slot while the calling thread captures", "[stream][stream_pool][capture]")
{
  const auto device = cuda::devices[0];

  for (const auto& c : capture_modes)
  {
    INFO("capture mode: " << c.name);

    const cuda::stream_pool pool{device, 2};
    const cuda::stream capturing{device};
    REQUIRE(begin_capture(capturing.get(), c.mode) == ::CUDA_SUCCESS);

    // First touch of both slots: the streams are created while this thread captures.
    const cuda::stream_ref s0 = pool.get_stream();
    const cuda::stream_ref s1 = pool.get_stream();
    REQUIRE(s0 != s1);
    REQUIRE(s0 != capturing);
    REQUIRE(s1 != capturing);

    // The capture was not invalidated by the creation.
    ::CUgraph graph = nullptr;
    REQUIRE(end_capture(capturing.get(), &graph) == ::CUDA_SUCCESS);
    destroy_graph(graph);

    // The streams work once the capture is over.
    ::test::pinned<int> value(0);
    ::test::launch_kernel_single_thread(s0, ::test::assign_42{}, value.get());
    s0.sync();
    REQUIRE(*value == 42);
  }

  SECTION("The thread's capture mode is restored after the slot is filled")
  {
    const cuda::stream_pool pool{device, 1};
    const cuda::stream capturing{device};
    REQUIRE(begin_capture(capturing.get(), ::CU_STREAM_CAPTURE_MODE_GLOBAL) == ::CUDA_SUCCESS);

    (void) pool.get_stream();

    // Exchanging the mode reports the mode this thread is in: it must be global again.
    ::CUstreamCaptureMode mode = ::CU_STREAM_CAPTURE_MODE_GLOBAL;
    cuda::__driver::__threadExchangeStreamCaptureMode(mode);
    REQUIRE(mode == ::CU_STREAM_CAPTURE_MODE_GLOBAL);

    ::CUgraph graph = nullptr;
    REQUIRE(end_capture(capturing.get(), &graph) == ::CUDA_SUCCESS);
    destroy_graph(graph);
  }
}

// Green contexts require CTK 12.5.
#if _CCCL_CTK_AT_LEAST(12, 5)

namespace
{
//! Create a green context that spans the whole of `device`. Returns an owning `__logical_device`.
cuda::__logical_device make_logical_device(cuda::device_ref device)
{
  const auto gctx = cuda::__driver::__greenCtxCreate(cuda::__driver::__deviceGet(device.get()));
  return cuda::__logical_device::from_native_handle(device, gctx);
}
} // namespace

C2H_CCCLRT_TEST("Stream pool on a green context", "[stream][stream_pool][logical_device]")
{
  if (test::cuda_driver_version() < 12050)
  {
    SUCCEED("Driver is too old for green context tests");
    return;
  }

  const auto device = cuda::devices[0];

  SECTION("The pool reports the green context and the device owning it")
  {
    auto ldev = ::make_logical_device(device);
    const cuda::stream_pool pool{ldev, 2};

    REQUIRE(pool.device() == device);
    REQUIRE(pool.__logical_device() == static_cast<const cuda::__logical_device_ref&>(ldev));
    REQUIRE(pool.__logical_device().kind() == cuda::__logical_device_ref::kinds::green_context);
  }

  SECTION("Streams are created on the green context and run work")
  {
    auto ldev = ::make_logical_device(device);
    const cuda::stream_pool pool{ldev, 2};

    for (cuda::std::size_t i = 0; i < pool.size(); ++i)
    {
      const cuda::stream_ref str = pool.get_stream();
      REQUIRE(str.device() == device);
      REQUIRE(str.__logical_device().kind() == cuda::__logical_device_ref::kinds::green_context);
      REQUIRE(str.__logical_device().green_context() == ldev.green_context());
      REQUIRE(cuda::__driver::__streamGetCtx(str.get()) == cuda::__driver::__ctxFromGreenCtx(ldev.green_context()));

      ::test::pinned<int> value(0);
      ::test::launch_kernel_single_thread(str, ::test::assign_42{}, value.get());
      str.sync();
      REQUIRE(*value == 42);
    }
  }

  SECTION("A device-backed logical device gives streams on the primary context")
  {
    const cuda::__logical_device_ref ldev{device};
    const cuda::stream_pool pool{ldev, 2};

    const cuda::stream_ref str = pool.get_stream();
    REQUIRE(str.device() == device);
    REQUIRE(cuda::__driver::__streamGetCtx(str.get()) == device.__primary_context());
  }

  SECTION("A slot is filled on the green context while the calling thread captures")
  {
    auto ldev = ::make_logical_device(device);
    const cuda::stream_pool pool{ldev, 1};
    const cuda::stream capturing{device};
    REQUIRE(begin_capture(capturing.get(), ::CU_STREAM_CAPTURE_MODE_GLOBAL) == ::CUDA_SUCCESS);

    const cuda::stream_ref str = pool.get_stream();
    REQUIRE(str.__logical_device().green_context() == ldev.green_context());

    ::CUgraph graph = nullptr;
    REQUIRE(end_capture(capturing.get(), &graph) == ::CUDA_SUCCESS);
    destroy_graph(graph);
  }
}

#endif // _CCCL_CTK_AT_LEAST(12, 5)
