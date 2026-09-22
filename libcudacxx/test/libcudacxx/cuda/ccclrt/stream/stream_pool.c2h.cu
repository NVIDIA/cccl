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
#include <stdexcept>
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

  // A pool must never be created by accident from a device in an argument list, and the size is mandatory.
  STATIC_REQUIRE(!cuda::std::is_default_constructible_v<cuda::stream_pool>);
  STATIC_REQUIRE(!cuda::std::is_convertible_v<cuda::device_ref, cuda::stream_pool>);
  STATIC_REQUIRE(!cuda::std::is_constructible_v<cuda::stream_pool, cuda::device_ref>);
  STATIC_REQUIRE(!cuda::std::is_constructible_v<cuda::stream_pool, cuda::__logical_device_ref>);
  STATIC_REQUIRE(cuda::std::is_constructible_v<cuda::stream_pool, cuda::device_ref, cuda::std::size_t>);
  STATIC_REQUIRE(cuda::std::is_constructible_v<cuda::stream_pool, cuda::__logical_device_ref, cuda::std::size_t>);
  STATIC_REQUIRE(cuda::std::is_constructible_v<cuda::stream_pool, const cuda::__logical_device&, cuda::std::size_t>);

  // The creation mode follows the size and the priority comes last; an int cannot slip into the mode slot.
  STATIC_REQUIRE(
    cuda::std::is_constructible_v<cuda::stream_pool, cuda::device_ref, cuda::std::size_t, cuda::stream_pool_creation>);
  STATIC_REQUIRE(
    cuda::std::
      is_constructible_v<cuda::stream_pool, cuda::device_ref, cuda::std::size_t, cuda::stream_pool_creation, int>);
  STATIC_REQUIRE(!cuda::std::is_constructible_v<cuda::stream_pool, cuda::device_ref, cuda::std::size_t, int>);
  STATIC_REQUIRE(
    cuda::std::is_constructible_v<cuda::stream_pool,
                                  cuda::__logical_device_ref,
                                  cuda::std::size_t,
                                  cuda::stream_pool_creation,
                                  int>);

  STATIC_REQUIRE(
    cuda::std::is_same_v<decltype(cuda::std::declval<const cuda::stream_pool&>().next_stream()), cuda::stream_ref>);
  STATIC_REQUIRE(
    cuda::std::is_same_v<decltype(cuda::std::declval<const cuda::stream_pool&>().at(0)), cuda::stream_ref>);
  STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<const cuda::stream_pool&>()[0]), cuda::stream_ref>);
}

C2H_CCCLRT_TEST("Stream pool on a device", "[stream][stream_pool]")
{
  const auto device = cuda::devices[0];

  SECTION("The pool reports its size, device and priority")
  {
    const cuda::stream_pool pool{device, 16};
    REQUIRE(pool.size() == 16);
    REQUIRE(pool.device() == device);
    REQUIRE(pool.priority() == cuda::stream::default_priority);
    REQUIRE(pool.__logical_device() == cuda::__logical_device_ref{device});
    REQUIRE(pool.__logical_device().kind() == cuda::__logical_device_ref::kinds::device);
  }

  SECTION("A size of zero is rejected")
  {
    REQUIRE_THROWS_AS((cuda::stream_pool{device, 0}), std::invalid_argument);
    REQUIRE_THROWS_AS((cuda::stream_pool{device, 0, cuda::stream_pool_creation::lazy}), std::invalid_argument);
    REQUIRE_THROWS_AS((cuda::stream_pool{device, 0, cuda::stream_pool_creation::eager}), std::invalid_argument);
    REQUIRE_THROWS_AS((cuda::stream_pool{cuda::__logical_device_ref{device}, 0}), std::invalid_argument);
  }

  SECTION("Requests for one slot always return the same stream")
  {
    const cuda::stream_pool pool{device, 4};
    REQUIRE(pool.size() == 4);
    REQUIRE(::test::count_driver_stack() == 0);

    const cuda::stream_ref s0 = pool.next_stream();
    REQUIRE(s0.get() != nullptr);
    REQUIRE(pool[0] == s0);
    REQUIRE(pool[4] == s0);

    const cuda::stream_ref s2 = pool[2];
    REQUIRE(s2 != s0);
    REQUIRE(pool[2] == s2);
    REQUIRE(pool[0] == s0);

    // at() is the same accessor, spelled out
    for (cuda::std::size_t i = 0; i < 2 * pool.size(); ++i)
    {
      REQUIRE(pool.at(i) == pool[i]);
    }
  }

  SECTION("A move takes over the streams and leaves an empty pool behind")
  {
    cuda::stream_pool source{device, 3};
    const cudaStream_t first  = source.next_stream().get();
    const cudaStream_t second = source.at(1).get();
    const cudaStream_t third  = source.at(2).get();

    cuda::stream_pool moved{std::move(source)};
    REQUIRE(source.size() == 0);
    REQUIRE(moved.size() == 3);
    REQUIRE(moved.device() == device);
    REQUIRE(moved.priority() == cuda::stream::default_priority);
    // The streams are the same, and the round-robin position carries over: slot 0 was already handed out.
    REQUIRE(moved.at(0).get() == first);
    REQUIRE(moved.at(1).get() == second);
    REQUIRE(moved.at(2).get() == third);
    REQUIRE(moved.next_stream().get() == second);

    // A stream_ref obtained before the move still runs work.
    const cuda::stream_ref before_move{first};
    ::test::pinned<int> value(0);
    ::test::launch_kernel_single_thread(before_move, ::test::assign_42{}, value.get());
    before_move.sync();
    REQUIRE(*value == 42);

    // Move assignment destroys the streams of the target and takes over those of the source.
    cuda::stream_pool target{device, 1, cuda::stream_pool_creation::lazy};
    target.next_stream().sync();
    target = std::move(moved);
    REQUIRE(moved.size() == 0);
    REQUIRE(target.size() == 3);
    REQUIRE(target.at(0).get() == first);
    REQUIRE(target.at(2).get() == third);
    target.at(2).sync();

    // A moved-from pool can be assigned to again.
    source = cuda::stream_pool{device, 2, cuda::stream_pool_creation::lazy};
    REQUIRE(source.size() == 2);
    REQUIRE(source.next_stream().device() == device);
  }

  SECTION("Pools can be stored by value in a container")
  {
    std::vector<cuda::stream_pool> pools;
    for (int i = 1; i <= 4; ++i)
    {
      pools.emplace_back(device, static_cast<cuda::std::size_t>(i), cuda::stream_pool_creation::lazy);
    }
    for (int i = 1; i <= 4; ++i)
    {
      REQUIRE(pools[i - 1].size() == static_cast<cuda::std::size_t>(i));
      REQUIRE(pools[i - 1].next_stream().device() == device);
    }
  }

  SECTION("Lazy creation is opted into")
  {
    const cuda::stream_pool pool{device, 3, cuda::stream_pool_creation::lazy};
    REQUIRE(pool.size() == 3);
    REQUIRE(pool.device() == device);
    REQUIRE(pool.priority() == cuda::stream::default_priority);

    const cuda::stream_ref s0 = pool.next_stream();
    REQUIRE(pool[0] == s0);
  }

  SECTION("Round-robin hands out every slot once before repeating")
  {
    const cuda::stream_pool pool{device, 3};

    const cuda::stream_ref s0 = pool.next_stream();
    const cuda::stream_ref s1 = pool.next_stream();
    const cuda::stream_ref s2 = pool.next_stream();

    REQUIRE(s0 != s1);
    REQUIRE(s1 != s2);
    REQUIRE(s0 != s2);

    REQUIRE(pool.next_stream() == s0);
    REQUIRE(pool.next_stream() == s1);
    REQUIRE(pool.next_stream() == s2);
    REQUIRE(pool.next_stream() == s0);
  }

  SECTION("Indexed access wraps around and does not advance the round-robin position")
  {
    const cuda::stream_pool pool{device, 3};

    const cuda::stream_ref s1 = pool[1];
    REQUIRE(pool[4] == s1);
    REQUIRE(pool[7] == s1);

    // Round-robin still starts at slot 0 and reaches slot 1 second.
    const cuda::stream_ref first = pool.next_stream();
    REQUIRE(first != s1);
    REQUIRE(pool.next_stream() == s1);
    REQUIRE(pool[0] == first);
  }

  SECTION("A pool of one stream always returns that stream")
  {
    const cuda::stream_pool pool{device, 1};
    const cuda::stream_ref only = pool.next_stream();
    REQUIRE(pool.next_stream() == only);
    REQUIRE(pool[0] == only);
    REQUIRE(pool[17] == only);
  }

  SECTION("Streams live on the requested device and run work")
  {
    const cuda::stream_pool pool{device, 2};

    for (cuda::std::size_t i = 0; i < 2 * pool.size(); ++i)
    {
      const cuda::stream_ref str = pool.next_stream();
      REQUIRE(str.device() == device);
      REQUIRE(str.__logical_device() == cuda::__logical_device_ref{device});

      ::test::pinned<int> value(0);
      ::test::launch_kernel_single_thread(str, ::test::assign_42{}, value.get());
      str.sync();
      REQUIRE(*value == 42);
    }
  }

  SECTION("Every stream is non-blocking, whether created eagerly or lazily")
  {
    for (const auto mode : {cuda::stream_pool_creation::eager, cuda::stream_pool_creation::lazy})
    {
      const cuda::stream_pool pool{device, 3, mode};
      for (cuda::std::size_t i = 0; i < pool.size(); ++i)
      {
        unsigned int flags{};
        {
          const cuda::__ensure_current_context guard(device);
          CUDART(cudaStreamGetFlags(pool[i].get(), &flags));
        }
        REQUIRE((flags & cudaStreamNonBlocking) != 0);
      }
    }
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
      const cuda::stream_pool pool{device, 2, cuda::stream_pool_creation::lazy, priority};
      REQUIRE(pool.priority() == priority);
      REQUIRE(pool.next_stream().priority() == priority);
      REQUIRE(pool.next_stream().priority() == priority);
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

      const cuda::stream_ref str = pool.next_stream();
      REQUIRE(str.device() == second);

      ::test::pinned<int> value(0);
      ::test::launch_kernel_single_thread(str, ::test::assign_42{}, value.get());
      str.sync();
      REQUIRE(*value == 42);
    }
  }

  SECTION("Every slot of a lazy pool yields a distinct, valid stream")
  {
    const cuda::stream_pool pool{device, 4, cuda::stream_pool_creation::lazy};

    std::vector<cuda::stream_ref> all;
    all.reserve(pool.size());
    for (cuda::std::size_t i = 0; i < pool.size(); ++i)
    {
      all.push_back(pool[i]);
    }

    // Every entry is a distinct, valid stream on the device.
    for (cuda::std::size_t i = 0; i < all.size(); ++i)
    {
      REQUIRE(all[i].get() != nullptr);
      REQUIRE(all[i].device() == device);
      for (cuda::std::size_t j = i + 1; j < all.size(); ++j)
      {
        REQUIRE(all[i] != all[j]);
      }
    }

    // The getters keep handing out the same streams.
    for (cuda::std::size_t i = 0; i < all.size(); ++i)
    {
      REQUIRE(pool[i] == all[i]);
      REQUIRE(pool.next_stream() == all[i]);
    }
  }

  SECTION("Eager creation, the default, creates every slot in the constructor")
  {
    const cuda::stream_pool pool{device, 3};
    REQUIRE(pool.size() == 3);
    REQUIRE(pool.device() == device);
    REQUIRE(pool.priority() == cuda::stream::default_priority);

    for (cuda::std::size_t i = 0; i < pool.size(); ++i)
    {
      const cuda::stream_ref str = pool[i];
      REQUIRE(str.get() != nullptr);
      REQUIRE(str.device() == device);
      REQUIRE(pool.next_stream() == str);
    }
  }

  SECTION("Every slot can be synchronized by index before the pool goes away")
  {
    const cuda::stream_pool pool{device, 3};
    ::test::pinned<int> value0(0);
    ::test::pinned<int> value1(0);
    ::test::pinned<int> value2(0);
    ::test::launch_kernel_single_thread(pool.next_stream(), ::test::assign_42{}, value0.get());
    ::test::launch_kernel_single_thread(pool.next_stream(), ::test::assign_42{}, value1.get());
    ::test::launch_kernel_single_thread(pool.next_stream(), ::test::assign_42{}, value2.get());

    for (cuda::std::size_t i = 0; i < pool.size(); ++i)
    {
      pool[i].sync();
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
    (void) pool.next_stream();
    (void) pool[1];
    REQUIRE(::test::count_driver_stack() == before);
  }
}

C2H_CCCLRT_TEST("Stream pool is usable from several threads", "[stream][stream_pool]")
{
  const auto device = cuda::devices[0];
  const cuda::stream_pool pool{device, 4};

  constexpr int num_threads             = 8;
  static constexpr int picks_per_thread = 64;
  std::vector<std::vector<cudaStream_t>> picks(num_threads);
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (int t = 0; t < num_threads; ++t)
  {
    threads.emplace_back([&pool, &picks, t] {
      picks[t].reserve(picks_per_thread);
      for (int i = 0; i < picks_per_thread; ++i)
      {
        picks[t].push_back(pool.next_stream().get());
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
    slots.insert(pool[i].get());
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

C2H_CCCLRT_TEST("Concurrent first requests for one slot create a single stream", "[stream][stream_pool]")
{
  const auto device = cuda::devices[0];

  constexpr int num_threads = 8;
  constexpr int num_rounds  = 16;
  for (int round = 0; round < num_rounds; ++round)
  {
    const cuda::stream_pool pool{device, 2, cuda::stream_pool_creation::lazy};
    std::vector<cudaStream_t> seen(num_threads, nullptr);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t)
    {
      threads.emplace_back([&pool, &seen, t] {
        seen[t] = pool[1].get();
      });
    }
    for (auto& thread : threads)
    {
      thread.join();
    }

    for (const auto handle : seen)
    {
      REQUIRE(handle == seen[0]);
    }
    REQUIRE(pool[1].get() == seen[0]);
  }
}

C2H_CCCLRT_TEST("Eager stream pool is usable from several threads", "[stream][stream_pool]")
{
  const auto device = cuda::devices[0];
  const cuda::stream_pool pool{device, 4, cuda::stream_pool_creation::eager};
  std::set<cudaStream_t> slots;
  for (cuda::std::size_t i = 0; i < pool.size(); ++i)
  {
    slots.insert(pool[i].get());
  }
  REQUIRE(slots.size() == 4);

  constexpr int num_threads             = 8;
  static constexpr int picks_per_thread = 64;
  std::vector<std::vector<cudaStream_t>> picks(num_threads);
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int t = 0; t < num_threads; ++t)
  {
    threads.emplace_back([&pool, &picks, t] {
      picks[t].reserve(picks_per_thread);
      for (int i = 0; i < picks_per_thread; ++i)
      {
        picks[t].push_back(pool.next_stream().get());
      }
    });
  }
  for (auto& thread : threads)
  {
    thread.join();
  }

  for (const auto& thread_picks : picks)
  {
    for (const auto handle : thread_picks)
    {
      REQUIRE(slots.count(handle) == 1);
    }
  }
}

C2H_CCCLRT_TEST("Stream pool creates its streams while the calling thread captures", "[stream][stream_pool][capture]")
{
  const auto device = cuda::devices[0];

  for (const auto& c : capture_modes)
  {
    INFO("capture mode: " << c.name);

    const cuda::stream_pool pool{device, 2, cuda::stream_pool_creation::lazy};
    const cuda::stream capturing{device};
    REQUIRE(begin_capture(capturing.get(), c.mode) == ::CUDA_SUCCESS);

    // First requests: the streams of the pool are created while this thread captures.
    const cuda::stream_ref s0 = pool.next_stream();
    const cuda::stream_ref s1 = pool.next_stream();
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

  SECTION("An eager pool is constructed while the calling thread captures")
  {
    const cuda::stream capturing{device};
    REQUIRE(begin_capture(capturing.get(), ::CU_STREAM_CAPTURE_MODE_GLOBAL) == ::CUDA_SUCCESS);

    const cuda::stream_pool pool{device, 2};
    REQUIRE(pool[0] != pool[1]);

    ::CUstreamCaptureMode mode = ::CU_STREAM_CAPTURE_MODE_GLOBAL;
    cuda::__driver::__threadExchangeStreamCaptureMode(mode);
    REQUIRE(mode == ::CU_STREAM_CAPTURE_MODE_GLOBAL);

    ::CUgraph graph = nullptr;
    REQUIRE(end_capture(capturing.get(), &graph) == ::CUDA_SUCCESS);
    destroy_graph(graph);
  }

  SECTION("The thread's capture mode is restored after the streams are created")
  {
    const cuda::stream_pool pool{device, 1, cuda::stream_pool_creation::lazy};
    const cuda::stream capturing{device};
    REQUIRE(begin_capture(capturing.get(), ::CU_STREAM_CAPTURE_MODE_GLOBAL) == ::CUDA_SUCCESS);

    (void) pool.next_stream();

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
      const cuda::stream_ref str = pool.next_stream();
      REQUIRE(str.device() == device);
      REQUIRE(str.__logical_device().kind() == cuda::__logical_device_ref::kinds::green_context);
      REQUIRE(str.__logical_device().green_context() == ldev.green_context());
      REQUIRE(cuda::__driver::__streamGetCtx(str.get()) == cuda::__driver::__ctxFromGreenCtx(ldev.green_context()));

      unsigned int flags{};
      {
        const cuda::__ensure_current_context guard(device);
        CUDART(cudaStreamGetFlags(str.get(), &flags));
      }
      REQUIRE((flags & cudaStreamNonBlocking) != 0);

      ::test::pinned<int> value(0);
      ::test::launch_kernel_single_thread(str, ::test::assign_42{}, value.get());
      str.sync();
      REQUIRE(*value == 42);
    }
  }

  SECTION("Eager creation creates every slot on the green context")
  {
    auto ldev = ::make_logical_device(device);
    const cuda::stream_pool pool{ldev, 2, cuda::stream_pool_creation::eager};

    for (cuda::std::size_t i = 0; i < pool.size(); ++i)
    {
      REQUIRE(pool[i].__logical_device().green_context() == ldev.green_context());
    }
  }

  SECTION("A device-backed logical device gives streams on the primary context")
  {
    const cuda::__logical_device_ref ldev{device};
    const cuda::stream_pool pool{ldev, 2};

    const cuda::stream_ref str = pool.next_stream();
    REQUIRE(str.device() == device);
    REQUIRE(cuda::__driver::__streamGetCtx(str.get()) == device.__primary_context());
  }

  SECTION("The streams are created on the green context while the calling thread captures")
  {
    auto ldev = ::make_logical_device(device);
    const cuda::stream_pool pool{ldev, 1, cuda::stream_pool_creation::lazy};
    const cuda::stream capturing{device};
    REQUIRE(begin_capture(capturing.get(), ::CU_STREAM_CAPTURE_MODE_GLOBAL) == ::CUDA_SUCCESS);

    const cuda::stream_ref str = pool.next_stream();
    REQUIRE(str.__logical_device().green_context() == ldev.green_context());

    ::CUgraph graph = nullptr;
    REQUIRE(end_capture(capturing.get(), &graph) == ::CUDA_SUCCESS);
    destroy_graph(graph);
  }
}

#endif // _CCCL_CTK_AT_LEAST(12, 5)
