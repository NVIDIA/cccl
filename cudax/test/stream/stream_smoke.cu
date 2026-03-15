//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/array>
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
#if _CCCL_COMPILER(NVHPC, <, 25, 11)
  CUDAX_REQUIRE(cuda::std::to_underlying(id1) != cuda::std::to_underlying(id2));
#else // ^^^ _CCCL_COMPILER(NVHPC, <, 25, 11) ^^^ / vvv !_CCCL_COMPILER(NVHPC, <, 25, 11) vvv
  CUDAX_REQUIRE(id1 != id2);
#endif // ^^^ !_CCCL_COMPILER(NVHPC, <, 25, 11) ^^^

  // Test that the same stream returns the same ID when called multiple times
#if _CCCL_COMPILER(NVHPC, <, 25, 11)
  CUDAX_REQUIRE(cuda::std::to_underlying(stream1.id()) == cuda::std::to_underlying(id1));
  CUDAX_REQUIRE(cuda::std::to_underlying(stream2.id()) == cuda::std::to_underlying(id2));
#else // ^^^ _CCCL_COMPILER(NVHPC, <, 25, 11) ^^^ / vvv !_CCCL_COMPILER(NVHPC, <, 25, 11) vvv
  CUDAX_REQUIRE(stream1.id() == id1);
  CUDAX_REQUIRE(stream2.id() == id2);
#endif // ^^^ !_CCCL_COMPILER(NVHPC, <, 25, 11) ^^^

  {
    // Test that stream_ref also supports id()
    // NULL stream needs a device to be set
    cuda::__ensure_current_context guard(cuda::device_ref{0});
    cuda::stream_ref ref1(::cudaStream_t{});
    cuda::stream_ref ref2(stream1);

#if _CCCL_COMPILER(NVHPC, <, 25, 11)
    CUDAX_REQUIRE(cuda::std::to_underlying(ref1.id()) != cuda::std::to_underlying(ref2.id()));
    CUDAX_REQUIRE(cuda::std::to_underlying(ref2.id()) == cuda::std::to_underlying(id1));
#else // ^^^ _CCCL_COMPILER(NVHPC, <, 25, 11) ^^^ / vvv !_CCCL_COMPILER(NVHPC, <, 25, 11) vvv
    CUDAX_REQUIRE(ref1.id() != ref2.id());
    CUDAX_REQUIRE(ref2.id() == id1);
#endif // ^^^ !_CCCL_COMPILER(NVHPC, <, 25, 11) ^^^
  }
}

C2H_CCCLRT_TEST("Replicate only creates streams", "[stream]")
{
  cudax::stream source{cuda::device_ref{0}};

  ::test::pinned<int> gate(0);
  ::test::pinned<int> out(0);
  ::cuda::atomic_ref atomic_gate(*gate);

  cudax::launch(source, ::test::one_thread_dims, ::test::spin_until_80{}, gate.get());
  auto [left] = cudax::replicate<1>(source);
  cudax::launch(left, ::test::one_thread_dims, ::test::assign_42{}, out.get());

  left.sync();
  CUDAX_REQUIRE(*out == 42);
  CUDAX_REQUIRE(!source.is_done());

  atomic_gate.store(80);
  source.sync();

  CUDAX_REQUIRE(source.device() == left.device());
}

C2H_CCCLRT_TEST("Replicate supports dynamic stream counts", "[stream]")
{
  cudax::stream source{cuda::device_ref{0}};
  constexpr size_t count = 3;
  auto streams           = cudax::replicate(source, count);

  CUDAX_REQUIRE(streams.size() == count);
  for (size_t idx = 0; idx < count; ++idx)
  {
    CUDAX_REQUIRE(streams[idx] != source);
    CUDAX_REQUIRE(streams[idx].device() == source.device());
  }
}

C2H_CCCLRT_TEST("Replicate prepend keeps moved stream at index zero", "[stream]")
{
  cudax::stream source{cuda::device_ref{0}};
  auto source_id         = source.id();
  constexpr size_t count = 2;

  auto streams = cudax::replicate_prepend(::cuda::std::move(source), count);

  CUDAX_REQUIRE(streams.size() == count + 1);
  CUDAX_REQUIRE(streams.front().id() == source_id);
  for (size_t idx = 1; idx < streams.size(); ++idx)
  {
    CUDAX_REQUIRE(streams[idx].device() == streams.front().device());
  }
}

C2H_CCCLRT_TEST("Replicate prepend static keeps moved stream at index zero", "[stream]")
{
  cudax::stream source{cuda::device_ref{0}};
  auto source_id = source.id();

  auto streams = cudax::replicate_prepend<2>(::cuda::std::move(source));

  CUDAX_REQUIRE(streams.size() == 3);
  CUDAX_REQUIRE(streams[0].id() == source_id);
  CUDAX_REQUIRE(streams[1].device() == streams[0].device());
  CUDAX_REQUIRE(streams[2].device() == streams[0].device());
}

C2H_CCCLRT_TEST("Join synchronizes stream groups", "[stream]")
{
  cudax::stream target1{cuda::device_ref{0}};
  cudax::stream target2{cuda::device_ref{0}};
  cudax::stream source{cuda::device_ref{0}};
  ::test::pinned<int> i(0);
  ::cuda::atomic_ref atomic_i(*i);

  cudax::launch(source, ::test::one_thread_dims, ::test::spin_until_80{}, i.get());
  cudax::launch(source, ::test::one_thread_dims, ::test::assign_42{}, i.get());

  auto targets = ::cuda::std::array<cudax::stream_ref, 2>{target1, target2};
  cudax::join(targets, cudax::stream_ref{source});

  cudax::launch(target1, ::test::one_thread_dims, ::test::verify_42{}, i.get());
  cudax::launch(target2, ::test::one_thread_dims, ::test::verify_42{}, i.get());
  CUDAX_REQUIRE(atomic_i.load() != 42);
  CUDAX_REQUIRE(!target1.is_done());
  CUDAX_REQUIRE(!target2.is_done());

  atomic_i.store(80);
  target1.sync();
  target2.sync();
  source.sync();
}

C2H_CCCLRT_TEST("Join overload accepts stream_ref single target", "[stream]")
{
  cudax::stream target{cuda::device_ref{0}};
  cudax::stream source1{cuda::device_ref{0}};
  cudax::stream source2{cuda::device_ref{0}};
  ::test::pinned<int> i(0);
  ::cuda::atomic_ref atomic_i(*i);

  cudax::launch(source2, ::test::one_thread_dims, ::test::spin_until_80{}, i.get());
  cudax::launch(source2, ::test::one_thread_dims, ::test::assign_42{}, i.get());

  auto sources = ::cuda::std::array<cudax::stream_ref, 2>{source1, source2};
  cudax::join(cudax::stream_ref{target}, sources);

  cudax::launch(target, ::test::one_thread_dims, ::test::verify_42{}, i.get());
  CUDAX_REQUIRE(atomic_i.load() != 42);
  CUDAX_REQUIRE(!target.is_done());

  atomic_i.store(80);
  target.sync();
  source1.sync();
  source2.sync();
}

C2H_CCCLRT_TEST("Join supports streams from multiple devices", "[stream]")
{
  if (cuda::devices.size() < 2)
  {
    return;
  }

  cudax::stream target{cuda::device_ref{0}};
  cudax::stream source0{cuda::device_ref{0}};
  cudax::stream source1{cuda::device_ref{static_cast<int>(cuda::devices.size() - 1)}};
  ::test::pinned<int> i(0);
  ::cuda::atomic_ref atomic_i(*i);

  cudax::launch(source0, ::test::one_thread_dims, ::test::spin_until_80{}, i.get());
  cudax::launch(source0, ::test::one_thread_dims, ::test::assign_42{}, i.get());
  cudax::launch(source1, ::test::one_thread_dims, ::test::spin_until_80{}, i.get());

  auto targets = ::cuda::std::array<cudax::stream_ref, 1>{target};
  auto sources = ::cuda::std::array<cudax::stream_ref, 2>{source0, source1};
  cudax::join(targets, sources);

  cudax::launch(target, ::test::one_thread_dims, ::test::verify_42{}, i.get());
  CUDAX_REQUIRE(atomic_i.load() != 42);
  CUDAX_REQUIRE(!target.is_done());

  atomic_i.store(80);
  target.sync();
  source0.sync();
  source1.sync();
}

C2H_CCCLRT_TEST("Join handles a stream present in both groups", "[stream]")
{
  cudax::stream shared{cuda::device_ref{0}};
  cudax::stream target{cuda::device_ref{0}};
  ::test::pinned<int> i(0);
  ::cuda::atomic_ref atomic_i(*i);

  cudax::launch(shared, ::test::one_thread_dims, ::test::spin_until_80{}, i.get());
  cudax::launch(shared, ::test::one_thread_dims, ::test::assign_42{}, i.get());

  auto to_streams   = ::cuda::std::array<cudax::stream_ref, 2>{shared, target};
  auto from_streams = ::cuda::std::array<cudax::stream_ref, 1>{shared};
  cudax::join(to_streams, from_streams);

  cudax::launch(target, ::test::one_thread_dims, ::test::verify_42{}, i.get());
  CUDAX_REQUIRE(atomic_i.load() != 42);
  CUDAX_REQUIRE(!target.is_done());

  atomic_i.store(80);
  target.sync();
  shared.sync();
}
