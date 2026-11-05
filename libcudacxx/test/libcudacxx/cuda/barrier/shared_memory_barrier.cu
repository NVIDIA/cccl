//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/barrier>
#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/ptx>
#include <cuda/std/chrono>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <new> // IWYU pragma: keep (needed for placement new)

#include <testing.cuh>

#include <c2h/device_and_stream.h>

#if _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC) && _CCCL_CUDACC_AT_LEAST(13, 4)

static_assert(cuda::std::is_default_constructible_v<cuda::shared_memory_barrier<>>);
static_assert(!cuda::std::is_constructible_v<cuda::shared_memory_barrier<>, cuda::std::ptrdiff_t>);
static_assert(!cuda::std::is_same_v<cuda::shared_memory_barrier<>::arrival_token, cuda::std::uint64_t>);
static_assert(cuda::std::is_default_constructible_v<cuda::shared_memory_barrier<>::arrival_token>);
static_assert(!cuda::std::is_constructible_v<cuda::shared_memory_barrier<>::arrival_token, int>);
static_assert(!cuda::std::is_convertible_v<int, cuda::shared_memory_barrier<>::arrival_token>);
static_assert(!cuda::std::is_copy_constructible_v<cuda::shared_memory_barrier<>::operation_status>);
static_assert(!cuda::std::is_copy_assignable_v<cuda::shared_memory_barrier<>::operation_status>);
static_assert(cuda::std::is_move_constructible_v<cuda::shared_memory_barrier<>::operation_status>);
static_assert(cuda::std::is_move_assignable_v<cuda::shared_memory_barrier<>::operation_status>);
static_assert(!cuda::std::is_convertible_v<cuda::shared_memory_barrier<>::operation_status, bool>);

static_assert(
  cuda::std::is_same_v<cuda::shared_memory_barrier<>, cuda::shared_memory_barrier<cuda::thread_scope_block>>);
static_assert(sizeof(cuda::shared_memory_barrier<cuda::thread_scope_block>) == sizeof(cuda::std::uint64_t));
static_assert(sizeof(cuda::shared_memory_barrier<cuda::thread_scope_cluster>) == sizeof(cuda::std::uint64_t));
static_assert(alignof(cuda::shared_memory_barrier<cuda::thread_scope_block>) == alignof(cuda::std::uint64_t));
static_assert(alignof(cuda::shared_memory_barrier<cuda::thread_scope_cluster>) == alignof(cuda::std::uint64_t));

template <class Fn>
TEST_DEVICE_FUNC void execute_on_thread_zero(Fn&& fn)
{
  if (threadIdx.x == 0)
  {
    fn();
  }
  __syncthreads();
}

template <class Fn0, class Fn1>
TEST_DEVICE_FUNC void concurrent_threads_launch(Fn0 fn0, Fn1 fn1)
{
  REQUIRE(blockDim.x == 2);

  __syncthreads();

  if (threadIdx.x == 0)
  {
    fn0();
  }
  else if (threadIdx.x == 1)
  {
    fn1();
  }

  __syncthreads();
}

template <int Id, cuda::thread_scope Scope>
TEST_DEVICE_FUNC cuda::shared_memory_barrier<Scope>* construct_barrier(int expected)
{
  // Each Id and scope create a distinct shared-memory object. The tests do not invalidate
  // these barriers because the memory is not reused for another purpose.
  alignas(cuda::shared_memory_barrier<Scope>) __shared__ char storage[sizeof(cuda::shared_memory_barrier<Scope>)];
  cuda::shared_memory_barrier<Scope>* bar = reinterpret_cast<cuda::shared_memory_barrier<Scope>*>(storage);
  execute_on_thread_zero([&] {
    new ((void*) bar) cuda::shared_memory_barrier<Scope>;
    init(bar, expected);
  });
  return bar;
}

template <cuda::thread_scope Scope>
TEST_DEVICE_FUNC void test_concurrent_arrive_and_wait()
{
  cuda::shared_memory_barrier<Scope>* bar = construct_barrier<0, Scope>(2);

  auto worker = [=] {
    for (int i = 0; i != 10; ++i)
    {
      bar->arrive_and_wait(cuda::ignore_status);
    }
  };

  concurrent_threads_launch(worker, worker);
}

template <cuda::thread_scope Scope>
TEST_DEVICE_FUNC void test_concurrent_arrive_wait()
{
  cuda::shared_memory_barrier<Scope>* bar = construct_barrier<1, Scope>(2);

  auto awaiter = [=] {
    auto token = bar->arrive();
    bar->wait(token, cuda::ignore_status);
  };
  auto arriver = [=] {
    (void) bar->arrive();
  };

  concurrent_threads_launch(awaiter, arriver);

  execute_on_thread_zero([&] {
    auto token = bar->arrive(2);
    bar->wait(token, cuda::ignore_status);
  });
}

template <cuda::thread_scope Scope>
TEST_DEVICE_FUNC void test_concurrent_arrive_and_drop()
{
  cuda::shared_memory_barrier<Scope>* bar = construct_barrier<2, Scope>(2);

  auto dropper = [=] {
    bar->arrive_and_drop();
  };
  auto arriver = [=] {
    bar->arrive_and_wait(cuda::ignore_status);
    bar->arrive_and_wait(cuda::ignore_status);
  };

  concurrent_threads_launch(dropper, arriver);
}

template <cuda::thread_scope Scope>
TEST_DEVICE_FUNC void test_concurrent_try_wait_for()
{
  cuda::shared_memory_barrier<Scope>* bar = construct_barrier<3, Scope>(2);
  cuda::std::chrono::nanoseconds delay(0);

  auto awaiter = [=] {
    auto token = bar->arrive();
    while (!bar->try_wait_for(token, delay, cuda::ignore_status))
    {
    }
  };
  auto arriver = [=] {
    (void) bar->arrive();
  };

  concurrent_threads_launch(awaiter, arriver);
}

template <cuda::thread_scope Scope>
TEST_DEVICE_FUNC void test_concurrent_try_wait_until()
{
  cuda::shared_memory_barrier<Scope>* bar = construct_barrier<4, Scope>(2);
  cuda::std::chrono::duration<int> delay(0);

  auto awaiter = [=] {
    auto token      = bar->arrive();
    auto until_time = cuda::std::chrono::system_clock::now() + delay;
    while (!bar->try_wait_until(token, until_time, cuda::ignore_status))
    {
    }
  };
  auto arriver = [=] {
    (void) bar->arrive();
  };

  concurrent_threads_launch(awaiter, arriver);
}

template <cuda::thread_scope Scope>
TEST_DEVICE_FUNC void test_concurrent_wait_phase()
{
  cuda::shared_memory_barrier<Scope>* bar = construct_barrier<5, Scope>(2);
  cuda::std::uint32_t phase               = 0;

  execute_on_thread_zero([&] {
    (void) bar->arrive();
  });

  auto awaiter = [=] {
    bar->wait(phase, cuda::ignore_status);
  };
  auto arriver = [=] {
    (void) bar->arrive();
  };

  concurrent_threads_launch(awaiter, arriver);

  execute_on_thread_zero([&] {
    (void) bar->arrive(2);
    bar->wait(phase + 1, cuda::ignore_status);
  });
}

template <cuda::thread_scope Scope>
TEST_DEVICE_FUNC void test_concurrent_try_wait_phase_for()
{
  cuda::shared_memory_barrier<Scope>* bar = construct_barrier<6, Scope>(2);
  cuda::std::uint32_t phase               = 0;
  cuda::std::chrono::nanoseconds delay(0);

  execute_on_thread_zero([&] {
    (void) bar->arrive();
  });

  auto awaiter = [=] {
    while (!bar->try_wait_for(phase, delay, cuda::ignore_status))
    {
    }
  };
  auto arriver = [=] {
    (void) bar->arrive();
  };

  concurrent_threads_launch(awaiter, arriver);
}

template <cuda::thread_scope Scope>
TEST_DEVICE_FUNC void test_concurrent_try_wait_phase_until()
{
  cuda::shared_memory_barrier<Scope>* bar = construct_barrier<7, Scope>(2);
  cuda::std::uint32_t phase               = 0;
  cuda::std::chrono::duration<int> delay(0);

  execute_on_thread_zero([&] {
    (void) bar->arrive();
  });

  auto awaiter = [=] {
    auto until_time = cuda::std::chrono::system_clock::now() + delay;
    while (!bar->try_wait_until(phase, until_time, cuda::ignore_status))
    {
    }
  };
  auto arriver = [=] {
    (void) bar->arrive();
  };

  concurrent_threads_launch(awaiter, arriver);
}

template <cuda::thread_scope Scope>
TEST_DEVICE_FUNC void test_concurrent_try_wait_conditional_phase_for()
{
  cuda::shared_memory_barrier<Scope>* bar = construct_barrier<11, Scope>(2);
  cuda::std::uint32_t phase               = 0;
  cuda::std::chrono::nanoseconds delay(0);

  execute_on_thread_zero([&] {
    (void) bar->arrive();
  });

  auto awaiter = [=] {
    while (!bar->try_wait_conditional_phase_for(phase, delay))
    {
    }
  };
  auto arriver = [=] {
    (void) bar->arrive();
  };

  concurrent_threads_launch(awaiter, arriver);
}

template <cuda::thread_scope Scope>
TEST_DEVICE_FUNC void test_concurrent_try_wait_conditional_phase_until()
{
  cuda::shared_memory_barrier<Scope>* bar = construct_barrier<12, Scope>(2);
  cuda::std::uint32_t phase               = 0;
  cuda::std::chrono::duration<int> delay(0);

  execute_on_thread_zero([&] {
    (void) bar->arrive();
  });

  auto awaiter = [=] {
    auto until_time = cuda::std::chrono::system_clock::now() + delay;
    while (!bar->try_wait_conditional_phase_until(phase, until_time))
    {
    }
  };
  auto arriver = [=] {
    (void) bar->arrive();
  };

  concurrent_threads_launch(awaiter, arriver);
}

template <cuda::thread_scope Scope>
TEST_DEVICE_FUNC void test_shared_memory_barrier_choreography()
{
  test_concurrent_arrive_and_wait<Scope>();
  test_concurrent_arrive_wait<Scope>();
  test_concurrent_arrive_and_drop<Scope>();
  test_concurrent_try_wait_for<Scope>();
  test_concurrent_try_wait_until<Scope>();
  test_concurrent_wait_phase<Scope>();
  test_concurrent_try_wait_phase_for<Scope>();
  test_concurrent_try_wait_phase_until<Scope>();
  test_concurrent_try_wait_conditional_phase_for<Scope>();
  test_concurrent_try_wait_conditional_phase_until<Scope>();
}

template <cuda::thread_scope Scope>
TEST_DEVICE_FUNC void check_success_status(const typename cuda::shared_memory_barrier<Scope>::operation_status& status)
{
  REQUIRE(status.complete());
  REQUIRE(!status.has_report());

  REQUIRE(status.get_error_count(cuda::status_source::generic_fabric) == 0);

  unsigned int visited = 0;
  status.for_each_error(cuda::status_source::generic_fabric, [&](cudaFabricOpStatusInfo) {
    ++visited;
  });
  REQUIRE(visited == 0);
}

template <cuda::thread_scope Scope>
TEST_DEVICE_FUNC void complete_tx(cuda::shared_memory_barrier<Scope>& bar, int transaction_count)
{
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_90,
    (cuda::ptx::mbarrier_complete_tx(
       cuda::ptx::sem_relaxed,
       cuda::ptx::scope_t<
         Scope == cuda::thread_scope_block ? cuda::ptx::dot_scope::cta : cuda::ptx::dot_scope::cluster>{},
       cuda::ptx::space_shared,
       cuda::device::barrier_native_handle(bar),
       static_cast<::cuda::std::uint32_t>(transaction_count));
     return;),
    NV_ANY_TARGET,
    (__trap();));
}

template <cuda::thread_scope Scope>
TEST_DEVICE_FUNC void test_tx_wait(cuda::shared_memory_barrier<Scope>* bar, int tx_count, int arrive_count = 1)
{
  auto token = bar->arrive_tx(arrive_count, tx_count);
  REQUIRE(!bar->test_wait(token, cuda::ignore_status));
  complete_tx<Scope>(*bar, tx_count);
  bar->wait(token, cuda::ignore_status);

  bar->expect_tx(tx_count);
  token = bar->arrive(arrive_count);
  complete_tx<Scope>(*bar, tx_count);
  bar->wait(token, cuda::ignore_status);

  token = bar->arrive_tx(arrive_count, tx_count);
  complete_tx<Scope>(*bar, tx_count);
  auto status = bar->wait(token, cuda::return_status);
  check_success_status<Scope>(status);
}

template <cuda::thread_scope Scope>
TEST_DEVICE_FUNC void test_tx_waits(cuda::shared_memory_barrier<Scope>* bar)
{
  test_tx_wait<Scope>(bar, 1);
  test_tx_wait<Scope>(bar, 1024);
}

template <cuda::thread_scope Scope>
TEST_DEVICE_FUNC void test_test_waits(cuda::shared_memory_barrier<Scope>* bar)
{
  typename cuda::shared_memory_barrier<Scope>::arrival_token token;
  execute_on_thread_zero([&] {
    token = bar->arrive();
    REQUIRE(!bar->test_wait(token, cuda::ignore_status));

    auto status = bar->test_wait(token, cuda::return_status);
    REQUIRE(!status.complete());
    REQUIRE(!status.has_report());

    // The other thread has not arrived, so every poll must return an incomplete status with no report.
    // Reassignment also checks that incomplete results can be overwritten without an inspection failure.
    for (int i = 0; i != 4; ++i)
    {
      status = bar->try_wait(token, cuda::return_status);
      REQUIRE(!status.complete());
      REQUIRE(!status.has_report());

      status = bar->try_wait_for(token, cuda::std::chrono::nanoseconds(1), cuda::return_status);
      REQUIRE(!status.complete());
      REQUIRE(!status.has_report());

      status = bar->try_wait_until(
        token, cuda::std::chrono::system_clock::now() - cuda::std::chrono::seconds(1), cuda::return_status);
      REQUIRE(!status.complete());
      REQUIRE(!status.has_report());

      status = bar->test_wait(0, cuda::return_status);
      REQUIRE(!status.complete());
      REQUIRE(!status.has_report());

      status = bar->try_wait(0, cuda::return_status);
      REQUIRE(!status.complete());
      REQUIRE(!status.has_report());

      status = bar->try_wait_for(0, cuda::std::chrono::nanoseconds(1), cuda::return_status);
      REQUIRE(!status.complete());
      REQUIRE(!status.has_report());

      status = bar->try_wait_until(
        0, cuda::std::chrono::system_clock::now() - cuda::std::chrono::seconds(1), cuda::return_status);
      REQUIRE(!status.complete());
      REQUIRE(!status.has_report());
    }
  });

  if (threadIdx.x != 0)
  {
    (void) bar->arrive();
  }

  __syncthreads();

  execute_on_thread_zero([&] {
    REQUIRE(bar->test_wait(token, cuda::ignore_status));

    auto status = bar->test_wait(token, cuda::return_status);
    check_success_status<Scope>(status);
  });
}

template <cuda::thread_scope Scope>
TEST_DEVICE_FUNC void test_ignore_status_waits(cuda::shared_memory_barrier<Scope>* bar)
{
  auto token = bar->arrive();
  while (!bar->try_wait(token, cuda::ignore_status))
  {
  }

  token = bar->arrive();
  bar->wait(token, cuda::ignore_status);

  bar->arrive_and_wait(cuda::ignore_status);
}

template <cuda::thread_scope Scope>
TEST_DEVICE_FUNC void test_status_waits(cuda::shared_memory_barrier<Scope>* bar)
{
  auto token       = bar->arrive();
  auto poll_status = bar->try_wait(token, cuda::return_status);
  if (poll_status.complete())
  {
    check_success_status<Scope>(poll_status);
  }

  auto status = bar->wait(token, cuda::return_status);
  check_success_status<Scope>(status);

  status = bar->arrive_and_wait(cuda::return_status);
  check_success_status<Scope>(status);

  bar->arrive_and_wait(cuda::ignore_status);
}

template <cuda::thread_scope Scope>
TEST_DEVICE_FUNC void test_phase_waits(cuda::shared_memory_barrier<Scope>* bar)
{
  (void) bar->arrive();
  bar->wait(0, cuda::ignore_status);

  REQUIRE(bar->test_wait(0, cuda::ignore_status));

  auto status = bar->test_wait(0, cuda::return_status);
  check_success_status<Scope>(status);

  bar->wait(0, cuda::ignore_status);

  status = bar->wait(0, cuda::return_status);
  check_success_status<Scope>(status);

  bar->wait_conditional_phase(0);
  REQUIRE(bar->test_wait_conditional_phase(0));
  REQUIRE(bar->try_wait_conditional_phase(0));
  REQUIRE(bar->try_wait_conditional_phase_for(0, cuda::std::chrono::nanoseconds(1)));
  REQUIRE(
    bar->try_wait_conditional_phase_until(0, cuda::std::chrono::system_clock::now() + cuda::std::chrono::seconds(1)));

  bar->arrive_and_wait(cuda::ignore_status);

  bar->wait(1, cuda::ignore_status);

  status = bar->wait(1, cuda::return_status);
  check_success_status<Scope>(status);

  bar->arrive_and_wait(cuda::ignore_status);

  REQUIRE(bar->try_wait_for(2, cuda::std::chrono::nanoseconds(1), cuda::ignore_status));

  status = bar->try_wait_for(2, cuda::std::chrono::nanoseconds(1), cuda::return_status);
  check_success_status<Scope>(status);

  bar->arrive_and_wait(cuda::ignore_status);

  REQUIRE(bar->try_wait_until(
    3, cuda::std::chrono::system_clock::now() + cuda::std::chrono::seconds(1), cuda::ignore_status));

  status =
    bar->try_wait_until(3, cuda::std::chrono::system_clock::now() + cuda::std::chrono::seconds(1), cuda::return_status);
  check_success_status<Scope>(status);
}

template <cuda::thread_scope Scope>
TEST_DEVICE_FUNC void test_shared_memory_barrier_common_extensions()
{
  cuda::shared_memory_barrier<Scope>* bar = construct_barrier<8, Scope>(blockDim.x);

  test_test_waits<Scope>(bar);
  test_ignore_status_waits<Scope>(bar);
  test_status_waits<Scope>(bar);
  test_phase_waits<Scope>(construct_barrier<10, Scope>(blockDim.x));
}

template <cuda::thread_scope Scope>
TEST_DEVICE_FUNC void test_shared_memory_barrier_sm90_extensions()
{
  cuda::shared_memory_barrier<Scope>* bar = construct_barrier<9, Scope>(blockDim.x);

  test_tx_waits<Scope>(bar);

  cuda::shared_memory_barrier<Scope>* batched_bar = construct_barrier<13, Scope>(2 * blockDim.x);
  test_tx_wait<Scope>(batched_bar, 1, 2);
  test_tx_wait<Scope>(batched_bar, 1024, 2);
}

template <cuda::thread_scope Scope>
__global__ void test_shared_memory_barrier_kernel()
{
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (test_shared_memory_barrier_choreography<Scope>(); test_shared_memory_barrier_common_extensions<Scope>();
                test_shared_memory_barrier_sm90_extensions<Scope>();))
}

#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC) && _CCCL_CUDACC_AT_LEAST(13, 4)

C2H_TEST("shared_memory_barrier", "[barrier]")
{
#if _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC) && _CCCL_CUDACC_AT_LEAST(13, 4)
  // Do not reference a kernel until the driver is known to provide the fabric status decoder device symbols.
  if (cuda::__driver::__version_below(13, 4))
  {
    return;
  }

  auto device = c2h::current_device();
  if (device.attribute(cuda::device_attributes::compute_capability_major) < 9)
  {
    return;
  }

  cuda::stream stream{device};
  const auto config = cuda::make_config(cuda::block_dims<2>(), cuda::grid_dims<1>());

  SECTION("block scope")
  {
    cuda::launch(stream, config, test_shared_memory_barrier_kernel<cuda::thread_scope_block>);
    stream.sync();
  }
  SECTION("cluster scope on local shared memory")
  {
    cuda::launch(stream, config, test_shared_memory_barrier_kernel<cuda::thread_scope_cluster>);
    stream.sync();
  }
#else // ^^^ CUDA 13.4+ ^^^ / vvv older toolkits vvv
  return;
#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC) && _CCCL_CUDACC_AT_LEAST(13, 4)
}
