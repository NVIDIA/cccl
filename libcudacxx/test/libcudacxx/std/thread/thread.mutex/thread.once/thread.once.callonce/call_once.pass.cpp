//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/mutex>

// struct once_flag;
// template<class Callable, class... Args>
//   void call_once(once_flag& flag, Callable&& func, Args&&... args);

// UNSUPPORTED: nvrtc
// UNSUPPORTED: force-tile
// The host checks below use std::thread. Device call_once is covered by the CUDA launch in main.

// UNSUPPORTED: libcpp-has-no-threads

#include <cuda/std/cassert>
#include <cuda/std/mutex>
#include <cuda/std/type_traits>

#include <chrono>
#include <thread>
#if _CCCL_CUDA_COMPILATION()
#  include <cuda_runtime.h>
#endif // _CCCL_CUDA_COMPILATION()

#include "test_macros.h"

constexpr cuda::std::once_flag global_flag{};

static_assert(sizeof(global_flag) >= sizeof(::cuda::std::__cccl_mutex_t), "");

static_assert(cuda::std::is_same_v<cuda::std::once_flag, cuda::once_flag<>>, "");
static_assert(cuda::std::is_same_v<cuda::std::once_flag, cuda::once_flag<cuda::thread_scope_system>>, "");
static_assert(cuda::once_flag<cuda::thread_scope_block>::_Scope == cuda::thread_scope_block, "");

static_assert(noexcept(cuda::std::once_flag{}), "");
static_assert(!cuda::std::is_copy_constructible_v<cuda::std::once_flag>, "");
static_assert(!cuda::std::is_copy_assignable_v<cuda::std::once_flag>, "");
static_assert(!cuda::std::is_move_constructible_v<cuda::std::once_flag>, "");
static_assert(!cuda::std::is_move_assignable_v<cuda::std::once_flag>, "");

void nothrow_init() noexcept {}

struct NothrowArgs
{
  void operator()(int, int) const noexcept {}
};

void maybe_throw_init() {}

cuda::std::once_flag flag0;

static_assert(noexcept(cuda::std::call_once(flag0, nothrow_init)), "");
static_assert(noexcept(cuda::std::call_once(flag0, NothrowArgs{}, 1, 2)), "");
static_assert(!noexcept(cuda::std::call_once(flag0, maybe_throw_init)), "");

int init0_called = 0;

void init0()
{
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  ++init0_called;
}

void f0()
{
  cuda::std::call_once(flag0, init0);
}

struct InitArgs
{
  static int called;

  void operator()(int i, int j)
  {
    called += i + j;
  }
};

int InitArgs::called = 0;

cuda::std::once_flag flag_args;

void f_args()
{
  cuda::std::call_once(flag_args, InitArgs(), 2, 3);
  cuda::std::call_once(flag_args, InitArgs(), 4, 5);
}

cuda::std::once_flag flag_exception;

int exception_called    = 0;
int exception_completed = 0;

void init_exception()
{
  ++exception_called;
  if (exception_called == 1)
  {
    TEST_THROW(1);
  }
  ++exception_completed;
}

void f_exception()
{
#if TEST_HAS_EXCEPTIONS()
  try
  {
    cuda::std::call_once(flag_exception, init_exception);
  }
  catch (...)
  {}
#endif // TEST_HAS_EXCEPTIONS()
}

cuda::std::once_flag flag41;
cuda::std::once_flag flag42;

int init41_called = 0;
int init42_called = 0;

void init41()
{
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  ++init41_called;
}

void init42()
{
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  ++init42_called;
}

void f41()
{
  cuda::std::call_once(flag41, init41);
  cuda::std::call_once(flag42, init42);
}

void f42()
{
  cuda::std::call_once(flag42, init42);
  cuda::std::call_once(flag41, init41);
}

#if _CCCL_CUDA_COMPILATION()
__device__ cuda::std::once_flag device_flag{};
__device__ int device_calls = 0;

__device__ void device_init()
{
  atomicAdd(&device_calls, 1);
}

__global__ void call_once_kernel()
{
  cuda::std::call_once(device_flag, device_init);
}
#endif // _CCCL_CUDA_COMPILATION()

void test_host()
{
  {
    std::thread t0(f0);
    std::thread t1(f0);
    t0.join();
    t1.join();
    assert(init0_called == 1);
  }

  {
    std::thread t0(f_args);
    std::thread t1(f_args);
    t0.join();
    t1.join();
    assert(InitArgs::called == 5);
  }

#if TEST_HAS_EXCEPTIONS()
  {
    std::thread t0(f_exception);
    std::thread t1(f_exception);
    t0.join();
    t1.join();
    assert(exception_called == 2);
    assert(exception_completed == 1);
  }
#endif // TEST_HAS_EXCEPTIONS()

  {
    std::thread t0(f41);
    std::thread t1(f42);
    t0.join();
    t1.join();
    assert(init41_called == 1);
    assert(init42_called == 1);
  }

#if _CCCL_CUDA_COMPILATION()
  {
    int zero = 0;
    assert(cudaMemcpyToSymbol(device_calls, &zero, sizeof(zero)) == cudaSuccess);
    call_once_kernel<<<1, 128>>>();
    assert(cudaDeviceSynchronize() == cudaSuccess);
    int calls = 0;
    assert(cudaMemcpyFromSymbol(&calls, device_calls, sizeof(calls)) == cudaSuccess);
    assert(calls == 1);
    call_once_kernel<<<1, 128>>>();
    assert(cudaDeviceSynchronize() == cudaSuccess);
    assert(cudaMemcpyFromSymbol(&calls, device_calls, sizeof(calls)) == cudaSuccess);
    assert(calls == 1);
  }
#endif // _CCCL_CUDA_COMPILATION()
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test_host();))
  return 0;
}
