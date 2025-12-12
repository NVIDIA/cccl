//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Include this first
#include <cuda/experimental/execution.cuh>

// Then include the test helpers
#include <thrust/equal.h>

#include <cuda/experimental/container.cuh>
#include <cuda/experimental/memory_resource.cuh>

#include <nv/target>

#include "testing.cuh" // IWYU pragma: keep

_CCCL_BEGIN_NV_DIAG_SUPPRESS(177) // function "_is_on_device" was declared but never referenced

namespace ex = cuda::experimental::execution;

__host__ __device__ bool _is_on_device() noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, //
                    ({ return false; }),
                    ({ return true; }));
}

struct _say_hello
{
  __device__ int operator()() const
  {
    CUDAX_CHECK(_is_on_device());
    printf("Hello from lambda on device!\n");
    return value;
  }

  int value;
};

// This is an "un-visitable" sender that does not have a tag type.
template <class Sndr>
struct _CCCL_TYPE_VISIBILITY_DEFAULT unknown_sender : Sndr
{
  _CCCL_API explicit unknown_sender(Sndr sndr) noexcept
      : Sndr(cuda::std::move(sndr))
  {}
};

void stream_context_test1()
{
  ex::stream_context ctx{cuda::device_ref{0}};
  auto sched = ctx.get_scheduler();
  static_assert(ex::__is_scheduler<decltype(sched)>);

  auto sndr = ex::schedule(sched) //
            | ex::then([] __host__ __device__() noexcept -> bool {
                return _is_on_device();
              });

  auto [on_device] = ex::sync_wait(std::move(sndr)).value();
  CHECK(on_device);
}

void stream_context_test2()
{
  ex::thread_context tctx;
  ex::stream_context sctx{cuda::device_ref{0}};
  auto sch = sctx.get_scheduler();

  auto start = //
    ex::schedule(sch) // begin work on the GPU
    | ex::then(_say_hello{42}) // enqueue a function object on the GPU
    | ex::then([] __device__(int i) noexcept -> int { // enqueue a lambda on the GPU
        CUDAX_CHECK(_is_on_device());
        printf("Hello again from lambda on device! i = %d\n", i);
        return i + 1;
      })
    | ex::continues_on(tctx.get_scheduler()) // continue work on the CPU
    | ex::then([] __host__ __device__(int i) -> int { // run a lambda on the CPU
        CUDAX_CHECK(!_is_on_device());
        NV_IF_TARGET(NV_IS_HOST,
                     (printf("Hello from lambda on host! i = %d\n", i);),
                     (printf("OOPS! still on the device! i = %d\n", i);))
        return i;
      });

  // run the ex, wait for it to finish, and get the result
  auto [i] = ex::sync_wait(std::move(start)).value();
  CHECK(i == 43);
  printf("All done on the host! result = %d\n", i);
}

void stream_ref_as_scheduler()
{
  ex::thread_context tctx;
  cudax::stream sctx{cuda::device_ref{0}};
  auto sch = sctx.get_scheduler();
  static_assert(ex::__is_scheduler<decltype(sch)>);

  auto start = //
    ex::schedule(sch) // begin work on the GPU
    | ex::then(_say_hello{42}) // enqueue a function object on the GPU
    | ex::then([] __device__(int i) noexcept -> int { // enqueue a lambda on the GPU
        CUDAX_CHECK(_is_on_device());
        printf("Hello again from lambda on device! i = %d\n", i);
        return i + 1;
      })
    | ex::continues_on(tctx.get_scheduler()) // continue work on the CPU
    | ex::then([] __host__ __device__(int i) noexcept -> int { // run a lambda on the CPU
        CUDAX_CHECK(!_is_on_device());
        NV_IF_TARGET(NV_IS_HOST,
                     (printf("Hello from lambda on host! i = %d\n", i);),
                     (printf("OOPS! still on the device! i = %d\n", i);))
        return i;
      });

  // run the ex, wait for it to finish, and get the result
  auto [i] = ex::sync_wait(std::move(start)).value();
  CHECK(i == 43);
  printf("All done on the host! result = %d\n", i);
}

void bulk_on_stream_scheduler()
{
  cuda::device_ref _dev{0};
  cudax::stream sctx{_dev};
  auto sch = sctx.get_scheduler();

  using _env_t = cudax::env_t<cuda::mr::device_accessible>;
  auto mr      = cuda::device_default_memory_pool(_dev);
  auto mr2     = cuda::mr::any_resource<cuda::mr::device_accessible>(mr);
  _env_t env{mr, cuda::get_stream(sch), ex::par_unseq};
  auto buf = cuda::make_buffer<int>(sctx, mr2, 10, 40, env); // a device buffer of 10 integers, initialized to 40
  cuda::std::span data{buf};

  auto start = //
    ex::schedule(sch) // begin work on the GPU
    | ex::then([data] __host__ __device__() -> cuda::std::span<int> {
        printf("Hello from lambda on device!\n");
        return data;
      })
    // enqueue a bulk kernel on the GPU
    | ex::bulk(ex::par_unseq, 10, [] __host__ __device__(int i, cuda::std::span<int> data) -> void {
        printf("Hello from bulk kernel on device! i = %d\n", i);
        CUDAX_CHECK(_is_on_device());
        CUDAX_CHECK(i < data.size());
        data[i] += 2;
      });

  auto expected = cuda::make_buffer<int>(sctx, mr2, 10, 42, env); // a device buffer of 10 integers, initialized
                                                                  // to 42

  // start the sender and wait for it to finish
  auto [span] = ex::sync_wait(std::move(start)).value();

  CHECK(thrust::equal(thrust::device, span.begin(), span.end(), expected.begin()));
}

void stream_adapt_non_visitable_sender()
{
  ex::stream_context ctx{cuda::device_ref{0}};
  auto with_sched = ex::prop{ex::get_scheduler, ctx.get_scheduler()};

  auto sndr = unknown_sender{ex::just(42)};
  auto [i]  = ex::sync_wait(sndr, with_sched).value();
  CHECK(i == 42);
}

void starts_on_with_stream_scheduler1()
{
  cuda::device_ref _dev{0};
  cudax::stream sctx{_dev};
  ex::thread_context tctx;
  auto sch = sctx.get_scheduler();

  auto start = ex::starts_on(sch, ex::just() | ex::then([] __device__() noexcept -> int {
                                    return 42;
                                  }));

  auto [i] = ex::sync_wait(std::move(start)).value();
  CHECK(i == 42);
}

void starts_on_with_stream_scheduler2()
{
  cuda::device_ref _dev{0};
  cudax::stream sctx{_dev};
  ex::thread_context tctx;
  auto sch = sctx.get_scheduler();

  auto start =
    ex::starts_on(sch, ex::just() | ex::then([] __device__() noexcept -> int {
                         return 42;
                       }))
    | ex::continues_on(tctx.get_scheduler()) // continue work on the CPU
    | ex::then([] __host__ __device__(int i) noexcept -> int {
        return i + 1;
      });

  auto [i] = ex::sync_wait(std::move(start)).value();
  CHECK(i == 43);
}

namespace
{
// Test code is placed in separate functions to avoid an nvc++ issue with
// extended lambdas in functions with internal linkage (as is the case
// with C2H tests).

C2H_TEST("a simple use of the stream context", "[context][stream]")
{
  REQUIRE_NOTHROW(stream_context_test1());
}

C2H_TEST("another simple use of the stream context", "[context][stream]")
{
  REQUIRE_NOTHROW(stream_context_test2());
}

C2H_TEST("use stream_ref as a scheduler", "[context][stream]")
{
  REQUIRE_NOTHROW(stream_ref_as_scheduler());
}

C2H_TEST("launch a bulk kernel", "[context][stream]")
{
  REQUIRE_NOTHROW(bulk_on_stream_scheduler());
}

C2H_TEST("run an unknown sender on a stream", "[context][stream]")
{
  REQUIRE_NOTHROW(stream_adapt_non_visitable_sender());
}

C2H_TEST("use starts_on with a stream scheduler", "[context][stream]")
{
  SECTION("starts_on that completes on the stream scheduler")
  {
    REQUIRE_NOTHROW(starts_on_with_stream_scheduler1());
  }

  SECTION("starts_on that completes on the thread scheduler")
  {
    REQUIRE_NOTHROW(starts_on_with_stream_scheduler2());
  }
}
} // namespace
