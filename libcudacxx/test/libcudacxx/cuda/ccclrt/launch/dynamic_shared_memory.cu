//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/std/cstddef>
#include <cuda/std/functional>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <testing.cuh>

#include "test_macros.h"

template <class T, class View>
struct TestKernel
{
  template <class Config>
  TEST_DEVICE_FUNC void operator()(const Config& config)
  {
    static_assert(cuda::std::is_same_v<View, decltype(cuda::dynamic_shared_memory(config))>);
    static_assert(noexcept(cuda::dynamic_shared_memory(config)));

    write_smem(cuda::dynamic_shared_memory(config));
  }

  TEST_DEVICE_FUNC void write_smem(T& view)
  {
    view = T{};
    CCCLRT_REQUIRE_DEVICE(view == T{});
  }

  template <cuda::std::size_t N>
  TEST_DEVICE_FUNC void write_smem(cuda::std::span<T, N> view)
  {
    for (cuda::std::size_t i = 0; i < view.size(); ++i)
    {
      view[i] = T{};
      CCCLRT_REQUIRE_DEVICE(view[i] == T{});
    }
  }
};

template <class T, class View, class Opt>
void test_opt_and_launch(cuda::stream_ref stream, Opt opt)
{
  static_assert(cuda::std::is_same_v<T, typename Opt::value_type>);
  static_assert(cuda::std::is_same_v<View, typename Opt::view_type>);

  const auto config = cuda::make_config(cuda::block_dims<1, 1>(), cuda::grid_dims<1, 1>(), opt);
  cuda::launch(stream, config, TestKernel<T, View>{});
  stream.sync();
}

template <class T>
void test_ref(cuda::stream_ref stream)
{
  static_assert(noexcept(cuda::dynamic_shared_memory<T>()));
  test_opt_and_launch<T, T&>(stream, cuda::dynamic_shared_memory<T>());
}

void test_ref(cuda::stream_ref stream)
{
  test_ref<int>(stream);
  test_ref<float>(stream);
  test_ref<double*>(stream);
  test_ref<void (*)()>(stream);
}

template <class T, cuda::std::size_t N>
void test_span(cuda::stream_ref stream)
{
  static_assert(!noexcept(cuda::dynamic_shared_memory<T[]>(N * 1024 * 1024)));
  test_opt_and_launch<T, cuda::std::span<T>>(stream, cuda::dynamic_shared_memory<T[]>(N));

  static_assert(noexcept(cuda::dynamic_shared_memory<T[N]>()));
  test_opt_and_launch<T, cuda::std::span<T, N>>(stream, cuda::dynamic_shared_memory<T[N]>());
}

void test_span(cuda::stream_ref stream)
{
  test_span<int, 1>(stream);
  test_span<int, 256>(stream);
  test_span<float, 1>(stream);
  test_span<float, 256>(stream);
  test_span<double*, 1>(stream);
  test_span<double*, 256>(stream);
  test_span<void (*)(), 1>(stream);
  test_span<void (*)(), 256>(stream);
}

C2H_TEST("Dynamic shared memory option", "[launch]")
{
  const cuda::device_ref device = cuda::devices[0];
  const cuda::stream stream{device};

  test_ref(stream);
  test_span(stream);
}

#if _CCCL_CTK_AT_LEAST(13, 2)
__global__ void shared_memory_mode_kernel()
{
  extern __shared__ int shared[];
  shared[0] = 42;
  CCCLRT_REQUIRE_DEVICE(shared[0] == 42);
}

C2H_TEST("Dynamic shared memory preserves the function limit", "[launch]")
{
  if (cuda::__driver::__version_below(13, 2))
  {
    SKIP("per-launch shared memory mode requires a CUDA 13.2 driver");
  }

  cuda::device_ref device = cuda::devices[0];
  cuda::stream stream{device};
  CUDART(cudaFuncSetAttribute(shared_memory_mode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 0));

  auto dims = cuda::make_hierarchy(cuda::block_dims<1>(), cuda::grid_dims<1>());
  SECTION("Portable shared memory")
  {
    auto config = cuda::make_config(dims, cuda::dynamic_shared_memory<int[128]>());
    cuda::launch(stream, config, shared_memory_mode_kernel);
  }
  SECTION("Non-portable shared memory")
  {
    constexpr int num_ints = 13 * 1024;
    int max_shared_memory{};
    CUDART(cudaDeviceGetAttribute(&max_shared_memory, cudaDevAttrMaxSharedMemoryPerBlockOptin, device.get()));
    if (static_cast<cuda::std::size_t>(max_shared_memory) < num_ints * sizeof(int))
    {
      SKIP("device does not support the requested non-portable shared memory size");
    }
    auto config = cuda::make_config(dims, cuda::dynamic_shared_memory<int[num_ints]>(cuda::non_portable));
    cuda::launch(stream, config, shared_memory_mode_kernel);
  }
  stream.sync();

  cudaFuncAttributes attributes{};
  CUDART(cudaFuncGetAttributes(&attributes, shared_memory_mode_kernel));
  CCCLRT_REQUIRE(attributes.maxDynamicSharedSizeBytes == 0);
}
#endif // _CCCL_CTK_AT_LEAST(13, 2)
