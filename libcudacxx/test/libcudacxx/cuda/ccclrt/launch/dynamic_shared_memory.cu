//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

template <class T, class View>
struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    static_assert(cuda::std::is_same_v<View, decltype(cuda::dynamic_shared_memory(config))>);
    static_assert(noexcept(cuda::dynamic_shared_memory(config)));

    write_smem(cuda::dynamic_shared_memory(config));
  }

  __device__ void write_smem(T& view)
  {
    view = T{};
    CCCLRT_REQUIRE_DEVICE(view == T{});
  }

  template <cuda::std::size_t N>
  __device__ void write_smem(cuda::std::span<T, N> view)
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
  cuda::device_ref device = cuda::devices[0];
  cuda::stream stream{device};

  test_ref(stream);
  test_span(stream);
}
