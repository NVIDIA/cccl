// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/thread/thread_load.cuh>

#include "cub_test_macros.h"

struct non_default_constructible
{
  int a;
  int b;

  __host__ __device__ explicit non_default_constructible(int value)
      : a(value)
      , b(-value)
  {}
};

template <cub::CacheLoadModifier Modifier, typename T>
__global__ void thread_load_kernel(const T* input, T* output)
{
  output[threadIdx.x] = cub::ThreadLoad<Modifier>(input + threadIdx.x);
}

using modifiers =
  c2h::enum_type_list<cub::CacheLoadModifier,
                      cub::LOAD_DEFAULT,
                      cub::LOAD_CA,
                      cub::LOAD_CG,
                      cub::LOAD_CS,
                      cub::LOAD_CV,
                      cub::LOAD_LDG,
                      cub::LOAD_VOLATILE>;

CUB_TEST("ThreadLoad works with non-default-constructible types", "[thread][load]", CUB_SMALL, modifiers)
{
  using T                 = non_default_constructible;
  constexpr auto modifier = c2h::get<0, TestType>::value;
  constexpr int num_items = 128;

  c2h::host_vector<T> h_input;
  for (int i = 0; i < num_items; i++)
  {
    h_input.push_back(T(i));
  }
  const c2h::device_vector<T> d_input(h_input);
  c2h::device_vector<T> d_output(num_items, T(0));

  thread_load_kernel<modifier>
    <<<1, num_items>>>(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_output.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  const c2h::host_vector<T> h_output(d_output);
  for (int i = 0; i < num_items; i++)
  {
    REQUIRE(h_output[i].a == i);
    REQUIRE(h_output[i].b == -i);
  }
}
