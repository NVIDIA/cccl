//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: nvrtc

#include <cuda/std/cmath>
#include <cuda/std/type_traits>

#include <cassert>
#include <cstdio>

#define CUDA_SAFE_CALL(...)                                                           \
  do                                                                                  \
  {                                                                                   \
    cudaError_t err = __VA_ARGS__;                                                    \
    if (err != cudaSuccess)                                                           \
    {                                                                                 \
      printf("CUDA ERROR: %s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err)); \
      exit(1);                                                                        \
    }                                                                                 \
  } while (false)

template <typename T, typename F>
void generate(const F& f, T* buffer, cuda::std::size_t size)
{
  for (auto i = 0ull; i < size; ++i)
  {
    buffer[i] = f(i);
  }
}

template <typename T, typename F>
void generate(const F& f, T* buffer, cuda::std::size_t size, T other)
{
  if (!cuda::std::isfinite(float(other)))
  {
    return;
  }
  for (auto i = 0ull; i < size; ++i)
  {
    buffer[i] = f(other, i);
  }
}

template <typename T, typename F, typename Head, typename... Args>
void generate(const F& f, T* buffer, cuda::std::size_t size, Head head, Args... args)
{
  for (auto i = 0ull; i < size; ++i)
  {
    buffer[i] = f(head, args..., i);
  }
}

template <typename T, typename F, typename... Args>
__global__ void generate_kernel(const F& f, T* buffer, cuda::std::size_t, Args... args)
{
  cuda::std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  buffer[index]           = f(args..., index);
}

template <typename T, cuda::std::size_t Dims, cuda::std::size_t Bitpatterns = 1ull << (sizeof(T) * CHAR_BIT)>
struct calculate_problem_sizes
{
  static const constexpr auto bitpatterns  = Bitpatterns;
  static const constexpr auto problem_size = Bitpatterns * calculate_problem_sizes<T, Dims - 1>::problem_size;

  template <typename F, typename... Args>
  static bool run(const F& f, Args... args)
  {
    bool good = true;
    for (auto i = 0ull; i < bitpatterns; ++i)
    {
      good = calculate_problem_sizes<T, Dims - 1>::run(f, args..., i) && good;
    }
    return good;
  }
};

template <typename T, cuda::std::size_t Bitpatterns>
struct calculate_problem_sizes<T, 1, Bitpatterns>
{
  static const constexpr auto bitpatterns  = Bitpatterns;
  static const constexpr auto problem_size = Bitpatterns;

  template <typename F, typename... Args>
  static bool run(const F& f, Args... args)
  {
    T* host_buffer = new T[problem_size]();

    T* device_buffer = nullptr;
    CUDA_SAFE_CALL(cudaMallocManaged(&device_buffer, sizeof(T) * problem_size));
    CUDA_SAFE_CALL(cudaMemset(device_buffer, 0, sizeof(T) * problem_size));

    generate(f, host_buffer, problem_size, args...);
    generate_kernel<<<problem_size / 256, 256>>>(f, device_buffer, problem_size, args...);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaGetLastError());

    bool good = true;

    for (cuda::std::size_t i = 0ull; i < problem_size; ++i)
    {
      if (memcmp(host_buffer + i, device_buffer + i, sizeof(T)) != 0)
      {
        printf("[%zu] unmatched, values = %+.10f, host = %+.10f, device = %+.10f\n",
               i,
               float(__half(__half_raw{(unsigned short) i})),
               (float) host_buffer[i],
               (float) device_buffer[i]);
        good = false;
      }
    }

    CUDA_SAFE_CALL(cudaFree(device_buffer));
    delete[] host_buffer;

    return good;
  }
};

template <typename T, cuda::std::size_t Bitpatterns>
struct calculate_problem_sizes<T, 0, Bitpatterns>
{
  static_assert(Bitpatterns == 0, "can't have 0 dims");
};

template <typename T, cuda::std::size_t Dims = 1, typename F>
void compare_host_device(const F& f)
{
  using sizes = calculate_problem_sizes<T, Dims>;

  auto good = sizes::run(f);
  fflush(stdout);
  assert(good);
}
