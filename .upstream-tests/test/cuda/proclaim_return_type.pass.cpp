//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03
// UNSUPPORTED: gcc-4

#include <cuda/functional>

#include <cuda/std/cassert>

template <class T, class Fn, class... As>
__host__ __device__
void test_proclaim_return_type(Fn&& fn, T expected, As... as)
{
  {
    auto f1 = cuda::proclaim_return_type<T>(cuda::std::forward<Fn>(fn));

    assert(f1(as...) == expected);

    auto f2{f1};
    assert(cuda::std::move(f2)(as...) == expected);
  }

  {
    const auto f1 = cuda::proclaim_return_type<T>(fn);

    assert(f1(as...) == expected);

    auto f2{f1};
    assert(cuda::std::move(f2)(as...) == expected);
  }
}

struct hd_callable
{
  __host__ __device__ int operator()() const& { return 42; }
  __host__ __device__ int operator()() const&& { return 42; }
};

#if !defined(__CUDACC_RTC__)
struct h_callable
{
  __host__ int operator()() const& { return 42; }
  __host__ int operator()() const&& { return 42; }
};
#endif

struct d_callable
{
  __device__ int operator()() const& { return 42; }
  __device__ int operator()() const&& { return 42; }
};

int main(int argc, char ** argv)
{
#ifdef __CUDA_ARCH__
#define TEST_SPECIFIER(...)                                                    \
  {                                                                            \
    test_proclaim_return_type<double>([] __VA_ARGS__ () { return 42.0; },      \
                                      42.0);                                   \
    test_proclaim_return_type<int>([] __VA_ARGS__ (int v) { return v * 2; },   \
                                   42, 21);                                    \
                                                                               \
    int v = 42;                                                                \
    int* vp = &v;                                                              \
    test_proclaim_return_type<int&>(                                           \
        [vp] __VA_ARGS__ () -> int& { return *vp; }, v);                       \
  }

#if !defined(__CUDACC_RTC__)
  TEST_SPECIFIER(__device__)
  TEST_SPECIFIER(__host__ __device__)
#endif
  TEST_SPECIFIER()
#undef TEST_SPECIFIER

  test_proclaim_return_type<int>(hd_callable{}, 42);
  test_proclaim_return_type<int>(d_callable{}, 42);
#else
  test_proclaim_return_type<int>(h_callable{}, 42);
#endif

  return 0;
}
