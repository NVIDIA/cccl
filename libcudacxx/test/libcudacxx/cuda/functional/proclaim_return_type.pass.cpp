//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: gcc-4

#include <cuda/functional>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

#if _CCCL_CUDACC_AT_LEAST(12, 3)
template <typename ReturnT>
__host__ __device__ void test_lambda_return_type()
{
  // Ensure type can be queried from cuda::std::invoke_result_t
  auto d_lm = [] __device__() -> ReturnT {
    return ReturnT{};
  };
  auto hd_lm = [] __host__ __device__() -> ReturnT { return ReturnT{}; };
  using Td   = decltype(d_lm);
  using Thd  = decltype(hd_lm);

  static_assert(cuda::std::is_same_v<cuda::std::invoke_result_t<Td>, ReturnT>);
  static_assert(cuda::std::is_same_v<cuda::std::invoke_result_t<Thd>, ReturnT>);
  unused(d_lm);
  unused(hd_lm);
}

struct custom_type
{
  char arr[16];
};
#endif // _CCCL_CUDACC_AT_LEAST(12, 3)

template <class T, class Fn, class... As>
__host__ __device__ void test_proclaim_return_type(Fn&& fn, T expected, As... as)
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
  __host__ __device__ int operator()() const&
  {
    return 42;
  }
  __host__ __device__ int operator()() const&&
  {
    return 42;
  }
};

#if !TEST_COMPILER(NVRTC)
struct h_callable
{
  __host__ int operator()() const&
  {
    return 42;
  }
  __host__ int operator()() const&&
  {
    return 42;
  }
};
#endif // !TEST_COMPILER(NVRTC)

struct d_callable
{
  __device__ int operator()() const&
  {
    return 42;
  }
  __device__ int operator()() const&&
  {
    return 42;
  }
};

int main(int argc, char** argv)
{
  int v   = 42;
  int* vp = &v;

  test_proclaim_return_type<int>(hd_callable{}, 42);
  test_proclaim_return_type<double>(
    [] {
      return 42.0;
    },
    42.0);
  test_proclaim_return_type<int>(
    [](const int v) {
      return v * 2;
    },
    42,
    21);
  test_proclaim_return_type<int&>(
    [vp]() -> int& {
      return *vp;
    },
    v);

  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (test_proclaim_return_type<int>(d_callable{}, 42);),
                    (test_proclaim_return_type<int>(h_callable{}, 42);))

  // execution space annotations on lambda require --extended-lambda flag with nvrtc
#if !TEST_COMPILER(NVRTC)
  NV_IF_TARGET(
    NV_IS_DEVICE,
    (test_proclaim_return_type<double>(
       [] __device__ {
         return 42.0;
       },
       42.0);
     test_proclaim_return_type<int>(
       [] __device__(const int v) {
         return v * 2;
       },
       42,
       21);
     test_proclaim_return_type<int&>(
       [vp] __device__() -> int& {
         return *vp;
       },
       v);

     test_proclaim_return_type<double>(
       [] __host__ __device__ {
         return 42.0;
       },
       42.0);
     test_proclaim_return_type<int>(
       [] __host__ __device__(const int v) {
         return v * 2;
       },
       42,
       21);
     test_proclaim_return_type<int&>(
       [vp] __host__ __device__() -> int& {
         return *vp;
       },
       v);))

  // Ensure that we can always declare functions even on host
  auto f = cuda::proclaim_return_type<bool>([] __device__() {
    return false;
  });
  auto g = cuda::proclaim_return_type<bool>([f] __device__() {
    return f();
  });
  unused(f);
  unused(g);

#  if _CCCL_CUDACC_AT_LEAST(12, 3)
  test_lambda_return_type<int>();
  test_lambda_return_type<long>();
  test_lambda_return_type<bool>();
  test_lambda_return_type<float>();
  test_lambda_return_type<double>();
  test_lambda_return_type<char>();
  test_lambda_return_type<custom_type>();
#  endif // _CCCL_CUDACC_AT_LEAST(12, 3)
#endif // !TEST_COMPILER(NVRTC)

  return 0;
}
