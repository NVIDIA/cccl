//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// todo(dabayer): Find a way to make this work for nvrtc.
// nvrtc doesn't allow accessing the static constexpr const auto& value member.
// UNSUPPORTED: nvrtc

// REQUIRES: !c++17

// constant_wrapper

// static constexpr decltype(auto) value = (X);
// using type = constant_wrapper;
// using value_type = decltype(X);

#include <cuda/std/algorithm>
#include <cuda/std/concepts>
#include <cuda/std/utility>

#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(20094) // a host member cannot be directly read in a __device__/__global__ function

static_assert(cuda::std::__constant_wrapper<42>::value == 42);
static_assert(cuda::std::same_as<decltype(cuda::std::__constant_wrapper<42>::value), const int>);
static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<42>::type, cuda::std::__constant_wrapper<42>>);
static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<42>::value_type, int>);

struct S
{
  int member = 42;
};

constexpr cuda::std::__constant_wrapper<S{5}> s_value;
using SValue = cuda::std::remove_const_t<decltype(s_value)>;

static_assert(s_value.value.member == 5);

// nvcc 12.0 fails to properly generate input file for host compiler.
#if !(TEST_CUDA_COMPILER(NVCC, ==, 12, 0) && _CCCL_HOST_COMPILATION())
static_assert(cuda::std::same_as<decltype(SValue::value), const S&>);
#endif // !(TEST_CUDA_COMPILER(NVCC, ==, 12, 0) && _CCCL_HOST_COMPILATION())

static_assert(cuda::std::same_as<SValue::type, SValue>);

// nvcc 12.9 + 13.0 fails to properly generate input file for host compiler.
#if !((TEST_CUDA_COMPILER(NVCC, ==, 12, 9) || TEST_CUDA_COMPILER(NVCC, ==, 13, 0)) && _CCCL_HOST_COMPILATION())
static_assert(cuda::std::same_as<SValue::value_type, S>);
#endif // !((TEST_CUDA_COMPILER(NVCC, ==, 12, 9) || TEST_CUDA_COMPILER(NVCC, ==, 13, 0)) && _CCCL_HOST_COMPILATION())

template <auto V>
TEST_FUNC constexpr bool value_ref_to_template_parameter_object()
{
  return &V == &cuda::std::__constant_wrapper<V>::value;
}

static_assert(value_ref_to_template_parameter_object<S{5}>());

constexpr int arr[] = {1, 2, 3, 4, 5};

static_assert(cuda::std::__constant_wrapper<arr>::value == arr);
static_assert(cuda::std::same_as<typename cuda::std::__constant_wrapper<arr>::type, cuda::std::__constant_wrapper<arr>>);

// nvcc < 13.3 incorrectly generates input file for host compiler.
#if !(TEST_CUDA_COMPILER(NVCC, <, 13, 3) && _CCCL_HOST_COMPILATION())
static_assert(cuda::std::same_as<typename cuda::std::__constant_wrapper<arr>::value_type, const int*>);
#endif // !(TEST_CUDA_COMPILER(NVCC, <, 13, 3) && _CCCL_HOST_COMPILATION())

int main(int, char**)
{
  return 0;
}
