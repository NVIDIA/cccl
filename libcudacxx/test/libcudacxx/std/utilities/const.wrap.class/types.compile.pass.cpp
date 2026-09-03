//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// todo(dabayer): nvrtc doesn't support non-trivial types as static data members without -default-device, fails with:
//   A class static data member with non-const type is considered a host variable, and host variables are not allowed in
//   JIT mode. Consider using -default-device flag to process such data members as __device__ variables in JIT mode

// constant_wrapper

// static constexpr decltype(auto) value = (X);
// using type = constant_wrapper;
// using value_type = decltype(X);

#include <cuda/std/algorithm>
#include <cuda/std/concepts>
#include <cuda/std/utility>

#include "test_macros.h"

static_assert(cuda::std::__constant_wrapper<42>::value == 42);
// todo(dabayer): This is failing with MSVC.
#if !_CCCL_COMPILER(MSVC)
static_assert(cuda::std::same_as<decltype(cuda::std::__constant_wrapper<42>::value), const int>);
#endif // !_CCCL_COMPILER(MSVC)
static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<42>::type, cuda::std::__constant_wrapper<42>>);
static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<42>::value_type, int>);

#if TEST_STD_VER >= 2020 && !TEST_COMPILER(NVRTC)

struct S
{
  int member = 42;
};

constexpr cuda::std::__constant_wrapper<S{5}> s_value;
using SValue = cuda::std::remove_const_t<decltype(s_value)>;

static_assert(s_value.value.member == 5);

// nvcc 12.0 fails to properly generate input file for host compiler.
#  if !(TEST_CUDA_COMPILER(NVCC, ==, 12, 0) && _CCCL_HOST_COMPILATION())
static_assert(cuda::std::same_as<decltype(SValue::value), const S&>);
#  endif // !(TEST_CUDA_COMPILER(NVCC, ==, 12, 0) && _CCCL_HOST_COMPILATION())

static_assert(cuda::std::same_as<SValue::type, SValue>);

// nvcc < 13.1 fails to properly generate input file for host compiler.
#  if !(TEST_CUDA_COMPILER(NVCC, <, 13, 1) && _CCCL_HOST_COMPILATION())
static_assert(cuda::std::same_as<SValue::value_type, S>);
#  endif // !(TEST_CUDA_COMPILER(NVCC, <, 13, 1) && _CCCL_HOST_COMPILATION())

template <auto V>
TEST_FUNC constexpr bool value_ref_to_template_parameter_object()
{
  // gcc < 13 evaluates this as taking address of rvalue.
#  if !TEST_COMPILER(GCC, <, 13)
  return &V == &cuda::std::__constant_wrapper<V>{};
#  else // ^^^ !TEST_COMPILER(GCC, <, 13) ^^^ / vvv TEST_COMPILER(GCC, <, 13) vvv
  return &V == &cuda::std::__constant_wrapper<V>::__get();
#  endif // ^^^ TEST_COMPILER(GCC, <, 13) ^^^
}

static_assert(value_ref_to_template_parameter_object<S{5}>());

#endif // TEST_STD_VER >= 2020 && !TEST_COMPILER(NVRTC)

constexpr int arr[] = {1, 2, 3, 4, 5};

static_assert(cuda::std::__constant_wrapper<arr>{} == arr);
static_assert(cuda::std::same_as<typename cuda::std::__constant_wrapper<arr>::type, cuda::std::__constant_wrapper<arr>>);

// nvcc < 13.3 incorrectly generates input file for host compiler.
#if !(TEST_CUDA_COMPILER(NVCC, <, 13, 3) && _CCCL_HOST_COMPILATION())
static_assert(cuda::std::same_as<typename cuda::std::__constant_wrapper<arr>::value_type, const int*>);
#endif // !(TEST_CUDA_COMPILER(NVCC, <, 13, 3) && _CCCL_HOST_COMPILATION())

int main(int, char**)
{
  return 0;
}
