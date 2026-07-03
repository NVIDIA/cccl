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

static_assert(cuda::std::__constant_wrapper<S{5}>::value.member == 5);
static_assert(cuda::std::same_as<decltype(cuda::std::__constant_wrapper<S{5}>::value), const S&>);
static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<S{5}>::type, cuda::std::__constant_wrapper<S{5}>>);
static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<S{5}>::value_type, S>);

template <auto V>
TEST_FUNC constexpr bool value_ref_to_template_parameter_object()
{
  return &V == &cuda::std::__constant_wrapper<V>::value;
}

static_assert(value_ref_to_template_parameter_object<S{5}>());

constexpr int arr[] = {1, 2, 3, 4, 5};

static_assert(cuda::std::__constant_wrapper<arr>::value == arr);
static_assert(cuda::std::same_as<typename cuda::std::__constant_wrapper<arr>::type, cuda::std::__constant_wrapper<arr>>);
static_assert(cuda::std::same_as<typename cuda::std::__constant_wrapper<arr>::value_type, const int*>);

int main(int, char**)
{
  return 0;
}
