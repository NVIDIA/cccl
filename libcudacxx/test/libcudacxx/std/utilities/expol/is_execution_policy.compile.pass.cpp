//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// template<class T> struct is_execution_policy;
// template<class T> constexpr bool is_execution_policy_v = is_execution_policy<T>::value;

#include <cuda/std/execution>

#include "test_macros.h"

static_assert(cuda::std::is_execution_policy<cuda::std::execution::sequenced_policy>::value);
static_assert(cuda::std::is_execution_policy<cuda::std::execution::parallel_policy>::value);
static_assert(cuda::std::is_execution_policy<cuda::std::execution::parallel_unsequenced_policy>::value);
static_assert(cuda::std::is_execution_policy<cuda::std::execution::unsequenced_policy>::value);

static_assert(cuda::std::is_execution_policy_v<cuda::std::execution::sequenced_policy>);
static_assert(cuda::std::is_execution_policy_v<cuda::std::execution::unsequenced_policy>);
static_assert(cuda::std::is_execution_policy_v<cuda::std::execution::parallel_policy>);
static_assert(cuda::std::is_execution_policy_v<cuda::std::execution::parallel_unsequenced_policy>);

static_assert(cuda::std::is_execution_policy_v<const cuda::std::execution::sequenced_policy>);
static_assert(cuda::std::is_execution_policy_v<volatile cuda::std::execution::sequenced_policy>);
static_assert(cuda::std::is_execution_policy_v<const volatile cuda::std::execution::sequenced_policy>);

static_assert(!cuda::std::__is_parallel_execution_policy_v<cuda::std::execution::sequenced_policy>);
static_assert(!cuda::std::__is_parallel_execution_policy_v<cuda::std::execution::unsequenced_policy>);
static_assert(cuda::std::__is_parallel_execution_policy_v<cuda::std::execution::parallel_policy>);
static_assert(cuda::std::__is_parallel_execution_policy_v<cuda::std::execution::parallel_unsequenced_policy>);

static_assert(cuda::std::__is_parallel_execution_policy_v<const cuda::std::execution::parallel_unsequenced_policy>);
static_assert(cuda::std::__is_parallel_execution_policy_v<volatile cuda::std::execution::parallel_unsequenced_policy>);
static_assert(
  cuda::std::__is_parallel_execution_policy_v<const volatile cuda::std::execution::parallel_unsequenced_policy>);

static_assert(!cuda::std::__is_unsequenced_execution_policy_v<cuda::std::execution::sequenced_policy>);
static_assert(cuda::std::__is_unsequenced_execution_policy_v<cuda::std::execution::unsequenced_policy>);
static_assert(!cuda::std::__is_unsequenced_execution_policy_v<cuda::std::execution::parallel_policy>);
static_assert(cuda::std::__is_unsequenced_execution_policy_v<cuda::std::execution::parallel_unsequenced_policy>);

static_assert(cuda::std::__is_unsequenced_execution_policy_v<const cuda::std::execution::parallel_unsequenced_policy>);
static_assert(
  cuda::std::__is_unsequenced_execution_policy_v<volatile cuda::std::execution::parallel_unsequenced_policy>);
static_assert(
  cuda::std::__is_unsequenced_execution_policy_v<const volatile cuda::std::execution::parallel_unsequenced_policy>);

int main(int, char**)
{
  return 0;
}
