//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// template<class F, class... Args>
// concept strict_weak_order;

#include <cuda/std/concepts>

using cuda::std::strict_weak_order;

static_assert(strict_weak_order<bool(int, int), int, int>, "");
static_assert(strict_weak_order<bool(int, int), double, double>, "");
static_assert(strict_weak_order<bool(int, double), double, double>, "");

static_assert(!strict_weak_order<bool (*)(), int, double>, "");
static_assert(!strict_weak_order<bool (*)(int), int, double>, "");
static_assert(!strict_weak_order<bool (*)(double), int, double>, "");

static_assert(!strict_weak_order<bool(double, double*), double, double*>, "");
static_assert(!strict_weak_order<bool(int&, int&), double&, double&>, "");

struct S1
{};
static_assert(strict_weak_order<bool (S1::*)(S1*), S1*, S1*>, "");
static_assert(strict_weak_order<bool (S1::*)(S1&), S1&, S1&>, "");

struct S2
{};

struct P1
{
  __host__ __device__ bool operator()(S1, S1) const;
};
static_assert(strict_weak_order<P1, S1, S1>, "");

struct P2
{
  __host__ __device__ bool operator()(S1, S1) const;
  __host__ __device__ bool operator()(S1, S2) const;
};
static_assert(!strict_weak_order<P2, S1, S2>, "");

struct P3
{
  __host__ __device__ bool operator()(S1, S1) const;
  __host__ __device__ bool operator()(S1, S2) const;
  __host__ __device__ bool operator()(S2, S1) const;
};
static_assert(!strict_weak_order<P3, S1, S2>, "");

struct P4
{
  __host__ __device__ bool operator()(S1, S1) const;
  __host__ __device__ bool operator()(S1, S2) const;
  __host__ __device__ bool operator()(S2, S1) const;
  __host__ __device__ bool operator()(S2, S2) const;
};
static_assert(strict_weak_order<P4, S1, S2>, "");

int main(int, char**)
{
  return 0;
}
