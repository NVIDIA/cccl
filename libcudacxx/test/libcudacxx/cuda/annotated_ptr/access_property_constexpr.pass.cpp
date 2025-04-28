//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: nvrtc

// error: expression must have a constant value annotated_ptr.h: note #2701-D: attempt to access run-time storage
// UNSUPPORTED: clang-14, gcc-9, gcc-8, gcc-7, msvc-19.29

#include <cuda/annotated_ptr>

#include "test_macros.h"

__host__ __device__ constexpr bool test_constexpr()
{
  using namespace cuda;
  access_property a{}; // default constructor
  access_property b{a}; // copy constructor
  access_property c{cuda::std::move(a)}; // move constructor
  // user-declared ctor
  access_property d1{access_property::global{}};
  access_property d2{access_property::normal{}};
  access_property d3{access_property::streaming{}};
  access_property d4{access_property::persisting{}};
  auto p1 = static_cast<cudaAccessProperty>(access_property::normal{});
  auto p2 = static_cast<cudaAccessProperty>(access_property::streaming{});
  auto p3 = static_cast<cudaAccessProperty>(access_property::persisting{});
  // fraction ctor
  access_property e1{access_property::normal{}, 1.0f};
  access_property e2{access_property::streaming{}, 1.0f};
  access_property e3{access_property::persisting{}, 1.0f};
  access_property e4{access_property::normal{}, 1.0f, access_property::streaming{}};
  access_property e5{access_property::persisting{}, 1.0f, access_property::streaming{}};
  b          = a; // copy assignment
  b          = cuda::std::move(a); // move assignment
  auto value = static_cast<uint64_t>(a);
  unused(p1, p2, p3, b, c, d1, d2, d3, d4, e1, e2, e3, e4, e5, value);
  return true;
}

int main(int, char**)
{
  static_assert(test_constexpr());
  return 0;
}
