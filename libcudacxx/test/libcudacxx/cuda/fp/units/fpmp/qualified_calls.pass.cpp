// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: every named fpmp2 free function is callable qualified.
//
//  The recommended way to reach the FP SDK is a namespace alias plus explicit
//  qualification, not a using-directive:
//
//      namespace cudax = cuda::experimental;
//      auto y = cudax::sqrt(x);
//
//  That only works if the named functions are members of the namespace. A
//  function defined as a hidden friend inside the class is found by ADL alone,
//  so the qualified spelling fails to compile with "is not a member of cudax",
//  which is why this file exists: it pins every named entry point to namespace
//  scope so none of them can regress to a hidden friend unnoticed.
//
//  The accuracy-selecting forms have no choice in the matter. A call that spells
//  out its template argument, add<fpmp2_accuracy::high>(a, b), needs the name
//  found by ordinary lookup for the '<' to parse as a template argument list
//  rather than a comparison, which ADL cannot provide before C++20. They are
//  therefore tested in their qualified form only, and the named functions match
//  them so that the whole public surface is qualifiable.
//
//  Each function is checked both ways where both are legal, and the two
//  spellings must agree, since they should resolve to the same function.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Equality on the full multi-precision value, not just its double image, so a
// wrong low word cannot pass unnoticed.
template <class T>
TEST_HOST_DEVICE_FUNC bool same(const T& __a, const T& __b)
{
  return !(__a != __b);
}

// ---- named functions: qualified and unqualified ---------------------------
//
// The operands are chosen so that every result is exactly representable in a
// double-float as well as a double-double, keeping the value checks valid in
// all four accuracy modes.
template <class T>
TEST_HOST_DEVICE_FUNC void test_named()
{
  const T a(3.0);
  const T b(1.5);
  const T c(4.0);

  // renormalize: the one that used to be a hidden friend. The qualified spelling is
  // itself the check that it is a namespace member, since qualified lookup does not
  // find a hidden friend - this line stops compiling if it moves back into the class,
  // even though ADL would still find the unqualified one next to it.
  assert(same(cudax::renormalize(a), renormalize(a)));
  assert(static_cast<double>(cudax::renormalize(a)) == 3.0);

  assert(same(cudax::sqrt(c), sqrt(c)));
  assert(static_cast<double>(cudax::sqrt(c)) == 2.0);

  // rsqrt is iterative, so only the two spellings are compared, not an exact value.
  assert(same(cudax::rsqrt(c), rsqrt(c)));

  assert(same(cudax::fma(a, b, c), fma(a, b, c)));
  assert(static_cast<double>(cudax::fma(a, b, c)) == 8.5); // 3 * 1.5 + 4

  assert(same(cudax::mad(a, b, c), mad(a, b, c)));
  assert(static_cast<double>(cudax::mad(a, b, c)) == 8.5);
}

// ---- accuracy-selecting functions: qualified only -------------------------
template <class T, cudax::fpmp2_accuracy Acc>
TEST_HOST_DEVICE_FUNC void test_accuracy_selected()
{
  const T a(3.0);
  const T b(1.5);
  const T c(4.0);

  assert(static_cast<double>(cudax::add<Acc>(a, b)) == 4.5);
  assert(static_cast<double>(cudax::sub<Acc>(a, b)) == 1.5);
  assert(static_cast<double>(cudax::mul<Acc>(a, b)) == 4.5);
  assert(static_cast<double>(cudax::div<Acc>(a, b)) == 2.0);
  assert(static_cast<double>(cudax::fma<Acc>(a, b, c)) == 8.5);
  assert(static_cast<double>(cudax::mad<Acc>(a, b, c)) == 8.5);
}

template <class T>
TEST_HOST_DEVICE_FUNC void test_type()
{
  test_named<T>();
  test_accuracy_selected<T, cudax::fpmp2_accuracy::low>();
  test_accuracy_selected<T, cudax::fpmp2_accuracy::mid>();
  test_accuracy_selected<T, cudax::fpmp2_accuracy::high>();
  test_accuracy_selected<T, cudax::fpmp2_accuracy::def>();
}

TEST_HOST_DEVICE_FUNC void test()
{
  test_type<cudax::fp32mp2>();
  test_type<cudax::fp32mp2_low>();
  test_type<cudax::fp32mp2_mid>();
  test_type<cudax::fp32mp2_high>();

  test_type<cudax::fp64mp2>();
  test_type<cudax::fp64mp2_low>();
  test_type<cudax::fp64mp2_mid>();
  test_type<cudax::fp64mp2_high>();
}

int main(int, char**)
{
  test();

  return 0;
}
