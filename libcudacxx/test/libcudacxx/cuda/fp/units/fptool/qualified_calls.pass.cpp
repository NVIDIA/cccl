// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: every named fpmp2_stat free function is callable qualified.
//
//  fpmp2_stat wraps fpmp2 and mirrors its API, so it has to mirror this property
//  too: the named free functions live at namespace scope and can be reached as
//  cudax::name(x), not only through ADL. See the companion test
//  units/fpmp/qualified_calls.pass.cpp for why that matters.
//
//  The reset and read entry points are host-only and CUDA-only, so only their
//  names are exercised here; stat.pass.cpp covers what they collect.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fptool>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

template <class T>
TEST_HOST_DEVICE_FUNC bool same(const T& __a, const T& __b)
{
  return !(__a != __b);
}

// The operands are exactly representable at both widths, so the value checks hold
// in every accuracy mode.
template <class T>
TEST_HOST_DEVICE_FUNC void test_named()
{
  const T a(3.0);
  const T b(1.5);
  const T c(4.0);

  // renormalize: the one that used to be a hidden friend, here and on the wrapped type.
  // The qualified spelling is itself the check that it is a namespace member, since
  // qualified lookup does not find a hidden friend - this line stops compiling if it
  // moves back into the class, even though ADL would still find the unqualified one.
  assert(same(cudax::renormalize(a), renormalize(a)));
  assert(static_cast<double>(cudax::renormalize(a)) == 3.0);

  assert(same(cudax::sqrt(c), sqrt(c)));
  assert(static_cast<double>(cudax::sqrt(c)) == 2.0);

  assert(same(cudax::rsqrt(c), rsqrt(c)));

  assert(same(cudax::fma(a, b, c), fma(a, b, c)));
  assert(static_cast<double>(cudax::fma(a, b, c)) == 8.5); // 3 * 1.5 + 4

  assert(same(cudax::mad(a, b, c), mad(a, b, c)));
  assert(static_cast<double>(cudax::mad(a, b, c)) == 8.5);
}

// An instrumented value must give the same answer as the plain one it wraps: the
// wrapper observes, it does not compute.
template <class Stat, class Plain>
TEST_HOST_DEVICE_FUNC void test_matches_plain()
{
  const Stat sa(3.0);
  const Plain pa(3.0);
  const Stat sc(4.0);
  const Plain pc(4.0);

  assert(static_cast<double>(cudax::renormalize(sa)) == static_cast<double>(cudax::renormalize(pa)));
  assert(static_cast<double>(cudax::sqrt(sc)) == static_cast<double>(cudax::sqrt(pc)));
}

template <class T>
TEST_HOST_DEVICE_FUNC void test_type()
{
  test_named<T>();
}

TEST_HOST_DEVICE_FUNC void test()
{
  test_type<cudax::fp32mp2_stat>();
  test_type<cudax::fp32mp2_stat_low>();
  test_type<cudax::fp32mp2_stat_mid>();
  test_type<cudax::fp32mp2_stat_high>();

  test_type<cudax::fp64mp2_stat>();
  test_type<cudax::fp64mp2_stat_low>();
  test_type<cudax::fp64mp2_stat_mid>();
  test_type<cudax::fp64mp2_stat_high>();

  test_matches_plain<cudax::fp32mp2_stat, cudax::fp32mp2>();
  test_matches_plain<cudax::fp64mp2_stat, cudax::fp64mp2>();
}

int main(int, char**)
{
  test();

  return 0;
}
