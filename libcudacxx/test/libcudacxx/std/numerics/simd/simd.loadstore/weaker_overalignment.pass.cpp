//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// SIMD host/device functions are not supported in tile mode.

// [simd.loadstore], an overalignment request may be weaker than natural alignment.

#include <cuda/simd>
#include <cuda/std/array>
#include <cuda/std/cassert>

#include "test_macros.h"

namespace simd = cuda::std::simd;

struct alternating_lane
{
  template <class Index>
  TEST_HOST_DEVICE_FUNC constexpr bool operator()(Index i) const
  {
    return i % 2 == 0;
  }
};

template <class Vec, class Array>
TEST_HOST_DEVICE_FUNC constexpr void check_load_store(
  const Vec& values, const Array& input, const Array& output, const typename Vec::mask_type& mask, int count)
{
  using T = typename Vec::value_type;
  for (int i = 0; i < Vec::size(); ++i)
  {
    const bool selected = i < count && mask[i];
    assert(values[i] == (selected ? input[i + 1] : T{}));
    assert(output[i + 1] == (selected ? input[i + 1] : T{-1}));
  }
  assert(output.front() == T{-1});
  assert(output.back() == T{-1});
}

template <class T, int N, class Flags>
TEST_HOST_DEVICE_FUNC constexpr void test_alignment(Flags flags)
{
  using Vec  = simd::vec<T, N>;
  using Mask = typename Vec::mask_type;
  cuda::std::array<T, N + 2> input{};
  cuda::std::array<T, N + 2> output{};
  for (int i = 0; i < N; ++i)
  {
    input[i + 1] = static_cast<T>(i + 1);
  }

  // Offset naturally aligned storage without promising vector alignment.
  auto* first = input.data() + 1;
  auto* dest  = output.data() + 1;
  const Mask all(true);

  auto values = simd::unchecked_load<Vec>(first, N, flags);
  output.fill(T{-1});
  simd::unchecked_store(values, dest, N, flags);
  check_load_store(values, input, output, all, N);

  const Mask alternating(alternating_lane{});
  auto masked = simd::unchecked_load<Vec>(first, N, alternating, flags);
  output.fill(T{-1});
  simd::unchecked_store(values, dest, N, alternating, flags);
  check_load_store(masked, input, output, alternating, N);

  for (const int count : {0, N - 1})
  {
    auto partial = simd::partial_load<Vec>(first, count, flags);
    output.fill(T{-1});
    simd::partial_store(values, dest, count, flags);
    check_load_store(partial, input, output, all, count);

    auto masked_partial = simd::partial_load<Vec>(first, count, alternating, flags);
    output.fill(T{-1});
    simd::partial_store(values, dest, count, alternating, flags);
    check_load_store(masked_partial, input, output, alternating, count);
  }
}

template <class T, int N>
TEST_HOST_DEVICE_FUNC constexpr void test_type()
{
  test_alignment<T, N>(simd::flag_overaligned<1>);
  if constexpr (alignof(T) > 2)
  {
    test_alignment<T, N>(simd::flag_overaligned<2>);
  }
  if constexpr (alignof(T) > 1)
  {
    test_alignment<T, N>(simd::flag_overaligned<alignof(T)>);
  }
}

template <int N>
TEST_HOST_DEVICE_FUNC constexpr void test_width()
{
  test_type<short, N>();
  test_type<int, N>();
  test_type<float, N>();
  test_type<double, N>();
}

TEST_HOST_DEVICE_FUNC constexpr bool test()
{
  test_width<1>();
  test_width<3>();
  test_width<4>();
  return true;
}

int main(int, char**)
{
  static_assert(test());
  assert(test()); // NOLINT(misc-static-assert)
  return 0;
}
