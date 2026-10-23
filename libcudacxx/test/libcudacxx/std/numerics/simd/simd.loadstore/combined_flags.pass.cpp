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

// [simd.loadstore], conversion and alignment across the complete flag pack

#include <cuda/std/__simd_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "test_macros.h"

namespace simd = cuda::std::simd;

struct mask_selection
{
  int selection;
  int last;

  template <class Index>
  TEST_HOST_DEVICE_FUNC constexpr bool operator()(Index i) const
  {
    return selection == 1 || (selection == 2 && i % 2 == 0) || (selection == 3 && i == last);
  }
};

template <class Vec, class Array>
TEST_HOST_DEVICE_FUNC constexpr void check_load_store(const Vec& values, const Array& input, const Array& output)
{
  using T             = typename Vec::value_type;
  using U             = typename Array::value_type;
  constexpr auto size = Vec::size();
  for (int i = 0; i < size; ++i)
  {
    assert(values[i] == static_cast<T>(input[i]));
    assert(output[i] == input[i]);
  }
  assert(output[size] == U{-1});
}

template <class Vec, class Array>
TEST_HOST_DEVICE_FUNC constexpr void check_masked_load_store(
  const Vec& values, const Array& input, const Array& output, const typename Vec::mask_type& mask, int count)
{
  using T             = typename Vec::value_type;
  using U             = typename Array::value_type;
  constexpr auto size = Vec::size();
  for (int i = 0; i < size; ++i)
  {
    const bool selected = i < count && mask[i];
    assert(values[i] == (selected ? static_cast<T>(input[i]) : T{}));
    assert(output[i] == (selected ? input[i] : U{-1}));
  }
  assert(output[size] == U{-1});
}

template <class T, class U, int N, class Flags>
TEST_HOST_DEVICE_FUNC constexpr void test_memory_flags(Flags flags)
{
  using Vec  = simd::vec<T, N>;
  using Mask = typename Vec::mask_type;
  alignas(512) cuda::std::array<U, N + 1> input{};
  alignas(512) cuda::std::array<U, N + 1> output{};
  for (int i = 0; i <= N; ++i)
  {
    input[i] = static_cast<U>(static_cast<float>(i + 1));
  }

  // Range overloads.
  const auto values = simd::unchecked_load<Vec>(input, flags);
  output.fill(U{-1});
  simd::unchecked_store(values, output, flags);
  check_load_store(values, input, output);

  // Iterator and count overloads.
  const auto count_values = simd::unchecked_load<Vec>(input.data(), N, flags);
  output.fill(U{-1});
  simd::unchecked_store(count_values, output.data(), N, flags);
  check_load_store(count_values, input, output);

  // Iterator and sentinel overloads.
  const auto sentinel_values = simd::unchecked_load<Vec>(input.data(), input.data() + N, flags);
  output.fill(U{-1});
  simd::unchecked_store(sentinel_values, output.data(), output.data() + N, flags);
  check_load_store(sentinel_values, input, output);

  // selection 0: Empty
  // selection 1: Full
  // selection 2: Alternating
  // selection 3: Last-lane-only
  for (int selection = 0; selection < 4; ++selection)
  {
    const Mask mask(mask_selection{selection, N - 1});
    output.fill(U{-1});

    // Full load/store
    const auto masked = simd::unchecked_load<Vec>(input, mask, flags);
    simd::unchecked_store(values, output, mask, flags);
    check_masked_load_store(masked, input, output, mask, N);

    // Partial load/store
    for (const int count : cuda::std::array<int, 2>{0, N - 1})
    {
      const auto partial = simd::partial_load<Vec>(input.data(), count, mask, flags);
      output.fill(U{-1});
      simd::partial_store(values, output.data(), count, mask, flags);
      check_masked_load_store(partial, input, output, mask, count);
    }
  }
}

// Compose flags outside function templates to avoid a GCC 11 ICE in C++20.
constexpr auto aligned_flags     = simd::flag_convert | simd::flag_aligned;
constexpr auto overaligned_flags = simd::flag_convert | simd::flag_overaligned<64>;

template <int N>
TEST_HOST_DEVICE_FUNC constexpr bool test_width()
{
  test_memory_flags<int, float, N>(aligned_flags);
  test_memory_flags<int, float, N>(overaligned_flags);
  return true;
}

TEST_HOST_DEVICE_FUNC constexpr bool test()
{
  test_width<1>();
  test_width<3>();
  test_width<4>();
  test_memory_flags<char, float, 3>(aligned_flags);
  test_memory_flags<float, double, 4>(aligned_flags);
  test_memory_flags<cuda::std::complex<float>, cuda::std::complex<float>, 3>(aligned_flags);
  return true;
}

int main(int, char**)
{
  static_assert(test());
  assert(test()); // NOLINT(cert-dcl03-c,misc-static-assert)
  return 0;
}
