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

template <class T, class U, int N, class Flags>
TEST_HOST_DEVICE_FUNC constexpr void test_memory_flags(Flags flags)
{
  using Vec  = simd::vec<T, N>;
  using Mask = typename Vec::mask_type;
  alignas(512) cuda::std::array<U, N + 1> input{};
  alignas(512) cuda::std::array<U, N + 1> output{};
  for (int i = 0; i <= N; ++i)
  {
    input[i] = static_cast<U>(i + 1);
  }

  // Exercise the full-load/store paths through each public overload family.
  for (int form = 0; form < 3; ++form)
  {
    const auto values = form == 0 ? simd::unchecked_load<Vec>(input, flags)
                      : form == 1 ? simd::unchecked_load<Vec>(input.data(), N, flags)
                                  : simd::unchecked_load<Vec>(input.data(), input.data() + N, flags);
    for (int i = 0; i < N; ++i)
    {
      assert(values[i] == static_cast<T>(input[i]));
    }
    output.fill(U{-1});
    if (form == 0)
    {
      simd::unchecked_store(values, output, flags);
    }
    else if (form == 1)
    {
      simd::unchecked_store(values, output.data(), N, flags);
    }
    else
    {
      simd::unchecked_store(values, output.data(), output.data() + N, flags);
    }
    for (int i = 0; i < N; ++i)
    {
      assert(output[i] == input[i]);
    }
    assert(output[N] == U{-1});
  }

  const auto values = simd::unchecked_load<Vec>(input, flags);
  // Empty, full, alternating, and last-lane-only selections.
  for (int selection = 0; selection < 4; ++selection)
  {
    const Mask mask(mask_selection{selection, N - 1});
    const auto masked = simd::unchecked_load<Vec>(input, mask, flags);
    output.fill(U{-1});
    simd::unchecked_store(values, output, mask, flags);
    for (int i = 0; i < N; ++i)
    {
      assert(masked[i] == (mask[i] ? static_cast<T>(input[i]) : T{}));
      assert(output[i] == (mask[i] ? input[i] : U{-1}));
    }
    assert(output[N] == U{-1});

    // Partial operations must also respect the count when the mask selects more lanes.
    for (int empty = 0; empty < 2; ++empty)
    {
      const int count    = empty ? 0 : N - 1;
      const auto partial = simd::partial_load<Vec>(input.data(), count, mask, flags);
      output.fill(U{-1});
      simd::partial_store(values, output.data(), count, mask, flags);
      for (int i = 0; i < N; ++i)
      {
        const bool selected = i < count && mask[i];
        assert(partial[i] == (selected ? static_cast<T>(input[i]) : T{}));
        assert(output[i] == (selected ? input[i] : U{-1}));
      }
      assert(output[N] == U{-1});
    }
  }
}

template <int N>
TEST_HOST_DEVICE_FUNC constexpr bool test_width()
{
  test_memory_flags<int, float, N>(simd::flag_convert | simd::flag_aligned);
  test_memory_flags<int, float, N>(simd::flag_aligned | simd::flag_convert);
  test_memory_flags<int, float, N>(simd::flag_convert | simd::flag_aligned | simd::flag_convert | simd::flag_aligned);
  test_memory_flags<int, float, N>(simd::flag_convert | simd::flag_overaligned<8> | simd::flag_overaligned<64>);
  test_memory_flags<int, float, N>(simd::flag_overaligned<64> | simd::flag_aligned | simd::flag_convert
                                   | simd::flag_overaligned<8> | simd::flag_overaligned<64>);
  return true;
}

template <int N>
TEST_HOST_DEVICE_FUNC void test()
{
  static_assert(test_width<N>());
  assert(test_width<N>()); // NOLINT(cert-dcl03-c,misc-static-assert)
}

TEST_HOST_DEVICE_FUNC constexpr bool test_types()
{
  test_memory_flags<char, float, 3>(simd::flag_convert | simd::flag_aligned);
  test_memory_flags<float, double, 4>(simd::flag_convert | simd::flag_aligned);
  test_memory_flags<cuda::std::complex<float>, cuda::std::complex<float>, 3>(simd::flag_convert | simd::flag_aligned);
  return true;
}

int main(int, char**)
{
  test<1>();
  test<3>();
  test<4>();
  test<63>();
  test<64>();
  static_assert(test_types());
  assert(test_types()); // NOLINT(cert-dcl03-c,misc-static-assert)
  return 0;
}
