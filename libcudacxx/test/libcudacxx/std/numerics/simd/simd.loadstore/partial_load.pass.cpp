//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/__simd_>

// [simd.loadstore], partial_load
//
// partial_load<V>(Range&&, flags<> = {});
// partial_load<V>(Range&&, const mask_type&, flags<> = {});
// partial_load<V>(I first, iter_difference_t<I> n, flags<> = {});
// partial_load<V>(I first, iter_difference_t<I> n, const mask_type&, flags<> = {});
// partial_load<V>(I first, S last, flags<> = {});
// partial_load<V>(I first, S last, const mask_type&, flags<> = {});

#include <cuda/std/__simd_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// partial_load: range

template <typename T, int N>
TEST_FUNC constexpr void test_partial_load_range()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  auto arr  = make_iota_array<T, N>();

  Vec vec = simd::partial_load<Vec>(arr);
  assert(vec == arr);

  Vec vec2 = simd::partial_load<Vec>(arr, simd::flag_default);
  assert(vec2 == arr);
}

//----------------------------------------------------------------------------------------------------------------------
// partial_load: range, masked

template <typename T, int N>
TEST_FUNC constexpr void test_partial_load_range_masked()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;
  auto arr   = make_iota_array<T, N>();

  Mask even_mask(is_even{});
  Vec vec = simd::partial_load<Vec>(arr, even_mask);
  for (int i = 0; i < N; ++i)
  {
    T expected = (i % 2 == 0) ? static_cast<T>(i + 1) : T{};
    assert(vec[i] == expected);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// partial_load: smaller range (count < Vec.size)

template <typename T, int N>
TEST_FUNC constexpr void test_partial_load_smaller_range()
{
  if constexpr (N > 1)
  {
    using Vec          = simd::basic_vec<T, simd::fixed_size<N>>;
    constexpr int Half = N / 2;
    auto small_arr     = make_iota_array<T, Half>();

    Vec vec = simd::partial_load<Vec>(small_arr);
    for (int i = 0; i < N; ++i)
    {
      if (i < Half)
      {
        assert(vec[i] == static_cast<T>(i + 1));
      }
      else
      {
        assert(vec[i] == T{});
      }
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// partial_load: iterator + count

template <typename T, int N>
TEST_FUNC constexpr void test_partial_load_iter_count()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  auto arr  = make_iota_array<T, N>();

  Vec vec = simd::partial_load<Vec>(arr.data(), N);
  assert(vec == arr);

  using Mask = typename Vec::mask_type;
  Mask even_mask(is_even{});
  Vec masked_vec = simd::partial_load<Vec>(arr.data(), N, even_mask);
  for (int i = 0; i < N; ++i)
  {
    T expected = (i % 2 == 0) ? static_cast<T>(i + 1) : T{};
    assert(masked_vec[i] == expected);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// partial_load: iterator + sentinel

template <typename T, int N>
TEST_FUNC constexpr void test_partial_load_iter_sentinel()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  auto arr  = make_iota_array<T, N>();

  Vec vec = simd::partial_load<Vec>(arr.data(), arr.data() + N);
  assert(vec == arr);

  using Mask = typename Vec::mask_type;
  Mask even_mask(is_even{});
  Vec masked_vec = simd::partial_load<Vec>(arr.data(), arr.data() + N, even_mask);
  for (int i = 0; i < N; ++i)
  {
    T expected = (i % 2 == 0) ? static_cast<T>(i + 1) : T{};
    assert(masked_vec[i] == expected);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// flag_convert: lossy load from wider type

template <typename T, int N>
TEST_FUNC constexpr void test_partial_load_convert()
{
  if constexpr (sizeof(T) <= sizeof(int) && cuda::std::is_integral_v<T>)
  {
    using Vec      = simd::basic_vec<T, simd::fixed_size<N>>;
    using WiderT   = cuda::std::conditional_t<cuda::std::is_signed_v<T>, int64_t, uint64_t>;
    auto wider_arr = make_iota_array<WiderT, N>();

    Vec vec = simd::partial_load<Vec>(wider_arr, simd::flag_convert);
    assert((vec == make_iota_array<T, N>()));
  }
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    using Vec      = simd::basic_vec<T, simd::fixed_size<N>>;
    auto wider_arr = make_iota_array<double, N>();

    Vec vec = simd::partial_load<Vec>(wider_arr, simd::flag_convert);
    assert((vec == make_iota_array<T, N>()));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// noexcept: public functions must NOT be noexcept

template <typename T, int N>
TEST_FUNC constexpr void test_partial_load_not_noexcept()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;
  cuda::std::array<T, N> arr{};
  Mask mask(true);
  unused(arr, mask);

  // range overloads
  static_assert(!noexcept(simd::partial_load<Vec>(arr, mask)));
  static_assert(!noexcept(simd::partial_load<Vec>(arr)));
  // iterator + count overloads
  static_assert(!noexcept(simd::partial_load<Vec>(arr.data(), N, mask)));
  static_assert(!noexcept(simd::partial_load<Vec>(arr.data(), N)));
  // iterator + sentinel overloads
  static_assert(!noexcept(simd::partial_load<Vec>(arr.data(), arr.data() + N, mask)));
  static_assert(!noexcept(simd::partial_load<Vec>(arr.data(), arr.data() + N)));
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  test_partial_load_range<T, N>();
  test_partial_load_range_masked<T, N>();
  test_partial_load_smaller_range<T, N>();
  test_partial_load_iter_count<T, N>();
  test_partial_load_iter_sentinel<T, N>();
  test_partial_load_convert<T, N>();
  test_partial_load_not_noexcept<T, N>();
}

DEFINE_BASIC_VEC_TEST()
DEFINE_BASIC_VEC_TEST_RUNTIME()

int main(int, char**)
{
  assert(test());
  static_assert(test());
  assert(test_runtime());
  return 0;
}
