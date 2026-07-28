//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: enable-tile
// error: asm statement is unsupported in tile code

// <cuda/std/__simd_>

// [simd.loadstore], unchecked_load
//
// unchecked_load<V>(Range&&, flags<> = {});
// unchecked_load<V>(Range&&, const mask_type&, flags<> = {});
// unchecked_load<V>(I first, iter_difference_t<I> n, flags<> = {});
// unchecked_load<V>(I first, iter_difference_t<I> n, const mask_type&, flags<> = {});
// unchecked_load<V>(I first, S last, flags<> = {});
// unchecked_load<V>(I first, S last, const mask_type&, flags<> = {});

#include <cuda/std/__simd_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// unchecked_load: range

template <typename T, int N>
TEST_FUNC constexpr void test_unchecked_load_range()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  auto arr  = make_iota_array<T, N>();

  Vec vec = simd::unchecked_load<Vec>(arr);
  assert(vec == arr);

  Vec vec2 = simd::unchecked_load<Vec>(arr, simd::flag_default);
  assert(vec2 == arr);
}

//----------------------------------------------------------------------------------------------------------------------
// unchecked_load: range, masked

template <typename T, int N>
TEST_FUNC constexpr void test_unchecked_load_range_masked()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;
  auto arr   = make_iota_array<T, N>();

  Mask even_mask(is_even{});
  Vec vec = simd::unchecked_load<Vec>(arr, even_mask);
  for (int i = 0; i < N; ++i)
  {
    T expected = (i % 2 == 0) ? static_cast<T>(i + 1) : T{};
    assert(vec[i] == expected);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// unchecked_load: iterator + count

template <typename T, int N>
TEST_FUNC constexpr void test_unchecked_load_iter_count()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  auto arr  = make_iota_array<T, N>();

  Vec vec = simd::unchecked_load<Vec>(arr.data(), N);
  assert(vec == arr);

  using Mask = typename Vec::mask_type;
  Mask even_mask(is_even{});
  Vec masked_vec = simd::unchecked_load<Vec>(arr.data(), N, even_mask);
  for (int i = 0; i < N; ++i)
  {
    T expected = (i % 2 == 0) ? static_cast<T>(i + 1) : T{};
    assert(masked_vec[i] == expected);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// unchecked_load: iterator + sentinel

template <typename T, int N>
TEST_FUNC constexpr void test_unchecked_load_iter_sentinel()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  auto arr  = make_iota_array<T, N>();

  Vec vec = simd::unchecked_load<Vec>(arr.data(), arr.data() + N);
  assert(vec == arr);

  using Mask = typename Vec::mask_type;
  Mask even_mask(is_even{});
  Vec masked_vec = simd::unchecked_load<Vec>(arr.data(), arr.data() + N, even_mask);
  for (int i = 0; i < N; ++i)
  {
    T expected = (i % 2 == 0) ? static_cast<T>(i + 1) : T{};
    assert(masked_vec[i] == expected);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// alignment flags

template <typename T, int N>
TEST_FUNC constexpr void test_unchecked_load_aligned()
{
  using Vec            = simd::basic_vec<T, simd::fixed_size<N>>;
  alignas(64) auto arr = make_iota_array<T, N>();

  Vec vec1 = simd::unchecked_load<Vec>(arr, simd::flag_aligned);
  Vec vec2 = simd::unchecked_load<Vec>(arr, simd::flag_overaligned<32>);
  assert(vec1 == arr);
  assert(vec2 == arr);
}

TEST_FUNC constexpr void test_unchecked_load_overaligned_non_power_of_two_size()
{
  using Vec            = simd::basic_vec<float, simd::fixed_size<3>>;
  alignas(16) auto arr = make_iota_array<float, 3>();

  Vec vec = simd::unchecked_load<Vec>(arr, simd::flag_overaligned<16>);
  assert(vec == arr);
}

//----------------------------------------------------------------------------------------------------------------------
// flag_convert: lossy load from wider type

template <typename T, int N>
TEST_FUNC constexpr void test_unchecked_load_convert()
{
  if constexpr (sizeof(T) <= sizeof(int) && cuda::std::is_integral_v<T>)
  {
    using Vec      = simd::basic_vec<T, simd::fixed_size<N>>;
    using WiderT   = cuda::std::conditional_t<cuda::std::is_signed_v<T>, int64_t, uint64_t>;
    auto wider_arr = make_iota_array<WiderT, N>();

    Vec vec = simd::unchecked_load<Vec>(wider_arr, simd::flag_convert);
    assert((vec == make_iota_array<T, N>()));
  }
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    using Vec      = simd::basic_vec<T, simd::fixed_size<N>>;
    auto wider_arr = make_iota_array<double, N>();

    Vec vec = simd::unchecked_load<Vec>(wider_arr, simd::flag_convert);
    assert((vec == make_iota_array<T, N>()));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// noexcept: public functions must NOT be noexcept

template <typename T, int N>
TEST_FUNC constexpr void test_unchecked_load_not_noexcept()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;
  cuda::std::array<T, N> arr{};
  Mask mask(true);
  unused(arr, mask);

  // range overloads
  static_assert(!noexcept(simd::unchecked_load<Vec>(arr, mask)));
  static_assert(!noexcept(simd::unchecked_load<Vec>(arr)));
  // iterator + count overloads
  static_assert(!noexcept(simd::unchecked_load<Vec>(arr.data(), N, mask)));
  static_assert(!noexcept(simd::unchecked_load<Vec>(arr.data(), N)));
  // iterator + sentinel overloads
  static_assert(!noexcept(simd::unchecked_load<Vec>(arr.data(), arr.data() + N, mask)));
  static_assert(!noexcept(simd::unchecked_load<Vec>(arr.data(), arr.data() + N)));
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  test_unchecked_load_range<T, N>();
  test_unchecked_load_range_masked<T, N>();
  test_unchecked_load_iter_count<T, N>();
  test_unchecked_load_iter_sentinel<T, N>();
  if constexpr (cuda::std::is_same_v<T, float> && N == 4)
  {
    test_unchecked_load_overaligned_non_power_of_two_size();
  }
  test_unchecked_load_aligned<T, N>();
  test_unchecked_load_convert<T, N>();
  test_unchecked_load_not_noexcept<T, N>();
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
