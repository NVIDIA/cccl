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

// [simd.loadstore], partial_store
//
// partial_store(const basic_vec&, Range&&, flags<> = {});
// partial_store(const basic_vec&, Range&&, const mask_type&, flags<> = {});
// partial_store(const basic_vec&, I first, iter_difference_t<I> n, flags<> = {});
// partial_store(const basic_vec&, I first, iter_difference_t<I> n, const mask_type&, flags<> = {});
// partial_store(const basic_vec&, I first, S last, flags<> = {});
// partial_store(const basic_vec&, I first, S last, const mask_type&, flags<> = {});

#include <cuda/std/__simd_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// partial_store: range

template <typename T, int N>
TEST_FUNC constexpr void test_partial_store_range()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec(iota_generator<T>{});

  cuda::std::array<T, N> dest{};
  simd::partial_store(vec, dest);
  for (int i = 0; i < N; ++i)
  {
    assert(dest[i] == static_cast<T>(i + 1));
  }

  cuda::std::array<T, N> dest2{};
  simd::partial_store(vec, dest2, simd::flag_default);
  for (int i = 0; i < N; ++i)
  {
    assert(dest2[i] == static_cast<T>(i + 1));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// partial_store: range, masked — verify unmasked lanes are preserved

template <typename T, int N>
TEST_FUNC constexpr void test_partial_store_range_masked()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;
  Vec vec(iota_generator<T>{});

  cuda::std::array<T, N> dest{};
  for (int i = 0; i < N; ++i)
  {
    dest[i] = static_cast<T>(99);
  }

  Mask even_mask(is_even{});
  simd::partial_store(vec, dest, even_mask);
  for (int i = 0; i < N; ++i)
  {
    T expected = (i % 2 == 0) ? static_cast<T>(i + 1) : static_cast<T>(99);
    assert(dest[i] == expected);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// partial_store: smaller range (count < Vec.size)

template <typename T, int N>
TEST_FUNC constexpr void test_partial_store_smaller_range()
{
  if constexpr (N > 1)
  {
    using Vec          = simd::basic_vec<T, simd::fixed_size<N>>;
    constexpr int Half = N / 2;
    Vec vec(iota_generator<T>{});

    cuda::std::array<T, Half> small_dest{};
    simd::partial_store(vec, small_dest);
    for (int i = 0; i < Half; ++i)
    {
      assert(small_dest[i] == static_cast<T>(i + 1));
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// partial_store: iterator + count

template <typename T, int N>
TEST_FUNC constexpr void test_partial_store_iter_count()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec(iota_generator<T>{});

  cuda::std::array<T, N> dest{};
  simd::partial_store(vec, dest.data(), N);
  for (int i = 0; i < N; ++i)
  {
    assert(dest[i] == static_cast<T>(i + 1));
  }

  using Mask = typename Vec::mask_type;
  cuda::std::array<T, N> masked_dest{};
  for (int i = 0; i < N; ++i)
  {
    masked_dest[i] = static_cast<T>(99);
  }
  Mask even_mask(is_even{});
  simd::partial_store(vec, masked_dest.data(), N, even_mask);
  for (int i = 0; i < N; ++i)
  {
    T expected = (i % 2 == 0) ? static_cast<T>(i + 1) : static_cast<T>(99);
    assert(masked_dest[i] == expected);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// partial_store: iterator + sentinel

template <typename T, int N>
TEST_FUNC constexpr void test_partial_store_iter_sentinel()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec vec(iota_generator<T>{});

  cuda::std::array<T, N> dest{};
  simd::partial_store(vec, dest.data(), dest.data() + N);
  for (int i = 0; i < N; ++i)
  {
    assert(dest[i] == static_cast<T>(i + 1));
  }

  using Mask = typename Vec::mask_type;
  cuda::std::array<T, N> masked_dest{};
  for (int i = 0; i < N; ++i)
  {
    masked_dest[i] = static_cast<T>(99);
  }
  Mask even_mask(is_even{});
  simd::partial_store(vec, masked_dest.data(), masked_dest.data() + N, even_mask);
  for (int i = 0; i < N; ++i)
  {
    T expected = (i % 2 == 0) ? static_cast<T>(i + 1) : static_cast<T>(99);
    assert(masked_dest[i] == expected);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// flag_convert: lossy store to narrower type

template <typename T, int N>
TEST_FUNC constexpr void test_partial_store_convert()
{
  if constexpr (sizeof(T) < 8 && cuda::std::is_integral_v<T>)
  {
    using WiderT = cuda::std::conditional_t<cuda::std::is_signed_v<T>, int64_t, uint64_t>;
    using Vec    = simd::basic_vec<WiderT, simd::fixed_size<N>>;
    Vec vec(iota_generator<WiderT>{});

    cuda::std::array<T, N> dest{};
    simd::partial_store(vec, dest, simd::flag_convert);
    for (int i = 0; i < N; ++i)
    {
      assert(dest[i] == static_cast<T>(static_cast<WiderT>(i + 1)));
    }
  }
  if constexpr (cuda::std::is_same_v<T, float>)
  {
    using Vec = simd::basic_vec<double, simd::fixed_size<N>>;
    Vec vec(iota_generator<double>{});

    cuda::std::array<T, N> dest{};
    simd::partial_store(vec, dest, simd::flag_convert);
    for (int i = 0; i < N; ++i)
    {
      assert(dest[i] == static_cast<T>(static_cast<double>(i + 1)));
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// noexcept: public functions must NOT be noexcept

template <typename T, int N>
TEST_FUNC constexpr void test_partial_store_not_noexcept()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Mask = typename Vec::mask_type;
  Vec vec(iota_generator<T>{});
  cuda::std::array<T, N> arr{};
  Mask mask(true);
  unused(vec, arr, mask);

  // range overloads
  static_assert(!noexcept(simd::partial_store(vec, arr, mask)));
  static_assert(!noexcept(simd::partial_store(vec, arr)));
  // iterator + count overloads
  static_assert(!noexcept(simd::partial_store(vec, arr.data(), N, mask)));
  static_assert(!noexcept(simd::partial_store(vec, arr.data(), N)));
  // iterator + sentinel overloads
  static_assert(!noexcept(simd::partial_store(vec, arr.data(), arr.data() + N, mask)));
  static_assert(!noexcept(simd::partial_store(vec, arr.data(), arr.data() + N)));
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  test_partial_store_range<T, N>();
  test_partial_store_range_masked<T, N>();
  test_partial_store_smaller_range<T, N>();
  test_partial_store_iter_count<T, N>();
  test_partial_store_iter_sentinel<T, N>();
  test_partial_store_convert<T, N>();
  test_partial_store_not_noexcept<T, N>();
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
