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

// [simd.permute.dynamic]
//
// template<simd-vec-type V, simd-integral I>
// constexpr resize_t<I::size(), V> permute(const V& v, const I& indices);
//
// template<simd-mask-type V, simd-integral I>
// constexpr resize_t<I::size(), V> permute(const V& v, const I& indices);

#include <cuda/std/__simd_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// basic_vec: same-size permutations (identity / reverse / broadcast / arbitrary)

template <typename T, typename Idx, int N>
TEST_FUNC constexpr void test_vec_same_size()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  using Ind = simd::basic_vec<Idx, simd::fixed_size<N>>;
  Vec src(iota_generator<T>{});

  // identity: indices = {0, 1, ..., N-1}
  {
    cuda::std::array<Idx, N> idx{};
    for (int i = 0; i < N; ++i)
    {
      idx[i] = static_cast<Idx>(i);
    }
    Ind indices(idx);
    Vec id = simd::permute(src, indices);
    static_assert(cuda::std::is_same_v<decltype(id), Vec>);
    for (int i = 0; i < N; ++i)
    {
      assert(id[i] == src[i]);
    }
  }
  // reverse: indices = {N-1, ..., 0}
  {
    cuda::std::array<Idx, N> idx{};
    for (int i = 0; i < N; ++i)
    {
      idx[i] = static_cast<Idx>(N - 1 - i);
    }
    Ind indices(idx);
    Vec rev = simd::permute(src, indices);
    for (int i = 0; i < N; ++i)
    {
      assert(rev[i] == src[N - 1 - i]);
    }
  }
  // broadcast lane 0: indices = {0, 0, ..., 0}
  {
    Ind zero_indices(Idx{0});
    Vec bcast = simd::permute(src, zero_indices);
    for (int i = 0; i < N; ++i)
    {
      assert(bcast[i] == src[0]);
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// basic_vec: size-changing permutations (I::size() != V::size())

template <typename T>
TEST_FUNC constexpr void test_vec_size_change()
{
  using Src = simd::basic_vec<T, simd::fixed_size<4>>;
  Src src(iota_generator<T>{});

  // larger 4 -> 8  indices = {0,1,2,3,0,1,2,3}
  {
    using Ind8 = simd::basic_vec<int, simd::fixed_size<8>>;
    cuda::std::array<int, 8> idx{0, 1, 2, 3, 0, 1, 2, 3};
    Ind8 indices(idx);

    using LargerVec      = simd::resize_t<8, Src>;
    LargerVec larger     = simd::permute(src, indices);
    using LargerExpected = simd::basic_vec<T, simd::fixed_size<8>>;
    static_assert(cuda::std::is_same_v<LargerVec, LargerExpected>);
    static_assert(cuda::std::is_same_v<decltype(larger), LargerVec>);
    for (int i = 0; i < 8; ++i)
    {
      assert(larger[i] == src[i % 4]);
    }
  }
  // smaller 4 -> 2: indices = {3, 1}
  {
    using Ind2 = simd::basic_vec<int, simd::fixed_size<2>>;
    cuda::std::array<int, 2> idx{3, 1};
    Ind2 indices(idx);

    using SmallerVec      = simd::resize_t<2, Src>;
    SmallerVec smaller    = simd::permute(src, indices);
    using SmallerExpected = simd::basic_vec<T, simd::fixed_size<2>>;
    static_assert(cuda::std::is_same_v<SmallerVec, SmallerExpected>);
    assert(smaller[0] == src[3]);
    assert(smaller[1] == src[1]);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// basic_vec: test with multiple integer index types for completeness

template <typename T, int N>
TEST_FUNC constexpr void test_vec_with_index_types()
{
  test_vec_same_size<T, int16_t, N>();
  test_vec_same_size<T, int, N>();
  test_vec_same_size<T, int64_t, N>();
  test_vec_same_size<T, uint32_t, N>();
}

//----------------------------------------------------------------------------------------------------------------------
// basic_mask: dynamic permute

template <int Bytes, int N>
TEST_FUNC constexpr void test_mask()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  using Ind  = simd::basic_vec<int, simd::fixed_size<N>>;
  Mask src(is_even{});

  // identity
  {
    cuda::std::array<int, N> idx{};
    for (int i = 0; i < N; ++i)
    {
      idx[i] = i;
    }
    Ind indices(idx);
    Mask id = simd::permute(src, indices);
    static_assert(cuda::std::is_same_v<decltype(id), Mask>);
    for (int i = 0; i < N; ++i)
    {
      assert(id[i] == src[i]);
    }
  }
  // reverse
  {
    cuda::std::array<int, N> idx{};
    for (int i = 0; i < N; ++i)
    {
      idx[i] = N - 1 - i;
    }
    Ind indices(idx);
    Mask rev = simd::permute(src, indices);
    for (int i = 0; i < N; ++i)
    {
      assert(rev[i] == src[N - 1 - i]);
    }
  }
  // size-change: larger 4 -> 8
  if constexpr (N == 4)
  {
    using Ind8 = simd::basic_vec<int, simd::fixed_size<8>>;
    cuda::std::array<int, 8> idx{0, 1, 2, 3, 0, 1, 2, 3};
    Ind8 indices(idx);

    using LargerMask     = simd::resize_t<8, Mask>;
    LargerMask larger    = simd::permute(src, indices);
    using LargerExpected = simd::basic_mask<Bytes, simd::fixed_size<8>>;
    static_assert(cuda::std::is_same_v<LargerMask, LargerExpected>);
    for (int i = 0; i < 8; ++i)
    {
      assert(larger[i] == src[i % 4]);
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// noexcept

TEST_FUNC constexpr void test_noexcept()
{
  using Vec     = simd::basic_vec<int, simd::fixed_size<4>>;
  using Mask    = simd::basic_mask<4, simd::fixed_size<4>>;
  using Indices = simd::basic_vec<int, simd::fixed_size<4>>;

  Vec v{};
  Mask m{};
  Indices idx{};
  unused(v, m, idx);

  static_assert(!noexcept(simd::permute(v, idx)));
  static_assert(!noexcept(simd::permute(m, idx)));
}

//----------------------------------------------------------------------------------------------------------------------
// Return-type

TEST_FUNC constexpr void test_return_type()
{
  using Vec4  = simd::basic_vec<int, simd::fixed_size<4>>;
  using Mask4 = simd::basic_mask<4, simd::fixed_size<4>>;
  using Ind2  = simd::basic_vec<int, simd::fixed_size<2>>;
  using Ind4  = simd::basic_vec<int, simd::fixed_size<4>>;
  using Ind8  = simd::basic_vec<int, simd::fixed_size<8>>;

  Vec4 v{};
  Mask4 m{};
  Ind2 i2{};
  Ind4 i4{};
  Ind8 i8{};
  unused(v, m, i2, i4, i8);

  static_assert(cuda::std::is_same_v<decltype(simd::permute(v, i4)), Vec4>);
  static_assert(cuda::std::is_same_v<decltype(simd::permute(v, i2)), simd::resize_t<2, Vec4>>);
  static_assert(cuda::std::is_same_v<decltype(simd::permute(v, i8)), simd::resize_t<8, Vec4>>);
  static_assert(cuda::std::is_same_v<decltype(simd::permute(m, i8)), simd::resize_t<8, Mask4>>);
}

//----------------------------------------------------------------------------------------------------------------------
// do not depend on element types
TEST_FUNC constexpr bool test_fixed_type()
{
  test_noexcept();
  test_return_type();
  test_mask<1, 4>();
  test_mask<4, 4>();
  test_mask<8, 4>();
  return true;
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  test_vec_with_index_types<T, N>();
  if constexpr (N == 4) // size-change tests do not depend on N
  {
    test_vec_size_change<T>();
  }
}

DEFINE_BASIC_VEC_TEST()
DEFINE_BASIC_VEC_TEST_RUNTIME()

int main(int, char**)
{
  assert(test());
  assert(test_fixed_type());
  static_assert(test());
  static_assert(test_fixed_type());
  assert(test_runtime());
  return 0;
}
