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

// [simd.permute.static]
//
// inline constexpr simd-size-type zero_element   = implementation-defined;
// inline constexpr simd-size-type uninit_element = implementation-defined;
//
// template<simd-size-type N = V::size(), simd-vec-type V, class IdxMap>
// constexpr resize_t<N, V> permute(const V& v, IdxMap&& idxmap);
//
// template<simd-size-type N = V::size(), simd-mask-type V, class IdxMap>
// constexpr resize_t<N, V> permute(const V& v, IdxMap&& idxmap);

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// Index-map generators

struct identity_gen
{
  template <typename I>
  TEST_FUNC constexpr ptrdiff_t operator()(I i) const
  {
    return i;
  }
};

struct reverse_gen_two_args
{
  template <typename I, typename S>
  TEST_FUNC constexpr ptrdiff_t operator()(I i, S size) const
  {
    return size - 1 - i;
  }
};

struct broadcast_lane0_gen
{
  template <typename I>
  TEST_FUNC constexpr ptrdiff_t operator()(I) const
  {
    return 0;
  }
};

struct repeat_modulo_gen
{
  template <typename I, typename S>
  TEST_FUNC constexpr ptrdiff_t operator()(I i, S size) const
  {
    return i % size;
  }
};

// Sentinel generators

// Zero out the odd lanes, keep the even lanes. Exercises zero_element
struct zero_odd_lanes_gen
{
  template <typename I>
  TEST_FUNC constexpr ptrdiff_t operator()(I i) const
  {
    return (i % 2 == 0) ? i : simd::zero_element;
  }
};

// Mark the odd lanes as uninit_element
struct uninit_odd_lanes_gen
{
  template <typename I>
  TEST_FUNC constexpr ptrdiff_t operator()(I i) const
  {
    const auto idx = static_cast<ptrdiff_t>(i);
    return (idx % 2 == 0) ? idx : simd::uninit_element;
  }
};

//----------------------------------------------------------------------------------------------------------------------
// basic_vec: default-N overload (N defaults to V::size())

template <typename T, int N>
TEST_FUNC constexpr void test_vec_default_n()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;

  Vec src(iota_generator<T>{});

  // identity: result matches source lane-for-lane
  Vec id = simd::permute(src, identity_gen{});
  static_assert(cuda::std::is_same_v<decltype(id), Vec>);
  for (int i = 0; i < N; ++i)
  {
    assert(id[i] == src[i]);
  }

  // reverse (2-arg form)
  Vec rev = simd::permute(src, reverse_gen_two_args{});
  for (int i = 0; i < N; ++i)
  {
    assert(rev[i] == src[N - 1 - i]);
  }

  // broadcast lane 0 (1-arg form)
  Vec broadcast = simd::permute(src, broadcast_lane0_gen{});
  for (int i = 0; i < N; ++i)
  {
    assert(broadcast[i] == src[0]);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// basic_vec: explicit-N overload (size change)

template <typename T>
TEST_FUNC constexpr void test_vec_size_change()
{
  using Src = simd::basic_vec<T, simd::fixed_size<4>>;
  Src src(iota_generator<T>{});

  // larger: 4 -> 8 via cyclic tiling
  using LargeVec       = simd::resize_t<8, Src>;
  LargeVec larger      = simd::permute<8>(src, repeat_modulo_gen{});
  using LargerExpected = simd::basic_vec<T, simd::fixed_size<8>>;
  static_assert(cuda::std::is_same_v<LargeVec, LargerExpected>);
  static_assert(cuda::std::is_same_v<decltype(larger), LargeVec>);
  for (int i = 0; i < 8; ++i)
  {
    assert(larger[i] == src[i % 4]);
  }

  // smaller: 4 -> 2 via identity (takes the first two source lanes)
  using SmallerVec         = simd::resize_t<2, Src>;
  const SmallerVec smaller = simd::permute<2>(src, identity_gen{});
  using SmallerExpected    = simd::basic_vec<T, simd::fixed_size<2>>;
  static_assert(cuda::std::is_same_v<SmallerVec, SmallerExpected>);
  for (int i = 0; i < 2; ++i)
  {
    assert(smaller[i] == src[i]);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// sentinels (zero_element / uninit_element)

template <typename T, int N>
TEST_FUNC constexpr void test_vec_sentinels()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  Vec src(iota_generator<T>{});

  // zero_element on odd lanes
  Vec zeroed = simd::permute(src, zero_odd_lanes_gen{});
  for (int i = 0; i < N; ++i)
  {
    auto expected = i % 2 == 0 ? src[i] : T{};
    assert(zeroed[i] == expected);
  }

  // uninit_element on odd lanes: unspecified-value
  Vec uninit = simd::permute(src, uninit_odd_lanes_gen{});
  for (int i = 0; i < N; i += 2)
  {
    assert(uninit[i] == src[i]);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// basic_mask

template <int Bytes, int N>
TEST_FUNC constexpr void test_mask()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  Mask src(is_even{});

  // identity
  Mask id = simd::permute(src, identity_gen{});
  static_assert(cuda::std::is_same_v<decltype(id), Mask>);
  for (int i = 0; i < N; ++i)
  {
    assert(id[i] == src[i]);
  }

  // reverse (2-arg form)
  Mask rev = simd::permute(src, reverse_gen_two_args{});
  for (int i = 0; i < N; ++i)
  {
    assert(rev[i] == src[N - 1 - i]);
  }

  // zero_element on odd lanes (become false)
  Mask zeroed = simd::permute(src, zero_odd_lanes_gen{});
  for (int i = 0; i < N; ++i)
  {
    auto expected = i % 2 == 0 ? src[i] : false;
    assert(zeroed[i] == expected);
  }

  // size change: larger 4 -> 8
  if constexpr (N == 4)
  {
    using LargerMask     = simd::resize_t<8, Mask>;
    LargerMask larger    = simd::permute<8>(src, repeat_modulo_gen{});
    using LargerExpected = simd::basic_mask<Bytes, simd::fixed_size<8>>;
    static_assert(cuda::std::is_same_v<LargerMask, LargerExpected>);
    for (int i = 0; i < 8; ++i)
    {
      assert(larger[i] == src[i % 4]);
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// Sentinel values

TEST_FUNC constexpr void test_sentinels_properties()
{
  static_assert(simd::zero_element != simd::uninit_element);
  static_assert(simd::zero_element < 0);
  static_assert(simd::uninit_element < 0);
  static_assert(cuda::std::is_same_v<decltype(simd::zero_element), const ptrdiff_t>);
  static_assert(cuda::std::is_same_v<decltype(simd::uninit_element), const ptrdiff_t>);
}

//----------------------------------------------------------------------------------------------------------------------

TEST_FUNC constexpr void test_noexcept()
{
  using Vec  = simd::basic_vec<int, simd::fixed_size<4>>;
  using Mask = simd::basic_mask<4, simd::fixed_size<4>>;

  Vec v{};
  Mask m{};
  unused(v, m);

  static_assert(!noexcept(simd::permute(v, identity_gen{})));
  static_assert(!noexcept(simd::permute(m, identity_gen{})));
}

//----------------------------------------------------------------------------------------------------------------------

// do not depend on types
TEST_FUNC constexpr bool test_fixed_type()
{
  test_sentinels_properties();
  test_noexcept();
  test_mask<1, 4>();
  test_mask<4, 4>();
  test_mask<8, 4>();
  return true;
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  test_vec_default_n<T, N>();
  test_vec_sentinels<T, N>();
  if constexpr (N == 4) // Size-change tests do not depend on N
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
