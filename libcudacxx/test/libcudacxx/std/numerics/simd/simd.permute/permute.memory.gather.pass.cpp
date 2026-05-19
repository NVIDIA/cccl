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

// [simd.permute.memory] gather
//
// template<class V = see below, ranges::contiguous_range R, class I, class IAbi, class... Flags>
// constexpr V partial_gather_from(R&& in, const basic_vec<I, IAbi>& indices, flags<Flags...> f = {});
// template<class V = see below, ranges::contiguous_range R, class I, class IAbi, class... Flags>
// constexpr V partial_gather_from(R&& in, const typename basic_vec<I, IAbi>::mask_type& mask,
//                                 const basic_vec<I, IAbi>& indices, flags<Flags...> f = {});
//
// template<class V = see below, ranges::contiguous_range R, class I, class IAbi, class... Flags>
// constexpr V unchecked_gather_from(R&& in, const basic_vec<I, IAbi>& indices, flags<Flags...> f = {});
// template<class V = see below, ranges::contiguous_range R, class I, class IAbi, class... Flags>
// constexpr V unchecked_gather_from(R&& in, const typename basic_vec<I, IAbi>::mask_type& mask,
//                                   const basic_vec<I, IAbi>& indices, flags<Flags...> f = {});

#include <cuda/std/__simd_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// partial_gather_from — unmasked

template <typename T, int N>
TEST_FUNC constexpr void test_partial_gather_unmasked()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  using Ind = simd::basic_vec<int, simd::fixed_size<N>>;

  auto buf = make_iota_array<T, N, 1>();

  // identity
  {
    auto idx = make_iota_array<int, N>();
    Ind indices(idx);
    Vec vec = simd::partial_gather_from(buf, indices);
    static_assert(cuda::std::is_same_v<decltype(vec), Vec>);
    assert(vec == buf);
  }
  // reverse
  {
    auto idx = make_reverse_iota_array<int, N>();
    Ind indices(idx);
    Vec vec = simd::partial_gather_from(buf, indices);
    for (int i = 0; i < N; ++i)
    {
      assert(vec[i] == buf[N - 1 - i]);
    }
  }
  // OOB indices produce T{}
  if constexpr (N >= 2)
  {
    cuda::std::array<int, N> idx{};
    idx[0] = -1; // OOB: negative
    idx[1] = N; // OOB: too large
    for (int i = 2; i < N; ++i)
    {
      idx[i] = i;
    }
    Ind indices(idx);
    Vec vec = simd::partial_gather_from(buf, indices);
    assert(vec[0] == T{});
    assert(vec[1] == T{});
    for (int i = 2; i < N; ++i)
    {
      assert(vec[i] == buf[i]);
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// partial_gather_from — masked

template <typename T, int N>
TEST_FUNC constexpr void test_partial_gather_masked()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Ind  = simd::basic_vec<int, simd::fixed_size<N>>;
  using Mask = typename Ind::mask_type;

  auto buf = make_iota_array<T, N, 1>();

  // mask all-true + identity indices
  {
    auto idx = make_iota_array<int, N>();
    Ind indices(idx);
    Mask all_true(true);
    Vec vec = simd::partial_gather_from(buf, all_true, indices);
    assert(vec == buf);
  }
  // mask all-false: every component becomes T{}
  {
    auto idx = make_iota_array<int, N>();
    Ind indices(idx);
    Mask all_false(false);
    Vec vec = simd::partial_gather_from(buf, all_false, indices);
    for (int i = 0; i < N; ++i)
    {
      assert(vec[i] == T{});
    }
  }
  // alternating mask: selected lanes load, unselected become T{}
  if constexpr (N >= 2)
  {
    auto idx = make_iota_array<int, N>();
    Ind indices(idx);
    Mask even_mask(is_even{});
    Vec vec = simd::partial_gather_from(buf, even_mask, indices);
    for (int i = 0; i < N; ++i)
    {
      auto expected = (i % 2 == 0) ? buf[i] : T{};
      assert(vec[i] == expected);
    }
  }
  // OOB indices -> T{}
  if constexpr (N >= 2)
  {
    cuda::std::array<int, N> idx{};
    idx[0] = -1;
    idx[1] = N;
    for (int i = 2; i < N; ++i)
    {
      idx[i] = i;
    }
    Ind indices(idx);
    Mask all_true(true);
    Vec vec = simd::partial_gather_from(buf, all_true, indices);
    assert(vec[0] == T{});
    assert(vec[1] == T{});
    for (int i = 2; i < N; ++i)
    {
      assert(vec[i] == buf[i]);
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// unchecked_gather_from

template <typename T, int N>
TEST_FUNC constexpr void test_unchecked_gather()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Ind  = simd::basic_vec<int, simd::fixed_size<N>>;
  using Mask = typename Ind::mask_type;

  auto buf = make_iota_array<T, N, 1>();

  // identity (unmasked)
  {
    auto idx = make_iota_array<int, N>();
    Ind indices(idx);
    Vec vec = simd::unchecked_gather_from(buf, indices);
    static_assert(cuda::std::is_same_v<decltype(vec), Vec>);
    assert(vec == buf);
  }
  // reverse (unmasked)
  {
    auto idx = make_reverse_iota_array<int, N>();
    Ind indices(idx);
    Vec vec = simd::unchecked_gather_from(buf, indices);
    for (int i = 0; i < N; ++i)
    {
      assert(vec[i] == buf[N - 1 - i]);
    }
  }
  // identity (masked, all-true)
  {
    auto idx = make_iota_array<int, N>();
    Ind indices(idx);
    Mask all_true(true);
    Vec vec = simd::unchecked_gather_from(buf, all_true, indices);
    assert(vec == buf);
  }
  // masked, alternating: unselected lanes get T{}; selected lanes must be in-range
  if constexpr (N >= 2)
  {
    auto idx = make_iota_array<int, N>();
    Ind indices(idx);
    Mask even_mask(is_even{});
    Vec vec = simd::unchecked_gather_from(buf, even_mask, indices);
    for (int i = 0; i < N; ++i)
    {
      auto expected = (i % 2 == 0) ? buf[i] : T{};
      assert(vec[i] == expected);
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// Default result deduction

TEST_FUNC constexpr void test_default_result_deduction()
{
  using Expected = simd::basic_vec<int, simd::fixed_size<4>>;
  using Ind      = simd::basic_vec<int, simd::fixed_size<4>>;

  cuda::std::array<int, 4> buf{10, 20, 30, 40};
  cuda::std::array<int, 4> idx{0, 1, 2, 3};
  Ind indices(idx);

  auto vec1 = simd::partial_gather_from(buf, indices);
  static_assert(cuda::std::is_same_v<decltype(vec1), Expected>);
  assert(vec1 == buf);

  typename Ind::mask_type all_true(true);
  auto vec2 = simd::partial_gather_from(buf, all_true, indices);
  static_assert(cuda::std::is_same_v<decltype(vec2), Expected>);
  assert(vec2 == buf);

  auto vec3 = simd::unchecked_gather_from(buf, indices);
  static_assert(cuda::std::is_same_v<decltype(vec3), Expected>);
  assert(vec3 == buf);

  auto vec4 = simd::unchecked_gather_from(buf, all_true, indices);
  static_assert(cuda::std::is_same_v<decltype(vec4), Expected>);
  assert(vec4 == buf);
}

//----------------------------------------------------------------------------------------------------------------------
// Explicit result with flag_convert (float to int)

TEST_FUNC constexpr void test_explicit_result_conversion()
{
  using Result = simd::basic_vec<int, simd::fixed_size<4>>;
  using Ind    = simd::basic_vec<int, simd::fixed_size<4>>;

  cuda::std::array<float, 4> float_buffer{10.0f, 20.0f, 30.0f, 40.0f};
  cuda::std::array<int, 4> idx{0, 1, 2, 3};
  Ind indices(idx);

  auto vec = simd::partial_gather_from<Result>(float_buffer, indices, simd::flag_convert);
  static_assert(cuda::std::is_same_v<decltype(vec), Result>);
  for (int i = 0; i < 4; ++i)
  {
    assert(vec[i] == static_cast<int>(float_buffer[i]));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// noexcept

TEST_FUNC constexpr void test_noexcept()
{
  using Ind  = simd::basic_vec<int, simd::fixed_size<4>>;
  using Mask = typename Ind::mask_type;

  cuda::std::array<int, 4> buf{};
  Ind indices{};
  Mask m{};
  unused(buf, indices, m);

  static_assert(!noexcept(simd::partial_gather_from(buf, indices)));
  static_assert(!noexcept(simd::partial_gather_from(buf, m, indices)));
  static_assert(!noexcept(simd::unchecked_gather_from(buf, indices)));
  static_assert(!noexcept(simd::unchecked_gather_from(buf, m, indices)));
}

//----------------------------------------------------------------------------------------------------------------------

// do not depend on element types
TEST_FUNC constexpr bool test_fixed_type()
{
  test_noexcept();
  test_default_result_deduction();
  test_explicit_result_conversion();
  return true;
}

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  test_partial_gather_unmasked<T, N>();
  test_partial_gather_masked<T, N>();
  test_unchecked_gather<T, N>();
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
