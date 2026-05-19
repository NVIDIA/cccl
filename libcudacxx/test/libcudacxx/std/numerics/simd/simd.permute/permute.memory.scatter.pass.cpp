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

// [simd.permute.memory] scatter
//
// template<class T, class Abi, ranges::contiguous_range R, class I, class IAbi, class... Flags>
// constexpr void partial_scatter_to(const basic_vec<T, Abi>& v, R&& out,
//                                   const basic_vec<I, IAbi>& indices, flags<Flags...> f = {});
// template<class T, class Abi, ranges::contiguous_range R, class I, class IAbi, class... Flags>
// constexpr void partial_scatter_to(const basic_vec<T, Abi>& v, R&& out,
//                                   const typename basic_vec<I, IAbi>::mask_type& mask,
//                                   const basic_vec<I, IAbi>& indices, flags<Flags...> f = {});
//
// template<class T, class Abi, ranges::contiguous_range R, class I, class IAbi, class... Flags>
// constexpr void unchecked_scatter_to(const basic_vec<T, Abi>& v, R&& out,
//                                     const basic_vec<I, IAbi>& indices, flags<Flags...> f = {});
// template<class T, class Abi, ranges::contiguous_range R, class I, class IAbi, class... Flags>
// constexpr void unchecked_scatter_to(const basic_vec<T, Abi>& v, R&& out,
//                                     const typename basic_vec<I, IAbi>::mask_type& mask,
//                                     const basic_vec<I, IAbi>& indices, flags<Flags...> f = {});

#include <cuda/std/__simd_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// partial_scatter_to — unmasked

template <typename T, int N>
TEST_FUNC constexpr void test_partial_scatter_unmasked()
{
  using Vec = simd::basic_vec<T, simd::fixed_size<N>>;
  using Ind = simd::basic_vec<int, simd::fixed_size<N>>;
  Vec src(iota_generator<T>{});

  // identity indices: arr[i] == src[i]
  {
    cuda::std::array<T, N> arr{};
    auto idx = make_iota_array<int, N>();
    Ind indices(idx);
    simd::partial_scatter_to(src, arr, indices);
    assert(src == arr);
  }
  // reverse indices: arr[N-1-i] == src[i]
  {
    cuda::std::array<T, N> arr{};
    auto idx = make_reverse_iota_array<int, N>();
    Ind indices(idx);
    simd::partial_scatter_to(src, arr, indices);
    for (int i = 0; i < N; ++i)
    {
      assert(arr[i] == src[N - 1 - i]);
    }
  }
  // OOB indices: those stores are dropped
  if constexpr (N >= 2)
  {
    cuda::std::array<T, N> arr{static_cast<T>(1), static_cast<T>(2)};
    cuda::std::array<int, N> idx{};
    idx[0] = -1;
    idx[1] = N;
    for (int i = 2; i < N; ++i)
    {
      idx[i] = i;
    }
    Ind indices(idx);
    simd::partial_scatter_to(src, arr, indices);
    assert(arr[0] == static_cast<T>(1));
    assert(arr[1] == static_cast<T>(2));
    for (int i = 2; i < N; ++i)
    {
      assert(arr[i] == src[i]);
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// partial_scatter_to — masked

template <typename T, int N>
TEST_FUNC constexpr void test_partial_scatter_masked()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Ind  = simd::basic_vec<int, simd::fixed_size<N>>;
  using Mask = typename Ind::mask_type;
  Vec src(iota_generator<T>{});

  // mask all-false: no stores
  {
    cuda::std::array<T, N> arr{};
    auto idx = make_iota_array<int, N>();
    Ind indices(idx);
    Mask all_false(false);
    simd::partial_scatter_to(src, arr, all_false, indices);
    assert((arr == cuda::std::array<T, N>{}));
  }
  // mask all-true + identity: arr == src
  {
    cuda::std::array<T, N> arr{};
    auto idx = make_iota_array<int, N>();
    Ind indices(idx);
    Mask all_true(true);
    simd::partial_scatter_to(src, arr, all_true, indices);
    assert(src == arr);
  }
  // alternating mask + identity: only selected lanes are stored
  if constexpr (N >= 2)
  {
    cuda::std::array<T, N> arr{};
    auto idx = make_iota_array<int, N>();
    Ind indices(idx);
    Mask even_mask(is_even{});
    simd::partial_scatter_to(src, arr, even_mask, indices);
    for (int i = 0; i < N; ++i)
    {
      auto expected = (i % 2 == 0) ? src[i] : T{};
      assert(arr[i] == expected);
    }
  }
  // masked + OOB
  if constexpr (N >= 2)
  {
    cuda::std::array<T, N> arr{static_cast<T>(1), static_cast<T>(2)};
    cuda::std::array<int, N> idx{};
    idx[0] = -1;
    idx[1] = N;
    for (int i = 2; i < N; ++i)
    {
      idx[i] = i;
    }
    Ind indices(idx);
    Mask all_true(true);
    simd::partial_scatter_to(src, arr, all_true, indices);
    assert(arr[0] == static_cast<T>(1));
    assert(arr[1] == static_cast<T>(2));
    for (int i = 2; i < N; ++i)
    {
      assert(arr[i] == src[i]);
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// unchecked_scatter_to

template <typename T, int N>
TEST_FUNC constexpr void test_unchecked_scatter()
{
  using Vec  = simd::basic_vec<T, simd::fixed_size<N>>;
  using Ind  = simd::basic_vec<int, simd::fixed_size<N>>;
  using Mask = typename Ind::mask_type;
  Vec src(iota_generator<T>{});

  // identity (unmasked)
  {
    cuda::std::array<T, N> arr{};
    auto idx = make_iota_array<int, N>();
    Ind indices(idx);
    simd::unchecked_scatter_to(src, arr, indices);
    assert(src == arr);
  }
  // reverse (unmasked)
  {
    cuda::std::array<T, N> arr{};
    auto idx = make_reverse_iota_array<int, N>();
    Ind indices(idx);
    simd::unchecked_scatter_to(src, arr, indices);
    for (int i = 0; i < N; ++i)
    {
      assert(arr[i] == src[N - 1 - i]);
    }
  }
  // identity (masked, all-true)
  {
    cuda::std::array<T, N> arr{};
    auto idx = make_iota_array<int, N>();
    Ind indices(idx);
    Mask all_true(true);
    simd::unchecked_scatter_to(src, arr, all_true, indices);
    assert(src == arr);
  }
  // alternating mask: only selected lanes get stored
  if constexpr (N >= 2)
  {
    cuda::std::array<T, N> arr{};
    auto idx = make_iota_array<int, N>();
    Ind indices(idx);
    Mask even_mask(is_even{});
    simd::unchecked_scatter_to(src, arr, even_mask, indices);
    for (int i = 0; i < N; ++i)
    {
      auto expected = (i % 2 == 0) ? src[i] : T{};
      assert(arr[i] == expected);
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// noexcept

TEST_FUNC constexpr void test_noexcept()
{
  using Vec  = simd::basic_vec<int, simd::fixed_size<4>>;
  using Ind  = simd::basic_vec<int, simd::fixed_size<4>>;
  using Mask = typename Ind::mask_type;

  cuda::std::array<int, 4> out{};
  Vec v{};
  Ind indices{};
  Mask m{};
  unused(out, v, indices, m);

  static_assert(!noexcept(simd::partial_scatter_to(v, out, indices)));
  static_assert(!noexcept(simd::partial_scatter_to(v, out, m, indices)));
  static_assert(!noexcept(simd::unchecked_scatter_to(v, out, indices)));
  static_assert(!noexcept(simd::unchecked_scatter_to(v, out, m, indices)));
}

//----------------------------------------------------------------------------------------------------------------------
// return-type

TEST_FUNC constexpr void test_return_type()
{
  using Vec  = simd::basic_vec<int, simd::fixed_size<4>>;
  using Ind  = simd::basic_vec<int, simd::fixed_size<4>>;
  using Mask = typename Ind::mask_type;

  cuda::std::array<int, 4> out{};
  Vec v{};
  Ind indices{};
  Mask m{};
  unused(out, v, indices, m);

  static_assert(cuda::std::is_same_v<decltype(simd::partial_scatter_to(v, out, indices)), void>);
  static_assert(cuda::std::is_same_v<decltype(simd::partial_scatter_to(v, out, m, indices)), void>);
  static_assert(cuda::std::is_same_v<decltype(simd::unchecked_scatter_to(v, out, indices)), void>);
  static_assert(cuda::std::is_same_v<decltype(simd::unchecked_scatter_to(v, out, m, indices)), void>);
}

//----------------------------------------------------------------------------------------------------------------------

// do not depend on element types
TEST_FUNC constexpr bool test_fixed_type()
{
  test_noexcept();
  test_return_type();
  return true;
}

template <typename T, int N>
TEST_FUNC constexpr void test_type()
{
  test_partial_scatter_unmasked<T, N>();
  test_partial_scatter_masked<T, N>();
  test_unchecked_scatter<T, N>();
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
