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

// [simd.creation], chunk (basic_vec)
//
// template<class T, class Abi> constexpr auto chunk(const basic_vec<typename T::value_type, Abi>&);
// template<simd-size-type N, class T, class Abi> constexpr auto chunk(const basic_vec<T, Abi>&);

#include <cuda/std/__simd_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// chunk<SubVec>(basic_vec) - exact divisor

template <typename T>
TEST_FUNC constexpr void test_chunk_exact_vec()
{
  using SrcAbi                   = simd::fixed_size<8>;
  using SubVec                   = simd::basic_vec<T, simd::fixed_size<4>>;
  simd::basic_vec<T, SrcAbi> src = make_iota_vec<T, 8>();

  auto chunks = simd::chunk<SubVec>(src);
  static_assert(cuda::std::is_same_v<decltype(chunks), cuda::std::array<SubVec, 2>>);
  for (int i = 0; i < 4; ++i)
  {
    assert(chunks[0][i] == static_cast<T>(i));
    assert(chunks[1][i] == static_cast<T>(i + 4));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// chunk<SubVec>(basic_vec) - remainder

template <typename T>
TEST_FUNC constexpr void test_chunk_remainder_vec()
{
  using SrcAbi                   = simd::fixed_size<6>;
  using SubVec                   = simd::basic_vec<T, simd::fixed_size<4>>;
  using TailVec                  = simd::resize_t<2, SubVec>;
  simd::basic_vec<T, SrcAbi> src = make_iota_vec<T, 6>();

  auto chunks = simd::chunk<SubVec>(src);
  static_assert(cuda::std::is_same_v<decltype(chunks), cuda::std::tuple<SubVec, TailVec>>);

  auto head = cuda::std::get<0>(chunks);
  auto tail = cuda::std::get<1>(chunks);
  for (int i = 0; i < 4; ++i)
  {
    assert(head[i] == static_cast<T>(i));
  }
  for (int i = 0; i < 2; ++i)
  {
    assert(tail[i] == static_cast<T>(i + 4));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// chunk<N>(basic_vec) - exact divisor

template <typename T>
TEST_FUNC constexpr void test_chunk_by_n_vec()
{
  using SrcAbi                   = simd::fixed_size<8>;
  using SubVec                   = simd::basic_vec<T, simd::fixed_size<2>>;
  simd::basic_vec<T, SrcAbi> src = make_iota_vec<T, 8>();

  auto chunks = simd::chunk<2>(src);
  static_assert(cuda::std::is_same_v<decltype(chunks), cuda::std::array<SubVec, 4>>);
  for (int k = 0; k < 4; ++k)
  {
    for (int i = 0; i < 2; ++i)
    {
      assert(chunks[k][i] == static_cast<T>(k * 2 + i));
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// chunk<N>(basic_vec) - remainder

template <typename T>
TEST_FUNC constexpr void test_chunk_by_n_remainder_vec()
{
  using SrcAbi                   = simd::fixed_size<6>;
  using SubVec                   = simd::basic_vec<T, simd::fixed_size<4>>;
  using TailVec                  = simd::resize_t<2, SubVec>;
  simd::basic_vec<T, SrcAbi> src = make_iota_vec<T, 6>();

  auto chunks = simd::chunk<4>(src);
  static_assert(cuda::std::is_same_v<decltype(chunks), cuda::std::tuple<SubVec, TailVec>>);

  auto head = cuda::std::get<0>(chunks);
  auto tail = cuda::std::get<1>(chunks);
  for (int i = 0; i < 4; ++i)
  {
    assert(head[i] == static_cast<T>(i));
  }
  for (int i = 0; i < 2; ++i)
  {
    assert(tail[i] == static_cast<T>(i + 4));
  }
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T>
TEST_FUNC constexpr void test_type()
{
  test_chunk_exact_vec<T>();
  test_chunk_remainder_vec<T>();
  test_chunk_by_n_vec<T>();
  test_chunk_by_n_remainder_vec<T>();
}

TEST_FUNC constexpr bool test()
{
  test_type<cuda::std::int16_t>();
  test_type<float>();
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
