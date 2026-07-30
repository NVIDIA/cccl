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

// [simd.creation], chunk (basic_mask)
//
// template<class T, class Abi> constexpr auto chunk(const basic_mask<mask-element-size<T>, Abi>&);
// template<simd-size-type N, size_t Bytes, class Abi> constexpr auto chunk(const basic_mask<Bytes, Abi>&);

#include <cuda/std/__simd_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// chunk<SubMask>(basic_mask) - exact divisor

template <typename T>
TEST_FUNC constexpr void test_chunk_exact_mask()
{
  constexpr cuda::std::size_t Bytes = sizeof(T);
  using SrcAbi                      = simd::fixed_size<8>;
  using SubMask                     = simd::basic_mask<Bytes, simd::fixed_size<4>>;
  using SrcMask                     = simd::basic_mask<Bytes, SrcAbi>;

  SrcMask src(is_even{});
  auto chunks = simd::chunk<SubMask>(src);
  static_assert(cuda::std::is_same_v<decltype(chunks), cuda::std::array<SubMask, 2>>);
  for (int i = 0; i < 4; ++i)
  {
    assert(chunks[0][i] == (i % 2 == 0));
    assert(chunks[1][i] == ((i + 4) % 2 == 0));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// chunk<SubMask>(basic_mask) - remainder

template <typename T>
TEST_FUNC constexpr void test_chunk_remainder_mask()
{
  constexpr cuda::std::size_t Bytes = sizeof(T);
  using SrcAbi                      = simd::fixed_size<6>;
  using SubMask                     = simd::basic_mask<Bytes, simd::fixed_size<4>>;
  using TailMask                    = simd::resize_t<2, SubMask>;
  using SrcMask                     = simd::basic_mask<Bytes, SrcAbi>;

  SrcMask src(is_even{});
  auto chunks = simd::chunk<SubMask>(src);
  static_assert(cuda::std::is_same_v<decltype(chunks), cuda::std::tuple<SubMask, TailMask>>);

  auto head = cuda::std::get<0>(chunks);
  auto tail = cuda::std::get<1>(chunks);
  for (int i = 0; i < 4; ++i)
  {
    assert(head[i] == (i % 2 == 0));
  }
  for (int i = 0; i < 2; ++i)
  {
    assert(tail[i] == ((i + 4) % 2 == 0));
  }
}

//----------------------------------------------------------------------------------------------------------------------
// chunk<N>(basic_mask) - exact divisor

template <typename T>
TEST_FUNC constexpr void test_chunk_by_n_mask()
{
  constexpr cuda::std::size_t Bytes = sizeof(T);
  using SrcAbi                      = simd::fixed_size<8>;
  using SubMask                     = simd::basic_mask<Bytes, simd::fixed_size<2>>;
  using SrcMask                     = simd::basic_mask<Bytes, SrcAbi>;

  SrcMask src(is_even{});
  auto chunks = simd::chunk<2>(src);
  static_assert(cuda::std::is_same_v<decltype(chunks), cuda::std::array<SubMask, 4>>);
  for (int k = 0; k < 4; ++k)
  {
    for (int i = 0; i < 2; ++i)
    {
      assert(chunks[k][i] == ((k * 2 + i) % 2 == 0));
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
// chunk<N>(basic_mask) - remainder

template <typename T>
TEST_FUNC constexpr void test_chunk_by_n_remainder_mask()
{
  constexpr cuda::std::size_t Bytes = sizeof(T);
  using SrcAbi                      = simd::fixed_size<6>;
  using SubMask                     = simd::basic_mask<Bytes, simd::fixed_size<4>>;
  using TailMask                    = simd::resize_t<2, SubMask>;
  using SrcMask                     = simd::basic_mask<Bytes, SrcAbi>;

  SrcMask src(is_even{});
  auto chunks = simd::chunk<4>(src);
  static_assert(cuda::std::is_same_v<decltype(chunks), cuda::std::tuple<SubMask, TailMask>>);

  auto head = cuda::std::get<0>(chunks);
  auto tail = cuda::std::get<1>(chunks);
  for (int i = 0; i < 4; ++i)
  {
    assert(head[i] == (i % 2 == 0));
  }
  for (int i = 0; i < 2; ++i)
  {
    assert(tail[i] == ((i + 4) % 2 == 0));
  }
}

//----------------------------------------------------------------------------------------------------------------------

template <typename T>
TEST_FUNC constexpr void test_type()
{
  test_chunk_exact_mask<T>();
  test_chunk_remainder_mask<T>();
  test_chunk_by_n_mask<T>();
  test_chunk_by_n_remainder_mask<T>();
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
