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

// [simd.flags], load and store flags
//
// template<class... Flags> struct flags;
//
// inline constexpr flags<> flag_default{};
// inline constexpr flags<convert-flag> flag_convert{};
// inline constexpr flags<aligned-flag> flag_aligned{};
// template<size_t N> constexpr flags<overaligned-flag<N>> flag_overaligned{};
//
// [simd.flags.oper], flags operators
// template<class... Other> friend constexpr flags<Flags..., Other...> operator|(flags, flags<Other...>);

#include <cuda/std/__simd_>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace simd = cuda::std::simd;

template <typename T, typename R>
TEST_FUNC constexpr bool is_same_flags(const T&, const R&)
{
  return cuda::std::is_same_v<cuda::std::remove_cvref_t<T>, cuda::std::remove_cvref_t<R>>;
}

TEST_FUNC void test()
{
  static_assert(is_same_flags(simd::flag_default, simd::flags<>{}));
  // default | X == X
  static_assert(is_same_flags(simd::flag_default | simd::flag_convert, simd::flag_convert));
  static_assert(is_same_flags(simd::flag_default | simd::flag_aligned, simd::flag_aligned));
  static_assert(is_same_flags(simd::flag_default | simd::flag_overaligned<32>, simd::flag_overaligned<32>));

  // X | default == X
  static_assert(is_same_flags(simd::flag_convert | simd::flag_default, simd::flag_convert));
  static_assert(is_same_flags(simd::flag_aligned | simd::flag_default, simd::flag_aligned));
  static_assert(is_same_flags(simd::flag_overaligned<32> | simd::flag_default, simd::flag_overaligned<32>));

  // two distinct flags
  static_assert(is_same_flags(simd::flag_convert | simd::flag_aligned, simd::flag_convert | simd::flag_aligned));
  static_assert(is_same_flags(simd::flag_aligned | simd::flag_convert, simd::flag_aligned | simd::flag_convert));
  static_assert(
    is_same_flags(simd::flag_convert | simd::flag_overaligned<32>, simd::flag_convert | simd::flag_overaligned<32>));
  static_assert(
    is_same_flags(simd::flag_overaligned<32> | simd::flag_convert, simd::flag_overaligned<32> | simd::flag_convert));
}

int main(int, char**)
{
  test();
  return 0;
}
