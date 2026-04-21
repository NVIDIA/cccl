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

// [simd.mask.binary], basic_mask binary operators
//
// friend constexpr basic_mask operator&&(const basic_mask&, const basic_mask&) noexcept;
// friend constexpr basic_mask operator||(const basic_mask&, const basic_mask&) noexcept;
// friend constexpr basic_mask operator&(const basic_mask&, const basic_mask&) noexcept;
// friend constexpr basic_mask operator|(const basic_mask&, const basic_mask&) noexcept;
// friend constexpr basic_mask operator^(const basic_mask&, const basic_mask&) noexcept;

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

template <int Bytes, int N>
TEST_FUNC constexpr void test_all_patterns()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  Mask all_true(true);
  Mask all_false(false);

  static_assert(cuda::std::is_same_v<decltype(all_true && all_false), Mask>);
  static_assert(cuda::std::is_same_v<decltype(all_true || all_false), Mask>);
  static_assert(cuda::std::is_same_v<decltype(all_true & all_false), Mask>);
  static_assert(cuda::std::is_same_v<decltype(all_true | all_false), Mask>);
  static_assert(cuda::std::is_same_v<decltype(all_true ^ all_false), Mask>);
  static_assert(noexcept(all_true && all_false));
  static_assert(noexcept(all_true || all_false));
  static_assert(noexcept(all_true & all_false));
  static_assert(noexcept(all_true | all_false));
  static_assert(noexcept(all_true ^ all_false));
  // logical AND
  {
    Mask and_result_tt = all_true && all_true;
    Mask and_result_ft = all_false && all_true;
    Mask and_result_tf = all_true && all_false;
    Mask and_result_ff = all_false && all_false;
    for (int i = 0; i < N; ++i)
    {
      assert(and_result_tt[i] == true);
      assert(and_result_ft[i] == false);
      assert(and_result_tf[i] == false);
      assert(and_result_ff[i] == false);
    }
  }
  // logical OR
  {
    Mask or_result_tt = all_true || all_true;
    Mask or_result_ft = all_false || all_true;
    Mask or_result_tf = all_true || all_false;
    Mask or_result_ff = all_false || all_false;
    for (int i = 0; i < N; ++i)
    {
      assert(or_result_tt[i] == true);
      assert(or_result_ft[i] == true);
      assert(or_result_tf[i] == true);
      assert(or_result_ff[i] == false);
    }
  }
  // bitwise AND
  {
    Mask bit_and_result_tt = all_true & all_true;
    Mask bit_and_result_ft = all_false & all_true;
    Mask bit_and_result_tf = all_true & all_false;
    Mask bit_and_result_ff = all_false & all_false;
    for (int i = 0; i < N; ++i)
    {
      assert(bit_and_result_tt[i] == true);
      assert(bit_and_result_ft[i] == false);
      assert(bit_and_result_tf[i] == false);
      assert(bit_and_result_ff[i] == false);
    }
  }
  // bitwise OR
  {
    Mask bit_or_result_tt = all_true | all_true;
    Mask bit_or_result_ft = all_false | all_true;
    Mask bit_or_result_tf = all_true | all_false;
    Mask bit_or_result_ff = all_false | all_false;
    for (int i = 0; i < N; ++i)
    {
      assert(bit_or_result_tt[i] == true);
      assert(bit_or_result_ft[i] == true);
      assert(bit_or_result_tf[i] == true);
      assert(bit_or_result_ff[i] == false);
    }
  }
  // bitwise XOR
  {
    Mask bit_xor_result_tt = all_true ^ all_true;
    Mask bit_xor_result_ft = all_false ^ all_true;
    Mask bit_xor_result_tf = all_true ^ all_false;
    Mask bit_xor_result_ff = all_false ^ all_false;
    for (int i = 0; i < N; ++i)
    {
      assert(bit_xor_result_tt[i] == false);
      assert(bit_xor_result_ft[i] == true);
      assert(bit_xor_result_tf[i] == true);
      assert(bit_xor_result_ff[i] == false);
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------

template <int Bytes>
TEST_FUNC constexpr void test_bytes()
{
  test_all_patterns<Bytes, 1>();
  test_all_patterns<Bytes, 4>();
}

TEST_FUNC constexpr bool test()
{
  test_bytes<1>();
  test_bytes<2>();
  test_bytes<4>();
  test_bytes<8>();
#if _CCCL_HAS_INT128()
  test_bytes<16>();
#endif
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
