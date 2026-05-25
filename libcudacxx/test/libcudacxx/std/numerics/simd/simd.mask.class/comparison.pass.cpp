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

// [simd.mask.comparison], basic_mask comparisons (element-wise)
//
// friend constexpr basic_mask operator==(const basic_mask&, const basic_mask&) noexcept;
// friend constexpr basic_mask operator!=(const basic_mask&, const basic_mask&) noexcept;
// friend constexpr basic_mask operator>=(const basic_mask&, const basic_mask&) noexcept;
// friend constexpr basic_mask operator<=(const basic_mask&, const basic_mask&) noexcept;
// friend constexpr basic_mask operator>(const basic_mask&, const basic_mask&) noexcept;
// friend constexpr basic_mask operator<(const basic_mask&, const basic_mask&) noexcept;

#include <cuda/std/__simd_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../simd_test_utils.h"
#include "test_macros.h"

//----------------------------------------------------------------------------------------------------------------------
// identical masks

template <int Bytes, int N>
TEST_FUNC constexpr void test_all_patterns()
{
  using Mask = simd::basic_mask<Bytes, simd::fixed_size<N>>;
  Mask all_true(true);
  Mask all_false(false);

  static_assert(cuda::std::is_same_v<decltype(all_true == all_true), Mask>);
  static_assert(cuda::std::is_same_v<decltype(all_true != all_false), Mask>);
  static_assert(cuda::std::is_same_v<decltype(all_true >= all_false), Mask>);
  static_assert(cuda::std::is_same_v<decltype(all_true <= all_false), Mask>);
  static_assert(cuda::std::is_same_v<decltype(all_true > all_false), Mask>);
  static_assert(cuda::std::is_same_v<decltype(all_true < all_false), Mask>);
  static_assert(noexcept(all_true == all_false));
  static_assert(noexcept(all_true != all_false));
  static_assert(noexcept(all_true >= all_false));
  static_assert(noexcept(all_true <= all_false));
  static_assert(noexcept(all_true > all_false));
  static_assert(noexcept(all_true < all_false));

  // operator==
  {
    Mask eq_result_tt = all_true == all_true;
    Mask eq_result_ft = all_false == all_true;
    Mask eq_result_tf = all_true == all_false;
    Mask eq_result_ff = all_false == all_false;
    for (int i = 0; i < N; ++i)
    {
      assert(eq_result_tt[i] == true);
      assert(eq_result_ft[i] == false);
      assert(eq_result_tf[i] == false);
      assert(eq_result_ff[i] == true);
    }
  }
  // operator!=
  {
    Mask ne_result_tt = all_true != all_true;
    Mask ne_result_ft = all_false != all_true;
    Mask ne_result_tf = all_true != all_false;
    Mask ne_result_ff = all_false != all_false;
    for (int i = 0; i < N; ++i)
    {
      assert(ne_result_tt[i] == false);
      assert(ne_result_ft[i] == true);
      assert(ne_result_tf[i] == true);
      assert(ne_result_ff[i] == false);
    }
  }
  // operator>=
  {
    Mask ge_result_tt = all_true >= all_true;
    Mask ge_result_ft = all_false >= all_true;
    Mask ge_result_tf = all_true >= all_false;
    Mask ge_result_ff = all_false >= all_false;
    for (int i = 0; i < N; ++i)
    {
      assert(ge_result_tt[i] == true);
      assert(ge_result_ft[i] == false);
      assert(ge_result_tf[i] == true);
      assert(ge_result_ff[i] == true);
    }
  }
  // operator<=
  {
    Mask le_result_tt = all_true <= all_true;
    Mask le_result_ft = all_false <= all_true;
    Mask le_result_tf = all_true <= all_false;
    Mask le_result_ff = all_false <= all_false;
    for (int i = 0; i < N; ++i)
    {
      assert(le_result_tt[i] == true);
      assert(le_result_ft[i] == true);
      assert(le_result_tf[i] == false);
      assert(le_result_ff[i] == true);
    }
  }
  // operator>
  {
    Mask gt_result_tt = all_true > all_true;
    Mask gt_result_ft = all_false > all_true;
    Mask gt_result_tf = all_true > all_false;
    Mask gt_result_ff = all_false > all_false;
    for (int i = 0; i < N; ++i)
    {
      assert(gt_result_tt[i] == false);
      assert(gt_result_ft[i] == false);
      assert(gt_result_tf[i] == true);
      assert(gt_result_ff[i] == false);
    }
  }
  // operator<
  {
    Mask lt_result_tt = all_true < all_true;
    Mask lt_result_ft = all_false < all_true;
    Mask lt_result_tf = all_true < all_false;
    Mask lt_result_ff = all_false < all_false;
    for (int i = 0; i < N; ++i)
    {
      assert(lt_result_tt[i] == false);
      assert(lt_result_ft[i] == true);
      assert(lt_result_tf[i] == false);
      assert(lt_result_ff[i] == false);
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
