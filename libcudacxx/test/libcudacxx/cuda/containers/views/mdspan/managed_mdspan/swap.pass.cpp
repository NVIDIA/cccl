//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>
//
// friend constexpr void swap(mdspan& x, mdspan& y) noexcept;
//
// Effects: Equivalent to:
//   swap(x.ptr_, y.ptr_);
//   swap(x.map_, y.map_);
//   swap(x.acc_, y.acc_);
#define _CCCL_DISABLE_MDSPAN_ACCESSOR_DETECT_INVALIDITY

#include <cuda/mdspan>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "../CustomTestLayouts.h"
#include "../MinimalElementType.h"
#include "test_macros.h"

template <class MDS>
__host__ __device__ constexpr void test_swap(MDS a, MDS b)
{
  auto org_a = a;
  auto org_b = b;
  swap(a, b);
  assert(a.extents() == org_b.extents());
  assert(b.extents() == org_a.extents());
  test_equality_handle(a, org_b.data_handle());
  test_equality_handle(b, org_a.data_handle());
  test_equality_mapping(a, org_b.mapping());
  test_equality_mapping(b, org_a.mapping());
  // This check uses a side effect of layout_wrapping_integral::swap to make sure
  // mdspan calls the underlying components' swap via ADL
  test_swap_counter<MDS>();
}

__host__ __device__ constexpr bool test()
{
  using extents_t    = cuda::std::extents<int, 4, cuda::std::dynamic_extent>;
  float data_a[1024] = {};
  float data_b[1024] = {};
  {
    cuda::managed_mdspan<float, extents_t> a(data_a, extents_t(12));
    cuda::managed_mdspan<float, extents_t> b(data_b, extents_t(5));
    test_swap(a, b);
  }
  {
    layout_wrapping_integral<4>::template mapping<extents_t> map_a(extents_t(12), not_extents_constructible_tag()),
      map_b(extents_t(5), not_extents_constructible_tag());
    cuda::managed_mdspan<float, extents_t, layout_wrapping_integral<4>> a(data_a, map_a);
    cuda::managed_mdspan<float, extents_t, layout_wrapping_integral<4>> b(data_b, map_b);
    test_swap(a, b);
  }
  return true;
}

int main(int, char**)
{
  test();
#if !_CCCL_COMPILER(GCC, <, 11) // gcc-10 complains about swap failing during constant evaluation
  static_assert(test(), "");
#endif // !_CCCL_COMPILER(GCC, <, 11)
  return 0;
}
