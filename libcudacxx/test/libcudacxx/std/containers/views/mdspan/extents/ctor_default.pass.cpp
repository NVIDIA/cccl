//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// Test default construction:
//
// constexpr extents() noexcept = default;
//
// Remarks: since the standard uses an exposition only array member, dynamic extents
// need to be zero initialized!

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/mdspan>

#include "../ConvertibleToIntegral.h"
#include "CtorTestCombinations.h"
#include "test_macros.h"

struct DefaultCtorTest
{
  template <class E,
            class AllExtents,
            class Extents,
            size_t... Indices,
            cuda::std::enable_if_t<sizeof...(Indices) == E::rank(), int> = 0>
  __host__ __device__ static constexpr void
  test_construction(AllExtents all_ext, Extents, cuda::std::index_sequence<Indices...>)
  {
    // This function gets called twice: once with Extents being just the dynamic ones, and once with all the extents
    // specified. We only test during the all extent case, since then Indices is the correct number. This allows us to
    // reuse the same testing machinery used in other constructor tests.
    static_assert(noexcept(E{}));
    // Need to construct new expected values, replacing dynamic values with 0
    cuda::std::array<typename AllExtents::value_type, E::rank()> expected_exts{
      ((E::static_extent(Indices) == cuda::std::dynamic_extent)
         ? typename AllExtents::value_type(0)
         : all_ext[Indices])...};
    test_runtime_observers(E{}, expected_exts);
  }

  template <class E,
            class AllExtents,
            class Extents,
            size_t... Indices,
            cuda::std::enable_if_t<sizeof...(Indices) != E::rank(), int> = 0>
  __host__ __device__ static constexpr void
  test_construction(AllExtents all_ext, Extents, cuda::std::index_sequence<Indices...>)
  {
    // nothing to do here
  }
};

int main(int, char**)
{
  test_index_type_combo<DefaultCtorTest>();
#if TEST_STD_VER >= 2017
  static_assert(test_index_type_combo<DefaultCtorTest>(), "");
#endif // TEST_STD_VER >= 2017
  return 0;
}
