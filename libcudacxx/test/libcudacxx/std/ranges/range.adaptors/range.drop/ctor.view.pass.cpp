//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: a non-__tile__ variable cannot be used in tile code

// constexpr explicit drop_view(V base, range_difference_t<V> count); // explicit since C++23

#include <cuda/std/ranges>

#include "test_convertible.h"
#include "test_macros.h"
#include "types.h"

static_assert(!test_convertible<cuda::std::ranges::drop_view<MoveOnlyView>,
                                MoveOnlyView,
                                cuda::std::ranges::range_difference_t<MoveOnlyView>>(),
              "This constructor must be explicit");

TEST_HOST_DEVICE_FUNC TEST_CONSTEXPR_CXX20 bool test()
{
  cuda::std::ranges::drop_view dropView1(MoveOnlyView(), 4);
  assert(dropView1.size() == 4);
  assert(dropView1.begin() == globalBuff + 4);

  cuda::std::ranges::drop_view dropView2(ForwardView(), 4);
  assert(base(dropView2.begin()) == globalBuff + 4);

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test());
#endif // TEST_STD_VER >= 2020 && defined(_CCCL_BUILTIN_ADDRESSOF)

  return 0;
}
