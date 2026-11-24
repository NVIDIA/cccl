//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr explicit common_view(V r);

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "types.h"

TEST_FUNC constexpr bool test()
{
  int buf[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    MoveOnlyView view{buf, buf + 8};
    cuda::std::ranges::common_view<MoveOnlyView> common(cuda::std::move(view));
    assert(cuda::std::move(common).base().begin_ == buf);
  }

  {
    CopyableView const view{buf, buf + 8};
    cuda::std::ranges::common_view<CopyableView> const common(view);
    assert(common.base().begin_ == buf);
  }

  {
    MoveOnlyView view{buf, buf + 8};
    cuda::std::ranges::common_view<MoveOnlyView> const common(cuda::std::move(view));
    assert(common.begin() == buf);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
