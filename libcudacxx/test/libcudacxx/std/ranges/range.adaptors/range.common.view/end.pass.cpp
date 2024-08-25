//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// constexpr auto end();
// constexpr auto end() const requires range<const V>;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  int buf[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    SizedRandomAccessView view{buf, buf + 8};
    cuda::std::ranges::common_view<SizedRandomAccessView> common(view);
    decltype(auto) end = common.end(); // Note this should NOT be the sentinel type.
    static_assert(cuda::std::same_as<decltype(end), RandomAccessIter>);
    assert(base(end) == buf + 8);
  }

  // const version
  {
    SizedRandomAccessView view{buf, buf + 8};
    cuda::std::ranges::common_view<SizedRandomAccessView> const common(view);
    decltype(auto) end = common.end(); // Note this should NOT be the sentinel type.
    static_assert(cuda::std::same_as<decltype(end), RandomAccessIter>);
    assert(base(end) == buf + 8);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  {
    int buf[8] = {1, 2, 3, 4, 5, 6, 7, 8};

    using CommonForwardIter = cuda::std::common_iterator<ForwardIter, sized_sentinel<ForwardIter>>;
    using CommonIntIter     = cuda::std::common_iterator<int*, sentinel_wrapper<int*>>;

    {
      SizedForwardView view{buf, buf + 8};
      cuda::std::ranges::common_view<SizedForwardView> common(view);
      decltype(auto) end = common.end();
      static_assert(cuda::std::same_as<decltype(end), CommonForwardIter>);
      assert(end == CommonForwardIter(cuda::std::ranges::end(view)));
    }
    {
      CopyableView view{buf, buf + 8};
      cuda::std::ranges::common_view<CopyableView> common(view);
      decltype(auto) end = common.end();
      static_assert(cuda::std::same_as<decltype(end), CommonIntIter>);
      assert(end == CommonIntIter(cuda::std::ranges::end(view)));
    }

    // const versions
    {
      SizedForwardView view{buf, buf + 8};
      cuda::std::ranges::common_view<SizedForwardView> const common(view);
      decltype(auto) end = common.end();
      static_assert(cuda::std::same_as<decltype(end), CommonForwardIter>);
      assert(end == CommonForwardIter(cuda::std::ranges::end(view)));
    }
    {
      CopyableView view{buf, buf + 8};
      cuda::std::ranges::common_view<CopyableView> const common(view);
      decltype(auto) end = common.end();
      static_assert(cuda::std::same_as<decltype(end), CommonIntIter>);
      assert(end == CommonIntIter(cuda::std::ranges::end(view)));
    }
  }

  return 0;
}
