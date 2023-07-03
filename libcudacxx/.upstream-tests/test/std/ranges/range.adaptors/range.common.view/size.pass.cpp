//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// constexpr auto size() requires sized_range<V>
// constexpr auto size() const requires sized_range<const V>

#include <cuda/std/ranges>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "types.h"

#if TEST_STD_VER > 17
template<class View>
concept SizeEnabled = requires(View v) { v.size(); };
#else
template <class View, class = void>
inline constexpr bool SizeEnabled = false;

template <class View>
inline constexpr bool SizeEnabled<View, cuda::std::void_t<decltype(cuda::std::declval<View>().size())>> = true;
#endif

__host__ __device__ constexpr bool test() {
  int buf[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    static_assert( SizeEnabled<cuda::std::ranges::common_view<SizedForwardView> const&>);
    static_assert(!SizeEnabled<cuda::std::ranges::common_view<CopyableView> const&>);
  }

  {
    SizedForwardView view{buf, buf + 8};
    cuda::std::ranges::common_view<SizedForwardView> common(view);
    assert(common.size() == 8);
  }

  {
    SizedForwardView view{buf, buf + 8};
    cuda::std::ranges::common_view<SizedForwardView> const common(view);
    assert(common.size() == 8);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");

  return 0;
}
