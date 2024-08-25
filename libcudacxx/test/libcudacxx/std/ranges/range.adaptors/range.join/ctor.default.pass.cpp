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

// join_view() requires default_initializable<V> = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "types.h"

struct DefaultView : cuda::std::ranges::view_base
{
  int i; // deliberately uninitialised

  __host__ __device__ ChildView* begin() const;
  __host__ __device__ ChildView* end() const;
};

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::ranges::join_view<ParentView<ChildView>> jv;
    assert(cuda::std::move(jv).base().ptr_ == globalChildren);
  }

  // Default constructor should value initialise underlying view
  {
    cuda::std::ranges::join_view<DefaultView> jv;
    assert(jv.base().i == 0);
  }

  static_assert(cuda::std::default_initializable<cuda::std::ranges::join_view<ParentView<ChildView>>>);
  static_assert(!cuda::std::default_initializable<cuda::std::ranges::join_view<CopyableParent>>);

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)

  return 0;
}
