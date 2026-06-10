//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// reverse_view() requires default_initializable<V> = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "types.h"

enum CtorKind
{
  DefaultCtor,
  PtrCtor
};
template <CtorKind CK>
struct BidirRangeWith : cuda::std::ranges::view_base
{
  int* ptr_ = nullptr;

  template <CtorKind CK2 = CK, cuda::std::enable_if_t<CK2 == DefaultCtor, int> = 0>
  TEST_FUNC constexpr BidirRangeWith() noexcept {};
  TEST_FUNC constexpr BidirRangeWith(int* ptr);

  TEST_FUNC constexpr bidirectional_iterator<int*> begin()
  {
    return bidirectional_iterator<int*>{ptr_};
  }
  TEST_FUNC constexpr bidirectional_iterator<const int*> begin() const
  {
    return bidirectional_iterator<const int*>{ptr_};
  }
  TEST_FUNC constexpr bidirectional_iterator<int*> end()
  {
    return bidirectional_iterator<int*>{ptr_ + 8};
  }
  TEST_FUNC constexpr bidirectional_iterator<const int*> end() const
  {
    return bidirectional_iterator<const int*>{ptr_ + 8};
  }
};

TEST_FUNC constexpr bool test()
{
  {
    static_assert(cuda::std::default_initializable<cuda::std::ranges::reverse_view<BidirRangeWith<DefaultCtor>>>);
    static_assert(!cuda::std::default_initializable<cuda::std::ranges::reverse_view<BidirRangeWith<PtrCtor>>>);
  }

  {
    cuda::std::ranges::reverse_view<BidirRangeWith<DefaultCtor>> rev;
    assert(rev.base().ptr_ == nullptr);
  }
  {
    const cuda::std::ranges::reverse_view<BidirRangeWith<DefaultCtor>> rev;
    assert(rev.base().ptr_ == nullptr);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
