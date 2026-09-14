//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class Ptr> constexpr auto to_address(const Ptr& p) noexcept;
//     Should not require a specialization of pointer_traits for Ptr.

#include <cuda/std/cassert>
#include <cuda/std/memory>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

struct IntPtr
{
  TEST_FUNC constexpr int* operator->() const
  {
    return ptr;
  }

  int* ptr;
};

template <class T, bool>
struct TemplatedPtr
{
  TEST_FUNC constexpr T* operator->() const
  {
    return ptr;
  }

  T* ptr;
};

TEST_FUNC constexpr bool test()
{
  int i = 0;
  assert(cuda::std::to_address(IntPtr{nullptr}) == nullptr);
  assert(cuda::std::to_address(IntPtr{&i}) == &i);

  bool b = false;
  assert(cuda::std::to_address(TemplatedPtr<bool, true>{nullptr}) == nullptr);
  assert(cuda::std::to_address(TemplatedPtr<bool, true>{&b}) == &b);

  static_assert(!cuda::std::__can_to_address<int>);
  static_assert(cuda::std::__can_to_address<IntPtr>);
  static_assert(cuda::std::__can_to_address<TemplatedPtr<bool, true>>);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
