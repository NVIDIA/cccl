//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr R& base() & noexcept { return r_; }
// constexpr const R& base() const& noexcept { return r_; }
// constexpr R&& base() && noexcept { return cuda::std::move(r_); }
// constexpr const R&& base() const&& noexcept { return cuda::std::move(r_); }

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>

#include "test_macros.h"

struct Base
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

__host__ __device__ constexpr bool test()
{
  using OwningView = cuda::std::ranges::owning_view<Base>;
  OwningView ov;
  decltype(auto) b1 = static_cast<OwningView&>(ov).base();
  decltype(auto) b2 = static_cast<OwningView&&>(ov).base();
  decltype(auto) b3 = static_cast<const OwningView&>(ov).base();
  decltype(auto) b4 = static_cast<const OwningView&&>(ov).base();

  static_assert(cuda::std::is_same_v<decltype(b1), Base&>);
  static_assert(cuda::std::is_same_v<decltype(b2), Base&&>);
  static_assert(cuda::std::is_same_v<decltype(b3), const Base&>);
  static_assert(cuda::std::is_same_v<decltype(b4), const Base&&>);

  assert(&b1 == &b2);
  assert(&b1 == &b3);
  assert(&b1 == &b4);

  static_assert(noexcept(static_cast<OwningView&>(ov).base()));
  static_assert(noexcept(static_cast<OwningView&&>(ov).base()));
  static_assert(noexcept(static_cast<const OwningView&>(ov).base()));
  static_assert(noexcept(static_cast<const OwningView&&>(ov).base()));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
