//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: enable-tile

// UNSUPPORTED: nvrtc

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"

struct host_bijection
{
  using index_type = uint32_t;

  host_bijection() {}
  host_bijection(const host_bijection&) {}
  host_bijection(host_bijection&&) {}
  host_bijection& operator=(const host_bijection&)
  {
    return *this;
  }
  host_bijection& operator=(host_bijection&&)
  {
    return *this;
  }
  ~host_bijection() {}

  template <class RNG>
  constexpr host_bijection(index_type, RNG&&) noexcept
  {}

  [[nodiscard]] constexpr index_type size() const noexcept
  {
    return 5;
  }

  [[nodiscard]] constexpr index_type operator()(index_type n) const noexcept
  {
    return __random_indices[n];
  }

  uint32_t __random_indices[5] = {4, 1, 2, 0, 3};
};

void test()
{
  // taken from host_bijection
  constexpr uint32_t random_indices[] = {4, 1, 2, 0, 3};

  { // constructors
    using shuffle_iterator = cuda::shuffle_iterator<int, host_bijection>;

    const shuffle_iterator default_constructed{};
    shuffle_iterator value_constructed{host_bijection{}};

    shuffle_iterator copy_constructed{default_constructed};
    shuffle_iterator move_constructed{::cuda::std::move(value_constructed)};

    [[maybe_unused]] shuffle_iterator copy_assigned{};
    copy_assigned = copy_constructed;

    [[maybe_unused]] shuffle_iterator move_assigned{};
    move_assigned = ::cuda::std::move(move_constructed);

    [[maybe_unused]] shuffle_iterator bijection_value_constructed{host_bijection{}, 0};
    [[maybe_unused]] shuffle_iterator size_bijection_value_constructed{4, host_bijection{}, 0};
  }

  cuda::shuffle_iterator iter1{host_bijection{}, 0};
  const cuda::shuffle_iterator iter2{host_bijection{}, 1};
  assert(iter1 != iter2);

  {
    assert(++iter1 == iter2);
    assert(--iter1 != iter2);
  }

  {
    assert(iter1++ != iter2);
    assert(iter1-- == iter2);
  }

  {
    assert(iter1 + 1 == iter2);
    assert(1 + iter1 == iter2);
    assert(iter1 - 1 != iter2);
    assert(iter2 - iter1 == 1);
  }

  {
    iter1 += 1;
    assert(iter1 == iter2);
    iter1 -= 1;
    assert(iter1 != iter2);
  }

  {
    assert(iter1[1] == random_indices[1]);
    assert(*iter1 == random_indices[0]);

    assert(iter2[1] == random_indices[2]);
    assert(*iter2 == random_indices[1]);
  }
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
