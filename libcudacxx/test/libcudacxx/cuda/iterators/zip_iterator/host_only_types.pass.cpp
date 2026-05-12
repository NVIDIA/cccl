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

#include "host_device_types.h"
#include "test_macros.h"

void test()
{
  host_only_container vec{};

  {
    using Iter         = typename host_only_container::iterator;
    using zip_iterator = cuda::zip_iterator<Iter, Iter>;

    const zip_iterator default_constructed{};
    zip_iterator value_constructed{vec.begin(), vec.begin()};

    zip_iterator copy_constructed{default_constructed};
    zip_iterator move_constructed{::cuda::std::move(value_constructed)};

    [[maybe_unused]] zip_iterator copy_assigned{};
    copy_assigned = copy_constructed;

    [[maybe_unused]] zip_iterator move_assigned{};
    move_assigned = ::cuda::std::move(move_constructed);
  }

  cuda::zip_iterator iter1{vec.begin(), vec.begin() + 1};
  const cuda::zip_iterator iter2{vec.begin() + 1, vec.begin() + 2};
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
    assert((iter1[1] == cuda::std::tuple{vec[1], vec[2]}));
    assert((iter2[1] == cuda::std::tuple{vec[2], vec[3]}));

    assert((*iter1 == cuda::std::tuple{vec[0], vec[1]}));
    assert((*iter2 == cuda::std::tuple{vec[1], vec[2]}));
  }
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
