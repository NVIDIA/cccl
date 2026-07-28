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
  {
    using constant_iterator = cuda::constant_iterator<host_only_type, int>;

    const constant_iterator default_constructed{};
    constant_iterator value_constructed{host_only_type{}};

    constant_iterator copy_constructed{default_constructed};
    constant_iterator move_constructed{::cuda::std::move(value_constructed)};

    [[maybe_unused]] constant_iterator copy_assigned{};
    copy_assigned = copy_constructed;

    [[maybe_unused]] constant_iterator move_assigned{};
    move_assigned = ::cuda::std::move(move_constructed);

    [[maybe_unused]] constant_iterator value_index_constructed{host_only_type{}, 42};
  }

  cuda::constant_iterator iter1{host_only_type{42}, 100};
  const cuda::constant_iterator iter2{host_only_type{1337}, 101};
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
    assert(iter1[1] == host_only_type{42});
    assert(*iter1 == host_only_type{42});

    assert(iter2[1] == host_only_type{1337});
    assert(*iter2 == host_only_type{1337});
  }
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
