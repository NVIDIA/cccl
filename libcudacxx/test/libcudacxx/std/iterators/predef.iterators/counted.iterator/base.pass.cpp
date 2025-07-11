//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr const I& base() const &;
// constexpr I base() &&;

#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

struct InputOrOutputArchetype
{
  using difference_type = int;

  int* ptr;

  __host__ __device__ int operator*()
  {
    return *ptr;
  }
  __host__ __device__ void operator++(int)
  {
    ++ptr;
  }
  __host__ __device__ InputOrOutputArchetype& operator++()
  {
    ++ptr;
    return *this;
  }
};

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    cuda::std::counted_iterator iter(cpp20_input_iterator<int*>{buffer}, 8);
    assert(base(iter.base()) == buffer);
    assert(base(cuda::std::move(iter).base()) == buffer);

    static_assert(noexcept(iter.base()));
    static_assert(cuda::std::is_same_v<decltype(iter.base()), const cpp20_input_iterator<int*>&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::move(iter).base()), cpp20_input_iterator<int*>>);
  }

  {
    cuda::std::counted_iterator iter(forward_iterator<int*>{buffer}, 8);
    assert(base(iter.base()) == buffer);
    assert(base(cuda::std::move(iter).base()) == buffer);

    static_assert(cuda::std::is_same_v<decltype(iter.base()), const forward_iterator<int*>&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::move(iter).base()), forward_iterator<int*>>);
  }

  {
    cuda::std::counted_iterator iter(contiguous_iterator<int*>{buffer}, 8);
    assert(base(iter.base()) == buffer);
    assert(base(cuda::std::move(iter).base()) == buffer);

    static_assert(cuda::std::is_same_v<decltype(iter.base()), const contiguous_iterator<int*>&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::move(iter).base()), contiguous_iterator<int*>>);
  }

  {
    cuda::std::counted_iterator iter(InputOrOutputArchetype{buffer}, 6);
    assert(iter.base().ptr == buffer);
    assert(cuda::std::move(iter).base().ptr == buffer);

    static_assert(cuda::std::is_same_v<decltype(iter.base()), const InputOrOutputArchetype&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::move(iter).base()), InputOrOutputArchetype>);
  }

  {
    const cuda::std::counted_iterator iter(cpp20_input_iterator<int*>{buffer}, 8);
    assert(base(iter.base()) == buffer);
    assert(base(cuda::std::move(iter).base()) == buffer);

    static_assert(cuda::std::is_same_v<decltype(iter.base()), const cpp20_input_iterator<int*>&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::move(iter).base()), const cpp20_input_iterator<int*>&>);
  }

  {
    const cuda::std::counted_iterator iter(forward_iterator<int*>{buffer}, 7);
    assert(base(iter.base()) == buffer);
    assert(base(cuda::std::move(iter).base()) == buffer);

    static_assert(cuda::std::is_same_v<decltype(iter.base()), const forward_iterator<int*>&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::move(iter).base()), const forward_iterator<int*>&>);
  }

  {
    const cuda::std::counted_iterator iter(contiguous_iterator<int*>{buffer}, 6);
    assert(base(iter.base()) == buffer);
    assert(base(cuda::std::move(iter).base()) == buffer);

    static_assert(cuda::std::is_same_v<decltype(iter.base()), const contiguous_iterator<int*>&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::move(iter).base()), const contiguous_iterator<int*>&>);
  }

  {
    const cuda::std::counted_iterator iter(InputOrOutputArchetype{buffer}, 6);
    assert(iter.base().ptr == buffer);
    assert(cuda::std::move(iter).base().ptr == buffer);

    static_assert(cuda::std::is_same_v<decltype(iter.base()), const InputOrOutputArchetype&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::move(iter).base()), const InputOrOutputArchetype&>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
