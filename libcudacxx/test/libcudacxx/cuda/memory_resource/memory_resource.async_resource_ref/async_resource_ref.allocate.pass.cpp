//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: nvrtc

// cuda::mr::async_resource_ref properties

#include <cuda/memory_resource>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

struct async_resource
{
  void* allocate(std::size_t, std::size_t)
  {
    return &_val;
  }

  void deallocate(void* ptr, std::size_t, std::size_t) noexcept
  {
    // ensure that we did get the right inputs forwarded
    _val = *static_cast<int*>(ptr);
  }

  void* allocate_async(std::size_t, std::size_t, cuda::stream_ref)
  {
    return &_val;
  }

  void deallocate_async(void* ptr, std::size_t, std::size_t, cuda::stream_ref)
  {
    // ensure that we did get the right inputs forwarded
    _val = *static_cast<int*>(ptr);
  }

  bool operator==(const async_resource& other) const
  {
    return _val == other._val;
  }
  bool operator!=(const async_resource& other) const
  {
    return _val != other._val;
  }

  int _val = 0;
};

void test_allocate()
{
  { // allocate(size)
    async_resource input{42};
    cuda::mr::async_resource_ref<> ref{input};

    // Ensure that we properly pass on the allocate function
    assert(input.allocate(0, 0) == ref.allocate(0));

    int expected_after_deallocate = 1337;
    ref.deallocate(static_cast<void*>(&expected_after_deallocate), 0);
    assert(input._val == expected_after_deallocate);
  }

  { // allocate(size, alignment)
    async_resource input{42};
    cuda::mr::async_resource_ref<> ref{input};

    // Ensure that we properly pass on the allocate function
    assert(input.allocate(0, 0) == ref.allocate(0, 0));

    int expected_after_deallocate = 1337;
    ref.deallocate(static_cast<void*>(&expected_after_deallocate), 0, 0);
    assert(input._val == expected_after_deallocate);
  }
}

void test_allocate_async()
{
  { // allocate(size)
    async_resource input{42};
    cuda::mr::async_resource_ref<> ref{input};

    // Ensure that we properly pass on the allocate function
    assert(input.allocate_async(0, 0, {}) == ref.allocate_async(0, {}));

    int expected_after_deallocate = 1337;
    ref.deallocate_async(static_cast<void*>(&expected_after_deallocate), 0, {});
    assert(input._val == expected_after_deallocate);
  }

  { // allocate(size, alignment)
    async_resource input{42};
    cuda::mr::async_resource_ref<> ref{input};

    // Ensure that we properly pass on the allocate function
    assert(input.allocate_async(0, 0, {}) == ref.allocate_async(0, 0, {}));

    int expected_after_deallocate = 1337;
    ref.deallocate_async(static_cast<void*>(&expected_after_deallocate), 0, 0, {});
    assert(input._val == expected_after_deallocate);
  }
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test_allocate(); test_allocate_async();))

  return 0;
}
