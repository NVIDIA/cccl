//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: nvrtc

// cuda::mr::resource_ref properties

#include <cuda/memory_resource>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "types.h"

template <class PropA, class PropB>
void test_conversion_from_async_resource_ref()
{
  test_resource<cuda::mr::host_accessible, PropA, PropB> input{42};
  cuda::mr::resource_ref<cuda::mr::host_accessible, PropA, PropB> ref_input{input};

  { // lvalue
    cuda::mr::resource_ref<cuda::mr::host_accessible, PropB> ref{ref_input};

    // Ensure that we properly pass on the allocate function
    assert(input.allocate(::cudaStream_t{}, 0, 0) == ref.allocate(::cudaStream_t{}, 0, 0));

    // Ensure we are deallocating properly
    int expected_after_deallocate = 1337;
    ref.deallocate(::cudaStream_t{}, static_cast<void*>(&expected_after_deallocate), 0, 0);
    assert(input._val == expected_after_deallocate);
  }

  { // prvalue
    cuda::mr::resource_ref<cuda::mr::host_accessible, PropB> ref{
      cuda::mr::resource_ref<cuda::mr::host_accessible, PropA, PropB>{input}};

    // Ensure that we properly pass on the allocate function
    assert(input.allocate(::cudaStream_t{}, 0, 0) == ref.allocate(::cudaStream_t{}, 0, 0));

    // Ensure we are deallocating properly
    int expected_after_deallocate = 1337;
    ref.deallocate(::cudaStream_t{}, static_cast<void*>(&expected_after_deallocate), 0, 0);
    assert(input._val == expected_after_deallocate);
  }
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST,
               (test_conversion_from_async_resource_ref<property_with_value<short>, property_with_value<int>>();
                test_conversion_from_async_resource_ref<property_with_value<short>, property_without_value<int>>();
                test_conversion_from_async_resource_ref<property_without_value<short>, property_without_value<int>>();))

  return 0;
}
