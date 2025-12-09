//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/devices>
#include <cuda/memory_resource>
#include <cuda/std/mdspan>
#include <cuda/std/utility>
#include <memory>

#include <cuda/experimental/__container/mdarray_device.cuh>

#include "testing.cuh"

C2H_CCCLRT_TEST("cudax::mdarray", "[container][mdarray]")
{
  cuda::device_memory_pool pool{cuda::device_ref{0}};
  using extents_type = cuda::std::dims<2>;
  using mdarray_type = cudax::device_mdarray<int, extents_type, cuda::std::layout_right>;
  mdarray_type mdarray{extents_type{2, 3}};

  CUDAX_REQUIRE(mdarray.size() == 6);
  CUDAX_REQUIRE(mdarray.extent(0) == 2);
  CUDAX_REQUIRE(mdarray.extent(1) == 3);
}

struct movable_allocator
{
  std::shared_ptr<cuda::device_memory_pool> pool;

  explicit movable_allocator()
      : pool(std::make_shared<cuda::device_memory_pool>(0))
  {}

  void* allocate_sync(std::size_t size) {
      return pool->allocate_sync(size);
  }

  void deallocate_sync(void* ptr, std::size_t size) {
      pool->deallocate_sync(ptr, size);
  }

  friend void get_property(const movable_allocator&, cuda::mr::device_accessible) {}
};


C2H_CCCLRT_TEST("cudax::mdarray move", "[container][mdarray][move]")
{
  using extents_type = cuda::std::dims<2>;
  using mdarray_type = cudax::device_mdarray<int, extents_type, cuda::std::layout_right, movable_allocator>;

  // Move Constructor
  {
    mdarray_type mdarray1{extents_type{2, 3}};
    int* ptr1 = mdarray1.data_handle();
    CUDAX_REQUIRE(ptr1 != nullptr);

    mdarray_type mdarray2{cuda::std::move(mdarray1)};
    int* ptr2 = mdarray2.data_handle();

    CUDAX_REQUIRE(ptr1 == ptr2);
    CUDAX_REQUIRE(mdarray2.extent(0) == 2);
    CUDAX_REQUIRE(mdarray2.extent(1) == 3);
    // Verify mdarray1 is empty/null (ptr is nulled out)
    CUDAX_REQUIRE(mdarray1.data_handle() == nullptr);
  }

  // Move Assignment (same extents required)
  {
    mdarray_type mdarray1{extents_type{2, 3}};
    mdarray_type mdarray2{extents_type{2, 3}};  // Same extents as mdarray1

    int* ptr1 = mdarray1.data_handle();
    int* ptr2_old = mdarray2.data_handle();

    mdarray2 = cuda::std::move(mdarray1);

    CUDAX_REQUIRE(mdarray2.data_handle() == ptr1);
    CUDAX_REQUIRE(mdarray2.data_handle() != ptr2_old);  // Different pointer (old storage released)
    CUDAX_REQUIRE(mdarray2.extent(0) == 2);
    CUDAX_REQUIRE(mdarray2.extent(1) == 3);
    CUDAX_REQUIRE(mdarray1.data_handle() == nullptr);
  }
}

C2H_CCCLRT_TEST("cudax::mdarray copy", "[container][mdarray][copy]")
{
  using extents_type = cuda::std::dims<2>;
  using mdarray_type = cudax::device_mdarray<int, extents_type, cuda::std::layout_right>;

  // Copy Assignment (same extents required)
  {
    mdarray_type mdarray1{extents_type{2, 3}};
    mdarray_type mdarray2{extents_type{2, 3}};  // Same extents as mdarray1

    mdarray2 = mdarray1;

    CUDAX_REQUIRE(mdarray2.extent(0) == 2);
    CUDAX_REQUIRE(mdarray2.extent(1) == 3);
    CUDAX_REQUIRE(mdarray2.size() == 6);
    // Ensure deep copy (ptrs different)
    CUDAX_REQUIRE(mdarray2.data_handle() != mdarray1.data_handle());
  }
}
