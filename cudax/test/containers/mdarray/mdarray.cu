//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/device_vector.h>

#include <cuda/algorithm>
#include <cuda/devices>
#include <cuda/memory_resource>
#include <cuda/std/algorithm>
#include <cuda/std/mdspan>
#include <cuda/std/utility>

#include <cuda/experimental/__container/mdarray_device.cuh>
#include <cuda/experimental/__container/mdarray_host.cuh>

#include "testing.cuh"

template <typename View, typename ElementType>
struct CheckInitOp
{
  View view;
  bool* d_result;

  template <typename IndexType, typename... Indices>
  _CCCL_DEVICE void operator()(IndexType, Indices... indices)
  {
    if (view(indices...) != ElementType{})
    {
      *d_result = false;
    }
  }
};

template <typename ElementType, typename ExtentsType, typename LayoutPolicy>
bool check_init(const cuda::device_mdspan<ElementType, ExtentsType, LayoutPolicy>& mdspan)
{
  using view_type = cuda::device_mdspan<ElementType, ExtentsType, LayoutPolicy>;
  thrust::device_vector<bool> d_result(1, true);
  CheckInitOp<view_type, ElementType> op{mdspan, thrust::raw_pointer_cast(d_result.data())};
  CUDAX_REQUIRE(cub::DeviceFor::ForEachInLayout(mdspan.mapping(), op) == cudaSuccess);
  return d_result[0];
}

C2H_TEST("cudax::mdarray", "[mdarray][constructor][default]")
{
  using extents_type = cuda::std::dims<2>;
  using d_mdarray_t  = cudax::device_mdarray<int, extents_type, cuda::std::layout_right>;
  using view_type    = typename d_mdarray_t::view_type;
  d_mdarray_t d_mdarray{};
  CUDAX_REQUIRE(d_mdarray.size() == 0);
  CUDAX_REQUIRE(d_mdarray.extent(0) == 0);
  CUDAX_REQUIRE(d_mdarray.extent(1) == 0);
}

C2H_TEST("cudax::mdarray", "[mdarray]")
{
  using extents_type = cuda::std::dims<2>;
  using d_mdarray_t  = cudax::device_mdarray<int, extents_type, cuda::std::layout_right>;
  using view_type    = typename d_mdarray_t::view_type;
  d_mdarray_t d_mdarray{extents_type{2, 3}};
  CUDAX_REQUIRE(d_mdarray.size() == 6);
  CUDAX_REQUIRE(d_mdarray.extent(0) == 2);
  CUDAX_REQUIRE(d_mdarray.extent(1) == 3);

  CUDAX_REQUIRE(check_init(d_mdarray.view()));
}

#if 0
struct movable_allocator
{
  std::shared_ptr<cuda::device_memory_pool> pool;

  explicit movable_allocator()
      : pool(std::make_shared<cuda::device_memory_pool>(cuda::device_ref{0}))
  {}

  void* allocate_sync(std::size_t size)
  {
    return pool->allocate_sync(size);
  }

  void deallocate_sync(void* ptr, std::size_t size)
  {
    pool->deallocate_sync(ptr, size);
  }

  bool operator==(const movable_allocator& other) const
  {
    return pool == other.pool;
  }

  bool operator!=(const movable_allocator& other) const
  {
    return !(*this == other);
  }

  friend void get_property(const movable_allocator&, cuda::mr::device_accessible) {}
};

C2H_TEST("cudax::mdarray move", "[container][mdarray][move]")
{
  using extents_type = cuda::std::dims<2>;
  using d_mdarray_t  = cudax::device_mdarray<int, extents_type, cuda::std::layout_right, movable_allocator>;

  // Move Constructor
  {
    d_mdarray_t mdarray1{extents_type{2, 3}};
    int* ptr1 = mdarray1.data_handle();
    CUDAX_REQUIRE(ptr1 != nullptr);

    d_mdarray_t mdarray2{cuda::std::move(mdarray1)};
    int* ptr2 = mdarray2.data_handle();

    CUDAX_REQUIRE(ptr1 == ptr2);
    CUDAX_REQUIRE(mdarray2.extent(0) == 2);
    CUDAX_REQUIRE(mdarray2.extent(1) == 3);
    // Verify mdarray1 is empty/null (ptr is nulled out)
    CUDAX_REQUIRE(mdarray1.data_handle() == nullptr);
  }

  // Move Assignment (same extents required)
  {
    d_mdarray_t mdarray1{extents_type{2, 3}};
    d_mdarray_t mdarray2{extents_type{2, 3}}; // Same extents as mdarray1

    int* ptr1     = mdarray1.data_handle();
    int* ptr2_old = mdarray2.data_handle();

    mdarray2 = cuda::std::move(mdarray1);

    CUDAX_REQUIRE(mdarray2.data_handle() == ptr1);
    CUDAX_REQUIRE(mdarray2.data_handle() != ptr2_old); // Different pointer (old storage released)
    CUDAX_REQUIRE(mdarray2.extent(0) == 2);
    CUDAX_REQUIRE(mdarray2.extent(1) == 3);
    CUDAX_REQUIRE(mdarray1.data_handle() == nullptr);
  }
}
C2H_TEST("cudax::mdarray copy", "[container][mdarray][copy]")
{
  using extents_type = cuda::std::dims<2>;
  using d_mdarray_t  = cudax::device_mdarray<int, extents_type, cuda::std::layout_right>;

  // Copy Assignment (same extents required)
  {
    d_mdarray_t mdarray1{extents_type{2, 3}};
    d_mdarray_t mdarray2{extents_type{2, 3}}; // Same extents as mdarray1

    mdarray2 = mdarray1;

    CUDAX_REQUIRE(mdarray2.extent(0) == 2);
    CUDAX_REQUIRE(mdarray2.extent(1) == 3);
    CUDAX_REQUIRE(mdarray2.size() == 6);
    // Ensure deep copy (ptrs different)
    CUDAX_REQUIRE(mdarray2.data_handle() != mdarray1.data_handle());
  }
}

C2H_TEST("cudax::mdarray properties", "[container][mdarray][properties]")
{
  using extents_type = cuda::std::dims<2>;
  using d_mdarray_t  = cudax::device_mdarray<int, extents_type, cuda::std::layout_right>;

  d_mdarray_t mdarray{extents_type{3, 4}};

  CUDAX_REQUIRE(mdarray.rank() == 2);
  CUDAX_REQUIRE(mdarray.rank_dynamic() == 2);
  CUDAX_REQUIRE(mdarray.static_extent(0) == cuda::std::dynamic_extent);
  CUDAX_REQUIRE(mdarray.static_extent(1) == cuda::std::dynamic_extent);

  CUDAX_REQUIRE(mdarray.extent(0) == 3);
  CUDAX_REQUIRE(mdarray.extent(1) == 4);

  CUDAX_REQUIRE(mdarray.stride(0) == 4);
  CUDAX_REQUIRE(mdarray.stride(1) == 1);

  CUDAX_REQUIRE(mdarray.size() == 12);
  CUDAX_REQUIRE(!mdarray.empty());

  CUDAX_REQUIRE(mdarray.is_unique());
  CUDAX_REQUIRE(mdarray.is_exhaustive());
  CUDAX_REQUIRE(!mdarray.is_strided());

  CUDAX_REQUIRE(mdarray.data_handle() != nullptr);

  // Const view properties
  const auto& const_mdarray = mdarray;
  CUDAX_REQUIRE(const_mdarray.size() == 12);
  CUDAX_REQUIRE(const_mdarray.extent(0) == 3);
}

C2H_TEST("cudax::mdarray host verification", "[container][mdarray][verification]")
{
  using extents_type        = cuda::std::dims<2>;
  using layout_type         = cuda::std::layout_right;
  using device_mdarray_type = cudax::device_mdarray<int, extents_type, layout_type>;
  using host_mdarray_type   = cudax::host_mdarray<int, extents_type, layout_type>;

  extents_type extents{4, 5};
  host_mdarray_type host_src{extents};

  // Fill host array
  int val = 0;
  for (int i = 0; i < host_src.extent(0); ++i)
  {
    for (int j = 0; j < host_src.extent(1); ++j)
    {
      host_src(i, j) = val++;
    }
  }

  device_mdarray_type device_dst{extents};

  cuda::stream stream{cuda::device_ref{0}};

  // Copy host -> device
  cuda::copy_bytes(stream, host_src, device_dst);
  stream.sync();

  // Verify on device (by copying back to host)
  host_mdarray_type host_dst{extents};
  cuda::copy_bytes(stream, device_dst, host_dst);
  stream.sync();

  for (int i = 0; i < host_dst.extent(0); ++i)
  {
    for (int j = 0; j < host_dst.extent(1); ++j)
    {
      CUDAX_REQUIRE(host_dst(i, j) == host_src(i, j));
    }
  }
}
#endif // 0
