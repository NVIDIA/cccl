//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/memory_resource.cuh>

#include <testing.cuh>

#include "test_resource.cuh"

#ifndef __CUDA_ARCH__

TEMPLATE_TEST_CASE_METHOD(test_fixture, "any_resource", "[container][resource]", big_resource, small_resource)
{
  using TestResource = TestType;
  static_assert(cuda::mr::synchronous_resource_with<TestResource, cudax::host_accessible>);
  constexpr bool is_big = sizeof(TestResource) > cuda::__default_small_object_size;

  SECTION("construct and destruct")
  {
    Counts expected{};
    CHECK(this->counts == expected);
    {
      cudax::any_resource<cudax::host_accessible> mr{TestResource{42, this}};
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.move_count;
      CHECK(this->counts == expected);
    }
    expected.delete_count += is_big;
    --expected.object_count;
    CHECK(this->counts == expected);
  }

  // Reset the counters:
  this->counts = Counts();

  SECTION("copy and move")
  {
    Counts expected{};
    CHECK(this->counts == expected);
    {
      cudax::any_resource<cudax::host_accessible> mr{TestResource{42, this}};
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.move_count;
      CHECK(this->counts == expected);

      auto mr2 = mr;
      expected.new_count += is_big;
      ++expected.copy_count;
      ++expected.object_count;
      CHECK(this->counts == expected);
      CHECK(mr == mr2);
      ++expected.equal_to_count;
      CHECK(this->counts == expected);

      auto mr3 = std::move(mr);
      expected.move_count += !is_big; // for big resources, move is a pointer swap
      CHECK(this->counts == expected);
      CHECK(mr2 == mr3);
      ++expected.equal_to_count;
      CHECK(this->counts == expected);
    }
    expected.delete_count += 2 * is_big;
    expected.object_count -= 2;
    CHECK(this->counts == expected);
  }

  // Reset the counters:
  this->counts = Counts();

  SECTION("allocate and deallocate_sync")
  {
    Counts expected{};
    CHECK(this->counts == expected);
    {
      cudax::any_resource<cudax::host_accessible> mr{TestResource{42, this}};
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.move_count;
      CHECK(this->counts == expected);

      void* ptr = mr.allocate_sync(bytes(50), align(8));
      CHECK(ptr == this);
      ++expected.allocate_count;
      CHECK(this->counts == expected);

      mr.deallocate_sync(ptr, bytes(50), align(8));
      ++expected.deallocate_count;
      CHECK(this->counts == expected);
    }
    expected.delete_count += is_big;
    --expected.object_count;
    CHECK(this->counts == expected);
  }

  // Reset the counters:
  this->counts = Counts();

  SECTION("allocate and deallocate")
  {
    Counts expected{};
    CHECK(this->counts == expected);
    {
      cudax::stream stream{cuda::device_ref{0}};
      cudax::any_resource<cudax::host_accessible> mr{TestResource{42, this}};
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.move_count;
      CHECK(this->counts == expected);

      void* ptr = mr.allocate(::cuda::stream_ref{stream}, bytes(50), align(8));
      CHECK(ptr == this);
      ++expected.allocate_async_count;
      CHECK(this->counts == expected);

      mr.deallocate(::cuda::stream_ref{stream}, ptr, bytes(50), align(8));
      ++expected.deallocate_async_count;
      CHECK(this->counts == expected);
    }
    expected.delete_count += is_big;
    --expected.object_count;
    CHECK(this->counts == expected);
  }

  // Reset the counters:
  this->counts = Counts();

  SECTION("conversion to synchronous_resource_ref")
  {
    Counts expected{};
    {
      cudax::any_resource<cudax::host_accessible> mr{TestResource{42, this}};
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.move_count;
      CHECK(this->counts == expected);

      cudax::synchronous_resource_ref<cudax::host_accessible> ref = mr;

      CHECK(this->counts == expected);
      auto* ptr = ref.allocate_sync(bytes(100), align(8));
      CHECK(ptr == this);
      ++expected.allocate_count;
      CHECK(this->counts == expected);
      ref.deallocate_sync(ptr, bytes(0), align(0));
      ++expected.deallocate_count;
      CHECK(this->counts == expected);
    }
    expected.delete_count += is_big;
    --expected.object_count;
    CHECK(this->counts == expected);
  }

  // Reset the counters:
  this->counts = Counts();

  SECTION("make_any_resource")
  {
    Counts expected{};
    CHECK(this->counts == expected);
    {
      cudax::any_resource<cudax::host_accessible> mr =
        cudax::make_any_resource<TestResource, cudax::host_accessible>(42, this);
      expected.new_count += is_big;
      ++expected.object_count;
      CHECK(this->counts == expected);
    }
    expected.delete_count += is_big;
    --expected.object_count;
    CHECK(this->counts == expected);
  }
  // Reset the counters:
  this->counts = Counts();
}

#endif // __CUDA_ARCH__
