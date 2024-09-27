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

#include "test_resource.h"
#include <catch2/catch.hpp>
#include <testing.cuh>

TEMPLATE_TEST_CASE_METHOD(test_fixture, "any_async_resource", "[container][resource]", big_resource, small_resource)
{
  using TestResource    = TestType;
  constexpr bool is_big = sizeof(TestResource) > sizeof(cuda::mr::_AnyResourceStorage);

  SECTION("construct and destruct")
  {
    Counts expected{};
    CHECK(this->counts == expected);
    {
      cudax::mr::any_async_resource<> mr{TestResource{42, this}};
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
      cudax::mr::any_async_resource<> mr{TestResource{42, this}};
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

  SECTION("allocate and deallocate")
  {
    Counts expected{};
    CHECK(this->counts == expected);
    {
      cudax::mr::any_async_resource<> mr{TestResource{42, this}};
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.move_count;
      CHECK(this->counts == expected);

      void* ptr = mr.allocate(bytes(50), align(8));
      CHECK(ptr == this);
      ++expected.allocate_count;
      CHECK(this->counts == expected);

      mr.deallocate(ptr, bytes(50), align(8));
      ++expected.deallocate_count;
      CHECK(this->counts == expected);
    }
    expected.delete_count += is_big;
    --expected.object_count;
    CHECK(this->counts == expected);
  }

  // Reset the counters:
  this->counts = Counts();

  SECTION("allocate_async and deallocate_async")
  {
    Counts expected{};
    CHECK(this->counts == expected);
    {
      cudax::stream stream{};
      cudax::mr::any_async_resource<> mr{TestResource{42, this}};
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.move_count;
      CHECK(this->counts == expected);

      void* ptr = mr.allocate_async(bytes(50), align(8), ::cuda::stream_ref{stream});
      CHECK(ptr == this);
      ++expected.allocate_async_count;
      CHECK(this->counts == expected);

      mr.deallocate_async(ptr, bytes(50), align(8), ::cuda::stream_ref{stream});
      ++expected.deallocate_async_count;
      CHECK(this->counts == expected);
    }
    expected.delete_count += is_big;
    --expected.object_count;
    CHECK(this->counts == expected);
  }

  // Reset the counters:
  this->counts = Counts();

  SECTION("conversion to resource_ref")
  {
    Counts expected{};
    {
      cudax::mr::any_async_resource<> mr{TestResource{42, this}};
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.move_count;
      CHECK(this->counts == expected);

      cuda::mr::resource_ref<> ref = mr;

      CHECK(this->counts == expected);
      auto* ptr = ref.allocate(bytes(100), align(8));
      CHECK(ptr == this);
      ++expected.allocate_count;
      CHECK(this->counts == expected);
      ref.deallocate(ptr, bytes(0), align(0));
      ++expected.deallocate_count;
      CHECK(this->counts == expected);
    }
    expected.delete_count += is_big;
    --expected.object_count;
    CHECK(this->counts == expected);
  }

  // Reset the counters:
  this->counts = Counts();

  SECTION("make_any_async_resource")
  {
    Counts expected{};
    CHECK(this->counts == expected);
    {
      cudax::mr::any_async_resource<> mr = cudax::mr::make_any_async_resource<TestResource>(42, this);
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
