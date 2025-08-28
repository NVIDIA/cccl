//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/memory_resource>

#include <cuda/experimental/memory_resource.cuh>

#include <testing.cuh>

#include "test_resource.cuh"

static_assert(
  cuda::has_property<cudax::any_synchronous_resource<cudax::host_accessible, get_data>, cudax::host_accessible>);
static_assert(cuda::has_property<cudax::any_synchronous_resource<cudax::host_accessible, get_data>, get_data>);
static_assert(
  !cuda::has_property<cudax::any_synchronous_resource<cudax::host_accessible, get_data>, cudax::device_accessible>);

struct unused_property
{};

TEMPLATE_TEST_CASE_METHOD(
  test_fixture, "any_synchronous_resource", "[container][resource]", big_resource, small_resource)
{
  using TestResource    = TestType;
  constexpr bool is_big = sizeof(TestResource) > cuda::__default_small_object_size;

  SECTION("construct and destruct")
  {
    Counts expected{};
    CHECK(this->counts == expected);
    {
      cudax::any_synchronous_resource<cudax::host_accessible, get_data> mr{TestResource{42, this}};
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.move_count;
      CHECK(this->counts == expected);
      CHECK(get_property(mr, get_data{}) == 42);
      get_property(mr, cudax::host_accessible{});
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
      cudax::any_synchronous_resource<cudax::host_accessible, get_data> mr{TestResource{42, this}};
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.move_count;
      CHECK(this->counts == expected);

      auto mr2 = mr;
      expected.new_count += is_big;
      ++expected.copy_count;
      ++expected.object_count;
      CHECK(this->counts == expected);
      CHECK((mr == mr2));
      ++expected.equal_to_count;
      CHECK(this->counts == expected);

      auto mr3 = std::move(mr);
      expected.move_count += !is_big; // for big resources, move is a pointer swap
      CHECK(this->counts == expected);
      CHECK((mr2 == mr3));
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
      cudax::any_synchronous_resource<cudax::host_accessible, get_data> mr{TestResource{42, this}};
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

  SECTION("equality comparable")
  {
    Counts expected{};
    CHECK(this->counts == expected);
    {
      cudax::legacy_managed_memory_resource managed1{}, managed2{};
      CHECK(managed1 == managed2);
      cudax::any_synchronous_resource<cudax::device_accessible> mr{managed1};
      CHECK(mr == managed1);
    }
    CHECK(this->counts == expected);
  }

  // Reset the counters:
  this->counts = Counts();

  SECTION("conversion from any_synchronous_resource to cudax::synchronous_resource_ref")
  {
    Counts expected{};
    {
      cudax::any_synchronous_resource<cudax::host_accessible, cudax::device_accessible, get_data> mr{
        TestResource{42, this}};
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.move_count;
      CHECK(this->counts == expected);

      // conversion from any_synchronous_resource to cuda::mr::synchronous_synchronous_resource_ref:
      cudax::synchronous_resource_ref<cudax::host_accessible, cudax::device_accessible, get_data> ref = mr;

      // conversion from any_synchronous_resource to cuda::mr::synchronous_synchronous_resource_ref with narrowing:
      cudax::synchronous_resource_ref<cudax::host_accessible, get_data> ref2 = mr;
      CHECK(get_property(ref2, get_data{}) == 42);

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

  SECTION("conversion from any_synchronous_resource to cuda::mr::synchronous_resource_ref")
  {
    Counts expected{};
    {
      cudax::any_synchronous_resource<cudax::host_accessible, cudax::device_accessible, get_data> mr{
        TestResource{42, this}};
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.move_count;
      CHECK(this->counts == expected);

      // conversion from any_synchronous_resource to cuda::mr::synchronous_resource_ref:
      cuda::mr::synchronous_resource_ref<cudax::host_accessible, cudax::device_accessible, get_data> ref = mr;

      // conversion from any_synchronous_resource to cuda::mr::synchronous_resource_ref with narrowing:
      cuda::mr::synchronous_resource_ref<cudax::host_accessible, get_data> ref2 = mr;
      CHECK(get_property(ref2, get_data{}) == 42);

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

  SECTION("conversion from synchronous_resource_ref to any_synchronous_resource")
  {
    Counts expected{};
    {
      TestResource test{42, this};
      ++expected.object_count;
      cudax::synchronous_resource_ref<cudax::host_accessible, get_data> ref{test};
      CHECK(this->counts == expected);

      cudax::any_synchronous_resource<cudax::host_accessible, get_data> mr = ref;
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.copy_count;
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
    expected.object_count -= 2;
    CHECK(this->counts == expected);
  }

  // Reset the counters:
  this->counts = Counts();

  SECTION("test slicing off of properties")
  {
    Counts expected{};
    CHECK(this->counts == expected);
    {
      cudax::any_synchronous_resource<cudax::host_accessible, cudax::device_accessible, get_data> mr{
        TestResource{42, this}};
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.move_count;
      CHECK(this->counts == expected);

      cudax::any_synchronous_resource<cudax::device_accessible, get_data> mr2 = mr;
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.copy_count;
      CHECK(this->counts == expected);

      CHECK(get_property(mr2, get_data{}) == 42);
      auto data = try_get_property(mr2, get_data{});
      static_assert(cuda::std::is_same_v<decltype(data), cuda::std::optional<int>>);
      CHECK(data.has_value());
      CHECK(data.value() == 42);

      auto host = try_get_property(mr2, cudax::host_accessible{});
      static_assert(cuda::std::is_same_v<decltype(host), bool>);
      CHECK(host);

      auto unused = try_get_property(mr2, unused_property{});
      static_assert(cuda::std::is_same_v<decltype(unused), bool>);
      CHECK(!unused);
    }
    expected.delete_count += 2 * is_big;
    expected.object_count -= 2;
    CHECK(this->counts == expected);
  }

  // Reset the counters:
  this->counts = Counts();

  SECTION("make_any_synchronous_resource")
  {
    Counts expected{};
    CHECK(this->counts == expected);
    {
      cudax::any_synchronous_resource<cudax::host_accessible, get_data> mr =
        cudax::make_any_synchronous_resource<TestResource, cudax::host_accessible, get_data>(42, this);
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
