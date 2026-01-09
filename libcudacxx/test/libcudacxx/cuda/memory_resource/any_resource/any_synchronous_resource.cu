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
#include <cuda/stream>

#include <testing.cuh>

#include "../test_resource.cuh"

static_assert(cuda::has_property<cuda::mr::any_synchronous_resource<cuda::mr::host_accessible, get_data>,
                                 ::cuda::mr::host_accessible>);
static_assert(cuda::has_property<cuda::mr::any_synchronous_resource<cuda::mr::host_accessible, get_data>, get_data>);
static_assert(!cuda::has_property<cuda::mr::any_synchronous_resource<cuda::mr::host_accessible, get_data>,
                                  ::cuda::mr::device_accessible>);

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
      cuda::mr::any_synchronous_resource<::cuda::mr::host_accessible, get_data> mr{TestResource{42, this}};
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.move_count;
      CHECK(this->counts == expected);
      CHECK(get_property(mr, get_data{}) == 42);
      get_property(mr, ::cuda::mr::host_accessible{});
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
      cuda::mr::any_synchronous_resource<::cuda::mr::host_accessible, get_data> mr{TestResource{42, this}};
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
      cuda::mr::any_synchronous_resource<::cuda::mr::host_accessible, get_data> mr{TestResource{42, this}};
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.move_count;
      CHECK(this->counts == expected);

      void* ptr = mr.allocate_sync(this->bytes(50), this->align(8));
      CHECK(ptr == this);
      ++expected.allocate_count;
      CHECK(this->counts == expected);

      mr.deallocate_sync(ptr, this->bytes(50), this->align(8));
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
      TestResource resource1{42, this};
      TestResource resource2{42, this};
      expected.object_count += 2;
      CHECK(this->counts == expected);
      CHECK(resource1 == resource2);
      ++expected.equal_to_count;
      CHECK(this->counts == expected);
      cuda::mr::any_synchronous_resource<::cuda::mr::host_accessible, get_data> mr{resource1};
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.copy_count;
      CHECK(this->counts == expected);
      CHECK(mr == resource1);
      ++expected.equal_to_count;
      CHECK(this->counts == expected);
    }
    expected.delete_count += is_big;
    expected.object_count -= 3;
    CHECK(this->counts == expected);
  }

  // Reset the counters:
  this->counts = Counts();

  SECTION("conversion from any_synchronous_resource to "
          "cuda::mr::synchronous_resource_ref")
  {
    Counts expected{};
    {
      cuda::mr::any_synchronous_resource<::cuda::mr::host_accessible, get_data> mr{TestResource{42, this}};
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.move_count;
      CHECK(this->counts == expected);

      // conversion from any_synchronous_resource to
      // cuda::mr::synchronous_synchronous_resource_ref:
      cuda::mr::synchronous_resource_ref<::cuda::mr::host_accessible, get_data> ref = mr;

      // conversion from any_synchronous_resource to
      // cuda::mr::synchronous_synchronous_resource_ref with narrowing:
      cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible, get_data> ref2 = mr;
      CHECK(get_property(ref2, get_data{}) == 42);

      CHECK(this->counts == expected);
      auto* ptr = ref.allocate_sync(this->bytes(100), this->align(8));
      CHECK(ptr == this);
      ++expected.allocate_count;
      CHECK(this->counts == expected);
      ref.deallocate_sync(ptr, this->bytes(0), this->align(0));
      ++expected.deallocate_count;
      CHECK(this->counts == expected);
    }
    expected.delete_count += is_big;
    --expected.object_count;
    CHECK(this->counts == expected);
  }

  SECTION("conversion from any_synchronous_resource to "
          "cuda::mr::synchronous_resource_ref")
  {
    Counts expected{};
    {
      cuda::mr::any_synchronous_resource<::cuda::mr::host_accessible, get_data> mr{TestResource{42, this}};
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.move_count;
      CHECK(this->counts == expected);

      // conversion from any_synchronous_resource to
      // cuda::mr::synchronous_resource_ref:
      cuda::mr::synchronous_resource_ref<::cuda::mr::host_accessible, get_data> ref = mr;

      // conversion from any_synchronous_resource to
      // cuda::mr::synchronous_resource_ref with narrowing:
      cuda::mr::synchronous_resource_ref<::cuda::mr::host_accessible, get_data> ref2 = mr;
      CHECK(get_property(ref2, get_data{}) == 42);

      CHECK(this->counts == expected);
      auto* ptr = ref.allocate_sync(this->bytes(100), this->align(8));
      CHECK(ptr == this);
      ++expected.allocate_count;
      CHECK(this->counts == expected);
      ref.deallocate_sync(ptr, this->bytes(0), this->align(0));
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
      cuda::mr::synchronous_resource_ref<::cuda::mr::host_accessible, get_data> ref{test};
      CHECK(this->counts == expected);

      cuda::mr::any_synchronous_resource<::cuda::mr::host_accessible, get_data> mr = ref;
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.copy_count;
      CHECK(this->counts == expected);

      auto* ptr = ref.allocate_sync(this->bytes(100), this->align(8));
      CHECK(ptr == this);
      ++expected.allocate_count;
      CHECK(this->counts == expected);
      ref.deallocate_sync(ptr, this->bytes(0), this->align(0));
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
      cuda::mr::any_synchronous_resource<::cuda::mr::host_accessible, extra_property, get_data> mr{
        TestResource{42, this}};
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.move_count;
      CHECK(this->counts == expected);

      cuda::mr::any_synchronous_resource<::cuda::mr::host_accessible, get_data> mr2 = mr;
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.copy_count;
      CHECK(this->counts == expected);

      CHECK(get_property(mr2, get_data{}) == 42);
      auto data = try_get_property(mr2, get_data{});
      static_assert(cuda::std::is_same_v<decltype(data), cuda::std::optional<int>>);
      CHECK(data.has_value());
      CHECK(data.value() == 42);

      auto host = try_get_property(mr2, extra_property{});
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
      cuda::mr::any_synchronous_resource<::cuda::mr::host_accessible, get_data> mr =
        cuda::mr::make_any_synchronous_resource<TestResource, ::cuda::mr::host_accessible, get_data>(42, this);
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

TEMPLATE_TEST_CASE_METHOD(
  test_fixture, "synchronous ref assignment operators", "[container][resource]", big_resource, small_resource)
{
  big_resource mr{42, this};
  cuda::mr::synchronous_resource_ref<::cuda::mr::host_accessible, get_data> ref{mr};
  CHECK(ref.allocate_sync(this->bytes(100), this->align(8)) == this);
  CHECK(get_property(ref, get_data{}) == 42);

  big_resource mr2{43, this};
  cuda::mr::synchronous_resource_ref<::cuda::mr::host_accessible, get_data> ref2{mr2};
  ref = ref2;
  CHECK(ref.allocate_sync(this->bytes(100), this->align(8)) == this);
  CHECK(get_property(ref, get_data{}) == 43);

  cuda::mr::synchronous_resource_ref<::cuda::mr::host_accessible, get_data, extra_property> ref3{mr};
  ref = ref3;
  CHECK(ref.allocate_sync(this->bytes(100), this->align(8)) == this);
  CHECK(get_property(ref, get_data{}) == 42);
}

TEMPLATE_TEST_CASE_METHOD(test_fixture, "Empty property set", "[container][resource]", big_resource, small_resource)
{
  using TestResource = TestType;
  {
    cuda::mr::any_synchronous_resource<> mr{TestResource{42, this}};
    CHECK(mr.allocate_sync(this->bytes(100), this->align(8)) == this);
    CHECK(!try_get_property(mr, get_data{}));
    CHECK(!try_get_property(mr, extra_property{}));
    mr.deallocate_sync(this, this->bytes(0), this->align(0));
  }

  {
    cuda::mr::any_synchronous_resource<get_data> mr{TestResource{42, this}};
    cuda::mr::any_synchronous_resource<> mr_sliced_off_to_empty{mr};
    CHECK(mr.allocate_sync(this->bytes(100), this->align(8)) == this);
    CHECK(try_get_property(mr, get_data{}).value() == 42);
    CHECK(!try_get_property(mr, extra_property{}));
    mr.deallocate_sync(this, this->bytes(0), this->align(0));
  }
}
