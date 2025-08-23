//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/container.cuh>
#include <cuda/experimental/memory_resource.cuh>

#include <testing.cuh>

#include "test_resource.cuh"

TEMPLATE_TEST_CASE_METHOD(test_fixture, "shared_resource", "[container][resource]", big_resource, small_resource)
{
  using TestResource = TestType;
  static_assert(cuda::mr::resource<cudax::shared_resource<TestResource>>);

  SECTION("construct and destruct")
  {
    Counts expected{};
    CHECK(this->counts == expected);
    {
      cudax::shared_resource mr{cuda::std::in_place_type<TestResource>, 42, this};
      ++expected.object_count;
      CHECK(this->counts == expected);
    }

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
      cudax::shared_resource mr{cuda::std::in_place_type<TestResource>, 42, this};
      ++expected.object_count;
      CHECK(this->counts == expected);

      auto mr2 = mr;
      CHECK(this->counts == expected);
      CHECK(mr == mr2); // pointers compare equal, no call to TestResource::operator==
      CHECK(this->counts == expected);

      cudax::shared_resource mr3{mr};
      CHECK(this->counts == expected);
      CHECK(mr == mr3); // pointers compare equal, no call to TestResource::operator==
      CHECK(this->counts == expected);

      auto mr4 = std::move(mr);
      CHECK(this->counts == expected);
      CHECK(mr2 == mr4); // pointers compare equal, no call to TestResource::operator==
      CHECK(this->counts == expected);

      cudax::shared_resource mr5{cuda::std::in_place_type<TestResource>, TestResource{42, this}};
      ++expected.object_count;
      ++expected.move_count;
      CHECK(mr4 == mr5); // pointers are not equal, calls TestResource::operator==
      ++expected.equal_to_count;
      CHECK(this->counts == expected);
    }

    expected.object_count -= 2;
    CHECK(this->counts == expected);
  }

  // Reset the counters:
  this->counts = Counts();

  SECTION("allocate_sync and deallocate_sync")
  {
    Counts expected{};
    CHECK(this->counts == expected);
    {
      cudax::shared_resource mr{cuda::std::in_place_type<TestResource>, 42, this};
      ++expected.object_count;
      CHECK(this->counts == expected);

      void* ptr = mr.allocate_sync(bytes(50), align(8));
      CHECK(ptr == this);
      ++expected.allocate_count;
      CHECK(this->counts == expected);

      mr.deallocate_sync(ptr, bytes(50), align(8));
      ++expected.deallocate_count;
      CHECK(this->counts == expected);
    }

    --expected.object_count;
    CHECK(this->counts == expected);
  }

  // Reset the counters:
  this->counts = Counts();

  SECTION("conversion to synchronous_resource_ref")
  {
    Counts expected{};
    {
      cudax::shared_resource mr{cuda::std::in_place_type<TestResource>, 42, this};
      ++expected.object_count;
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
    --expected.object_count;
    CHECK(this->counts == expected);
  }

  // Reset the counters:
  this->counts = Counts();

  SECTION("basic sanity test about shared resource handling")
  {
    Counts expected{};
    align(alignof(int) * 4);
    {
      bytes(42 * sizeof(int));
      cudax::uninitialized_buffer<int, cudax::host_accessible> buffer{
        cudax::shared_resource<TestResource>(cuda::std::in_place_type<TestResource>, 42, this), 42};
      ++expected.object_count;
      ++expected.allocate_count;
      CHECK(this->counts == expected);

      // copying the shared_resource should not copy the stored resource
      {
        // accounting for new storage
        bytes(1337 * sizeof(int));
        cudax::uninitialized_buffer<int, cudax::host_accessible> other_buffer{buffer.memory_resource(), 1337};
        ++expected.allocate_count;
        CHECK(this->counts == expected);
      }

      // The original resource is still alive, but the second allocation was released
      bytes(42 * sizeof(int));
      ++expected.deallocate_count;
      CHECK(this->counts == expected);

      {
        // Moving the resource should not do anything
        cudax::uninitialized_buffer<int, cudax::host_accessible> third_buffer = ::cuda::std::move(buffer);
        CHECK(this->counts == expected);
      }

      // The original shared_resource has been moved from so everything is gone already
      --expected.object_count;
      ++expected.deallocate_count;
      CHECK(this->counts == expected);
    }

    // Nothing changes here as the first shared_resources has been moved from
    CHECK(this->counts == expected);
  }

  // Reset the counters:
  this->counts = Counts();
}
