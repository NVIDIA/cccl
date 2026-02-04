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

#include "../test_resource.cuh" // IWYU pragma: keep

#ifndef __CUDA_ARCH__

TEMPLATE_TEST_CASE_METHOD(test_fixture, "any_resource", "[container][resource]", big_resource, small_resource)
{
  using TestResource = TestType;
  static_assert(cuda::mr::synchronous_resource_with<TestResource, ::cuda::mr::host_accessible>);
  constexpr bool is_big = sizeof(TestResource) > cuda::__default_small_object_size;

  SECTION("construct and destruct")
  {
    Counts expected{};
    CHECK(this->counts == expected);
    {
      cuda::mr::any_resource<::cuda::mr::host_accessible> mr{TestResource{42, this}};
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
      cuda::mr::any_resource<::cuda::mr::host_accessible> mr{TestResource{42, this}};
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
      cuda::mr::any_resource<::cuda::mr::host_accessible> mr{TestResource{42, this}};
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

  SECTION("allocate and deallocate")
  {
    Counts expected{};
    CHECK(this->counts == expected);
    {
      cuda::stream stream{cuda::device_ref{0}};
      cuda::mr::any_resource<::cuda::mr::host_accessible> mr{TestResource{42, this}};
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.move_count;
      CHECK(this->counts == expected);

      void* ptr = mr.allocate(::cuda::stream_ref{stream}, this->bytes(50), this->align(8));
      CHECK(ptr == this);
      ++expected.allocate_async_count;
      CHECK(this->counts == expected);

      mr.deallocate(::cuda::stream_ref{stream}, ptr, this->bytes(50), this->align(8));
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
      cuda::mr::any_resource<::cuda::mr::host_accessible> mr{TestResource{42, this}};
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.move_count;
      CHECK(this->counts == expected);

      cuda::mr::synchronous_resource_ref<::cuda::mr::host_accessible> ref = mr;

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

  SECTION("make_any_resource")
  {
    Counts expected{};
    CHECK(this->counts == expected);
    {
      cuda::mr::any_resource<::cuda::mr::host_accessible> mr =
        cuda::mr::make_any_resource<TestResource, ::cuda::mr::host_accessible>(42, this);
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
  test_fixture, "ref assignment operators", "[container][resource]", big_resource, small_resource)
{
  big_resource mr{42, this};
  cuda::mr::resource_ref<::cuda::mr::host_accessible, get_data> ref{mr};
  CHECK(ref.allocate_sync(this->bytes(100), this->align(8)) == this);
  CHECK(get_property(ref, get_data{}) == 42);

  big_resource mr2{43, this};
  cuda::mr::resource_ref<::cuda::mr::host_accessible, get_data> ref2{mr2};
  ref = ref2;
  CHECK(ref.allocate_sync(this->bytes(100), this->align(8)) == this);
  CHECK(get_property(ref, get_data{}) == 43);

  cuda::mr::resource_ref<::cuda::mr::host_accessible, get_data, extra_property> ref3{mr};
  ref = ref3;
  CHECK(ref.allocate_sync(this->bytes(100), this->align(8)) == this);
  CHECK(get_property(ref, get_data{}) == 42);
}

struct host_device_resource
{
  void* allocate(cuda::stream_ref, size_t, size_t)
  {
    return nullptr;
  }
  void deallocate(cuda::stream_ref, void*, size_t, size_t) noexcept {}
  void* allocate_sync(size_t, size_t)
  {
    return nullptr;
  }
  void deallocate_sync(void*, size_t, size_t) noexcept {}
  friend bool operator==(const host_device_resource&, const host_device_resource&) noexcept
  {
    return true;
  }
  friend bool operator!=(const host_device_resource&, const host_device_resource&) noexcept
  {
    return false;
  }
  friend constexpr void get_property(const host_device_resource&, cuda::mr::host_accessible) noexcept {}
  friend constexpr void get_property(const host_device_resource&, cuda::mr::device_accessible) noexcept {}
};
static_assert(cuda::has_property<host_device_resource, cuda::mr::device_accessible>);
static_assert(cuda::has_property<host_device_resource, cuda::mr::host_accessible>);

void requires_host(cuda::mr::resource_ref<cuda::mr::host_accessible>) {}
void requires_device(cuda::mr::resource_ref<cuda::mr::device_accessible>) {}

bool checks_device_runtime_any_resource(cuda::mr::any_resource<cuda::mr::host_accessible> res)
{
  if (try_get_property(res, cuda::mr::device_accessible{}))
  {
    std::cout << "Dynamically determined that we are device accessible" << std::endl;
    return true;
  }
  return false;
}

bool checks_device_runtime_resource_ref(cuda::mr::resource_ref<cuda::mr::host_accessible> ref)
{
  if (try_get_property(ref, cuda::mr::device_accessible{}))
  {
    std::cout << "Dynamically determined that we are device accessible" << std::endl;
    return true;
  }
  return false;
}

TEST_CASE("resource_ref regression test for cccl#6839", "[container][resource]")
{
  // Test for https://github.com/NVIDIA/cccl/issues/6839
  auto host_device_mr =
    cuda::mr::make_any_resource<host_device_resource, cuda::mr::host_accessible, cuda::mr::device_accessible>();
  cuda::mr::resource_ref<cuda::mr::host_accessible> ref = host_device_mr;
  requires_host(host_device_mr); // compile-time enforced, good.
  requires_device(host_device_mr); // compile-time enforced, good

  cuda::mr::any_resource<cuda::mr::host_accessible> res = host_device_mr;
  CHECK(checks_device_runtime_any_resource(res)); // Test that we are device accessible
  CHECK(checks_device_runtime_resource_ref(ref)); // Test that we are device accessible
}

TEMPLATE_TEST_CASE_METHOD(test_fixture, "Empty property set", "[container][resource]", big_resource, small_resource)
{
  using TestResource = TestType;
  {
    cuda::mr::any_resource<> mr{TestResource{42, this}};
    CHECK(mr.allocate_sync(this->bytes(100), this->align(8)) == this);
    CHECK(!try_get_property(mr, get_data{}));
    CHECK(!try_get_property(mr, extra_property{}));
    mr.deallocate_sync(this, this->bytes(0), this->align(0));
  }

  {
    cuda::mr::any_resource<get_data> mr{TestResource{42, this}};
    cuda::mr::any_resource<> mr_sliced_off_to_empty{mr};
    CHECK(mr.allocate_sync(this->bytes(100), this->align(8)) == this);
    CHECK(try_get_property(mr, get_data{}).value() == 42);
    CHECK(!try_get_property(mr, extra_property{}));
    mr.deallocate_sync(this, this->bytes(0), this->align(0));
  }
}
#endif // __CUDA_ARCH__
