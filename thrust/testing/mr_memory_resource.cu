// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/mr/allocator.h>
#include <thrust/mr/memory_resource.h>
#include <thrust/mr/new.h>

#include <cstddef>

#include <unittest/unittest.h>

class forwarding_resource : public thrust::mr::memory_resource<>
{
public:
  void* do_allocate(std::size_t bytes, std::size_t alignment) override
  {
    return upstream.do_allocate(bytes, alignment);
  }

  void do_deallocate(void* pointer, std::size_t bytes, std::size_t alignment) override
  {
    upstream.do_deallocate(pointer, bytes, alignment);
  }

private:
  thrust::mr::new_delete_resource upstream;
};

class identity_resource final : public forwarding_resource
{};

class always_equal_resource final : public forwarding_resource
{
public:
  _CCCL_HOST_DEVICE bool do_is_equal(const thrust::mr::memory_resource<>&) const noexcept override
  {
    return true;
  }
};

void TestMemoryResourceIdentityEquality()
{
  identity_resource first;
  identity_resource second;

  REQUIRE(first == first);
  REQUIRE_FALSE(first != first);
  REQUIRE_FALSE(first == second);
  REQUIRE_FALSE(second == first);

  using allocator = thrust::mr::allocator<int, identity_resource>;

  const allocator first_allocator(&first);
  const allocator same_allocator(&first);
  const allocator second_allocator(&second);

  REQUIRE(first_allocator == same_allocator);
  REQUIRE_FALSE(first_allocator != same_allocator);
  REQUIRE_FALSE(first_allocator == second_allocator);
  REQUIRE_FALSE(second_allocator == first_allocator);
}
DECLARE_UNITTEST(TestMemoryResourceIdentityEquality);

void TestMemoryResourceEquivalentEquality()
{
  always_equal_resource first;
  always_equal_resource second;

  REQUIRE(first == second);
  REQUIRE(second == first);
  REQUIRE_FALSE(first != second);

  using allocator = thrust::mr::allocator<int, always_equal_resource>;

  const allocator first_allocator(&first);
  const allocator second_allocator(&second);

  REQUIRE(first_allocator == second_allocator);
  REQUIRE(second_allocator == first_allocator);
  REQUIRE_FALSE(first_allocator != second_allocator);
}
DECLARE_UNITTEST(TestMemoryResourceEquivalentEquality);
