// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/mr/allocator.h>
#include <thrust/mr/memory_resource.h>
#include <thrust/mr/new.h>

#include <cstddef>

#include <unittest/unittest.h>

namespace
{
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

  ASSERT_EQUAL(first == first, true);
  ASSERT_EQUAL(first != first, false);
  ASSERT_EQUAL(first == second, false);
  ASSERT_EQUAL(second == first, false);

  using allocator = thrust::mr::allocator<int, identity_resource>;

  const allocator first_allocator(&first);
  const allocator same_allocator(&first);
  const allocator second_allocator(&second);

  ASSERT_EQUAL(first_allocator == same_allocator, true);
  ASSERT_EQUAL(first_allocator != same_allocator, false);
  ASSERT_EQUAL(first_allocator == second_allocator, false);
  ASSERT_EQUAL(second_allocator == first_allocator, false);
}
DECLARE_UNITTEST(TestMemoryResourceIdentityEquality);

void TestMemoryResourceEquivalentEquality()
{
  always_equal_resource first;
  always_equal_resource second;

  ASSERT_EQUAL(first == second, true);
  ASSERT_EQUAL(second == first, true);
  ASSERT_EQUAL(first != second, false);

  using allocator = thrust::mr::allocator<int, always_equal_resource>;

  const allocator first_allocator(&first);
  const allocator second_allocator(&second);

  ASSERT_EQUAL(first_allocator == second_allocator, true);
  ASSERT_EQUAL(second_allocator == first_allocator, true);
  ASSERT_EQUAL(first_allocator != second_allocator, false);
}
DECLARE_UNITTEST(TestMemoryResourceEquivalentEquality);
} // namespace
