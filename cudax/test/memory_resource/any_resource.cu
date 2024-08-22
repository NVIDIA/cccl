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

#include "cuda/std/detail/libcxx/include/cstddef"
#include <catch2/catch.hpp>
#include <testing.cuh>

struct Counts
{
  int object_count     = 0;
  int move_count       = 0;
  int copy_count       = 0;
  int allocate_count   = 0;
  int deallocate_count = 0;
  int equal_to_count   = 0;
  int new_count        = 0;
  int delete_count     = 0;

  friend bool operator==(const Counts& lhs, const Counts& rhs) noexcept
  {
    return lhs.object_count == rhs.object_count && //
           lhs.move_count == rhs.move_count && //
           lhs.copy_count == rhs.copy_count && //
           lhs.allocate_count == rhs.allocate_count && //
           lhs.deallocate_count == rhs.deallocate_count && //
           lhs.equal_to_count == rhs.equal_to_count && //
           lhs.new_count == rhs.new_count && //
           lhs.delete_count == rhs.delete_count; //
  }

  friend bool operator!=(const Counts& lhs, const Counts& rhs) noexcept
  {
    return !(lhs == rhs);
  }
};

struct big_resource
{
  int data;
  unsigned long cookie[3] = {0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF};

  static Counts counts;

  explicit big_resource(int i) noexcept
      : data(i)
  {
    ++counts.object_count;
  }

  big_resource(big_resource&& other) noexcept
      : data(other.data)
  {
    other._assert_valid();
    ++counts.move_count;
    ++counts.object_count;
    other.cookie[0] = other.cookie[1] = other.cookie[2] = 0x0C07FEFE;
  }

  big_resource(const big_resource& other) noexcept
      : data(other.data)
  {
    other._assert_valid();
    ++counts.copy_count;
    ++counts.object_count;
  }

  ~big_resource()
  {
    --counts.object_count;
  }

  void* allocate(std::size_t, std::size_t)
  {
    _assert_valid();
    ++counts.allocate_count;
    return nullptr;
  }

  void deallocate(void*, std::size_t, std::size_t) noexcept
  {
    _assert_valid();
    ++counts.deallocate_count;
    return;
  }

  friend bool operator==(const big_resource& lhs, const big_resource& rhs)
  {
    lhs._assert_valid();
    rhs._assert_valid();
    ++counts.equal_to_count;
    return lhs.data == rhs.data;
  }

  friend bool operator!=(const big_resource& lhs, const big_resource& rhs)
  {
    FAIL("any_resource should only be calling operator==");
    return lhs.data != rhs.data;
  }

  void _assert_valid() const noexcept
  {
    REQUIRE(cookie[0] == 0xDEADBEEF);
    REQUIRE(cookie[1] == 0xDEADBEEF);
    REQUIRE(cookie[2] == 0xDEADBEEF);
  }

  static void* operator new(::cuda::std::size_t size)
  {
    ++counts.new_count;
    return ::operator new(size);
  }

  static void operator delete(void* pv) noexcept
  {
    ++counts.delete_count;
    return ::operator delete(pv);
  }
};

static_assert(sizeof(big_resource) > sizeof(cuda::mr::_AnyResourceStorage));

Counts big_resource::counts{};

TEST_CASE("any_resource", "[container][resource]")
{
  SECTION("big resources")
  {
    SECTION("construct and destruct")
    {
      Counts expected{};
      CHECK(big_resource::counts == expected);
      {
        cudax::mr::any_resource<> mr{big_resource{42}};
        ++expected.new_count;
        ++expected.object_count;
        ++expected.move_count;
        CHECK(big_resource::counts == expected);

        void* ptr = mr.allocate(0, 0);
        ++expected.allocate_count;
        CHECK(big_resource::counts == expected);

        mr.deallocate(ptr, 0, 0);
        ++expected.deallocate_count;
        CHECK(big_resource::counts == expected);
      }
      ++expected.delete_count;
      --expected.object_count;
      CHECK(big_resource::counts == expected);
    }

    // Reset the counters:
    big_resource::counts = Counts();

    SECTION("conversion to resource_ref")
    {
      Counts expected{};
      {
        cudax::mr::any_resource<> mr{big_resource{42}};
        ++expected.new_count;
        ++expected.object_count;
        ++expected.move_count;
        CHECK(big_resource::counts == expected);

        cuda::mr::resource_ref<> ref = mr;

        CHECK(big_resource::counts == expected);
        auto* ptr = ref.allocate(0, 0);
        ++expected.allocate_count;
        CHECK(big_resource::counts == expected);
        ref.deallocate(ptr, 0, 0);
        ++expected.deallocate_count;
        CHECK(big_resource::counts == expected);
      }
      ++expected.delete_count;
      --expected.object_count;
      CHECK(big_resource::counts == expected);
    }
  }
}
