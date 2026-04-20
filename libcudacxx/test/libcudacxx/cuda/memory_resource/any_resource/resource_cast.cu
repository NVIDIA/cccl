//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/memory_resource>

#include <testing.cuh>

#include "../test_resource.cuh" // IWYU pragma: keep

#ifndef __CUDA_ARCH__

TEST_CASE("resource_cast from any_resource", "[container][resource]")
{
  test_fixture_ fixture;
  big_resource stored{42, &fixture};
  cuda::mr::any_resource<cuda::mr::host_accessible> mr{stored};

  SECTION("matching type returns non-null")
  {
    auto* ptr = cuda::mr::resource_cast<big_resource>(&mr);
    REQUIRE(ptr != nullptr);
    CHECK(ptr->data == 42);
  }

  SECTION("mismatched type returns null")
  {
    auto* ptr = cuda::mr::resource_cast<small_resource>(&mr);
    CHECK(ptr == nullptr);
  }

  SECTION("const overload")
  {
    const auto& cmr = mr;
    auto* ptr       = cuda::mr::resource_cast<big_resource>(&cmr);
    REQUIRE(ptr != nullptr);
    CHECK(ptr->data == 42);
  }
}

TEST_CASE("resource_cast from any_synchronous_resource", "[container][resource]")
{
  test_fixture_ fixture;
  cuda::mr::any_synchronous_resource<cuda::mr::host_accessible> mr{big_resource{7, &fixture}};

  auto* ptr = cuda::mr::resource_cast<big_resource>(&mr);
  REQUIRE(ptr != nullptr);
  CHECK(ptr->data == 7);

  CHECK(cuda::mr::resource_cast<small_resource>(&mr) == nullptr);
}

TEST_CASE("resource_cast from resource_ref", "[container][resource]")
{
  test_fixture_ fixture;
  big_resource stored{99, &fixture};
  cuda::mr::resource_ref<cuda::mr::host_accessible> ref{stored};

  auto* ptr = cuda::mr::resource_cast<big_resource>(&ref);
  REQUIRE(ptr != nullptr);
  CHECK(ptr->data == 99);
  CHECK(ptr == &stored);

  CHECK(cuda::mr::resource_cast<small_resource>(&ref) == nullptr);
}

TEST_CASE("resource_cast from synchronous_resource_ref", "[container][resource]")
{
  test_fixture_ fixture;
  big_resource stored{55, &fixture};
  cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible> ref{stored};

  auto* ptr = cuda::mr::resource_cast<big_resource>(&ref);
  REQUIRE(ptr != nullptr);
  CHECK(ptr->data == 55);

  CHECK(cuda::mr::resource_cast<small_resource>(&ref) == nullptr);
}

TEST_CASE("resource_cast with void extracts raw pointer", "[container][resource]")
{
  test_fixture_ fixture;
  cuda::mr::any_resource<cuda::mr::host_accessible> mr{big_resource{1, &fixture}};

  auto* vptr = cuda::mr::resource_cast<void>(&mr);
  CHECK(vptr != nullptr);

  const auto& cmr   = mr;
  const auto* cvptr = cuda::mr::resource_cast<void>(&cmr);
  CHECK(cvptr != nullptr);
  CHECK(cvptr == vptr);
}

TEST_CASE("resource_cast on empty any_resource returns null", "[container][resource]")
{
  cuda::mr::any_resource<cuda::mr::host_accessible> mr{};
  CHECK(cuda::mr::resource_cast<big_resource>(&mr) == nullptr);
}

TEST_CASE("resource_cast on nullptr input returns null", "[container][resource]")
{
  cuda::mr::any_resource<cuda::mr::host_accessible>* ptr = nullptr;
  CHECK(cuda::mr::resource_cast<big_resource>(ptr) == nullptr);

  const cuda::mr::any_resource<cuda::mr::host_accessible>* cptr = nullptr;
  CHECK(cuda::mr::resource_cast<big_resource>(cptr) == nullptr);
}

TEST_CASE("resource_cast on moved-from any_resource returns null", "[container][resource]")
{
  test_fixture_ fixture;
  cuda::mr::any_resource<cuda::mr::host_accessible> src{big_resource{42, &fixture}};
  auto dst = ::cuda::std::move(src);

  // moved-from source should be empty
  CHECK(cuda::mr::resource_cast<big_resource>(&src) == nullptr);

  // destination should have the value
  auto* ptr = cuda::mr::resource_cast<big_resource>(&dst);
  REQUIRE(ptr != nullptr);
  CHECK(ptr->data == 42);
}

// A derived resource type for testing that resource_cast uses exact type matching
struct derived_resource : big_resource
{
  using big_resource::big_resource;
  using default_queries = cuda::mr::properties_list<>;
};

TEST_CASE("resource_cast uses exact type matching, not derived-to-base", "[container][resource]")
{
  test_fixture_ fixture;
  derived_resource stored{99, &fixture};
  cuda::mr::any_resource<cuda::mr::host_accessible> mr{stored};

  // Exact type match succeeds
  auto* exact = cuda::mr::resource_cast<derived_resource>(&mr);
  REQUIRE(exact != nullptr);
  CHECK(exact->data == 99);

  // Base type does NOT match — resource_cast uses exact type identity
  auto* base = cuda::mr::resource_cast<big_resource>(&mr);
  CHECK(base == nullptr);
}

// ── dynamic_resource_cast ───────────────────────────────────────────────────

TEST_CASE("dynamic_resource_cast any_resource narrowing", "[container][resource]")
{
  test_fixture_ fixture;
  big_resource stored{42, &fixture};

  SECTION("narrow to one property")
  {
    cuda::mr::any_resource<cuda::mr::host_accessible, extra_property> src{stored};
    auto dst = cuda::mr::dynamic_resource_cast<extra_property>(::cuda::std::move(src));
    CHECK(dst.has_value());
    auto* ptr = cuda::mr::resource_cast<big_resource>(&dst);
    REQUIRE(ptr != nullptr);
    CHECK(ptr->data == 42);
  }

  SECTION("narrow to the other property")
  {
    cuda::mr::any_resource<cuda::mr::host_accessible, extra_property> src{stored};
    auto dst = cuda::mr::dynamic_resource_cast<cuda::mr::host_accessible>(::cuda::std::move(src));
    CHECK(dst.has_value());
    auto* ptr = cuda::mr::resource_cast<big_resource>(&dst);
    REQUIRE(ptr != nullptr);
    CHECK(ptr->data == 42);
  }

#  if _CCCL_HAS_EXCEPTIONS()
  SECTION("cast to unsupported property throws")
  {
    cuda::mr::any_resource<cuda::mr::host_accessible> src{stored};
    CHECK_THROWS_AS(cuda::mr::dynamic_resource_cast<cuda::mr::device_accessible>(::cuda::std::move(src)),
                    cuda::__bad_any_cast);
  }

  SECTION("cast to undeclared property throws")
  {
    // extra_property is supported by big_resource but not in the wrapper's interface
    cuda::mr::any_resource<cuda::mr::host_accessible> src{stored};
    CHECK_THROWS_AS(cuda::mr::dynamic_resource_cast<extra_property>(::cuda::std::move(src)), cuda::__bad_any_cast);
  }
#  endif // _CCCL_HAS_EXCEPTIONS()
}

TEST_CASE("dynamic_resource_cast any_resource narrow-then-widen round-trip", "[container][resource]")
{
  test_fixture_ fixture;
  big_resource stored{42, &fixture};

  // Start wide, narrow, then widen back — vtable retains the original entries
  cuda::mr::any_resource<cuda::mr::host_accessible, extra_property> wide{stored};
  auto narrow = cuda::mr::dynamic_resource_cast<cuda::mr::host_accessible>(::cuda::std::move(wide));
  CHECK(narrow.has_value());
  auto restored = cuda::mr::dynamic_resource_cast<extra_property>(::cuda::std::move(narrow));
  CHECK(restored.has_value());
  CHECK(try_get_property(restored, extra_property{}) == true);
  auto* ptr = cuda::mr::resource_cast<big_resource>(&restored);
  REQUIRE(ptr != nullptr);
  CHECK(ptr->data == 42);
}

TEST_CASE("dynamic_resource_cast any_synchronous_resource", "[container][resource]")
{
  test_fixture_ fixture;
  big_resource stored{7, &fixture};

  SECTION("valid narrowing cast")
  {
    cuda::mr::any_synchronous_resource<cuda::mr::host_accessible, extra_property> src{stored};
    auto dst = cuda::mr::dynamic_resource_cast<extra_property>(::cuda::std::move(src));
    CHECK(dst.has_value());
    auto* ptr = cuda::mr::resource_cast<big_resource>(&dst);
    REQUIRE(ptr != nullptr);
    CHECK(ptr->data == 7);
  }

#  if _CCCL_HAS_EXCEPTIONS()
  SECTION("invalid cast throws")
  {
    cuda::mr::any_synchronous_resource<cuda::mr::host_accessible> src{stored};
    CHECK_THROWS_AS(cuda::mr::dynamic_resource_cast<cuda::mr::device_accessible>(::cuda::std::move(src)),
                    cuda::__bad_any_cast);
  }
#  endif // _CCCL_HAS_EXCEPTIONS()
}

TEST_CASE("dynamic_resource_cast resource_ref", "[container][resource]")
{
  test_fixture_ fixture;
  big_resource stored{99, &fixture};

  SECTION("valid narrowing cast")
  {
    cuda::mr::resource_ref<cuda::mr::host_accessible, extra_property> ref{stored};
    auto dst  = cuda::mr::dynamic_resource_cast<extra_property>(&ref);
    auto* ptr = cuda::mr::resource_cast<big_resource>(&dst);
    REQUIRE(ptr != nullptr);
    CHECK(ptr->data == 99);
  }

#  if _CCCL_HAS_EXCEPTIONS()
  SECTION("invalid cast throws")
  {
    cuda::mr::resource_ref<cuda::mr::host_accessible> ref{stored};
    CHECK_THROWS_AS(cuda::mr::dynamic_resource_cast<cuda::mr::device_accessible>(&ref), cuda::__bad_any_cast);
  }
#  endif // _CCCL_HAS_EXCEPTIONS()
}

TEST_CASE("dynamic_resource_cast synchronous_resource_ref", "[container][resource]")
{
  test_fixture_ fixture;
  big_resource stored{55, &fixture};

  SECTION("valid narrowing cast")
  {
    cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible, extra_property> ref{stored};
    auto dst  = cuda::mr::dynamic_resource_cast<extra_property>(&ref);
    auto* ptr = cuda::mr::resource_cast<big_resource>(&dst);
    REQUIRE(ptr != nullptr);
    CHECK(ptr->data == 55);
  }

#  if _CCCL_HAS_EXCEPTIONS()
  SECTION("invalid cast throws")
  {
    cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible> ref{stored};
    CHECK_THROWS_AS(cuda::mr::dynamic_resource_cast<cuda::mr::device_accessible>(&ref), cuda::__bad_any_cast);
  }
#  endif // _CCCL_HAS_EXCEPTIONS()
}

// ── void property vtable entry verification ─────────────────────────────────

TEST_CASE("try_get_property for void properties respects concrete type", "[container][resource]")
{
  // big_resource provides host_accessible and extra_property, but NOT device_accessible
  test_fixture_ fixture;
  big_resource stored{1, &fixture};
  cuda::mr::any_resource<cuda::mr::host_accessible> mr{stored};

  // Provided by big_resource — should be true
  CHECK(try_get_property(mr, cuda::mr::host_accessible{}) == true);

  // NOT provided by big_resource — should be false
  CHECK(try_get_property(mr, cuda::mr::device_accessible{}) == false);
}

#endif // __CUDA_ARCH__
