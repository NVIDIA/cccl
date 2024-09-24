#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include <catch2/catch.hpp>
#include <testing.cuh>

using std::size_t;
using std::uintptr_t;

struct Counts
{
  int object_count           = 0;
  int move_count             = 0;
  int copy_count             = 0;
  int allocate_count         = 0;
  int deallocate_count       = 0;
  int allocate_async_count   = 0;
  int deallocate_async_count = 0;
  int equal_to_count         = 0;
  int new_count              = 0;
  int delete_count           = 0;

  friend std::ostream& operator<<(std::ostream& os, const Counts& counts)
  {
    return os
        << "object: " << counts.object_count << ", " //
        << "move: " << counts.move_count << ", " //
        << "copy: " << counts.copy_count << ", " //
        << "allocate: " << counts.allocate_count << ", " //
        << "deallocate: " << counts.deallocate_count << ", " //
        << "allocate_async: " << counts.allocate_async_count << ", " //
        << "deallocate_async: " << counts.deallocate_async_count << ", " //
        << "equal_to: " << counts.equal_to_count << ", " //
        << "new: " << counts.new_count << ", " //
        << "delete: " << counts.delete_count;
  }

  friend bool operator==(const Counts& lhs, const Counts& rhs) noexcept
  {
    return lhs.object_count == rhs.object_count && //
           lhs.move_count == rhs.move_count && //
           lhs.copy_count == rhs.copy_count && //
           lhs.allocate_count == rhs.allocate_count && //
           lhs.deallocate_count == rhs.deallocate_count && //
           lhs.allocate_async_count == rhs.allocate_async_count && //
           lhs.deallocate_async_count == rhs.deallocate_async_count && //
           lhs.equal_to_count == rhs.equal_to_count && //
           lhs.new_count == rhs.new_count && //
           lhs.delete_count == rhs.delete_count; //
  }

  friend bool operator!=(const Counts& lhs, const Counts& rhs) noexcept
  {
    return !(lhs == rhs);
  }
};

struct test_fixture_
{
  Counts counts;
  size_t bytes_ = 0;
  size_t align_ = 0;
  static thread_local Counts* counts_;

  test_fixture_() noexcept
      : counts()
  {
    counts_ = &counts;
  }

  size_t bytes(size_t sz) noexcept
  {
    bytes_ = sz;
    return bytes_;
  }

  size_t align(size_t align) noexcept
  {
    align_ = align;
    return align_;
  }
};

inline thread_local Counts* test_fixture_::counts_ = nullptr;

template <class>
using test_fixture = test_fixture_;

template <class T>
struct test_resource
{
  int data;
  test_fixture_* fixture;
  T cookie[2] = {0xDEADBEEF, 0xDEADBEEF};

  explicit test_resource(int i, test_fixture_* fix) noexcept
      : data(i)
      , fixture(fix)
  {
    ++fixture->counts.object_count;
  }

  test_resource(test_resource&& other) noexcept
      : data(other.data)
      , fixture(other.fixture)
  {
    other._assert_valid();
    ++fixture->counts.move_count;
    ++fixture->counts.object_count;
    other.cookie[0] = other.cookie[1] = 0x0C07FEFE;
  }

  test_resource(const test_resource& other) noexcept
      : data(other.data)
      , fixture(other.fixture)
  {
    other._assert_valid();
    ++fixture->counts.copy_count;
    ++fixture->counts.object_count;
  }

  ~test_resource()
  {
    --fixture->counts.object_count;
  }

  void* allocate(std::size_t bytes, std::size_t align)
  {
    _assert_valid();
    CHECK(bytes == fixture->bytes_);
    CHECK(align == fixture->align_);
    ++fixture->counts.allocate_count;
    return fixture;
  }

  void deallocate(void* ptr, std::size_t bytes, std::size_t align) noexcept
  {
    _assert_valid();
    CHECK(ptr == fixture);
    CHECK(bytes == fixture->bytes_);
    CHECK(align == fixture->align_);
    ++fixture->counts.deallocate_count;
    return;
  }

  void* allocate_async(std::size_t bytes, std::size_t align, ::cuda::stream_ref)
  {
    _assert_valid();
    CHECK(bytes == fixture->bytes_);
    CHECK(align == fixture->align_);
    ++fixture->counts.allocate_async_count;
    return fixture;
  }

  void deallocate_async(void* ptr, std::size_t bytes, std::size_t align, ::cuda::stream_ref) noexcept
  {
    _assert_valid();
    CHECK(ptr == fixture);
    CHECK(bytes == fixture->bytes_);
    CHECK(align == fixture->align_);
    ++fixture->counts.deallocate_async_count;
    return;
  }

  friend bool operator==(const test_resource& lhs, const test_resource& rhs)
  {
    lhs._assert_valid();
    rhs._assert_valid();
    ++lhs.fixture->counts.equal_to_count;
    return lhs.data == rhs.data;
  }

  friend bool operator!=(const test_resource& lhs, const test_resource& rhs)
  {
    FAIL("any_resource should only be calling operator==");
    return lhs.data != rhs.data;
  }

  void _assert_valid() const noexcept
  {
    REQUIRE(cookie[0] == 0xDEADBEEF);
    REQUIRE(cookie[1] == 0xDEADBEEF);
  }

  static void* operator new(::cuda::std::size_t size)
  {
    ++test_fixture_::counts_->new_count;
    return ::operator new(size);
  }

  static void operator delete(void* pv) noexcept
  {
    ++test_fixture_::counts_->delete_count;
    return ::operator delete(pv);
  }

  friend constexpr void get_property(const test_resource&, cuda::mr::host_accessible) noexcept {}
};

using big_resource   = test_resource<uintptr_t>;
using small_resource = test_resource<unsigned int>;

static_assert(sizeof(big_resource) > sizeof(cuda::mr::_AnyResourceStorage));
static_assert(sizeof(small_resource) <= sizeof(cuda::mr::_AnyResourceStorage));
