//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: nvrtc

// cuda::mr::resource

#include <cuda/memory_resource>
#include <cuda/std/cstdint>

#include "test_macros.h"

struct invalid_argument
{};

struct valid_resource
{
  void* allocate(std::size_t, std::size_t)
  {
    return nullptr;
  }
  void deallocate(void*, std::size_t, std::size_t) noexcept {}
  bool operator==(const valid_resource&) const
  {
    return true;
  }
  bool operator!=(const valid_resource&) const
  {
    return false;
  }
};
static_assert(cuda::mr::resource<valid_resource>, "");

struct invalid_allocate_argument
{
  void* allocate(invalid_argument, std::size_t)
  {
    return nullptr;
  }
  void deallocate(void*, std::size_t, std::size_t) noexcept {}
  bool operator==(const invalid_allocate_argument&)
  {
    return true;
  }
  bool operator!=(const invalid_allocate_argument&)
  {
    return false;
  }
};
static_assert(!cuda::mr::resource<invalid_allocate_argument>, "");

struct invalid_allocate_return
{
  int allocate(std::size_t, std::size_t)
  {
    return 42;
  }
  void deallocate(void*, std::size_t, std::size_t) noexcept {}
  bool operator==(const invalid_allocate_return&)
  {
    return true;
  }
  bool operator!=(const invalid_allocate_return&)
  {
    return false;
  }
};
static_assert(!cuda::mr::resource<invalid_allocate_return>, "");

struct invalid_deallocate_argument
{
  void* allocate(std::size_t, std::size_t)
  {
    return nullptr;
  }
  void deallocate(void*, invalid_argument, std::size_t) noexcept {}
  bool operator==(const invalid_deallocate_argument&)
  {
    return true;
  }
  bool operator!=(const invalid_deallocate_argument&)
  {
    return false;
  }
};
static_assert(!cuda::mr::resource<invalid_deallocate_argument>, "");

struct non_comparable
{
  void* allocate(std::size_t, std::size_t)
  {
    return nullptr;
  }
  void deallocate(void*, std::size_t, std::size_t) noexcept {}
};
static_assert(!cuda::mr::resource<non_comparable>, "");

struct non_eq_comparable
{
  void* allocate(std::size_t, std::size_t)
  {
    return nullptr;
  }
  void deallocate(void*, std::size_t, std::size_t) noexcept {}
  bool operator!=(const non_eq_comparable&)
  {
    return false;
  }
};
static_assert(!cuda::mr::resource<non_eq_comparable>, "");

#if TEST_STD_VER < 2020
struct non_neq_comparable
{
  void* allocate(std::size_t, std::size_t)
  {
    return nullptr;
  }
  void deallocate(void*, std::size_t, std::size_t) noexcept {}
  bool operator==(const non_neq_comparable&)
  {
    return true;
  }
};
static_assert(!cuda::mr::resource<non_neq_comparable>, "");
#endif // TEST_STD_VER <20

int main(int, char**)
{
  return 0;
}
