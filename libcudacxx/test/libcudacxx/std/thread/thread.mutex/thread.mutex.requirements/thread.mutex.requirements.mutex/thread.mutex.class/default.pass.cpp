//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/mutex>

// class mutex;

// constexpr mutex() noexcept;

// UNSUPPORTED: pre-sm-70
// Device atomics used by mutex require sm_70.

// UNSUPPORTED: force-tile
// error: asm statement is not supported

#include <cuda/std/cassert>
#include <cuda/std/mutex>
#include <cuda/std/type_traits>

#include "test_macros.h"

static_assert(noexcept(cuda::std::mutex{}), "");
static_assert(cuda::std::is_nothrow_default_constructible_v<cuda::std::mutex>, "");
static_assert(!cuda::std::is_copy_constructible_v<cuda::std::mutex>, "");
static_assert(!cuda::std::is_copy_assignable_v<cuda::std::mutex>, "");
static_assert(!cuda::std::is_move_constructible_v<cuda::std::mutex>, "");
static_assert(!cuda::std::is_move_assignable_v<cuda::std::mutex>, "");
static_assert(cuda::std::is_same_v<cuda::std::mutex, cuda::mutex<>>, "");
static_assert(cuda::std::is_same_v<cuda::std::mutex, cuda::mutex<cuda::thread_scope_device>>, "");
static_assert(cuda::std::mutex::_Scope == cuda::thread_scope_device, "");
static_assert(cuda::std::is_standard_layout_v<cuda::std::mutex>, "");

TEST_HOST_DEVICE_FUNC constexpr bool test_constexpr_mutex()
{
  cuda::std::mutex mutex{};
  (void) mutex;
  return true;
}
static_assert(test_constexpr_mutex(), "");

int main(int, char**)
{
  cuda::std::mutex mutex;
  assert(mutex.try_lock());
  assert(!mutex.try_lock());
  mutex.unlock();
  assert(mutex.try_lock());
  mutex.unlock();
  return 0;
}
