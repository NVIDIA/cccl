//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/mutex>

// template <class Mutex>
// class lock_guard;

// explicit lock_guard(mutex_type& m);
// lock_guard(mutex_type& m, adopt_lock_t);

// UNSUPPORTED: pre-sm-70
// Device atomics used by mutex require sm_70.

// UNSUPPORTED: force-tile
// error: asm statement is not supported

#include <cuda/std/cassert>
#include <cuda/std/mutex>
#include <cuda/std/type_traits>

#include "test_macros.h"

static_assert(cuda::std::is_same_v<cuda::std::lock_guard<cuda::std::mutex>::mutex_type, cuda::std::mutex>, "");
static_assert(!cuda::std::is_copy_constructible_v<cuda::std::lock_guard<cuda::std::mutex>>, "");
static_assert(!cuda::std::is_copy_assignable_v<cuda::std::lock_guard<cuda::std::mutex>>, "");
static_assert(!cuda::std::is_convertible_v<cuda::std::mutex&, cuda::std::lock_guard<cuda::std::mutex>>, "");

int main(int, char**)
{
  cuda::std::mutex mutex;
  {
    cuda::std::lock_guard<cuda::std::mutex> guard(mutex);
    assert(!mutex.try_lock());
  }
  assert(mutex.try_lock());
  mutex.unlock();

  mutex.lock();
  {
    cuda::std::lock_guard<cuda::std::mutex> guard(mutex, cuda::std::adopt_lock);
    assert(!mutex.try_lock());
  }
  assert(mutex.try_lock());
  mutex.unlock();
  return 0;
}
