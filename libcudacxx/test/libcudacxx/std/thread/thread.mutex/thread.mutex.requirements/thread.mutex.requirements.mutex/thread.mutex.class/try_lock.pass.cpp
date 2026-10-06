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

// bool try_lock();

// UNSUPPORTED: nvrtc
// UNSUPPORTED: libcpp-has-no-threads
// The checks below use std::thread.

// UNSUPPORTED: pre-sm-70
// Device atomics used by mutex require sm_70.

// UNSUPPORTED: force-tile
// error: asm statement is not supported

#include <cuda/std/cassert>
#include <cuda/std/mutex>

#include <thread>

#include "test_macros.h"

cuda::std::mutex held_mutex;

void try_lock_while_held()
{
  for (int i = 0; i != 10; ++i)
  {
    assert(!held_mutex.try_lock());
  }
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, ({
                 {
                   cuda::std::mutex mutex;
                   assert(mutex.try_lock());
                   assert(!mutex.try_lock());
                   mutex.unlock();
                   assert(mutex.try_lock());
                   mutex.unlock();
}

{
  held_mutex.lock();
  std::thread worker(try_lock_while_held);
  worker.join();
  held_mutex.unlock();
  assert(held_mutex.try_lock());
  held_mutex.unlock();
}
}))
  return 0;
}
