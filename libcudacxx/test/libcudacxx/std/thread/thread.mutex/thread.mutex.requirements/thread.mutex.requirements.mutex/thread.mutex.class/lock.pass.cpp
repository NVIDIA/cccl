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

// void lock();

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

cuda::std::mutex blocking_mutex;
int blocking_ready        = 0;
int blocking_held_by_main = 1;

void blocking_lock()
{
  blocking_ready = 1;
  blocking_mutex.lock();
  assert(blocking_held_by_main == 0);
  blocking_mutex.unlock();
}

cuda::std::mutex exclusive_mutex;
int exclusive_counter = 0;

void exclusive_section()
{
  exclusive_mutex.lock();
  ++exclusive_counter;
  assert(exclusive_counter == 1);
  --exclusive_counter;
  exclusive_mutex.unlock();
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, ({
                 {
                   cuda::std::mutex mutex;
                   mutex.lock();
                   mutex.unlock();
}

{
  blocking_mutex.lock();
  std::thread worker(blocking_lock);
  while (blocking_ready == 0)
  {
  }
  blocking_held_by_main = 0;
  blocking_mutex.unlock();
  worker.join();
}

{
  std::thread first(exclusive_section);
  std::thread second(exclusive_section);
  first.join();
  second.join();
  assert(exclusive_counter == 0);
}
}))
  return 0;
}
