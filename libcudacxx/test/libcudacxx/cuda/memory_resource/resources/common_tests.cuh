//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <testing.cuh>
#include <utility.cuh>

template <typename ResourceType>
void test_deallocate_async([[maybe_unused]] ResourceType& resource)
{
  /* disable until we move the launch API to libcudacxx
  cudax::stream stream{cuda::device_ref{0}};
  test::pinned<int> i(0);
  cuda::atomic_ref atomic_i(*i);

  int* allocation = static_cast<int*>(resource.allocate_sync(sizeof(int)));

  cudax::launch(stream, test::one_thread_dims, test::spin_until_80{}, i.get());
  cudax::launch(stream, test::one_thread_dims, test::assign_42{}, allocation);
  cudax::launch(stream, test::one_thread_dims, test::verify_42{}, allocation);

  resource.deallocate(stream, allocation, sizeof(int));

  atomic_i.store(80);
  stream.sync();
  */
}
