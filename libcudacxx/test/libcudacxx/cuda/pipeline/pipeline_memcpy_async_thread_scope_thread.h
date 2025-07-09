//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-70

#include <cuda/pipeline>

#include "cuda_space_selector.h"
#include "large_type.h"

template <class T, template <typename, typename> class SourceSelector, template <typename, typename> class DestSelector>
__host__ __device__ __noinline__ void test_fully_specialized()
{
  SourceSelector<T, constructor_initializer> source_sel;
  typename DestSelector<T, constructor_initializer>::template offsetted<decltype(source_sel)::shared_offset> dest_sel;

  T* source = source_sel.construct(static_cast<T>(12));
  T* dest   = dest_sel.construct(static_cast<T>(0));

  auto pipe = cuda::make_pipeline();

  assert(*source == 12);
  assert(*dest == 0);

  pipe.producer_acquire();
  cuda::memcpy_async(dest, source, sizeof(T), pipe);
  pipe.producer_commit();
  pipe.consumer_wait();

  assert(*source == 12);
  assert(*dest == 12);

  pipe.consumer_release();

  *source = 24;

  pipe.producer_acquire();
  cuda::memcpy_async(static_cast<void*>(dest), static_cast<void*>(source), sizeof(T), pipe);
  pipe.producer_commit();
  pipe.consumer_wait();

  assert(*source == 24);
  assert(*dest == 24);

  pipe.consumer_release();
}

template <class T, template <typename, typename> class SourceSelector>
__host__ __device__ __noinline__ void test_select_destination()
{
  test_fully_specialized<T, SourceSelector, local_memory_selector>();
  NV_IF_TARGET(NV_IS_DEVICE,
               (test_fully_specialized<T, SourceSelector, shared_memory_selector>();
                test_fully_specialized<T, SourceSelector, global_memory_selector>();))
}

template <class T>
__host__ __device__ __noinline__ void test_select_source()
{
  test_select_destination<T, local_memory_selector>();
  NV_IF_TARGET(
    NV_IS_DEVICE,
    (test_select_destination<T, shared_memory_selector>(); test_select_destination<T, global_memory_selector>();))
}
