// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>

#include <cuda/buffer>
#include <cuda/cccl_runtime_test_helper.cuh>
#include <cuda/std/initializer_list>
#include <cuda/stream>

#include <unittest/unittest.h>

void TestDeviceBufferShuffleCudaStreams()
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto buffer       = cuda::make_device_buffer<int>(stream, device, 5, cuda::no_init);
  const auto policy = thrust::cuda::par_nosync.on(stream.get());

  thrust::sequence(policy, buffer.begin(), buffer.end());

  thrust::default_random_engine rng{2};
  thrust::shuffle(policy, buffer.begin(), buffer.end(), rng);
  thrust::sort(policy, buffer.begin(), buffer.end());

  test_runtime::assert_equal(stream, buffer, {0, 1, 2, 3, 4});
}
DECLARE_UNITTEST(TestDeviceBufferShuffleCudaStreams);

void TestDeviceBufferSortCudaStreams()
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto buffer       = cuda::make_device_buffer<int>(stream, device, cuda::std::initializer_list<int>{3, 1, 4, 0, 2});
  const auto policy = thrust::cuda::par_nosync.on(stream.get());

  thrust::sort(policy, buffer.begin(), buffer.end());

  test_runtime::assert_equal(stream, buffer, {0, 1, 2, 3, 4});
}
DECLARE_UNITTEST(TestDeviceBufferSortCudaStreams);
