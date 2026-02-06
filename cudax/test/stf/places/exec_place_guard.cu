//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Unit tests for exec_place_guard RAII helper
 */

#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

#include <thread>
#include <vector>

using namespace cuda::experimental::stf;

// Test 1: Basic guard functionality - single device switch
void test_basic_guard(int ndevs)
{
  if (ndevs < 2)
  {
    fprintf(stderr, "test_basic_guard: skipping (need at least 2 devices).\n");
    return;
  }

  // Start on device 0
  cuda_safe_call(cudaSetDevice(0));

  int dev_before = -1;
  cuda_safe_call(cudaGetDevice(&dev_before));
  EXPECT(dev_before == 0);

  // Use guard to switch to device 1
  {
    exec_place_guard guard(exec_place::device(1));

    int dev_inside = -1;
    cuda_safe_call(cudaGetDevice(&dev_inside));
    EXPECT(dev_inside == 1);
  }

  // After guard destruction, should be back to device 0
  int dev_after = -1;
  cuda_safe_call(cudaGetDevice(&dev_after));
  EXPECT(dev_after == 0);
}

// Test 2: Nested guards
void test_nested_guards(int ndevs)
{
  if (ndevs < 3)
  {
    fprintf(stderr, "test_nested_guards: skipping (need at least 3 devices).\n");
    return;
  }

  cuda_safe_call(cudaSetDevice(0));

  {
    exec_place_guard guard1(exec_place::device(1));

    int dev = -1;
    cuda_safe_call(cudaGetDevice(&dev));
    EXPECT(dev == 1);

    {
      exec_place_guard guard2(exec_place::device(2));

      cuda_safe_call(cudaGetDevice(&dev));
      EXPECT(dev == 2);
    }

    // After inner guard destruction, should be back to device 1
    cuda_safe_call(cudaGetDevice(&dev));
    EXPECT(dev == 1);
  }

  // After outer guard destruction, should be back to device 0
  int dev = -1;
  cuda_safe_call(cudaGetDevice(&dev));
  EXPECT(dev == 0);
}

// Test 3: Guard with host execution place (should be no-op for device)
void test_host_place_guard(int ndevs)
{
  if (ndevs < 1)
  {
    fprintf(stderr, "test_host_place_guard: skipping (need at least 1 device).\n");
    return;
  }

  cuda_safe_call(cudaSetDevice(0));

  int dev_before = -1;
  cuda_safe_call(cudaGetDevice(&dev_before));

  {
    exec_place_guard guard(exec_place::host());

    // Device should remain unchanged when using host place
    int dev_inside = -1;
    cuda_safe_call(cudaGetDevice(&dev_inside));
    EXPECT(dev_inside == dev_before);
  }

  int dev_after = -1;
  cuda_safe_call(cudaGetDevice(&dev_after));
  EXPECT(dev_after == dev_before);
}

// Test 4: Guard with same device (should be efficient no-op)
void test_same_device_guard(int ndevs)
{
  if (ndevs < 1)
  {
    fprintf(stderr, "test_same_device_guard: skipping (need at least 1 device).\n");
    return;
  }

  cuda_safe_call(cudaSetDevice(0));

  {
    exec_place_guard guard(exec_place::device(0));

    int dev = -1;
    cuda_safe_call(cudaGetDevice(&dev));
    EXPECT(dev == 0);
  }

  int dev = -1;
  cuda_safe_call(cudaGetDevice(&dev));
  EXPECT(dev == 0);
}

// Test 5: Stream creation within guard scope
void test_stream_creation_in_guard(int ndevs)
{
  if (ndevs < 2)
  {
    fprintf(stderr, "test_stream_creation_in_guard: skipping (need at least 2 devices).\n");
    return;
  }

  cuda_safe_call(cudaSetDevice(0));

  cudaStream_t stream;

  {
    exec_place_guard guard(exec_place::device(1));
    cuda_safe_call(cudaStreamCreate(&stream));
  }

#if _CCCL_CTK_AT_LEAST(12, 8)
  // Verify stream was created on device 1 (cudaStreamGetDevice requires CUDA 12.8+)
  int stream_dev = -1;
  cuda_safe_call(cudaStreamGetDevice(stream, &stream_dev));
  EXPECT(stream_dev == 1);
#endif // _CCCL_CTK_AT_LEAST(12, 8)

  // Clean up (need to be on correct device for some operations)
  {
    exec_place_guard guard(exec_place::device(1));
    cuda_safe_call(cudaStreamDestroy(stream));
  }
}

// Test 6: Multiple threads with guards
void test_multithreaded_guards(int ndevs)
{
  if (ndevs < 2)
  {
    fprintf(stderr, "test_multithreaded_guards: skipping (need at least 2 devices).\n");
    return;
  }

  const int num_threads = ::std::min(ndevs, 4);
  ::std::vector<bool> results(num_threads, false);

  ::std::vector<::std::thread> threads;
  for (int i = 0; i < num_threads; ++i)
  {
    threads.emplace_back([&results, i, ndevs]() {
      // Each thread starts on device 0
      cuda_safe_call(cudaSetDevice(0));

      int target_dev = i % ndevs;

      {
        exec_place_guard guard(exec_place::device(target_dev));

        int dev = -1;
        cuda_safe_call(cudaGetDevice(&dev));
        if (dev != target_dev)
        {
          return;
        }

        // Do some work
        cudaStream_t stream;
        cuda_safe_call(cudaStreamCreate(&stream));
        cuda_safe_call(cudaStreamSynchronize(stream));
        cuda_safe_call(cudaStreamDestroy(stream));
      }

      // Verify restoration
      int dev_after = -1;
      cuda_safe_call(cudaGetDevice(&dev_after));
      if (dev_after != 0)
      {
        return;
      }

      results[i] = true;
    });
  }

  for (auto& th : threads)
  {
    th.join();
  }

  for (int i = 0; i < num_threads; ++i)
  {
    EXPECT(results[i]);
  }
}

// Test 7: Stress test with multiple iterations
void test_stress_iterations(int ndevs)
{
  if (ndevs < 2)
  {
    fprintf(stderr, "test_stress_iterations: skipping (need at least 2 devices).\n");
    return;
  }

  cuda_safe_call(cudaSetDevice(0));

  const int iterations = 100;
  for (int iter = 0; iter < iterations; ++iter)
  {
    int target_dev = iter % ndevs;

    {
      exec_place_guard guard(exec_place::device(target_dev));

      int dev = -1;
      cuda_safe_call(cudaGetDevice(&dev));
      EXPECT(dev == target_dev);
    }

    int dev_after = -1;
    cuda_safe_call(cudaGetDevice(&dev_after));
    EXPECT(dev_after == 0);
  }
}

int main()
{
  // Initialize CUDA
  cuda_safe_call(cudaFree(0));

  int ndevs;
  cuda_safe_call(cudaGetDeviceCount(&ndevs));

  test_basic_guard(ndevs);
  test_nested_guards(ndevs);
  test_host_place_guard(ndevs);
  test_same_device_guard(ndevs);
  test_stream_creation_in_guard(ndevs);
  test_multithreaded_guards(ndevs);
  test_stress_iterations(ndevs);

  return 0;
}
