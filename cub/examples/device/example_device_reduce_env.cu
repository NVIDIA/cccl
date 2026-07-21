// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// Simple example of DeviceReduce::Sum() using an environment
/// Sums an array of int keys.

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_reduce.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/stream>

#include <cstdio>

#include "../../test/test_util.h"

bool g_verbose = false; // Whether to display input/output to console

void Initialize(int* h_in, int num_items)
{
  for (int i = 0; i < num_items; ++i)
  {
    h_in[i] = i;
  }

  if (g_verbose)
  {
    printf("Input:\n");
    DisplayResults(h_in, num_items);
    printf("\n\n");
  }
}

void Solve(int* h_in, int& h_reference, int num_items)
{
  for (int i = 0; i < num_items; ++i)
  {
    if (i == 0)
    {
      h_reference = h_in[0];
    }
    else
    {
      h_reference += h_in[i];
    }
  }
}

int main(int argc, char** argv)
{
  // Initialize command line and print usage
  CommandLineArgs args(argc, argv);
  if (args.CheckCmdLineFlag("help"))
  {
    printf("%s "
           "[--n=<input items> "
           "[--device=<device-id>] "
           "[--v] "
           "\n",
           argv[0]);
    std::exit(0);
  }

  // Parse command line options
  int num_items      = 150;
  int device_ordinal = 0;
  g_verbose          = args.CheckCmdLineFlag("v");
  args.GetCmdLineArgument("n", num_items);
  args.GetCmdLineArgument("d", device_ordinal);

  // example-begin env-overload-setup
  // Setup device, stream, memory resource, determinism
  auto device          = cuda::devices[device_ordinal];
  auto stream          = cuda::stream{device};
  auto memory_resource = cuda::device_default_memory_pool(device);
  auto determinism     = cuda::execution::require(cuda::execution::determinism::run_to_run);

  // Create environment
  auto env = cuda::std::execution::env{cuda::stream_ref{stream}, memory_resource, determinism};
  // example-end env-overload-setup

  printf("cub::DeviceReduce::Sum() %d items (%d-byte elements)\n", num_items, (int) sizeof(int));
  fflush(stdout);

  // Allocate host arrays
  std::vector<int> h_in(num_items);
  int h_reference = 0;

  // Initialize problem and solution
  Initialize(h_in.data(), num_items);
  Solve(h_in.data(), h_reference, num_items);

  // Allocate problem device arrays
  auto d_in = thrust::device_vector<int>(num_items, thrust::no_init);

  // Initialize device input
  thrust::copy(h_in.begin(), h_in.end(), d_in.begin());

  // Allocate device output array
  auto d_out = thrust::device_vector<int>(1);

  // example-begin env-overload-run
  // Run
  CubDebugExit(cub::DeviceReduce::Sum(d_in.data(), d_out.data(), num_items, env));
  // example-end env-overload-run

  // Check for correctness
  // Check for correctness (and display results, if specified)
  const int compare =
    CompareDeviceResults(&h_reference, thrust::raw_pointer_cast(d_out.data()), 1, g_verbose, g_verbose);
  printf("\t%s", compare ? "FAIL" : "PASS");
  printf("\n\n");

  return 0;
}
