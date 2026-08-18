// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/******************************************************************************
 * Simple example of DeviceScan::ExclusiveSum().
 *
 * Computes an exclusive sum of int keys.
 *
 * To compile using the command line:
 *   nvcc -arch=sm_XX example_device_scan.cu -I../.. -lcudart -O3
 *
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_scan.cuh>

#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/std/cstddef>
#include <cuda/stream>

#include <cstdio>

#include "../../test/test_util.h"

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and aliases
//---------------------------------------------------------------------

bool g_verbose = false; // Whether to display input/output to console

//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

/**
 * Initialize problem
 */
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

/**
 * Solve exclusive-scan problem
 */
int Solve(int* h_in, int* h_reference, int num_items)
{
  int inclusive = 0;
  int aggregate = 0;

  for (int i = 0; i < num_items; ++i)
  {
    h_reference[i] = inclusive;
    inclusive += h_in[i];
    aggregate += h_in[i];
  }

  return aggregate;
}

//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char** argv)
{
  int num_items = 150;

  // Initialize command line
  CommandLineArgs args(argc, argv);
  g_verbose = args.CheckCmdLineFlag("v");
  args.GetCmdLineArgument("n", num_items);

  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
    printf("%s "
           "[--n=<input items> "
           "[--v] "
           "\n",
           argv[0]);
    exit(0);
  }

  // Set up device, stream, and memory resource
  const auto device                 = cuda::devices[0];
  const auto stream                 = cuda::stream{device};
  const auto device_memory_resource = cuda::device_default_memory_pool(device);

  printf("cub::DeviceScan::ExclusiveSum %d items (%d-byte elements)\n", num_items, (int) sizeof(int));
  fflush(stdout);

  // Allocate host arrays
  int* h_in        = new int[num_items];
  int* h_reference = new int[num_items];

  // Initialize problem and solution
  Initialize(h_in, num_items);
  Solve(h_in, h_reference, num_items);

  // Allocate and initialize problem device arrays
  auto d_in = cuda::make_buffer<int>(stream, device_memory_resource, h_in, h_in + num_items);

  // Allocate device output array
  auto d_out = cuda::make_buffer<int>(stream, device_memory_resource, num_items, cuda::no_init);

  // Allocate temporary storage
  size_t temp_storage_bytes = 0;
  CubDebugExit(
    DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, d_in.data(), d_out.data(), num_items, stream.get()));
  auto d_temp_storage =
    cuda::make_buffer<cuda::std::byte>(stream, device_memory_resource, temp_storage_bytes, cuda::no_init);

  // Run
  CubDebugExit(DeviceScan::ExclusiveSum(
    d_temp_storage.data(), temp_storage_bytes, d_in.data(), d_out.data(), num_items, stream.get()));

  // Check for correctness (and display results, if specified)
  stream.sync();
  int compare = CompareDeviceResults(h_reference, d_out.data(), num_items, true, g_verbose);
  printf("\t%s", compare ? "FAIL" : "PASS");
  AssertEquals(0, compare);

  // Cleanup
  if (h_in)
  {
    delete[] h_in;
  }
  if (h_reference)
  {
    delete[] h_reference;
  }

  printf("\n\n");

  return 0;
}
