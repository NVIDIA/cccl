// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/******************************************************************************
 * Simple example of DeviceRadixSort::SortPairs().
 *
 * Sorts an array of float keys paired with a corresponding array of int values.
 *
 * To compile using the command line:
 *   nvcc -arch=sm_XX example_device_radix_sort.cu -I../.. -lcudart -O3
 *
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_radix_sort.cuh>

#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/std/cstddef>
#include <cuda/stream>

#include <algorithm>
#include <cstdio>

#include "../../test/test_util.h"

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and aliases
//---------------------------------------------------------------------

namespace
{
bool g_verbose = false; // Whether to display input/output to console

//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

/**
 * Simple key-value pairing for floating point types.
 * Treats positive and negative zero as equivalent.
 */
struct Pair
{
  float key;
  int value;

  bool operator<(const Pair& b) const
  {
    return key < b.key;
  }
};

/**
 * Initialize key-value sorting problem.
 */
void Initialize(float* h_keys, int* h_values, float* h_reference_keys, int* h_reference_values, int num_items)
{
  Pair* h_pairs = new Pair[num_items];

  for (int i = 0; i < num_items; ++i)
  {
    RandomBits(h_keys[i]);
    RandomBits(h_values[i]);
    h_pairs[i].key   = h_keys[i];
    h_pairs[i].value = h_values[i];
  }

  if (g_verbose)
  {
    printf("Input keys:\n");
    DisplayResults(h_keys, num_items);
    printf("\n\n");

    printf("Input values:\n");
    DisplayResults(h_values, num_items);
    printf("\n\n");
  }

  std::stable_sort(h_pairs, h_pairs + num_items);

  for (int i = 0; i < num_items; ++i)
  {
    h_reference_keys[i]   = h_pairs[i].key;
    h_reference_values[i] = h_pairs[i].value;
  }

  delete[] h_pairs;
}

//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------
} // namespace

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

  printf("cub::DeviceRadixSort::SortPairs() %d items (%d-byte keys %d-byte values)\n",
         num_items,
         int(sizeof(float)),
         int(sizeof(int)));
  fflush(stdout);

  // Allocate host arrays
  float* h_keys           = new float[num_items];
  float* h_reference_keys = new float[num_items];
  int* h_values           = new int[num_items];
  int* h_reference_values = new int[num_items];

  // Initialize problem and solution on host
  Initialize(h_keys, h_values, h_reference_keys, h_reference_values, num_items);

  // Allocate and initialize device arrays
  auto d_keys_0   = cuda::make_buffer<float>(stream, device_memory_resource, h_keys, h_keys + num_items);
  auto d_keys_1   = cuda::make_buffer<float>(stream, device_memory_resource, num_items, cuda::no_init);
  auto d_values_0 = cuda::make_buffer<int>(stream, device_memory_resource, h_values, h_values + num_items);
  auto d_values_1 = cuda::make_buffer<int>(stream, device_memory_resource, num_items, cuda::no_init);
  DoubleBuffer<float> d_keys{d_keys_0.data(), d_keys_1.data()};
  DoubleBuffer<int> d_values{d_values_0.data(), d_values_1.data()};

  // Allocate temporary storage
  size_t temp_storage_bytes = 0;
  CubDebugExit(DeviceRadixSort::SortPairs(
    nullptr, temp_storage_bytes, d_keys, d_values, num_items, 0, sizeof(float) * 8, stream.get()));
  auto d_temp_storage =
    cuda::make_buffer<cuda::std::byte>(stream, device_memory_resource, temp_storage_bytes, cuda::no_init);

  // Run
  CubDebugExit(DeviceRadixSort::SortPairs(
    d_temp_storage.data(), temp_storage_bytes, d_keys, d_values, num_items, 0, sizeof(float) * 8, stream.get()));

  // Check for correctness (and display results, if specified)
  stream.sync();
  int compare = CompareDeviceResults(h_reference_keys, d_keys.Current(), num_items, true, g_verbose);
  printf("\t Compare keys (selector %d): %s\n", d_keys.selector, compare ? "FAIL" : "PASS");
  AssertEquals(0, compare);
  compare = CompareDeviceResults(h_reference_values, d_values.Current(), num_items, true, g_verbose);
  printf("\t Compare values (selector %d): %s\n", d_values.selector, compare ? "FAIL" : "PASS");
  AssertEquals(0, compare);

  // Cleanup
  if (h_keys)
  {
    delete[] h_keys;
  }
  if (h_reference_keys)
  {
    delete[] h_reference_keys;
  }
  if (h_values)
  {
    delete[] h_values;
  }
  if (h_reference_values)
  {
    delete[] h_reference_values;
  }

  printf("\n\n");

  return 0;
}
