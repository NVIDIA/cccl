// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/******************************************************************************
 * Simple example of DevicePartition::Flagged().
 *
 * Partition flagged items from from a sequence of int keys using a
 * corresponding sequence of unsigned char flags.
 *
 * To compile using the command line:
 *   nvcc -arch=sm_XX example_device_partition_flagged.cu -I../.. -lcudart -O3
 *
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_partition.cuh>

#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/std/cstddef>
#include <cuda/std/limits>
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
 * Initialize problem, setting flags at distances of random length
 * chosen from [1..max_segment]
 */
void Initialize(int* h_in, unsigned char* h_flags, int num_items, int max_segment)
{
  int key = 0;
  int i   = 0;
  while (i < num_items)
  {
    // Select number of repeating occurrences
    unsigned short bits;
    RandomBits(bits);
    const int repeat = cuda::std::max(
      1,
      static_cast<int>(static_cast<float>(bits)
                       * (static_cast<float>(max_segment) / cuda::std::numeric_limits<unsigned short>::max())));

    int j = i;
    while (j < cuda::std::min(i + repeat, num_items))
    {
      h_flags[j] = 0;
      h_in[j]    = key;
      j++;
    }

    h_flags[i] = 1;
    i          = j;
    key++;
  }

  if (g_verbose)
  {
    printf("Input:\n");
    DisplayResults(h_in, num_items);
    printf("Flags:\n");
    DisplayResults(h_flags, num_items);
    printf("\n\n");
  }
}

/**
 * Solve unique problem
 */
int Solve(int* h_in, unsigned char* h_flags, int* h_reference, int num_items)
{
  int num_selected = 0;
  for (int i = 0; i < num_items; ++i)
  {
    if (h_flags[i])
    {
      h_reference[num_selected] = h_in[i];
      num_selected++;
    }
    else
    {
      h_reference[num_items - (i - num_selected) - 1] = h_in[i];
    }
  }

  return num_selected;
}

//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char** argv)
{
  int num_items   = 150;
  int max_segment = 40; // Maximum segment length

  // Initialize command line
  CommandLineArgs args(argc, argv);
  g_verbose = args.CheckCmdLineFlag("v");
  args.GetCmdLineArgument("n", num_items);
  args.GetCmdLineArgument("maxseg", max_segment);

  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
    printf("%s "
           "[--n=<input items> "
           "[--maxseg=<max segment length>] "
           "[--v] "
           "\n",
           argv[0]);
    exit(0);
  }

  // Set up device, stream, and memory resource
  const auto device                 = cuda::devices[0];
  const auto stream                 = cuda::stream{device};
  const auto device_memory_resource = cuda::device_default_memory_pool(device);

  // Allocate host arrays
  int* h_in              = new int[num_items];
  int* h_reference       = new int[num_items];
  unsigned char* h_flags = new unsigned char[num_items];

  // Initialize problem and solution
  Initialize(h_in, h_flags, num_items, max_segment);
  int num_selected = Solve(h_in, h_flags, h_reference, num_items);

  printf("cub::DevicePartition::Flagged %d items, %d selected (avg distance %d), %d-byte elements\n",
         num_items,
         num_selected,
         (num_selected > 0) ? num_items / num_selected : 0,
         (int) sizeof(int));
  fflush(stdout);

  // Allocate and initialize problem device arrays
  auto d_in    = cuda::make_buffer<int>(stream, device_memory_resource, h_in, h_in + num_items);
  auto d_flags = cuda::make_buffer<unsigned char>(stream, device_memory_resource, h_flags, h_flags + num_items);

  // Allocate device output array and num selected
  auto d_out              = cuda::make_buffer<int>(stream, device_memory_resource, num_items, cuda::no_init);
  auto d_num_selected_out = cuda::make_buffer<int>(stream, device_memory_resource, 1, cuda::no_init);

  // Allocate temporary storage
  size_t temp_storage_bytes = 0;
  CubDebugExit(DevicePartition::Flagged(
    nullptr,
    temp_storage_bytes,
    d_in.data(),
    d_flags.data(),
    d_out.data(),
    d_num_selected_out.data(),
    num_items,
    stream));
  auto d_temp_storage =
    cuda::make_buffer<cuda::std::byte>(stream, device_memory_resource, temp_storage_bytes, cuda::no_init);

  // Run
  CubDebugExit(DevicePartition::Flagged(
    d_temp_storage.data(),
    temp_storage_bytes,
    d_in.data(),
    d_flags.data(),
    d_out.data(),
    d_num_selected_out.data(),
    num_items,
    stream));

  // Check for correctness (and display results, if specified)
  stream.sync();
  int compare = CompareDeviceResults(h_reference, d_out.data(), num_items, true, g_verbose);
  printf("\t Data %s ", compare ? "FAIL" : "PASS");
  compare |= CompareDeviceResults(&num_selected, d_num_selected_out.data(), 1, true, g_verbose);
  printf("\t Count %s ", compare ? "FAIL" : "PASS");
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
  if (h_flags)
  {
    delete[] h_flags;
  }

  printf("\n\n");

  return 0;
}
