// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/******************************************************************************
 * Simple example of DevicePartition::If().
 *
 * Partitions items from from a sequence of int keys using a
 * section functor (greater-than)
 *
 * To compile using the command line:
 *   nvcc -arch=sm_XX example_device_select_if.cu -I../.. -lcudart -O3
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

namespace
{
bool g_verbose = false; // Whether to display input/output to console

/// Selection functor type
struct GreaterThan
{
  int compare;

  __host__ __device__ __forceinline__ GreaterThan(int compare)
      : compare(compare)
  {}

  __host__ __device__ __forceinline__ bool operator()(const int& a) const
  {
    return (a > compare);
  }
};

//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

/**
 * Initialize problem, setting runs of random length chosen from [1..max_segment]
 */
void Initialize(int* h_in, int num_items, int max_segment)
{
  int key = 0;
  int i   = 0;
  while (i < num_items)
  {
    // Randomly select number of repeating occurrences uniformly from [1..max_segment]
    unsigned short bits;
    RandomBits(bits);
    const int repeat = cuda::std::max(
      1,
      static_cast<int>(static_cast<float>(bits)
                       * (static_cast<float>(max_segment) / cuda::std::numeric_limits<unsigned short>::max())));

    int j = i;
    while (j < cuda::std::min(i + repeat, num_items))
    {
      h_in[j] = key;
      j++;
    }

    i = j;
    key++;
  }

  if (g_verbose)
  {
    printf("Input:\n");
    DisplayResults(h_in, num_items);
    printf("\n\n");
  }
}

/**
 * Solve unique problem
 */
template <typename SelectOp>
int Solve(int* h_in, SelectOp select_op, int* h_reference, int num_items)
{
  int num_selected = 0;
  for (int i = 0; i < num_items; ++i)
  {
    if (select_op(h_in[i]))
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
} // namespace

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
           "[--maxseg=<max segment length>]"
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
  int* h_in        = new int[num_items];
  int* h_reference = new int[num_items];

  // DevicePartition a pivot index
  unsigned int pivot_index;
  unsigned int max_int = (unsigned int) -1;
  RandomBits(pivot_index);
  pivot_index = (unsigned int) ((float(pivot_index) * (float(num_items - 1) / float(max_int))));
  printf("Pivot idx: %d\n", pivot_index);
  fflush(stdout);

  // Initialize problem and solution
  Initialize(h_in, num_items, max_segment);
  GreaterThan select_op(h_in[pivot_index]);

  int num_selected = Solve(h_in, select_op, h_reference, num_items);

  printf("cub::DevicePartition::If %d items, %d selected (avg run length %d), %d-byte elements\n",
         num_items,
         num_selected,
         (num_selected > 0) ? num_items / num_selected : 0,
         (int) sizeof(int));
  fflush(stdout);

  // Allocate and initialize problem device arrays
  auto d_in = cuda::make_buffer<int>(stream, device_memory_resource, h_in, h_in + num_items);

  // Allocate device output array and num selected
  auto d_out              = cuda::make_buffer<int>(stream, device_memory_resource, num_items, cuda::no_init);
  auto d_num_selected_out = cuda::make_buffer<int>(stream, device_memory_resource, 1, cuda::no_init);

  // Allocate temporary storage
  size_t temp_storage_bytes = 0;
  CubDebugExit(DevicePartition::If(
    nullptr, temp_storage_bytes, d_in.data(), d_out.data(), d_num_selected_out.data(), num_items, select_op, stream));
  auto d_temp_storage =
    cuda::make_buffer<cuda::std::byte>(stream, device_memory_resource, temp_storage_bytes, cuda::no_init);

  // Run
  CubDebugExit(DevicePartition::If(
    d_temp_storage.data(),
    temp_storage_bytes,
    d_in.data(),
    d_out.data(),
    d_num_selected_out.data(),
    num_items,
    select_op,
    stream));

  // Check for correctness (and display results, if specified)
  stream.sync();
  int compare = CompareDeviceResults(h_reference, d_out.data(), num_items, true, g_verbose);
  printf("\t Data %s ", compare ? "FAIL" : "PASS");
  compare = compare | CompareDeviceResults(&num_selected, d_num_selected_out.data(), 1, true, g_verbose);
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

  printf("\n\n");

  return 0;
}
