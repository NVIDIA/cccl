// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/******************************************************************************
 * Simple example of sorting a sequence of keys and values (each pair is a
 * randomly-selected int32 paired with its original offset in the unsorted sequence), and then
 * isolating all maximal, non-trivial (having length > 1) "runs" of duplicates.
 *
 * To compile using the command line:
 *   nvcc -arch=sm_XX example_device_sort_find_non_trivial_runs.cu -I../.. -lcudart -O3
 *
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>

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

bool g_verbose = false; // Whether to display input/output to console

//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

/**
 * Simple key-value pairing for using std::sort on key-value pairs.
 */
template <typename Key, typename Value>
struct Pair
{
  Key key;
  Value value;

  bool operator<(const Pair& b) const
  {
    return (key < b.key);
  }
};

/**
 * Pair ostream operator
 */
template <typename Key, typename Value>
std::ostream& operator<<(std::ostream& os, const Pair<Key, Value>& val)
{
  os << '<' << val.key << ',' << val.value << '>';
  return os;
}

/**
 * Initialize problem
 */
template <typename Key, typename Value>
void Initialize(Key* h_keys, Value* h_values, int num_items, int max_key)
{
  float scale = float(max_key) / float(UINT_MAX);
  for (int i = 0; i < num_items; ++i)
  {
    Key sample;
    RandomBits(sample);
    h_keys[i]   = (max_key == -1) ? i : (Key) (scale * sample);
    h_values[i] = i;
  }

  if (g_verbose)
  {
    printf("Keys:\n");
    DisplayResults(h_keys, num_items);
    printf("\n\n");

    printf("Values:\n");
    DisplayResults(h_values, num_items);
    printf("\n\n");
  }
}

/**
 * Solve sorted non-trivial subrange problem.  Returns the number
 * of non-trivial runs found.
 */
template <typename Key, typename Value>
int Solve(Key* h_keys, Value* h_values, int num_items, int* h_offsets_reference, int* h_lengths_reference)
{
  // Sort

  Pair<Key, Value>* h_pairs = new Pair<Key, Value>[num_items];
  for (int i = 0; i < num_items; ++i)
  {
    h_pairs[i].key   = h_keys[i];
    h_pairs[i].value = h_values[i];
  }

  std::stable_sort(h_pairs, h_pairs + num_items);

  if (g_verbose)
  {
    printf("Sorted pairs:\n");
    DisplayResults(h_pairs, num_items);
    printf("\n\n");
  }

  // Find non-trivial runs

  Key previous  = h_pairs[0].key;
  int length    = 1;
  int num_runs  = 0;
  int run_begin = 0;

  for (int i = 1; i < num_items; ++i)
  {
    if (previous != h_pairs[i].key)
    {
      if (length > 1)
      {
        h_offsets_reference[num_runs] = run_begin;
        h_lengths_reference[num_runs] = length;
        num_runs++;
      }
      length    = 1;
      run_begin = i;
    }
    else
    {
      length++;
    }
    previous = h_pairs[i].key;
  }

  if (length > 1)
  {
    h_offsets_reference[num_runs] = run_begin;
    h_lengths_reference[num_runs] = length;
    num_runs++;
  }

  delete[] h_pairs;

  return num_runs;
}

//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char** argv)
{
  using Key   = unsigned int;
  using Value = int;

  int timing_iterations = 0;
  int num_items         = 40;
  Key max_key           = 20; // Max item

  // Initialize command line
  CommandLineArgs args(argc, argv);
  g_verbose = args.CheckCmdLineFlag("v");
  args.GetCmdLineArgument("n", num_items);
  args.GetCmdLineArgument("maxkey", max_key);
  args.GetCmdLineArgument("i", timing_iterations);

  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
    printf("%s "
           "[--i=<timing iterations> "
           "[--n=<input items, default 40> "
           "[--maxkey=<max key, default 20 (use -1 to test only unique keys)>]"
           "[--v] "
           "\n",
           argv[0]);
    exit(0);
  }

  // Set up device, stream, and memory resource
  const auto device                 = cuda::devices[0];
  const auto stream                 = cuda::stream{device};
  const auto device_memory_resource = cuda::device_default_memory_pool(device);

  // Allocate host arrays (problem and reference solution)

  Key* h_keys              = new Key[num_items];
  Value* h_values          = new Value[num_items];
  int* h_offsets_reference = new int[num_items];
  int* h_lengths_reference = new int[num_items];

  // Initialize key-value pairs and compute reference solution (sort them, and identify non-trivial runs)
  printf("Computing reference solution on CPU for %d items (max key %d)\n", num_items, max_key);
  fflush(stdout);

  Initialize(h_keys, h_values, num_items, static_cast<int>(max_key));
  int num_runs = Solve(h_keys, h_values, num_items, h_offsets_reference, h_lengths_reference);

  printf("%d non-trivial runs\n", num_runs);
  fflush(stdout);

  // Repeat for performance timing
  float elapsed_millis     = 0.0;
  float elapsed_rle_millis = 0.0;
  for (int i = 0; i <= timing_iterations; ++i)
  {
    // Allocate and initialize device arrays for sorting
    auto d_keys_0   = cuda::make_buffer<Key>(stream, device_memory_resource, h_keys, h_keys + num_items);
    auto d_keys_1   = cuda::make_buffer<Key>(stream, device_memory_resource, num_items, cuda::no_init);
    auto d_values_0 = cuda::make_buffer<Value>(stream, device_memory_resource, h_values, h_values + num_items);
    auto d_values_1 = cuda::make_buffer<Value>(stream, device_memory_resource, num_items, cuda::no_init);
    DoubleBuffer<Key> d_keys{d_keys_0.data(), d_keys_1.data()};
    DoubleBuffer<Value> d_values{d_values_0.data(), d_values_1.data()};

    // Start timer
    const auto start_event = stream.record_timed_event();

    // Allocate temporary storage for sorting
    size_t temp_storage_bytes = 0;
    void* d_temp_storage      = nullptr;
    CubDebugExit(DeviceRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items, 0, sizeof(Key) * 8, stream.get()));
    auto sort_temp_storage =
      cuda::make_buffer<cuda::std::byte>(stream, device_memory_resource, temp_storage_bytes, cuda::no_init);

    // Do the sort
    CubDebugExit(DeviceRadixSort::SortPairs(
      sort_temp_storage.data(), temp_storage_bytes, d_keys, d_values, num_items, 0, sizeof(Key) * 8, stream.get()));

    // Start timer
    const auto rle_start_event = stream.record_timed_event();

    // Allocate device arrays for enumerating non-trivial runs
    auto d_offsets_out = cuda::make_buffer<int>(stream, device_memory_resource, num_items, cuda::no_init);
    auto d_lengths_out = cuda::make_buffer<int>(stream, device_memory_resource, num_items, cuda::no_init);
    auto d_num_runs    = cuda::make_buffer<int>(stream, device_memory_resource, 1, cuda::no_init);

    // Allocate temporary storage for isolating non-trivial runs
    d_temp_storage     = nullptr;
    temp_storage_bytes = 0;
    CubDebugExit(DeviceRunLengthEncode::NonTrivialRuns(
      d_temp_storage,
      temp_storage_bytes,
      d_keys.d_buffers[d_keys.selector],
      d_offsets_out.data(),
      d_lengths_out.data(),
      d_num_runs.data(),
      num_items,
      stream.get()));
    auto rle_temp_storage =
      cuda::make_buffer<cuda::std::byte>(stream, device_memory_resource, temp_storage_bytes, cuda::no_init);

    // Do the isolation
    CubDebugExit(DeviceRunLengthEncode::NonTrivialRuns(
      rle_temp_storage.data(),
      temp_storage_bytes,
      d_keys.d_buffers[d_keys.selector],
      d_offsets_out.data(),
      d_lengths_out.data(),
      d_num_runs.data(),
      num_items,
      stream.get()));

    //
    // Hypothetically do stuff with the original key-indices corresponding to non-trivial runs of identical keys
    //

    // Stop sort timer
    const auto end_event = stream.record_timed_event();
    stream.sync();

    if (i == 0)
    {
      // First iteration is a warmup: // Check for correctness (and display results, if specified)

      printf("\nRUN OFFSETS: \n");
      int compare = CompareDeviceResults(h_offsets_reference, d_offsets_out.data(), num_runs, true, g_verbose);
      printf("\t\t %s ", compare ? "FAIL" : "PASS");

      printf("\nRUN LENGTHS: \n");
      compare |= CompareDeviceResults(h_lengths_reference, d_lengths_out.data(), num_runs, true, g_verbose);
      printf("\t\t %s ", compare ? "FAIL" : "PASS");

      printf("\nNUM RUNS: \n");
      compare |= CompareDeviceResults(&num_runs, d_num_runs.data(), 1, true, g_verbose);
      printf("\t\t %s ", compare ? "FAIL" : "PASS");

      AssertEquals(0, compare);
    }
    else
    {
      elapsed_millis += static_cast<float>((end_event - start_event).count()) / 1.0e6f;
      elapsed_rle_millis += static_cast<float>((end_event - rle_start_event).count()) / 1.0e6f;
    }
  }

  // Host cleanup
  if (h_keys)
  {
    delete[] h_keys;
  }
  if (h_values)
  {
    delete[] h_values;
  }
  if (h_offsets_reference)
  {
    delete[] h_offsets_reference;
  }
  if (h_lengths_reference)
  {
    delete[] h_lengths_reference;
  }

  printf("\n\n");

  if (timing_iterations > 0)
  {
    printf("%d timing iterations, average time to sort and isolate non-trivial duplicates: %.3f ms (%.3f ms spent in "
           "RLE isolation)\n",
           timing_iterations,
           elapsed_millis / static_cast<float>(timing_iterations),
           elapsed_rle_millis / static_cast<float>(timing_iterations));
  }

  return 0;
}
