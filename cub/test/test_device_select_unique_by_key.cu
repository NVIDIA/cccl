/******************************************************************************
 * Copyright (c) NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 * Test of DeviceSelect::Unique utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_select.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/util_allocator.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include "test_util.h"

#include <cstdio>
#include <typeinfo>

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose               = false;
int                     g_timing_iterations     = 0;
int                     g_repeat                = 0;
float                   g_device_giga_bandwidth;
CachingDeviceAllocator  g_allocator(true);

// Dispatch types
enum Backend
{
    CUB,        // CUB method
    CDP,        // GPU-based (dynamic parallelism) dispatch to CUB method
};


//---------------------------------------------------------------------
// Dispatch to different CUB DeviceSelect entrypoints
//---------------------------------------------------------------------


/**
 * Dispatch to unique entrypoint
 */
template <typename KeyInputIteratorT, typename ValueInputIteratorT, typename KeyOutputIteratorT, typename ValueOutputIteratorT, typename NumSelectedIteratorT, typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>               /*dispatch_to*/,
    int                         timing_timing_iterations,
    size_t                      */*d_temp_storage_bytes*/,
    cudaError_t                 */*d_cdp_error*/,

    void*                       d_temp_storage,
    size_t                      &temp_storage_bytes,
    KeyInputIteratorT           d_keys_in,
    ValueInputIteratorT         d_values_in,
    KeyOutputIteratorT          d_keys_out,
    ValueOutputIteratorT        d_values_out,
    NumSelectedIteratorT        d_num_selected_out,
    OffsetT                     num_items)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceSelect::UniqueByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_keys_out, d_values_out, d_num_selected_out, num_items);
    }
    return error;
}

//---------------------------------------------------------------------
// CUDA Nested Parallelism Test Kernel
//---------------------------------------------------------------------

#if TEST_CDP == 1

/**
 * Simple wrapper kernel to invoke DeviceSelect
 */
template <int CubBackend,
          typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueOutputIteratorT,
          typename NumSelectedIteratorT,
          typename OffsetT>
__global__ void CDPDispatchKernel(Int2Type<CubBackend> cub_backend,
                                  int                  timing_timing_iterations,
                                  size_t              *d_temp_storage_bytes,
                                  cudaError_t         *d_cdp_error,

                                  void                *d_temp_storage,
                                  size_t               temp_storage_bytes,
                                  KeyInputIteratorT    d_keys_in,
                                  ValueInputIteratorT  d_values_in,
                                  KeyOutputIteratorT   d_keys_out,
                                  ValueOutputIteratorT d_values_out,
                                  NumSelectedIteratorT d_num_selected_out,
                                  OffsetT              num_items)
{
  *d_cdp_error = Dispatch(cub_backend,
                          timing_timing_iterations,
                          d_temp_storage_bytes,
                          d_cdp_error,
                          d_temp_storage,
                          temp_storage_bytes,
                          d_keys_in,
                          d_values_in,
                          d_keys_out,
                          d_values_out,
                          d_num_selected_out,
                          num_items);

  *d_temp_storage_bytes = temp_storage_bytes;
}

/**
 * Dispatch to CDP kernel
 */
template <typename KeyInputIteratorT,
          typename ValueInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueOutputIteratorT,
          typename NumSelectedIteratorT,
          typename OffsetT>
cudaError_t Dispatch(Int2Type<CDP> /*dispatch_to*/,
                     int          timing_timing_iterations,
                     size_t      *d_temp_storage_bytes,
                     cudaError_t *d_cdp_error,

                     void                *d_temp_storage,
                     size_t              &temp_storage_bytes,
                     KeyInputIteratorT    d_keys_in,
                     ValueInputIteratorT  d_values_in,
                     KeyOutputIteratorT   d_keys_out,
                     ValueOutputIteratorT d_values_out,
                     NumSelectedIteratorT d_num_selected_out,
                     OffsetT              num_items)
{
  // Invoke kernel to invoke device-side dispatch
  cudaError_t retval =
    thrust::cuda_cub::launcher::triple_chevron(1, 1, 0, 0)
      .doit(CDPDispatchKernel<CUB,
                              KeyInputIteratorT,
                              ValueInputIteratorT,
                              KeyOutputIteratorT,
                              ValueOutputIteratorT,
                              NumSelectedIteratorT,
                              OffsetT>,
            Int2Type<CUB>{},
            timing_timing_iterations,
            d_temp_storage_bytes,
            d_cdp_error,
            d_temp_storage,
            temp_storage_bytes,
            d_keys_in,
            d_values_in,
            d_keys_out,
            d_values_out,
            d_num_selected_out,
            num_items);
  CubDebugExit(retval);

  // Copy out temp_storage_bytes
  CubDebugExit(cudaMemcpy(&temp_storage_bytes,
                          d_temp_storage_bytes,
                          sizeof(size_t) * 1,
                          cudaMemcpyDeviceToHost));

  // Copy out error
  CubDebugExit(cudaMemcpy(&retval,
                          d_cdp_error,
                          sizeof(cudaError_t) * 1,
                          cudaMemcpyDeviceToHost));
  return retval;
}

#endif // TEST_CDP

//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------


/**
 * Initialize problem
 */
template <typename T>
void Initialize(
    int         entropy_reduction,
    T           *h_in,
    int         num_items,
    int         max_segment)
{
    unsigned int max_int = (unsigned int) -1;

    int key = 0;
    int i = 0;
    while (i < num_items)
    {
        // Select number of repeating occurrences for the current run
        int repeat;
        if (max_segment < 0)
        {
            repeat = num_items;
        }
        else if (max_segment < 2)
        {
            repeat = 1;
        }
        else
        {
            RandomBits(repeat, entropy_reduction);
            repeat = (int) ((double(repeat) * double(max_segment)) / double(max_int));
            repeat = CUB_MAX(1, repeat);
        }

        int j = i;
        while (j < CUB_MIN(i + repeat, num_items))
        {
            InitValue(INTEGER_SEED, h_in[j], key);
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
template <
    typename        KeyInputIteratorT,
    typename        ValueInputIteratorT,
    typename        KeyT,
    typename        ValueT>
int Solve(
    KeyInputIteratorT    h_keys_in,
    ValueInputIteratorT  h_values_in,
    KeyT                 *h_keys_reference,
    ValueT               *h_values_reference,
    int                  num_items)
{
    int num_selected = 0;
    if (num_items > 0)
    {
        h_keys_reference[num_selected] = h_keys_in[0];
        h_values_reference[num_selected] = h_values_in[0];
        num_selected++;
    }

    for (int i = 1; i < num_items; ++i)
    {
        if (h_keys_in[i] != h_keys_in[i - 1])
        {
            h_keys_reference[num_selected] = h_keys_in[i];
            h_values_reference[num_selected] = h_values_in[i];
            num_selected++;
        }
    }

    return num_selected;
}



/**
 * Test DeviceSelect for a given problem input
 */
template <
    Backend             BACKEND,
    typename            KeyInputIteratorT,
    typename            ValueInputIteratorT,
    typename            KeyT,
    typename            ValueT>
void Test(
    KeyInputIteratorT    d_keys_in,
    ValueInputIteratorT  d_values_in,
    KeyT                 *h_keys_reference,
    ValueT               *h_values_reference,
    int                  num_selected,
    int                  num_items)
{
    // Allocate device output array and num selected
    KeyT    *d_keys_out = NULL;
    ValueT  *d_values_out = NULL;
    int     *d_num_selected_out = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys_out, sizeof(KeyT) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values_out, sizeof(ValueT) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_num_selected_out, sizeof(int)));

    // Allocate CDP device arrays
    size_t          *d_temp_storage_bytes = NULL;
    cudaError_t     *d_cdp_error = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage_bytes,  sizeof(size_t) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cdp_error,           sizeof(cudaError_t) * 1));

    // Allocate temporary storage
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), 1, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_keys_out, d_values_out, d_num_selected_out, num_items));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Clear device output array
    CubDebugExit(cudaMemset(d_keys_out, 0, sizeof(KeyT) * num_items));
    CubDebugExit(cudaMemset(d_values_out, 0, sizeof(ValueT) * num_items));
    CubDebugExit(cudaMemset(d_num_selected_out, 0, sizeof(int)));

    // Run warmup/correctness iteration
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), 1, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_keys_out, d_values_out, d_num_selected_out, num_items));

    // Check for correctness (and display results, if specified)
    int compare11 = CompareDeviceResults(h_keys_reference, d_keys_out, num_selected, true, g_verbose);
    int compare12 = CompareDeviceResults(h_values_reference, d_values_out, num_selected, true, g_verbose);
    int compare1 = compare11 && compare12;
    printf("\t Data %s ", compare1 ? "FAIL" : "PASS");

    int compare2 = CompareDeviceResults(&num_selected, d_num_selected_out, 1, true, g_verbose);
    printf("\t Count %s ", compare2 ? "FAIL" : "PASS");

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Performance
    GpuTimer gpu_timer;
    gpu_timer.Start();
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), g_timing_iterations, d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_keys_out, d_values_out, d_num_selected_out, num_items));
    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    // Display performance
    if (g_timing_iterations > 0)
    {
        float avg_millis        = elapsed_millis / g_timing_iterations;
        float giga_rate         = float(num_items) / avg_millis / 1000.0f / 1000.0f;
        float giga_bandwidth    = float((num_items + num_selected) * (sizeof(KeyT) + sizeof(ValueT))) / avg_millis / 1000.0f / 1000.0f;
        printf(", %.3f avg ms, %.3f billion items/s, %.3f logical GB/s, %.1f%% peak", avg_millis, giga_rate, giga_bandwidth, giga_bandwidth / g_device_giga_bandwidth * 100.0);
    }
    printf("\n\n");

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Cleanup
    if (d_keys_out) CubDebugExit(g_allocator.DeviceFree(d_keys_out));
    if (d_values_out) CubDebugExit(g_allocator.DeviceFree(d_values_out));
    if (d_num_selected_out) CubDebugExit(g_allocator.DeviceFree(d_num_selected_out));
    if (d_temp_storage_bytes) CubDebugExit(g_allocator.DeviceFree(d_temp_storage_bytes));
    if (d_cdp_error) CubDebugExit(g_allocator.DeviceFree(d_cdp_error));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    // Correctness asserts
    AssertEquals(0, compare1 | compare2);
}


/**
 * Test DeviceSelect on pointer type
 */
template <
    Backend         BACKEND,
    typename        KeyT,
    typename        ValueT>
void TestPointer(
    int             num_items,
    int             entropy_reduction,
    int             max_segment)
{
    // Allocate host arrays
    KeyT*    h_keys_in        = new KeyT[num_items];
    ValueT*  h_values_in      = new ValueT[num_items];
    KeyT*    h_keys_reference = new KeyT[num_items];
    ValueT*  h_values_reference = new ValueT[num_items];

    // Initialize problem and solution
    Initialize(entropy_reduction, h_keys_in, num_items, max_segment);
    Initialize(entropy_reduction, h_values_in, num_items, max_segment);
    int num_selected = Solve(h_keys_in, h_values_in, h_keys_reference, h_values_reference, num_items);

    printf("\nPointer %s cub::DeviceSelect::Unique %d items, %d selected (avg run length %.3f), %s %d-byte elements, entropy_reduction %d\n",
        (BACKEND == CDP) ? "CDP CUB" : "CUB",
        num_items, num_selected, float(num_items) / num_selected,
        typeid(KeyT).name(),
        (int) sizeof(KeyT),
        entropy_reduction);
    fflush(stdout);

    // Allocate problem device arrays
    KeyT *d_keys_in = NULL;
    ValueT *d_values_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys_in, sizeof(KeyT) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values_in, sizeof(ValueT) * num_items));

    // Initialize device input
    CubDebugExit(cudaMemcpy(d_keys_in, h_keys_in, sizeof(KeyT) * num_items, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_values_in, h_values_in, sizeof(ValueT) * num_items, cudaMemcpyHostToDevice));

    // Run Test
    Test<BACKEND>(d_keys_in, d_values_in, h_keys_reference, h_values_reference, num_selected, num_items);

    // Cleanup
    if (h_keys_in) delete[] h_keys_in;
    if (h_values_in) delete[] h_values_in;
    if (h_keys_reference) delete[] h_keys_reference;
    if (h_values_reference) delete[] h_values_reference;
    if (d_keys_in) CubDebugExit(g_allocator.DeviceFree(d_keys_in));
    if (d_values_in) CubDebugExit(g_allocator.DeviceFree(d_values_in));
}


/**
 * Test DeviceSelect on iterator type
 */
template <
    Backend         BACKEND,
    typename        KeyT,
    typename        ValueT>
void TestIterator(
    int             num_items)
{
    // Use a counting iterator as the input
    CountingInputIterator<KeyT, int> h_keys_in(0);
    CountingInputIterator<ValueT, int> h_values_in(0);

    // Allocate host arrays
    KeyT*    h_keys_reference   = new KeyT[num_items];
    ValueT*  h_values_reference = new ValueT[num_items];

    // Initialize problem and solution
    int num_selected = Solve(h_keys_in, h_values_in, h_keys_reference, h_values_reference, num_items);

    printf("\nIterator %s cub::DeviceSelect::Unique %d items, %d selected (avg run length %.3f), %s %d-byte elements\n",
        (BACKEND == CDP) ? "CDP CUB" : "CUB",
        num_items, num_selected, float(num_items) / num_selected,
        typeid(KeyT).name(),
        (int) sizeof(ValueT));
    fflush(stdout);

    // Run Test
    Test<BACKEND>(h_keys_in, h_values_in, h_keys_reference, h_values_reference, num_selected, num_items);

    // Cleanup
    if (h_keys_reference) delete[] h_keys_reference;
    if (h_values_reference) delete[] h_values_reference;
}


/**
 * Test different gen modes
 */
template <
    Backend         BACKEND,
    typename        KeyT,
    typename        ValueT>
void Test(
    int             num_items)
{
    for (int max_segment = 1; ((max_segment > 0) && (max_segment < num_items)); max_segment *= 11)
    {
        TestPointer<BACKEND, KeyT, ValueT>(num_items, 0, max_segment);
        TestPointer<BACKEND, KeyT, ValueT>(num_items, 2, max_segment);
        TestPointer<BACKEND, KeyT, ValueT>(num_items, 7, max_segment);
    }
}


/**
 * Test different dispatch
 */
template <
    typename        KeyT,
    typename        ValueT>
void TestOp(
    int             num_items)
{
#if TEST_CDP == 0
    Test<CUB, KeyT, ValueT>(num_items);
#elif TEST_CDP == 1
    Test<CDP, KeyT, ValueT>(num_items);
#endif // TEST_CDP
}


/**
 * Test different input sizes
 */
template <
    typename        KeyT,
    typename        ValueT>
void Test(
    int             num_items)
{
    if (num_items < 0)
    {
        TestOp<KeyT, ValueT>(0);
        TestOp<KeyT, ValueT>(1);
        TestOp<KeyT, ValueT>(100);
        TestOp<KeyT, ValueT>(10000);
        TestOp<KeyT, ValueT>(1000000);
    }
    else
    {
        TestOp<KeyT, ValueT>(num_items);
    }
}



//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char** argv)
{
    int num_items           = -1;
    int entropy_reduction   = 0;
    int maxseg              = 1000;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("n", num_items);
    args.GetCmdLineArgument("i", g_timing_iterations);
    args.GetCmdLineArgument("repeat", g_repeat);
    args.GetCmdLineArgument("maxseg", maxseg);
    args.GetCmdLineArgument("entropy", entropy_reduction);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--n=<input items> "
            "[--i=<timing iterations> "
            "[--device=<device-id>] "
            "[--maxseg=<max segment length>]"
            "[--entropy=<segment length bit entropy reduction rounds>]"
            "[--v] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());
    g_device_giga_bandwidth = args.device_giga_bandwidth;
    printf("\n");

    // %PARAM% TEST_CDP cdp 0:1

    // Test different input types
    Test<unsigned char, int>(num_items);
    Test<unsigned char, long>(num_items);
    Test<unsigned short, int>(num_items);
    Test<unsigned short, long>(num_items);
    Test<unsigned int, int>(num_items);
    Test<unsigned int, long>(num_items);
    Test<unsigned long long, int>(num_items);
    Test<unsigned long long, long>(num_items);

    Test<uchar2, uint2>(num_items);
    Test<uchar2, ulonglong2>(num_items);
    Test<ushort2, uint2>(num_items);
    Test<ushort2, ulonglong2>(num_items);
    Test<uint2, uint2>(num_items);
    Test<uint2, ulonglong2>(num_items);
    Test<ulonglong2, uint2>(num_items);
    Test<ulonglong2, ulonglong2>(num_items);

    Test<uchar4, uint4>(num_items);
    Test<uchar4, ulonglong4>(num_items);
    Test<ushort4, uint4>(num_items);
    Test<ushort4, ulonglong4>(num_items);
    Test<uint4, uint4>(num_items);
    Test<uint4, ulonglong4>(num_items);
    Test<ulonglong4, uint4>(num_items);
    Test<ulonglong4, ulonglong4>(num_items);

    Test<TestFoo, TestFoo>(num_items);
    Test<TestFoo, TestBar>(num_items);
    Test<TestFoo, int>(num_items);
    Test<TestBar, TestFoo>(num_items);
    Test<TestBar, TestBar>(num_items);
    Test<TestBar, int>(num_items);

    return 0;
}



