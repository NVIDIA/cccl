/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
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
 * Test of DeviceSelect::If and DevicePartition::If utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_select.cuh>
#include <cub/device/device_partition.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/util_allocator.cuh>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>

#include "test_util.h"

#include <cstdio>
#include <typeinfo>

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose               = false;
int                     g_timing_iterations     = 0;
float                   g_device_giga_bandwidth;
CachingDeviceAllocator  g_allocator(true);

// Dispatch types
enum Backend
{
    CUB,        // CUB method
    CDP,        // GPU-based (dynamic parallelism) dispatch to CUB method
};


// Selection functor type
template <typename T>
struct LessThan
{
    T compare;

    __host__ __device__ __forceinline__
    LessThan(T compare) : compare(compare) {}

    __host__ __device__ __forceinline__
    bool operator()(const T &a) const {
        return (a < compare);
    }
};

//---------------------------------------------------------------------
// Dispatch to different CUB DeviceSelect entrypoints
//---------------------------------------------------------------------


/**
 * Dispatch to select if entrypoint
 */
template <typename InputIteratorT, typename FlagIteratorT, typename SelectOpT, typename OutputIteratorT, typename NumSelectedIteratorT, typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>               /*dispatch_to*/,
    Int2Type<false>             /*is_flagged*/,
    Int2Type<false>             /*is_partition*/,
    int                         timing_timing_iterations,
    size_t*                     /*d_temp_storage_bytes*/,
    cudaError_t*                /*d_cdp_error*/,

    void*                       d_temp_storage,
    size_t&                     temp_storage_bytes,
    InputIteratorT              d_in,
    FlagIteratorT               /*d_flags*/,
    OutputIteratorT             d_out,
    NumSelectedIteratorT        d_num_selected_out,
    OffsetT                     num_items,
    SelectOpT                   select_op)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items, select_op);
    }
    return error;
}


/**
 * Dispatch to partition if entrypoint
 */
template <typename InputIteratorT, typename FlagIteratorT, typename SelectOpT, typename OutputIteratorT, typename NumSelectedIteratorT, typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>               /*dispatch_to*/,
    Int2Type<false>             /*is_flagged*/,
    Int2Type<true>              /*is_partition*/,
    int                         timing_timing_iterations,
    size_t*                     /*d_temp_storage_bytes*/,
    cudaError_t*                /*d_cdp_error*/,

    void*                       d_temp_storage,
    size_t&                     temp_storage_bytes,
    InputIteratorT              d_in,
    FlagIteratorT               /*d_flags*/,
    OutputIteratorT             d_out,
    NumSelectedIteratorT        d_num_selected_out,
    OffsetT                     num_items,
    SelectOpT                   select_op)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DevicePartition::If(d_temp_storage, temp_storage_bytes, d_in, d_out, d_num_selected_out, num_items, select_op);
    }
    return error;
}


/**
 * Dispatch to select flagged entrypoint
 */
template <typename InputIteratorT, typename FlagIteratorT, typename SelectOpT, typename OutputIteratorT, typename NumSelectedIteratorT, typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>               /*dispatch_to*/,
    Int2Type<true>              /*is_flagged*/,
    Int2Type<false>             /*partition*/,
    int                         timing_timing_iterations,
    size_t*                     /*d_temp_storage_bytes*/,
    cudaError_t*                /*d_cdp_error*/,

    void*                       d_temp_storage,
    size_t&                     temp_storage_bytes,
    InputIteratorT              d_in,
    FlagIteratorT               d_flags,
    OutputIteratorT             d_out,
    NumSelectedIteratorT        d_num_selected_out,
    OffsetT                     num_items,
    SelectOpT                   /*select_op*/)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items);
    }
    return error;
}


/**
 * Dispatch to partition flagged entrypoint
 */
template <typename InputIteratorT, typename FlagIteratorT, typename SelectOpT, typename OutputIteratorT, typename NumSelectedIteratorT, typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>               /*dispatch_to*/,
    Int2Type<true>              /*is_flagged*/,
    Int2Type<true>              /*partition*/,
    int                         timing_timing_iterations,
    size_t*                     /*d_temp_storage_bytes*/,
    cudaError_t*                /*d_cdp_error*/,

    void*                       d_temp_storage,
    size_t&                     temp_storage_bytes,
    InputIteratorT              d_in,
    FlagIteratorT               d_flags,
    OutputIteratorT             d_out,
    NumSelectedIteratorT        d_num_selected_out,
    OffsetT                     num_items,
    SelectOpT                   /*select_op*/)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items);
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
          typename IsFlaggedTag,
          typename IsPartitionTag,
          typename InputIteratorT,
          typename FlagIteratorT,
          typename SelectOpT,
          typename OutputIteratorT,
          typename NumSelectedIteratorT,
          typename OffsetT>
__global__ void CDPDispatchKernel(Int2Type<CubBackend> cub_backend,
                                  IsFlaggedTag         is_flagged,
                                  IsPartitionTag       is_partition,
                                  int                  timing_timing_iterations,
                                  size_t              *d_temp_storage_bytes,
                                  cudaError_t         *d_cdp_error,

                                  void                *d_temp_storage,
                                  size_t               temp_storage_bytes,
                                  InputIteratorT       d_in,
                                  FlagIteratorT        d_flags,
                                  OutputIteratorT      d_out,
                                  NumSelectedIteratorT d_num_selected_out,
                                  OffsetT              num_items,
                                  SelectOpT            select_op)
{
  *d_cdp_error = Dispatch(cub_backend,
                          is_flagged,
                          is_partition,
                          timing_timing_iterations,
                          d_temp_storage_bytes,
                          d_cdp_error,
                          d_temp_storage,
                          temp_storage_bytes,
                          d_in,
                          d_flags,
                          d_out,
                          d_num_selected_out,
                          num_items,
                          select_op);

  *d_temp_storage_bytes = temp_storage_bytes;
}

/**
 * Dispatch to CDP kernel
 */
template <typename IsFlaggedTag,
          typename IsPartitionTag,
          typename InputIteratorT,
          typename FlagIteratorT,
          typename SelectOpT,
          typename OutputIteratorT,
          typename NumSelectedIteratorT,
          typename OffsetT>
cudaError_t Dispatch(Int2Type<CDP> /*dispatch_to*/,
                     IsFlaggedTag   is_flagged,
                     IsPartitionTag is_partition,
                     int            timing_timing_iterations,
                     size_t        *d_temp_storage_bytes,
                     cudaError_t   *d_cdp_error,

                     void                *d_temp_storage,
                     size_t              &temp_storage_bytes,
                     InputIteratorT       d_in,
                     FlagIteratorT        d_flags,
                     OutputIteratorT      d_out,
                     NumSelectedIteratorT d_num_selected_out,
                     OffsetT              num_items,
                     SelectOpT            select_op)
{
  // Invoke kernel to invoke device-side dispatch
  cudaError_t retval =
    thrust::cuda_cub::launcher::triple_chevron(1, 1, 0, 0)
      .doit(CDPDispatchKernel<CUB,
                              IsFlaggedTag,
                              IsPartitionTag,
                              InputIteratorT,
                              FlagIteratorT,
                              SelectOpT,
                              OutputIteratorT,
                              NumSelectedIteratorT,
                              OffsetT>,
            Int2Type<CUB>{},
            is_flagged,
            is_partition,
            timing_timing_iterations,
            d_temp_storage_bytes,
            d_cdp_error,
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_flags,
            d_out,
            d_num_selected_out,
            num_items,
            select_op);
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
    T*  h_in,
    int num_items)
{
    for (int i = 0; i < num_items; ++i)
    {
        // Initialize each item to a randomly selected value from [0..126]
        unsigned int value;
        RandomBits(value, 0, 0, 7);
        if (value == 127)
            value = 126;
        InitValue(INTEGER_SEED, h_in[i], value);
    }

    if (g_verbose)
    {
        printf("Input:\n");
        DisplayResults(h_in, num_items);
        printf("\n\n");
    }
}


/**
 * Solve selection problem (and set corresponding flags)
 */
template <
    typename        InputIteratorT,
    typename        FlagIteratorT,
    typename        SelectOpT,
    typename        T>
int Solve(
    InputIteratorT  h_in,
    SelectOpT       select_op,
    T*              h_reference,
    FlagIteratorT   h_flags,
    int             num_items)
{
    int num_selected = 0;
    for (int i = 0; i < num_items; ++i)
    {
        if ((h_flags[i] = select_op(h_in[i])))
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



/**
 * Test DeviceSelect for a given problem input
 */
template <
    Backend             BACKEND,
    bool                IS_FLAGGED,
    bool                IS_PARTITION,
    typename            DeviceInputIteratorT,
    typename            FlagT,
    typename            SelectOpT,
    typename            T>
void Test(
    DeviceInputIteratorT    d_in,
    FlagT*                  h_flags,
    SelectOpT               select_op,
    T*                      h_reference,
    int                     num_selected,
    int                     num_items)
{
    // Allocate device flags, output, and num-selected
    FlagT*      d_flags = NULL;
    T*          d_out = NULL;
    int*        d_num_selected_out = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_flags, sizeof(FlagT) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(T) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_num_selected_out, sizeof(int)));

    // Allocate CDP device arrays
    size_t*         d_temp_storage_bytes = NULL;
    cudaError_t*    d_cdp_error = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage_bytes,  sizeof(size_t) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cdp_error,           sizeof(cudaError_t) * 1));

    // Allocate temporary storage
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), Int2Type<IS_FLAGGED>(), Int2Type<IS_PARTITION>(), 1, d_temp_storage_bytes, d_cdp_error,
    d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items, select_op));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Copy flags and clear device output array
    CubDebugExit(cudaMemcpy(d_flags, h_flags, sizeof(FlagT) * num_items, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemset(d_out, 0, sizeof(T) * num_items));
    CubDebugExit(cudaMemset(d_num_selected_out, 0, sizeof(int)));

    // Run warmup/correctness iteration
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), Int2Type<IS_FLAGGED>(), Int2Type<IS_PARTITION>(), 1, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items, select_op));

    // Check for correctness (and display results, if specified)
    int compare1 = (IS_PARTITION) ?
        CompareDeviceResults(h_reference, d_out, num_items, true, g_verbose) :
        CompareDeviceResults(h_reference, d_out, num_selected, true, g_verbose);
    printf("\t Data %s\n", compare1 ? "FAIL" : "PASS");

    int compare2 = CompareDeviceResults(&num_selected, d_num_selected_out, 1, true, g_verbose);
    printf("\t Count %s\n", compare2 ? "FAIL" : "PASS");

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Performance
    GpuTimer gpu_timer;
    gpu_timer.Start();
    CubDebugExit(Dispatch(Int2Type<BACKEND>(), Int2Type<IS_FLAGGED>(), Int2Type<IS_PARTITION>(), g_timing_iterations, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items, select_op));
    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    // Display performance
    if (g_timing_iterations > 0)
    {
        float   avg_millis          = elapsed_millis / g_timing_iterations;
        float   giga_rate           = float(num_items) / avg_millis / 1000.0f / 1000.0f;
        int     num_output_items    = (IS_PARTITION) ? num_items : num_selected;
        int     num_flag_items      = (IS_FLAGGED) ? num_items : 0;
        size_t  num_bytes           = sizeof(T) * (num_items + num_output_items) + sizeof(FlagT) * num_flag_items;
        float   giga_bandwidth      = float(num_bytes) / avg_millis / 1000.0f / 1000.0f;

        printf(", %.3f avg ms, %.3f billion items/s, %.3f logical GB/s, %.1f%% peak", avg_millis, giga_rate, giga_bandwidth, giga_bandwidth / g_device_giga_bandwidth * 100.0);
    }
    printf("\n\n");

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Cleanup
    if (d_flags) CubDebugExit(g_allocator.DeviceFree(d_flags));
    if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
    if (d_num_selected_out) CubDebugExit(g_allocator.DeviceFree(d_num_selected_out));
    if (d_temp_storage_bytes) CubDebugExit(g_allocator.DeviceFree(d_temp_storage_bytes));
    if (d_cdp_error) CubDebugExit(g_allocator.DeviceFree(d_cdp_error));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    // Correctness asserts
    AssertEquals(0, compare1 | compare2);
}


/**
 * Test on pointer type
 */
template <
    Backend         BACKEND,
    bool            IS_FLAGGED,
    bool            IS_PARTITION,
    typename        T>
void TestPointer(
    int             num_items,
    float           select_ratio)
{
    typedef char FlagT;

    // Allocate host arrays
    T*      h_in        = new T[num_items];
    FlagT*  h_flags     = new FlagT[num_items];
    T*      h_reference = new T[num_items];

    // Initialize input
    Initialize(h_in, num_items);

    // Select a comparison value that is select_ratio through the space of [0,127]
    T compare;
    if (select_ratio <= 0.0)
        InitValue(INTEGER_SEED, compare, 0);        // select none
    else if (select_ratio >= 1.0)
        InitValue(INTEGER_SEED, compare, 127);      // select all
    else
        InitValue(INTEGER_SEED, compare, int(double(double(127) * select_ratio)));

    LessThan<T> select_op(compare);
    int num_selected = Solve(h_in, select_op, h_reference, h_flags, num_items);

    if (g_verbose) std::cout << "\nComparison item: " << compare << "\n";
    printf("\nPointer %s cub::%s::%s %d items, %d selected (select ratio %.3f), %s %d-byte elements\n",
        (IS_PARTITION) ? "DevicePartition" : "DeviceSelect",
        (IS_FLAGGED) ? "Flagged" : "If",
        (BACKEND == CDP) ? "CDP CUB" : "CUB",
        num_items, num_selected, float(num_selected) / num_items, typeid(T).name(), (int) sizeof(T));
    fflush(stdout);

    // Allocate problem device arrays
    T *d_in = NULL;

    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(T) * num_items));

    // Initialize device input
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(T) * num_items, cudaMemcpyHostToDevice));

    // Run Test
    Test<BACKEND, IS_FLAGGED, IS_PARTITION>(d_in, h_flags, select_op, h_reference, num_selected, num_items);

    // Cleanup
    if (h_in) delete[] h_in;
    if (h_reference) delete[] h_reference;
    if (h_flags) delete[] h_flags;
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
}


/**
 * Test on iterator type
 */
template <
    Backend         BACKEND,
    bool            IS_FLAGGED,
    bool            IS_PARTITION,
    typename        T>
void TestIterator(
    int             num_items,
    float           select_ratio)
{
    typedef char FlagT;

    // Allocate host arrays
    T*      h_reference = new T[num_items];
    FlagT*  h_flags = new FlagT[num_items];

    // Use counting iterator as the input
    CountingInputIterator<T, int> h_in(0);

    // Select a comparison value that is select_ratio through the space of [0,127]
    T compare;
    if (select_ratio <= 0.0)
        InitValue(INTEGER_SEED, compare, 0);        // select none
    else if (select_ratio >= 1.0)
        InitValue(INTEGER_SEED, compare, 127);      // select all
    else
        InitValue(INTEGER_SEED, compare, int(double(double(127) * select_ratio)));

    LessThan<T> select_op(compare);
    int num_selected = Solve(h_in, select_op, h_reference, h_flags, num_items);

    if (g_verbose) std::cout << "\nComparison item: " << compare << "\n";
    printf("\nIterator %s cub::%s::%s %d items, %d selected (select ratio %.3f), %s %d-byte elements\n",
        (IS_PARTITION) ? "DevicePartition" : "DeviceSelect",
        (IS_FLAGGED) ? "Flagged" : "If",
        (BACKEND == CDP) ? "CDP CUB" : "CUB",
        num_items, num_selected, float(num_selected) / num_items, typeid(T).name(), (int) sizeof(T));
    fflush(stdout);

    // Run Test
    Test<BACKEND, IS_FLAGGED, IS_PARTITION>(h_in, h_flags, select_op, h_reference, num_selected, num_items);

    // Cleanup
    if (h_reference) delete[] h_reference;
    if (h_flags) delete[] h_flags;
}


/**
 * Test different selection ratios
 */
template <
    Backend         BACKEND,
    bool            IS_FLAGGED,
    bool            IS_PARTITION,
    typename        T>
void Test(
    int             num_items)
{
    for (float select_ratio = 0.0f; select_ratio <= 1.0f; select_ratio += 0.2f)
    {
        TestPointer<BACKEND, IS_FLAGGED, IS_PARTITION, T>(num_items, select_ratio);
    }
}


/**
 * Test (select vs. partition) and (flagged vs. functor)
 */
template <
    Backend         BACKEND,
    typename        T>
void TestMethod(
    int             num_items)
{
    // Functor
    Test<BACKEND, false, false, T>(num_items);
    Test<BACKEND, false, true, T>(num_items);

    // Flagged
    Test<BACKEND, true, false, T>(num_items);
    Test<BACKEND, true, true, T>(num_items);
}


/**
 * Test different dispatch
 */
template <
    typename        T>
void TestOp(
    int             num_items)
{
#if TEST_CDP == 0
    TestMethod<CUB, T>(num_items);
#elif TEST_CDP == 1
    TestMethod<CDP, T>(num_items);
#endif // TEST_CDP
}


/**
 * Test different input sizes
 */
template <typename T>
void Test(
    int             num_items)
{
    if (num_items < 0)
    {
        TestOp<T>(0);
        TestOp<T>(1);
        TestOp<T>(100);
        TestOp<T>(10000);
        TestOp<T>(1000000);
    }
    else
    {
        TestOp<T>(num_items);
    }
}

template<class T0, class T1>
struct pair_to_col_t
{
    __host__ __device__ T0 operator()(const thrust::tuple<T0, T1> &in)
    {
        return thrust::get<0>(in);
    }
};

template<class T0, class T1>
struct select_t {
    __host__ __device__ bool operator()(const thrust::tuple<T0, T1> &in) {
        return static_cast<T1>(thrust::get<0>(in)) > thrust::get<1>(in);
    }
};

template <typename T0, typename T1>
void TestMixedOp(int num_items)
{
    const T0 target_value = static_cast<T0>(42);
    thrust::device_vector<T0> col_a(num_items, target_value);
    thrust::device_vector<T1> col_b(num_items, static_cast<T1>(4.2));

    thrust::device_vector<T0> result(num_items);

    auto in = thrust::make_zip_iterator(col_a.begin(), col_b.begin());
    auto out = thrust::make_transform_output_iterator(result.begin(), pair_to_col_t<T0, T1>{});

    void *d_tmp_storage {};
    std::size_t tmp_storage_size{};
    cub::DeviceSelect::If(
            d_tmp_storage, tmp_storage_size,
            in, out, thrust::make_discard_iterator(),
            num_items, select_t<T0, T1>{});

    thrust::device_vector<char> tmp_storage(tmp_storage_size);
    d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

    cub::DeviceSelect::If(
            d_tmp_storage, tmp_storage_size,
            in, out, thrust::make_discard_iterator(),
            num_items, select_t<T0, T1>{});

    AssertEquals(num_items, thrust::count(result.begin(), result.end(), target_value));
}

/**
 * Test different input sizes
 */
template <typename T0, typename T1>
void TestMixed(int num_items)
{
    if (num_items < 0)
    {
        TestMixedOp<T0, T1>(0);
        TestMixedOp<T0, T1>(1);
        TestMixedOp<T0, T1>(100);
        TestMixedOp<T0, T1>(10000);
        TestMixedOp<T0, T1>(1000000);
    }
    else
    {
        TestMixedOp<T0, T1>(num_items);
    }
}

void TestFlagsNormalization()
{
  const int num_items = 1024 * 1024;
  thrust::device_vector<int> result(num_items);

  void *d_tmp_storage{};
  std::size_t tmp_storage_size{};
  CubDebugExit(
    cub::DeviceSelect::Flagged(d_tmp_storage,
                               tmp_storage_size,
                               cub::CountingInputIterator<int>(0),      // in
                               cub::CountingInputIterator<int>(1),      // flags
                               thrust::raw_pointer_cast(result.data()), // out
                               thrust::make_discard_iterator(), // num_out
                               num_items));

  thrust::device_vector<char> tmp_storage(tmp_storage_size);
  d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  CubDebugExit(
    cub::DeviceSelect::Flagged(d_tmp_storage,
                               tmp_storage_size,
                               cub::CountingInputIterator<int>(0),      // in
                               cub::CountingInputIterator<int>(1),      // flags
                               thrust::raw_pointer_cast(result.data()), // out
                               thrust::make_discard_iterator(), // num_out
                               num_items));

  AssertTrue(thrust::equal(result.begin(),
                           result.end(),
                           thrust::make_counting_iterator(0)));
}

void TestFlagsAliasingInPartition()
{
  int h_items[]{0, 1, 0, 2, 0, 3, 0, 4, 0, 5};
  constexpr int num_items = sizeof(h_items) / sizeof(h_items[0]);

  int *d_in{};
  int *d_out{};

  CubDebugExit(g_allocator.DeviceAllocate((void **)&d_in, sizeof(h_items)));
  CubDebugExit(g_allocator.DeviceAllocate((void **)&d_out, sizeof(h_items)));

  CubDebugExit(
    cudaMemcpy(d_in, h_items, sizeof(h_items), cudaMemcpyHostToDevice));

  // alias flags and keys
  int *d_flags = d_in;

  void *d_tmp_storage{};
  std::size_t tmp_storage_size{};

  CubDebugExit(
    cub::DevicePartition::Flagged(d_tmp_storage,
                                  tmp_storage_size,
                                  d_in,
                                  d_flags,
                                  d_out,
                                  thrust::make_discard_iterator(), // num_out
                                  num_items));

  thrust::device_vector<char> tmp_storage(tmp_storage_size);
  d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  CubDebugExit(
    cub::DevicePartition::Flagged(d_tmp_storage,
                                  tmp_storage_size,
                                  d_in,
                                  d_flags,
                                  d_out,
                                  thrust::make_discard_iterator(), // num_out
                                  num_items));

  AssertTrue(thrust::equal(thrust::device,
                           d_out,
                           d_out + num_items / 2,
                           thrust::make_counting_iterator(1)));

  AssertEquals(
    thrust::count(thrust::device, d_out + num_items / 2, d_out + num_items, 0),
    num_items / 2);

  CubDebugExit(g_allocator.DeviceFree(d_out));
  CubDebugExit(g_allocator.DeviceFree(d_in));
}

struct Odd
{
  __host__ __device__ bool operator()(int v) const { return v % 2; }
};

void TestIfInPlace()
{
  const int num_items = 4 * 1024 * 1024;
  const int num_iters = 42;

  thrust::device_vector<int> num_out(1);
  thrust::device_vector<int> data(num_items);
  thrust::device_vector<int> reference(num_items);
  thrust::device_vector<int> reference_out(1);

  thrust::sequence(data.begin(), data.end());

  Odd op{};

  int *d_num_out = thrust::raw_pointer_cast(num_out.data());
  int *d_data = thrust::raw_pointer_cast(data.data());
  int *d_reference = thrust::raw_pointer_cast(reference.data());
  int *d_reference_out = thrust::raw_pointer_cast(reference_out.data());

  void *d_tmp_storage{};
  std::size_t tmp_storage_size{};

  CubDebugExit(
    cub::DeviceSelect::If(d_tmp_storage,
                          tmp_storage_size,
                          d_data,
                          d_num_out,
                          num_items,
                          op));

  thrust::device_vector<char> tmp_storage(tmp_storage_size);
  d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  thrust::default_random_engine g{};

  for (int iter = 0; iter < num_iters; iter++)
  {
    thrust::shuffle(data.begin(), data.end(), g);

    CubDebugExit(
      cub::DeviceSelect::If(d_tmp_storage,
                            tmp_storage_size,
                            d_data,
                            d_reference,
                            d_reference_out,
                            num_items,
                            op));

    CubDebugExit(
      cub::DeviceSelect::If(d_tmp_storage,
                            tmp_storage_size,
                            d_data,
                            d_num_out,
                            num_items,
                            op));

    AssertEquals(num_out, reference_out);
    const int num_selected = num_out[0];

    const bool match_reference = thrust::equal(reference.begin(),
                                               reference.begin() + num_selected,
                                               data.begin());
    AssertTrue(match_reference);
  }
}

void TestFlaggedInPlace()
{
  const int num_items = 4 * 1024 * 1024;
  const int num_iters = 42;

  thrust::device_vector<int> num_out(1);
  thrust::device_vector<int> data(num_items);
  thrust::device_vector<bool> flags(num_items);

  int h_num_out{};
  int *d_num_out = thrust::raw_pointer_cast(num_out.data());
  int *d_data = thrust::raw_pointer_cast(data.data());
  bool *d_flags = thrust::raw_pointer_cast(flags.data());

  void *d_tmp_storage{};
  std::size_t tmp_storage_size{};

  CubDebugExit(
    cub::DeviceSelect::Flagged(d_tmp_storage,
                               tmp_storage_size,
                               d_data,
                               d_flags,
                               d_num_out,
                               num_items));

  thrust::device_vector<char> tmp_storage(tmp_storage_size);
  d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  thrust::default_random_engine g{};

  for (int iter = 0; iter < num_iters; iter++)
  {
    const int num_selected = RandomValue(num_items);

    thrust::sequence(data.begin(), data.end());
    thrust::fill(flags.begin(), flags.begin() + num_selected, true);
    thrust::fill(flags.begin() + num_selected, flags.end(), false);
    thrust::shuffle(flags.begin(), flags.end(), g);

    CubDebugExit(
      cub::DeviceSelect::Flagged(d_tmp_storage,
                                 tmp_storage_size,
                                 d_data,
                                 d_flags,
                                 d_num_out,
                                 num_items));

    cudaMemcpy(&h_num_out, d_num_out, sizeof(int), cudaMemcpyDeviceToHost);

    AssertEquals(num_selected, h_num_out);

    auto selection_perm_begin = thrust::make_permutation_iterator(flags.begin(),
                                                                  data.begin());
    auto selection_perm_end = selection_perm_begin + num_selected;

    AssertEquals(num_selected,
                 thrust::count(selection_perm_begin, selection_perm_end, true));
  }
}

void TestFlaggedInPlaceWithAliasedFlags()
{
  const int num_items = 1024 * 1024;
  const int num_iters = 42;

  thrust::device_vector<int> num_out(1);
  thrust::device_vector<int> data(num_items);
  thrust::device_vector<int> reference(num_items);
  thrust::device_vector<int> flags(num_items);

  int h_num_out{};
  int *d_num_out = thrust::raw_pointer_cast(num_out.data());
  int *d_data = thrust::raw_pointer_cast(data.data());
  int *d_flags = d_data; // alias
  int *d_allocated_flags = thrust::raw_pointer_cast(data.data());
  int *d_reference = thrust::raw_pointer_cast(reference.data());

  void *d_tmp_storage{};
  std::size_t tmp_storage_size{};

  CubDebugExit(
    cub::DeviceSelect::Flagged(d_tmp_storage,
                               tmp_storage_size,
                               d_data,
                               d_flags,
                               d_num_out,
                               num_items));

  thrust::device_vector<char> tmp_storage(tmp_storage_size);
  d_tmp_storage = thrust::raw_pointer_cast(tmp_storage.data());

  thrust::default_random_engine g{};

  for (int iter = 0; iter < num_iters; iter++)
  {
    const int num_selected = RandomValue(num_items);

    thrust::sequence(data.begin(), data.begin() + num_selected, 1);
    thrust::fill(data.begin() + num_selected, data.end(), 0);
    thrust::shuffle(data.begin(), data.end(), g);

    CubDebugExit(
      cub::DeviceSelect::Flagged(d_tmp_storage,
                                 tmp_storage_size,
                                 d_data,      // in
                                 d_allocated_flags,
                                 d_reference, // out
                                 d_num_out,
                                 num_items));

    CubDebugExit(
      cub::DeviceSelect::Flagged(d_tmp_storage,
                                 tmp_storage_size,
                                 d_data,
                                 d_flags,
                                 d_num_out,
                                 num_items));

    cudaMemcpy(&h_num_out, d_num_out, sizeof(int), cudaMemcpyDeviceToHost);

    AssertEquals(num_selected, h_num_out);

    const bool match_reference = thrust::equal(reference.begin(),
                                               reference.begin() + num_selected,
                                               data.begin());
    AssertTrue(match_reference);
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
    float select_ratio      = 0.5;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("n", num_items);
    args.GetCmdLineArgument("i", g_timing_iterations);
    args.GetCmdLineArgument("ratio", select_ratio);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--n=<input items> "
            "[--i=<timing iterations> "
            "[--device=<device-id>] "
            "[--ratio=<selection ratio, default 0.5>] "
            "[--v] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());
    g_device_giga_bandwidth = args.device_giga_bandwidth;
    printf("\n");

    // %PARAM% TEST_CDP cdp 0:1

    TestFlagsAliasingInPartition();

    TestFlaggedInPlace();
    TestFlaggedInPlaceWithAliasedFlags();
    TestIfInPlace();

    Test<unsigned char>(num_items);
    Test<unsigned short>(num_items);
    Test<unsigned int>(num_items);
    Test<unsigned long long>(num_items);

    Test<uchar2>(num_items);
    Test<ushort2>(num_items);
    Test<uint2>(num_items);
    Test<ulonglong2>(num_items);

    Test<uchar4>(num_items);
    Test<ushort4>(num_items);
    Test<uint4>(num_items);
    Test<ulonglong4>(num_items);

    Test<TestFoo>(num_items);
    Test<TestBar>(num_items);

    TestMixed<int, double>(num_items);
    TestFlagsNormalization();

    return 0;
}



