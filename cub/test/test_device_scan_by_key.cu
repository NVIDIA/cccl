/******************************************************************************
 * Copyright (c) 2021 NVIDIA CORPORATION.  All rights reserved.
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
 * Test of DeviceScan utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_scan.cuh>
#include <cub/iterator/constant_input_iterator.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/discard_output_iterator.cuh>
#include <cub/iterator/transform_input_iterator.cuh>
#include <cub/util_allocator.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include "test_util.h"

#include <cstdio>
#include <limits>
#include <typeinfo>

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose           = false;
int                     g_timing_iterations = 0;
double                  g_device_giga_bandwidth;
CachingDeviceAllocator  g_allocator(true);

// Dispatch types
enum Backend
{
    CUB,        // CUB method
    CDP,        // GPU-based (dynamic parallelism) dispatch to CUB method
};


enum AliasMode
{
  AliasNone,  // output is allocated
  AliasKeys,  // output is an alias of input keys
  AliasValues // output is an alias of input values
};


/**
 * \brief WrapperFunctor (for precluding test-specialized dispatch to *Sum variants)
 */
template<typename OpT>
struct WrapperFunctor
{
    OpT op;

    WrapperFunctor(OpT op) : op(op) {}

    template <typename T, typename U>
    __host__ __device__ __forceinline__ auto operator()(const T &a, const U &b) const
      -> decltype(op(a, b))
    {
        return static_cast<T>(op(a, b));
    }
};

/**
 * \brief DivideByFiveFunctor (used by TestIterator)
 */
template<typename OutputT>
struct DivideByFiveFunctor
{
    template <typename T>
    __host__ __device__ __forceinline__ OutputT operator()(const T &a) const
    {
        return static_cast<OutputT>(a / 5);
    }
};

/**
 * \brief Mod2Equality (used for non-bool keys to make keys more likely to equal each other)
 */
struct Mod2Equality
{
    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const
    {
        return (a % 2) == (b % 2);
    }
};


//---------------------------------------------------------------------
// Dispatch to different CUB DeviceScan entrypoints
//---------------------------------------------------------------------

/**
 * Dispatch to exclusive scan entrypoint
 */
template <typename IsPrimitiveT, typename KeysInputIteratorT, typename ValuesInputIteratorT, typename ValuesOutputIteratorT, typename ScanOpT, typename InitialValueT, typename OffsetT, typename EqualityOpT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>         /*dispatch_to*/,
    IsPrimitiveT          /*is_primitive*/,
    int                   timing_timing_iterations,
    size_t                */*d_temp_storage_bytes*/,
    cudaError_t           */*d_cdp_error*/,

    void*                 d_temp_storage,
    size_t&               temp_storage_bytes,
    KeysInputIteratorT    d_keys_in,
    ValuesInputIteratorT  d_values_in,
    ValuesOutputIteratorT d_values_out,
    ScanOpT               scan_op,
    InitialValueT         initial_value,
    OffsetT               num_items,
    EqualityOpT           equality_op)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceScan::ExclusiveScanByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_values_out, scan_op, initial_value, num_items, equality_op);
    }
    return error;
}


/**
 * Dispatch to exclusive sum entrypoint
 */
template <typename KeysInputIteratorT, typename ValuesInputIteratorT, typename ValuesOutputIteratorT, typename InitialValueT, typename OffsetT, typename EqualityOpT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>         /*dispatch_to*/,
    Int2Type<true>        /*is_primitive*/,
    int                   timing_timing_iterations,
    size_t                */*d_temp_storage_bytes*/,
    cudaError_t           */*d_cdp_error*/,

    void*                 d_temp_storage,
    size_t&               temp_storage_bytes,
    KeysInputIteratorT    d_keys_in,
    ValuesInputIteratorT  d_values_in,
    ValuesOutputIteratorT d_values_out,
    Sum                   /*scan_op*/,
    InitialValueT         /*initial_value*/,
    OffsetT               num_items,
    EqualityOpT           equality_op)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceScan::ExclusiveSumByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_values_out, num_items, equality_op);
    }
    return error;
}


/**
 * Dispatch to inclusive scan entrypoint
 */
template <typename IsPrimitiveT, typename KeysInputIteratorT, typename ValuesInputIteratorT, typename ValuesOutputIteratorT, typename ScanOpT, typename OffsetT, typename EqualityOpT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>         /*dispatch_to*/,
    IsPrimitiveT          /*is_primitive*/,
    int                   timing_timing_iterations,
    size_t                */*d_temp_storage_bytes*/,
    cudaError_t           */*d_cdp_error*/,

    void*                 d_temp_storage,
    size_t&               temp_storage_bytes,
    KeysInputIteratorT    d_keys_in,
    ValuesInputIteratorT  d_values_in,
    ValuesOutputIteratorT d_values_out,
    ScanOpT               scan_op,
    NullType              /*initial_value*/,
    OffsetT               num_items,
    EqualityOpT           equality_op)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceScan::InclusiveScanByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_values_out, scan_op, num_items, equality_op);
    }
    return error;
}

/**
 * Dispatch to inclusive sum entrypoint
 */
template <typename KeysInputIteratorT, typename ValuesInputIteratorT, typename ValuesOutputIteratorT, typename OffsetT, typename EqualityOpT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>         /*dispatch_to*/,
    Int2Type<true>        /*is_primitive*/,
    int                   timing_timing_iterations,
    size_t                */*d_temp_storage_bytes*/,
    cudaError_t           */*d_cdp_error*/,

    void*                 d_temp_storage,
    size_t&               temp_storage_bytes,
    KeysInputIteratorT    d_keys_in,
    ValuesInputIteratorT  d_values_in,
    ValuesOutputIteratorT d_values_out,
    Sum                   /*scan_op*/,
    NullType              /*initial_value*/,
    OffsetT               num_items,
    EqualityOpT           equality_op)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceScan::InclusiveSumByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_values_out, num_items, equality_op);
    }
    return error;
}

//---------------------------------------------------------------------
// CUDA Nested Parallelism Test Kernel
//---------------------------------------------------------------------

#if TEST_CDP == 1

/**
 * Simple wrapper kernel to invoke DeviceScan
 */
template <int CubBackend,
          typename IsPrimitiveT,
          typename KeysInputIteratorT,
          typename ValuesInputIteratorT,
          typename ValuesOutputIteratorT,
          typename ScanOpT,
          typename InitialValueT,
          typename OffsetT,
          typename EqualityOpT>
__global__ void CDPDispatchKernel(Int2Type<CubBackend> cub_backend,
                                  IsPrimitiveT         is_primitive,
                                  int                  timing_timing_iterations,
                                  size_t              *d_temp_storage_bytes,
                                  cudaError_t         *d_cdp_error,

                                  void                 *d_temp_storage,
                                  size_t                temp_storage_bytes,
                                  KeysInputIteratorT    d_keys_in,
                                  ValuesInputIteratorT  d_values_in,
                                  ValuesOutputIteratorT d_values_out,
                                  ScanOpT               scan_op,
                                  InitialValueT         initial_value,
                                  OffsetT               num_items,
                                  EqualityOpT           equality_op)
{
  *d_cdp_error = Dispatch(cub_backend,
                          is_primitive,
                          timing_timing_iterations,
                          d_temp_storage_bytes,
                          d_cdp_error,
                          d_temp_storage,
                          temp_storage_bytes,
                          d_keys_in,
                          d_values_in,
                          d_values_out,
                          scan_op,
                          initial_value,
                          num_items,
                          equality_op);

  *d_temp_storage_bytes = temp_storage_bytes;
}

/**
 * Dispatch to CDP kernel
 */
template <typename IsPrimitiveT,
          typename KeysInputIteratorT,
          typename ValuesInputIteratorT,
          typename ValuesOutputIteratorT,
          typename ScanOpT,
          typename InitialValueT,
          typename OffsetT,
          typename EqualityOpT>
cudaError_t Dispatch(Int2Type<CDP> /*dispatch_to*/,
                     IsPrimitiveT is_primitive,
                     int          timing_timing_iterations,
                     size_t      *d_temp_storage_bytes,
                     cudaError_t *d_cdp_error,

                     void                 *d_temp_storage,
                     size_t               &temp_storage_bytes,
                     KeysInputIteratorT    d_keys_in,
                     ValuesInputIteratorT  d_values_in,
                     ValuesOutputIteratorT d_values_out,
                     ScanOpT               scan_op,
                     InitialValueT         initial_value,
                     OffsetT               num_items,
                     EqualityOpT           equality_op)
{
  // Invoke kernel to invoke device-side dispatch
  cudaError_t retval =
    thrust::cuda_cub::launcher::triple_chevron(1, 1, 0, 0)
      .doit(CDPDispatchKernel<CUB,
                              IsPrimitiveT,
                              KeysInputIteratorT,
                              ValuesInputIteratorT,
                              ValuesOutputIteratorT,
                              ScanOpT,
                              InitialValueT,
                              OffsetT,
                              EqualityOpT>,
            Int2Type<CUB>{},
            is_primitive,
            timing_timing_iterations,
            d_temp_storage_bytes,
            d_cdp_error,
            d_temp_storage,
            temp_storage_bytes,
            d_keys_in,
            d_values_in,
            d_values_out,
            scan_op,
            initial_value,
            num_items,
            equality_op);
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
    GenMode      gen_mode,
    T            *h_in,
    int          num_items)
{
    for (int i = 0; i < num_items; ++i)
    {
        InitValue(gen_mode, h_in[i], i);
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
template <
    typename        KeysInputIteratorT,
    typename        ValuesInputIteratorT,
    typename        OutputT,
    typename        ScanOpT,
    typename        InitialValueT,
    typename        EqualityOpT>
void Solve(
    KeysInputIteratorT    h_keys_in,
    ValuesInputIteratorT  h_values_in,
    OutputT               *h_reference,
    int                   num_items,
    ScanOpT               scan_op,
    InitialValueT         initial_value,
    EqualityOpT           equality_op)
{
    using ValueT = cub::detail::value_t<ValuesInputIteratorT>;
    using AccumT = cub::detail::accumulator_t<ScanOpT, InitialValueT, ValueT>;

    if (num_items > 0)
    {
        for (int i = 0; i < num_items;) {
            AccumT val         = static_cast<AccumT>(h_values_in[i]);
            h_reference[i]     = initial_value;
            AccumT inclusive   = static_cast<AccumT>(scan_op(initial_value, val));

            ++i;

            for (; i < num_items && equality_op(h_keys_in[i - 1], h_keys_in[i]); ++i)
            {
                val = static_cast<AccumT>(h_values_in[i]);
                h_reference[i] = static_cast<OutputT>(inclusive);
                inclusive = static_cast<AccumT>(scan_op(inclusive, val));
            }
        }
    }
}


/**
 * Solve inclusive-scan problem
 */
template <
    typename        KeysInputIteratorT,
    typename        ValuesInputIteratorT,
    typename        OutputT,
    typename        ScanOpT,
    typename        EqualityOpT>
void Solve(
    KeysInputIteratorT    h_keys_in,
    ValuesInputIteratorT  h_values_in,
    OutputT               *h_reference,
    int                   num_items,
    ScanOpT               scan_op,
    NullType              /*initial_value*/,
    EqualityOpT           equality_op)
{
    using ValueT = cub::detail::value_t<ValuesInputIteratorT>;
    using AccumT = cub::detail::accumulator_t<ScanOpT, ValueT, ValueT>;

    if (num_items > 0)
    {
        for (int i = 0; i < num_items;) {
            AccumT inclusive    = h_values_in[i];
            h_reference[i]      = static_cast<OutputT>(inclusive);

            ++i;

            for (; i < num_items && equality_op(h_keys_in[i - 1], h_keys_in[i]); ++i)
            {
                AccumT val = h_values_in[i];
                inclusive = static_cast<AccumT>(scan_op(inclusive, val));
                h_reference[i] = static_cast<OutputT>(inclusive);
            }
        }
    }
}

template<typename OutputT, typename DeviceInputIteratorT, bool InPlace>
struct AllocateOutput {
    static void run(OutputT *&d_out, DeviceInputIteratorT, int num_items) {
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(OutputT) * num_items));
    }
};

template<typename OutputT>
struct AllocateOutput<OutputT, OutputT *, true> {
    static void run(OutputT *&d_out, OutputT *d_in, int /* num_items */) {
        d_out = d_in;
    }
};

/**
 * Test DeviceScan for a given problem input
 */
template <
    Backend             BACKEND,
    typename            KeysInputIteratorT,
    typename            ValuesInputIteratorT,
    typename            OutputT,
    typename            ScanOpT,
    typename            InitialValueT,
    typename            EqualityOpT,
    AliasMode           Mode=AliasNone>
void Test(
    KeysInputIteratorT      d_keys_in,
    ValuesInputIteratorT    d_values_in,
    OutputT                 *h_reference,
    int                     num_items,
    ScanOpT                 scan_op,
    InitialValueT           initial_value,
    EqualityOpT             equality_op)
{
    using KeyT = cub::detail::value_t<KeysInputIteratorT>;
    using InputT = cub::detail::value_t<ValuesInputIteratorT>;

    // Allocate device output array
    OutputT *d_values_out = NULL;

    if (Mode == AliasKeys)
    {
      AllocateOutput<OutputT, KeysInputIteratorT, Mode == AliasKeys>::run(
        d_values_out,
        d_keys_in,
        num_items);
    }
    else
    {
      AllocateOutput<OutputT, ValuesInputIteratorT, Mode == AliasValues>::run(
        d_values_out,
        d_values_in,
        num_items);
    }

    // Allocate CDP device arrays
    size_t          *d_temp_storage_bytes = NULL;
    cudaError_t     *d_cdp_error = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage_bytes,  sizeof(size_t) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cdp_error,   sizeof(cudaError_t) * 1));

    // Allocate temporary storage
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;
    CubDebugExit(Dispatch(
        Int2Type<BACKEND>(),
        Int2Type<Traits<OutputT>::PRIMITIVE>(),
        1,
        d_temp_storage_bytes,
        d_cdp_error,
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in,
        d_values_in,
        d_values_out,
        scan_op,
        initial_value,
        num_items,
        equality_op));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Clear device output array
    if (Mode == AliasNone)
    {
      CubDebugExit(cudaMemset(d_values_out, 0, sizeof(OutputT) * num_items));
    }

    // Run warmup/correctness iteration
    CubDebugExit(Dispatch(
        Int2Type<BACKEND>(),
        Int2Type<Traits<OutputT>::PRIMITIVE>(),
        1,
        d_temp_storage_bytes,
        d_cdp_error,
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in,
        d_values_in,
        d_values_out,
        scan_op,
        initial_value,
        num_items,
        equality_op));

    // Check for correctness (and display results, if specified)
    const int compare = CompareDeviceResults(h_reference,
                                             d_values_out,
                                             num_items,
                                             true,
                                             g_verbose);

    printf("\t%s", compare ? "FAIL" : "PASS");

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Display performance
    if (g_timing_iterations > 0)
    {
      // Performance
      GpuTimer gpu_timer;
      gpu_timer.Start();
      CubDebugExit(Dispatch(Int2Type<BACKEND>(),
          Int2Type<Traits<OutputT>::PRIMITIVE>(),
          g_timing_iterations,
          d_temp_storage_bytes,
          d_cdp_error,
          d_temp_storage,
          temp_storage_bytes,
          d_keys_in,
          d_values_in,
          d_values_out,
          scan_op,
          initial_value,
          num_items,
          equality_op));

      gpu_timer.Stop();
      float elapsed_millis = gpu_timer.ElapsedMillis();
        float avg_millis = elapsed_millis / g_timing_iterations;
        float giga_rate = float(num_items) / avg_millis / 1000.0f / 1000.0f;
        float giga_bandwidth = giga_rate * (sizeof(InputT) + sizeof(OutputT));
        printf(", %.3f avg ms, %.3f billion items/s, %.3f logical GB/s, %.1f%% peak",
            avg_millis, giga_rate, giga_bandwidth, giga_bandwidth / g_device_giga_bandwidth * 100.0);
    }

    printf("\n\n");

    // Cleanup
    if (Mode == AliasNone)
    {
      if (d_values_out) 
      {
        CubDebugExit(g_allocator.DeviceFree(d_values_out));
      }
    }

    if (d_temp_storage_bytes) CubDebugExit(g_allocator.DeviceFree(d_temp_storage_bytes));
    if (d_cdp_error) CubDebugExit(g_allocator.DeviceFree(d_cdp_error));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    // Correctness asserts
    AssertEquals(0, compare);
}

template <Backend BACKEND,
          typename KeysInputIteratorT,
          typename OutputT,
          typename ScanOpT,
          typename InitialValueT,
          typename EqualityOpT>
void TestInplaceValues(KeysInputIteratorT d_keys_in,
                       OutputT *d_values_in,
                       OutputT *h_reference,
                       int num_items,
                       ScanOpT scan_op,
                       InitialValueT initial_value,
                       EqualityOpT equality_op)
{
  Test<BACKEND,
       KeysInputIteratorT,
       OutputT *,
       OutputT,
       ScanOpT,
       InitialValueT,
       EqualityOpT,
       AliasValues>(d_keys_in,
                    d_values_in,
                    h_reference,
                    num_items,
                    scan_op,
                    initial_value,
                    equality_op);
}

template <Backend BACKEND,
          typename KeysInputIteratorT,
          typename ValuesInputIteratorT,
          typename OutputT,
          typename ScanOpT,
          typename InitialValueT,
          typename EqualityOpT>
void TestInplaceValues(KeysInputIteratorT,
                       ValuesInputIteratorT,
                       OutputT *,
                       int,
                       ScanOpT,
                       InitialValueT,
                       EqualityOpT)
{}

template <Backend BACKEND,
          typename T,
          typename ValuesInputIteratorT,
          typename ScanOpT,
          typename InitialValueT,
          typename EqualityOpT>
void TestInplaceKeys(T *d_keys_in,
                     ValuesInputIteratorT d_values_in,
                     T *h_reference,
                     int num_items,
                     ScanOpT scan_op,
                     InitialValueT initial_value,
                     EqualityOpT equality_op)
{
  Test<BACKEND,
       T *,
       ValuesInputIteratorT,
       T,
       ScanOpT,
       InitialValueT,
       EqualityOpT,
       AliasKeys>(d_keys_in,
                  d_values_in,
                  h_reference,
                  num_items,
                  scan_op,
                  initial_value,
                  equality_op);
}

template <Backend BACKEND,
          typename KeysInputIteratorT,
          typename ValuesInputIteratorT,
          typename OutputT,
          typename ScanOpT,
          typename InitialValueT,
          typename EqualityOpT>
void TestInplaceKeys(KeysInputIteratorT,
                     ValuesInputIteratorT,
                     OutputT *,
                     int,
                     ScanOpT,
                     InitialValueT,
                     EqualityOpT)
{}

/**
 * Test DeviceScan on pointer type
 */
template <
    Backend         BACKEND,
    typename        KeyT,
    typename        InputT,
    typename        OutputT,
    typename        ScanOpT,
    typename        InitialValueT,
    typename        EqualityOpT>
void TestPointer(
    int             num_items,
    GenMode         gen_mode,
    ScanOpT         scan_op,
    InitialValueT   initial_value,
    EqualityOpT     equality_op)
{
    printf("\nPointer %s %s cub::DeviceScan::%s %d items, %s->%s (%d->%d bytes) , gen-mode %s\n",
        (BACKEND == CDP) ? "CDP CUB" : "CUB",
        (std::is_same<InitialValueT, NullType>::value) ? "Inclusive" : "Exclusive",
        (std::is_same<ScanOpT, Sum>::value) ? "Sum" : "Scan",
        num_items,
        typeid(InputT).name(), typeid(OutputT).name(), (int) sizeof(InputT), (int) sizeof(OutputT),
        (gen_mode == RANDOM) ? "RANDOM" : (gen_mode == INTEGER_SEED) ? "SEQUENTIAL" : "HOMOGENOUS");
    fflush(stdout);

    // Allocate host arrays
    KeyT*       h_keys_in   = new KeyT[num_items];
    InputT*     h_values_in = new InputT[num_items];
    OutputT*    h_reference = new OutputT[num_items];

    // Initialize problem and solution
    Initialize(gen_mode, h_keys_in, num_items);
    Initialize(gen_mode, h_values_in, num_items);

    // If the output type is primitive and the operator is cub::Sum, the test
    // dispatcher throws away scan_op and initial_value for exclusive scan.
    // Without an initial_value arg, the accumulator switches to the input value
    // type.
    // Do the same thing here:
    if (Traits<OutputT>::PRIMITIVE &&
        std::is_same<ScanOpT, cub::Sum>::value &&
        !std::is_same<InitialValueT, NullType>::value)
    {
      Solve(h_keys_in, h_values_in, h_reference, num_items, cub::Sum{}, InputT{}, equality_op);
    }
    else
    {
      Solve(h_keys_in, h_values_in, h_reference, num_items, scan_op, initial_value, equality_op);
    }

    // Allocate problem device arrays
    KeyT *d_keys_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys_in, sizeof(KeyT) * num_items));
    InputT *d_values_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values_in, sizeof(InputT) * num_items));

    // Initialize device input
    CubDebugExit(cudaMemcpy(d_keys_in, h_keys_in, sizeof(KeyT) * num_items, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_values_in, h_values_in, sizeof(InputT) * num_items, cudaMemcpyHostToDevice));

    // Run Test
    Test<BACKEND>(d_keys_in,
                  d_values_in,
                  h_reference,
                  num_items,
                  scan_op,
                  initial_value,
                  equality_op);

    // Test in/out values aliasing
    TestInplaceValues<BACKEND>(d_keys_in,
                               d_values_in, 
                               h_reference,
                               num_items,
                               scan_op,
                               initial_value,
                               equality_op);

    CubDebugExit(cudaMemcpy(d_values_in, h_values_in, sizeof(InputT) * num_items, cudaMemcpyHostToDevice));

    // Test keys/values aliasing (should go last, changes keys)
    TestInplaceKeys<BACKEND>(d_keys_in,
                             d_values_in,
                             h_reference,
                             num_items,
                             scan_op,
                             initial_value,
                             equality_op);

    // Cleanup
    if (h_keys_in) delete[] h_keys_in;
    if (h_values_in) delete[] h_values_in;
    if (h_reference) delete[] h_reference;
    if (d_keys_in) CubDebugExit(g_allocator.DeviceFree(d_keys_in));
    if (d_values_in) CubDebugExit(g_allocator.DeviceFree(d_values_in));
}


/**
 * Test DeviceScan on iterator type
 */
template <
    Backend         BACKEND,
    typename        KeyT,
    typename        InputT,
    typename        OutputT,
    typename        ScanOpT,
    typename        InitialValueT,
    typename        EqualityOpT>
void TestIterator(
    int             num_items,
    ScanOpT         scan_op,
    InitialValueT   initial_value,
    EqualityOpT     equality_op)
{
    printf("\nIterator %s %s cub::DeviceScan::%s %d items, %s->%s (%d->%d bytes)\n",
        (BACKEND == CDP) ? "CDP CUB" : "CUB",
        (std::is_same<InitialValueT, NullType>::value) ? "Inclusive" : "Exclusive",
        (std::is_same<ScanOpT, Sum>::value) ? "Sum" : "Scan",
        num_items,
        typeid(InputT).name(), typeid(OutputT).name(), (int) sizeof(InputT), (int) sizeof(OutputT));
    fflush(stdout);

    // Use a counting iterator followed by div as the keys
    using CountingIterT = CountingInputIterator<int, int>;
    CountingIterT h_keys_in_helper(0);
    TransformInputIterator<KeyT, DivideByFiveFunctor<KeyT>, CountingIterT> h_keys_in(h_keys_in_helper, DivideByFiveFunctor<KeyT>());

    // Use a constant iterator as the input
    InputT val = InputT();
    ConstantInputIterator<InputT, int> h_values_in(val);

    // Allocate host arrays
    OutputT*  h_reference = new OutputT[num_items];

    // Initialize problem and solution
    Solve(h_keys_in, h_values_in, h_reference, num_items, scan_op, initial_value, equality_op);

    // Run Test
    Test<BACKEND>(h_keys_in, h_values_in, h_reference, num_items, scan_op, initial_value, equality_op);

    // Cleanup
    if (h_reference) delete[] h_reference;
}


/**
 * Test different gen modes
 */
template <
    Backend         BACKEND,
    typename        KeyT,
    typename        InputT,
    typename        OutputT,
    typename        ScanOpT,
    typename        InitialValueT,
    typename        EqualityOpT>
void Test(
    int             num_items,
    ScanOpT         scan_op,
    InitialValueT   initial_value,
    EqualityOpT     equality_op)
{
    TestPointer<BACKEND, KeyT, InputT, OutputT>(  num_items, UNIFORM, scan_op, initial_value, equality_op);
    TestPointer<BACKEND, KeyT, InputT, OutputT>(  num_items, RANDOM,  scan_op, initial_value, equality_op);
    TestIterator<BACKEND, KeyT, InputT, OutputT>( num_items, scan_op, initial_value, equality_op);
}


/**
 * Test different dispatch
 */
template <
    typename        KeyT,
    typename        InputT,
    typename        OutputT,
    typename        ScanOpT,
    typename        InitialValueT,
    typename        EqualityOpT>
void Test(
    int             num_items,
    ScanOpT         scan_op,
    InitialValueT   initial_value,
    EqualityOpT     equality_op)
{
#if TEST_CDP == 0
    Test<CUB, KeyT, InputT, OutputT>(num_items, scan_op, initial_value, equality_op);
#elif TEST_CDP == 1
    Test<CDP, KeyT, InputT, OutputT>(num_items, scan_op, initial_value, equality_op);
#endif // TEST_CDP
}


/**
 * Test different operators
 */
template <typename KeyT, typename InputT, typename OutputT, typename EqualityOpT>
void TestOp(
    int             num_items,
    OutputT         identity,
    OutputT         initial_value,
    EqualityOpT     equality_op)
{
    // Exclusive (use identity as initial value because it will dispatch to *Sum variants that don't take initial values)
    Test<KeyT, InputT, OutputT>(num_items, cub::Sum(), identity, equality_op);
    Test<KeyT, InputT, OutputT>(num_items, cub::Max(), identity, equality_op);

    // Exclusive (non-specialized, so we can test initial-value)
    Test<KeyT, InputT, OutputT>(num_items, WrapperFunctor<cub::Sum>(cub::Sum()), initial_value, equality_op);
    Test<KeyT, InputT, OutputT>(num_items, WrapperFunctor<cub::Max>(cub::Max()), initial_value, equality_op);

    // Inclusive (no initial value)
    Test<KeyT, InputT, OutputT>(num_items, cub::Sum(), NullType(), equality_op);
    Test<KeyT, InputT, OutputT>(num_items, cub::Max(), NullType(), equality_op);
}

/**
 * Test different key type and equality operator
 */
template <typename InputT, typename OutputT>
void TestKeyTAndEqualityOp(
    int             num_items,
    OutputT         identity,
    OutputT         initial_value)
{
    TestOp<bool, InputT>(num_items, identity, initial_value, Equality());
    TestOp<unsigned int, InputT>( num_items, identity, initial_value, Mod2Equality());
}

/**
 * Test different input sizes
 */
template <
    typename InputT,
    typename OutputT>
void TestSize(
    int         num_items,
    OutputT     identity,
    OutputT     initial_value)
{
    if (num_items < 0)
    {
        TestKeyTAndEqualityOp<InputT>(0,        identity, initial_value);
        TestKeyTAndEqualityOp<InputT>(1,        identity, initial_value);
        TestKeyTAndEqualityOp<InputT>(100,      identity, initial_value);
        TestKeyTAndEqualityOp<InputT>(10000,    identity, initial_value);
        TestKeyTAndEqualityOp<InputT>(1000000,  identity, initial_value);
    }
    else
    {
        TestKeyTAndEqualityOp<InputT>(num_items, identity, initial_value);
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
    int num_items = -1;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("n", num_items);
    args.GetCmdLineArgument("i", g_timing_iterations);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--n=<input items> "
            "[--i=<timing iterations> "
            "[--device=<device-id>] "
            "[--v] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());
    g_device_giga_bandwidth = args.device_giga_bandwidth;
    printf("\n");

    // %PARAM% TEST_CDP cdp 0:1
    // %PARAM% TEST_VALUE_TYPES types 0:1:2:3:4:5

#if TEST_VALUE_TYPES == 0

    // Test different input+output data types
    TestSize<unsigned char>(num_items, (int)0, (int)99);

    // Test same input+output data types
    TestSize<unsigned char>(num_items, (unsigned char)0, (unsigned char)99);
    TestSize<signed char>(num_items, (char)0, (char)99);

#elif TEST_VALUE_TYPES == 1

    TestSize<unsigned short>(num_items, (unsigned short)0, (unsigned short)99);
    TestSize<unsigned int>(num_items, (unsigned int)0, (unsigned int)99);
    TestSize<unsigned long long>(num_items,
                                 (unsigned long long)0,
                                 (unsigned long long)99);
#elif TEST_VALUE_TYPES == 2

    TestSize<uchar2>(num_items, make_uchar2(0, 0), make_uchar2(17, 21));
    TestSize<char2>(num_items, make_char2(0, 0), make_char2(17, 21));
    TestSize<ushort2>(num_items, make_ushort2(0, 0), make_ushort2(17, 21));

#elif TEST_VALUE_TYPES == 3

    TestSize<uint2>(num_items, make_uint2(0, 0), make_uint2(17, 21));
    TestSize<ulonglong2>(num_items,
                         make_ulonglong2(0, 0),
                         make_ulonglong2(17, 21));
    TestSize<uchar4>(num_items,
                     make_uchar4(0, 0, 0, 0),
                     make_uchar4(17, 21, 32, 85));

#elif TEST_VALUE_TYPES == 4

    TestSize<char4>(num_items,
                    make_char4(0, 0, 0, 0),
                    make_char4(17, 21, 32, 85));

    TestSize<ushort4>(num_items,
                      make_ushort4(0, 0, 0, 0),
                      make_ushort4(17, 21, 32, 85));
    TestSize<uint4>(num_items,
                    make_uint4(0, 0, 0, 0),
                    make_uint4(17, 21, 32, 85));

#elif TEST_VALUE_TYPES == 5

    TestSize<ulonglong4>(num_items,
                         make_ulonglong4(0, 0, 0, 0),
                         make_ulonglong4(17, 21, 32, 85));

    TestSize<TestFoo>(num_items,
                      TestFoo::MakeTestFoo(0, 0, 0, 0),
                      TestFoo::MakeTestFoo(std::numeric_limits<TestFoo::x_t>::max(),
                                           std::numeric_limits<TestFoo::y_t>::max(),
                                           std::numeric_limits<TestFoo::z_t>::max(),
                                           std::numeric_limits<TestFoo::w_t>::max()));

    TestSize<TestBar>(num_items, 
                      TestBar(0, 0), 
                      TestBar(std::numeric_limits<long long>::max(), 
                              std::numeric_limits<int>::max()));

#endif

    return 0;
}
