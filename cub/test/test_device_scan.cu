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
 * Test of DeviceScan utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_scan.cuh>
#include <cub/iterator/constant_input_iterator.cuh>
#include <cub/iterator/discard_output_iterator.cuh>
#include <cub/util_allocator.cuh>
#include <cub/util_type.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include "test_util.h"

#include <cstdio>
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


//---------------------------------------------------------------------
// Dispatch to different CUB DeviceScan entrypoints
//---------------------------------------------------------------------

/**
 * Dispatch to exclusive scan entrypoint
 */
template <typename IsPrimitiveT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename ScanOpT,
          typename InitialValueT,
          typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t
Dispatch(Int2Type<true> /*in_place*/,
         Int2Type<CUB> /*dispatch_to*/,
         IsPrimitiveT /*is_primitive*/,
         int timing_timing_iterations,
         size_t * /* d_temp_storage_bytes */,
         cudaError_t * /* d_cdp_error */,
         void *d_temp_storage,
         size_t &temp_storage_bytes,
         InputIteratorT d_in,
         OutputIteratorT /* d_out */,
         ScanOpT scan_op,
         InitialValueT initial_value,
         OffsetT num_items)
{
  cudaError_t error = cudaSuccess;
  for (int i = 0; i < timing_timing_iterations; ++i)
  {
    error = DeviceScan::ExclusiveScan(d_temp_storage,
                                      temp_storage_bytes,
                                      d_in,
                                      scan_op,
                                      initial_value,
                                      num_items);
  }
  return error;
}

template <typename IsPrimitiveT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename ScanOpT,
          typename InitialValueT,
          typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t
Dispatch(Int2Type<false> /*in_place*/,
         Int2Type<CUB> /*dispatch_to*/,
         IsPrimitiveT /*is_primitive*/,
         int timing_timing_iterations,
         size_t * /*d_temp_storage_bytes*/,
         cudaError_t * /*d_cdp_error*/,
         void *d_temp_storage,
         size_t &temp_storage_bytes,
         InputIteratorT d_in,
         OutputIteratorT d_out,
         ScanOpT scan_op,
         InitialValueT initial_value,
         OffsetT num_items)
{
  cudaError_t error = cudaSuccess;
  for (int i = 0; i < timing_timing_iterations; ++i)
  {
    error = DeviceScan::ExclusiveScan(d_temp_storage,
                                      temp_storage_bytes,
                                      d_in,
                                      d_out,
                                      scan_op,
                                      initial_value,
                                      num_items);
  }
  return error;
}

/**
 * Dispatch to exclusive sum entrypoint
 */
template <typename InputIteratorT,
          typename OutputIteratorT,
          typename InitialValueT,
          typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t
Dispatch(Int2Type<true> /*in_place*/,
         Int2Type<CUB> /*dispatch_to*/,
         Int2Type<true> /*is_primitive*/,
         int timing_timing_iterations,
         size_t * /*d_temp_storage_bytes*/,
         cudaError_t * /*d_cdp_error*/,
         void *d_temp_storage,
         size_t &temp_storage_bytes,
         InputIteratorT d_in,
         OutputIteratorT /* d_out */,
         Sum /*scan_op*/,
         InitialValueT /*initial_value*/,
         OffsetT num_items)
{
  cudaError_t error = cudaSuccess;
  for (int i = 0; i < timing_timing_iterations; ++i)
  {
    error = DeviceScan::ExclusiveSum(d_temp_storage,
                                     temp_storage_bytes,
                                     d_in,
                                     num_items);
  }
  return error;
}

template <typename InputIteratorT,
          typename OutputIteratorT,
          typename InitialValueT,
          typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t
Dispatch(Int2Type<false> /*in_place*/,
         Int2Type<CUB> /*dispatch_to*/,
         Int2Type<true> /*is_primitive*/,
         int timing_timing_iterations,
         size_t * /*d_temp_storage_bytes*/,
         cudaError_t * /*d_cdp_error*/,
         void *d_temp_storage,
         size_t &temp_storage_bytes,
         InputIteratorT d_in,
         OutputIteratorT d_out,
         Sum /*scan_op*/,
         InitialValueT /*initial_value*/,
         OffsetT num_items)
{
  cudaError_t error = cudaSuccess;
  for (int i = 0; i < timing_timing_iterations; ++i)
  {
    error = DeviceScan::ExclusiveSum(d_temp_storage,
                                     temp_storage_bytes,
                                     d_in,
                                     d_out,
                                     num_items);
  }
  return error;
}

/**
 * Dispatch to inclusive scan entrypoint
 */
template <typename IsPrimitiveT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename ScanOpT,
          typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t
Dispatch(Int2Type<true> /*in_place*/,
         Int2Type<CUB> /*dispatch_to*/,
         IsPrimitiveT /*is_primitive*/,
         int timing_timing_iterations,
         size_t * /*d_temp_storage_bytes*/,
         cudaError_t * /*d_cdp_error*/,
         void *d_temp_storage,
         size_t &temp_storage_bytes,
         InputIteratorT d_in,
         OutputIteratorT /* d_out */,
         ScanOpT scan_op,
         NullType /* initial_value */,
         OffsetT num_items)
{
  cudaError_t error = cudaSuccess;
  for (int i = 0; i < timing_timing_iterations; ++i)
  {
    error = DeviceScan::InclusiveScan(d_temp_storage,
                                      temp_storage_bytes,
                                      d_in,
                                      scan_op,
                                      num_items);
  }
  return error;
}

template <typename IsPrimitiveT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename ScanOpT,
          typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t
Dispatch(Int2Type<false> /*in_place*/,
         Int2Type<CUB> /*dispatch_to*/,
         IsPrimitiveT /*is_primitive*/,
         int timing_timing_iterations,
         size_t * /*d_temp_storage_bytes*/,
         cudaError_t * /*d_cdp_error*/,
         void *d_temp_storage,
         size_t &temp_storage_bytes,
         InputIteratorT d_in,
         OutputIteratorT d_out,
         ScanOpT scan_op,
         NullType /*initial_value*/,
         OffsetT num_items)
{
  cudaError_t error = cudaSuccess;
  for (int i = 0; i < timing_timing_iterations; ++i)
  {
    error = DeviceScan::InclusiveScan(d_temp_storage,
                                      temp_storage_bytes,
                                      d_in,
                                      d_out,
                                      scan_op,
                                      num_items);
  }
  return error;
}

/**
 * Dispatch to inclusive sum entrypoint
 */
template <typename InputIteratorT, typename OutputIteratorT, typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t
Dispatch(Int2Type<true> /*in_place*/,
         Int2Type<CUB> /*dispatch_to*/,
         Int2Type<true> /*is_primitive*/,
         int timing_timing_iterations,
         size_t * /*d_temp_storage_bytes*/,
         cudaError_t * /*d_cdp_error*/,
         void *d_temp_storage,
         size_t &temp_storage_bytes,
         InputIteratorT d_in,
         OutputIteratorT /* d_out */,
         Sum /*scan_op*/,
         NullType /*initial_value*/,
         OffsetT num_items)
{
  cudaError_t error = cudaSuccess;
  for (int i = 0; i < timing_timing_iterations; ++i)
  {
    error = DeviceScan::InclusiveSum(d_temp_storage,
                                     temp_storage_bytes,
                                     d_in,
                                     num_items);
  }
  return error;
}

template <typename InputIteratorT, typename OutputIteratorT, typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t
Dispatch(Int2Type<false> /*in_place*/,
         Int2Type<CUB> /*dispatch_to*/,
         Int2Type<true> /*is_primitive*/,
         int timing_timing_iterations,
         size_t * /*d_temp_storage_bytes*/,
         cudaError_t * /*d_cdp_error*/,
         void *d_temp_storage,
         size_t &temp_storage_bytes,
         InputIteratorT d_in,
         OutputIteratorT d_out,
         Sum /*scan_op*/,
         NullType /*initial_value*/,
         OffsetT num_items)
{
  cudaError_t error = cudaSuccess;
  for (int i = 0; i < timing_timing_iterations; ++i)
  {
    error = DeviceScan::InclusiveSum(d_temp_storage,
                                     temp_storage_bytes,
                                     d_in,
                                     d_out,
                                     num_items);
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
template <typename InPlaceT,
          typename CubBackendT,
          typename IsPrimitiveT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename ScanOpT,
          typename InitialValueT,
          typename OffsetT>
__global__ void CDPDispatchKernel(InPlaceT     in_place,
                                  CubBackendT  cub_backend,
                                  IsPrimitiveT is_primitive,
                                  int          timing_timing_iterations,
                                  size_t      *d_temp_storage_bytes,
                                  cudaError_t *d_cdp_error,

                                  void           *d_temp_storage,
                                  size_t          temp_storage_bytes,
                                  InputIteratorT  d_in,
                                  OutputIteratorT d_out,
                                  ScanOpT         scan_op,
                                  InitialValueT   initial_value,
                                  OffsetT         num_items)
{
  *d_cdp_error = Dispatch(in_place,
                          cub_backend,
                          is_primitive,
                          timing_timing_iterations,
                          d_temp_storage_bytes,
                          d_cdp_error,
                          d_temp_storage,
                          temp_storage_bytes,
                          d_in,
                          d_out,
                          scan_op,
                          initial_value,
                          num_items);

  *d_temp_storage_bytes = temp_storage_bytes;
}

/**
 * Dispatch to CDP kernel
 */
template <typename InPlaceT,
          typename IsPrimitiveT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename ScanOpT,
          typename InitialValueT,
          typename OffsetT>
cudaError_t Dispatch(InPlaceT      in_place,
                     Int2Type<CDP> dispatch_to,
                     IsPrimitiveT  is_primitive,
                     int           timing_timing_iterations,
                     size_t       *d_temp_storage_bytes,
                     cudaError_t  *d_cdp_error,

                     void           *d_temp_storage,
                     size_t         &temp_storage_bytes,
                     InputIteratorT  d_in,
                     OutputIteratorT d_out,
                     ScanOpT         scan_op,
                     InitialValueT   initial_value,
                     OffsetT         num_items)
{
  // Invoke kernel to invoke device-side dispatch to CUB backend:
  (void)dispatch_to;
  using CubBackendT = Int2Type<CUB>;
  CubBackendT cub_backend;
  cudaError_t   retval =
    thrust::cuda_cub::launcher::triple_chevron(1, 1, 0, 0)
      .doit(CDPDispatchKernel<InPlaceT,
                              CubBackendT,
                              IsPrimitiveT,
                              InputIteratorT,
                              OutputIteratorT,
                              ScanOpT,
                              InitialValueT,
                              OffsetT>,
            in_place,
            cub_backend,
            is_primitive,
            timing_timing_iterations,
            d_temp_storage_bytes,
            d_cdp_error,
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            scan_op,
            initial_value,
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
    typename        InputIteratorT,
    typename        OutputT,
    typename        ScanOpT,
    typename        InitialValueT>
void Solve(
    InputIteratorT  h_in,
    OutputT         *h_reference,
    int             num_items,
    ScanOpT         scan_op,
    InitialValueT   initial_value)
{
    using AccumT = 
      cub::detail::accumulator_t<
        ScanOpT, 
        InitialValueT, 
        cub::detail::value_t<InputIteratorT>>;

    if (num_items > 0)
    {
        AccumT val         = static_cast<AccumT>(h_in[0]);
        h_reference[0]     = initial_value;
        AccumT inclusive   = static_cast<AccumT>(scan_op(initial_value, val));

        for (int i = 1; i < num_items; ++i)
        {
            val = static_cast<AccumT>(h_in[i]);
            h_reference[i] = static_cast<OutputT>(inclusive);
            inclusive = static_cast<AccumT>(scan_op(inclusive, val));
        }
    }
}


/**
 * Solve inclusive-scan problem
 */
template <
    typename        InputIteratorT,
    typename        OutputT,
    typename        ScanOpT>
void Solve(
    InputIteratorT  h_in,
    OutputT         *h_reference,
    int             num_items,
    ScanOpT         scan_op,
    NullType)
{
    using AccumT = 
      cub::detail::accumulator_t<
        ScanOpT, 
        cub::detail::value_t<InputIteratorT>, 
        cub::detail::value_t<InputIteratorT>>;

    if (num_items > 0)
    {
        AccumT inclusive    = h_in[0];
        h_reference[0]      = static_cast<OutputT>(inclusive);

        for (int i = 1; i < num_items; ++i)
        {
            AccumT val = h_in[i];
            inclusive = static_cast<AccumT>(scan_op(inclusive, val));
            h_reference[i] = static_cast<OutputT>(inclusive);
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
    typename            DeviceInputIteratorT,
    typename            OutputT,
    typename            ScanOpT,
    typename            InitialValueT,
    bool                InPlace=false>
void Test(
    DeviceInputIteratorT    d_in,
    OutputT                 *h_reference,
    int                     num_items,
    ScanOpT                 scan_op,
    InitialValueT           initial_value)
{
    using InputT = cub::detail::value_t<DeviceInputIteratorT>;

    // Allocate device output array
    OutputT *d_out = NULL;
    AllocateOutput<OutputT, DeviceInputIteratorT, InPlace>::run(d_out, d_in, num_items);

    // Allocate CDP device arrays
    size_t          *d_temp_storage_bytes = NULL;
    cudaError_t     *d_cdp_error = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage_bytes,  sizeof(size_t) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cdp_error,   sizeof(cudaError_t) * 1));

    // Allocate temporary storage
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;
    CubDebugExit(Dispatch(
        Int2Type<InPlace>(),
        Int2Type<BACKEND>(),
        Int2Type<Traits<OutputT>::PRIMITIVE>(),
        1,
        d_temp_storage_bytes,
        d_cdp_error,
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        scan_op,
        initial_value,
        num_items));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    if (!InPlace)
    {
      // Clear device output array
      CubDebugExit(cudaMemset(d_out, 0, sizeof(OutputT) * num_items));
    }

    // Run warmup/correctness iteration
    CubDebugExit(Dispatch(
        Int2Type<InPlace>(),
        Int2Type<BACKEND>(),
        Int2Type<Traits<OutputT>::PRIMITIVE>(),
        1,
        d_temp_storage_bytes,
        d_cdp_error,
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        scan_op,
        initial_value,
        num_items));

    // Check for correctness (and display results, if specified)
    int compare = CompareDeviceResults(h_reference, d_out, num_items, true, g_verbose);
    printf("\t%s", compare ? "FAIL" : "PASS");

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Performance
    if (g_timing_iterations > 0)
    {
      GpuTimer gpu_timer;
      gpu_timer.Start();
      CubDebugExit(Dispatch(Int2Type<InPlace>(),
                            Int2Type<BACKEND>(),
                            Int2Type<Traits<OutputT>::PRIMITIVE>(),
                            g_timing_iterations,
                            d_temp_storage_bytes,
                            d_cdp_error,
                            d_temp_storage,
                            temp_storage_bytes,
                            d_in,
                            d_out,
                            scan_op,
                            initial_value,
                            num_items));
      gpu_timer.Stop();
      float elapsed_millis = gpu_timer.ElapsedMillis();

      // Display performance
      float avg_millis     = elapsed_millis / g_timing_iterations;
      float giga_rate      = float(num_items) / avg_millis / 1000.0f / 1000.0f;
      float giga_bandwidth = giga_rate * (sizeof(InputT) + sizeof(OutputT));
      printf(", %.3f avg ms, %.3f billion items/s, %.3f logical GB/s, %.1f%% "
             "peak",
             avg_millis,
             giga_rate,
             giga_bandwidth,
             giga_bandwidth / g_device_giga_bandwidth * 100.0);
    }

    printf("\n\n");

    // Cleanup
    if (!InPlace)
    {
      if (d_out)
      {
        CubDebugExit(g_allocator.DeviceFree(d_out));
      }
    }

    if (d_temp_storage_bytes) CubDebugExit(g_allocator.DeviceFree(d_temp_storage_bytes));
    if (d_cdp_error) CubDebugExit(g_allocator.DeviceFree(d_cdp_error));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    // Correctness asserts
    AssertEquals(0, compare);
}

template <typename InitialValueT>
__global__ void FillInitValue(InitialValueT *ptr, InitialValueT initial_value) {
    *ptr = initial_value;
}

template <
    Backend             BACKEND,
    typename            DeviceInputIteratorT,
    typename            OutputT,
    typename            ScanOpT,
    typename            InitialValueT>
typename std::enable_if<!std::is_same<InitialValueT, cub::NullType>::value>::type
TestFutureInitValue(
    DeviceInputIteratorT    d_in,
    OutputT                 *h_reference,
    int                     num_items,
    ScanOpT                 scan_op,
    InitialValueT           initial_value)
{
    // Allocate device initial_value
    InitialValueT *d_initial_value = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_initial_value, sizeof(InitialValueT)));
    FillInitValue<<<1, 1>>>(d_initial_value, initial_value);

    // Run test
    auto future_init_value = cub::FutureValue<InitialValueT>(d_initial_value);
    Test<BACKEND>(d_in, h_reference, num_items, scan_op, future_init_value);

    // Cleanup
    if (d_initial_value) CubDebugExit(g_allocator.DeviceFree(d_initial_value));
}

template <
    Backend             BACKEND,
    typename            DeviceInputIteratorT,
    typename            OutputT,
    typename            ScanOpT,
    typename            InitialValueT>
typename std::enable_if<std::is_same<InitialValueT, cub::NullType>::value>::type
TestFutureInitValue(
    DeviceInputIteratorT,
    OutputT *,
    int,
    ScanOpT,
    InitialValueT)
{
    // cub::NullType does not have device pointer, so nothing to do here
}

template <
  Backend             BACKEND,
  typename            DeviceInputIteratorT,
  typename            OutputT,
  typename            ScanOpT,
  typename            InitialValueT>
typename std::enable_if<!std::is_same<InitialValueT, cub::NullType>::value>::type
TestFutureInitValueIter(
    DeviceInputIteratorT    d_in,
    OutputT                 *h_reference,
    int                     num_items,
    ScanOpT                 scan_op,
    InitialValueT           initial_value)
{
    using IterT = cub::ConstantInputIterator<InitialValueT>;
    IterT iter(initial_value);
    auto future_init_value = cub::FutureValue<InitialValueT, IterT>(iter);
    Test<BACKEND>(d_in, h_reference, num_items, scan_op, future_init_value);
}

template <
    Backend             BACKEND,
    typename            DeviceInputIteratorT,
    typename            OutputT,
    typename            ScanOpT,
    typename            InitialValueT>
typename std::enable_if<std::is_same<InitialValueT, cub::NullType>::value>::type
TestFutureInitValueIter(
    DeviceInputIteratorT,
    OutputT *,
    int,
    ScanOpT,
    InitialValueT)
{
    // cub::NullType does not have device pointer, so nothing to do here
}

template <Backend BACKEND,
          typename OutputT,
          typename ScanOpT,
          typename InitialValueT>
void TestInplace(OutputT *d_in,
                 OutputT *h_reference,
                 int num_items,
                 ScanOpT scan_op,
                 InitialValueT initial_value)
{
  Test<BACKEND, OutputT *, OutputT, ScanOpT, InitialValueT, true>(d_in,
                                                                  h_reference,
                                                                  num_items,
                                                                  scan_op,
                                                                  initial_value);
}

template <Backend BACKEND,
          typename DeviceInputIteratorT,
          typename OutputT,
          typename ScanOpT,
          typename InitialValueT>
void TestInplace(DeviceInputIteratorT, OutputT *, int, ScanOpT, InitialValueT)
{}

/**
 * Test DeviceScan on pointer type
 */
template <
    Backend         BACKEND,
    typename        InputT,
    typename        OutputT,
    typename        ScanOpT,
    typename        InitialValueT>
void TestPointer(
    int             num_items,
    GenMode         gen_mode,
    ScanOpT         scan_op,
    InitialValueT   initial_value)
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
    InputT*     h_in        = new InputT[num_items];
    OutputT*    h_reference = new OutputT[num_items];

    // Initialize problem and solution
    Initialize(gen_mode, h_in, num_items);

    // If the output type is primitive and the operator is cub::Sum, the test
    // dispatcher throws away scan_op and initial_value for exclusive scan.
    // Without an initial_value arg, the accumulator switches to the input value
    // type.
    // Do the same thing here:
    if (Traits<OutputT>::PRIMITIVE &&
        std::is_same<ScanOpT, cub::Sum>::value &&
        !std::is_same<InitialValueT, NullType>::value)
    {
      Solve(h_in, h_reference, num_items, cub::Sum{}, InputT{});
    }
    else
    {
      Solve(h_in, h_reference, num_items, scan_op, initial_value);
    }

    // Allocate problem device arrays
    InputT *d_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(InputT) * num_items));

    // Initialize device input
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(InputT) * num_items, cudaMemcpyHostToDevice));

    // Run Test
    Test<BACKEND>(d_in, h_reference, num_items, scan_op, initial_value);
    TestFutureInitValue<BACKEND>(d_in, h_reference, num_items, scan_op, initial_value);
    TestInplace<BACKEND>(d_in, h_reference, num_items, scan_op, initial_value);

    // Cleanup
    if (h_in) delete[] h_in;
    if (h_reference) delete[] h_reference;
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
}


/**
 * Test DeviceScan on iterator type
 */
template <
    Backend         BACKEND,
    typename        InputT,
    typename        OutputT,
    typename        ScanOpT,
    typename        InitialValueT>
void TestIterator(
    int             num_items,
    ScanOpT         scan_op,
    InitialValueT   initial_value)
{
    printf("\nIterator %s %s cub::DeviceScan::%s %d items, %s->%s (%d->%d bytes)\n",
        (BACKEND == CDP) ? "CDP CUB" : "CUB",
        (std::is_same<InitialValueT, NullType>::value) ? "Inclusive" : "Exclusive",
        (std::is_same<ScanOpT, Sum>::value) ? "Sum" : "Scan",
        num_items,
        typeid(InputT).name(), typeid(OutputT).name(), (int) sizeof(InputT), (int) sizeof(OutputT));
    fflush(stdout);

    // Use a constant iterator as the input
    InputT val = InputT();
    ConstantInputIterator<InputT, int> h_in(val);

    // Allocate host arrays
    OutputT*  h_reference = new OutputT[num_items];

    // Initialize problem and solution
    Solve(h_in, h_reference, num_items, scan_op, initial_value);

    // Run Test
    Test<BACKEND>(h_in, h_reference, num_items, scan_op, initial_value);
    TestFutureInitValueIter<BACKEND>(h_in, h_reference, num_items, scan_op, initial_value);

    // Cleanup
    if (h_reference) delete[] h_reference;
}


/**
 * Test different gen modes
 */
template <
    Backend         BACKEND,
    typename        InputT,
    typename        OutputT,
    typename        ScanOpT,
    typename        InitialValueT>
void Test(
    int             num_items,
    ScanOpT         scan_op,
    InitialValueT   initial_value)
{
    TestPointer<BACKEND, InputT, OutputT>(  num_items, UNIFORM, scan_op, initial_value);
    TestPointer<BACKEND, InputT, OutputT>(  num_items, RANDOM,  scan_op, initial_value);
    TestIterator<BACKEND, InputT, OutputT>( num_items, scan_op, initial_value);
}


/**
 * Test different dispatch
 */
template <
    typename        InputT,
    typename        OutputT,
    typename        ScanOpT,
    typename        InitialValueT>
void Test(
    int             num_items,
    ScanOpT         scan_op,
    InitialValueT   initial_value)
{
#if TEST_CDP == 0
    Test<CUB, InputT, OutputT>(num_items, scan_op, initial_value);
#elif TEST_CDP == 1
    Test<CDP, InputT, OutputT>(num_items, scan_op, initial_value);
#endif // TEST_CDP
}


/**
 * Test different operators
 */
template <typename InputT, typename OutputT>
void TestOp(
    int             num_items,
    OutputT         identity,
    OutputT         initial_value)
{
    // Exclusive (use identity as initial value because it will dispatch to *Sum variants that don't take initial values)
    Test<InputT, OutputT>(num_items, cub::Sum(), identity);
    Test<InputT, OutputT>(num_items, cub::Max(), identity);

    // Exclusive (non-specialized, so we can test initial-value)
    Test<InputT, OutputT>(num_items, WrapperFunctor<cub::Sum>(cub::Sum()), initial_value);
    Test<InputT, OutputT>(num_items, WrapperFunctor<cub::Max>(cub::Max()), initial_value);

    // Inclusive (no initial value)
    Test<InputT, OutputT>(num_items, cub::Sum(), NullType());
    Test<InputT, OutputT>(num_items, cub::Max(), NullType());
}


/**
 * Test different input sizes
 */
template <
    typename InputT,
    typename OutputT>
void TestSize(
    int     num_items,
    OutputT identity,
    OutputT initial_value)
{
    if (num_items < 0)
    {
        TestOp<InputT>(0,        identity, initial_value);
        TestOp<InputT>(1,        identity, initial_value);
        TestOp<InputT>(100,      identity, initial_value);
        TestOp<InputT>(10000,    identity, initial_value);
        TestOp<InputT>(1000000,  identity, initial_value);
    }
    else
    {
        TestOp<InputT>(num_items, identity, initial_value);
    }
}

class CustomInputT
{
  char m_val{};

public:
  __host__ __device__ explicit CustomInputT(char val)
      : m_val(val)
  {}

  __host__ __device__ int get() const { return static_cast<int>(m_val); }
};

class CustomAccumulatorT
{
  int m_val{0};
  int m_magic_value{42};

  __host__ __device__ CustomAccumulatorT(int val)
      : m_val(val)
  {}

public:
  __host__ __device__ CustomAccumulatorT()
  {}

  __host__ __device__ CustomAccumulatorT(const CustomAccumulatorT &in)
    : m_val(in.is_valid() * in.get())
    , m_magic_value(in.is_valid() * 42)
  {}

  __host__ __device__ CustomAccumulatorT(const CustomInputT &in)
    : m_val(in.get())
    , m_magic_value(42)
  {}

  __host__ __device__ void operator=(const CustomInputT &in)
  {
    if (this->is_valid())
    {
      m_val = in.get();
    }
  }

  __host__ __device__ void operator=(const CustomAccumulatorT &in)
  {
    if (this->is_valid() && in.is_valid())
    {
      m_val = in.get();
    }
  }

  __host__ __device__ CustomAccumulatorT 
  operator+(const CustomInputT &in) const
  {
    const int multiplier = this->is_valid();
    return {(m_val + in.get()) * multiplier};
  }

  __host__ __device__ CustomAccumulatorT
  operator+(const CustomAccumulatorT &in) const
  {
    const int multiplier = this->is_valid() && in.is_valid();
    return {(m_val + in.get()) * multiplier};
  }

  __host__ __device__ int get() const { return m_val; }

  __host__ __device__ bool is_valid() const { return m_magic_value == 42; }
};

class CustomOutputT
{
  int *m_d_ok_count{};
  int m_expected{};

public:
  __host__ __device__ CustomOutputT(int *d_ok_count, int expected)
      : m_d_ok_count(d_ok_count)
      , m_expected(expected)
  {}

  __device__ void operator=(const CustomAccumulatorT &accum) const
  {
    const int ok = accum.is_valid() && (accum.get() == m_expected);
    atomicAdd(m_d_ok_count, ok);
  }
};

__global__ void InitializeTestAccumulatorTypes(int num_items,
                                               int *d_ok_count,
                                               CustomInputT *d_in,
                                               CustomOutputT *d_out)
{
  const int idx = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);

  if (idx < num_items)
  {
    d_in[idx] = CustomInputT(1);
    d_out[idx] = CustomOutputT{d_ok_count, idx};
  }
}

void TestAccumulatorTypes()
{
  const int num_items  = 2 * 1024 * 1024;
  const int block_size = 256;
  const int grid_size  = (num_items + block_size - 1) / block_size;

  CustomInputT *d_in{};
  CustomOutputT *d_out{};
  CustomAccumulatorT init{};
  int *d_ok_count{};

  CubDebugExit(g_allocator.DeviceAllocate((void **)&d_ok_count, sizeof(int)));
  CubDebugExit(g_allocator.DeviceAllocate((void **)&d_out,
                                          sizeof(CustomOutputT) * num_items));
  CubDebugExit(g_allocator.DeviceAllocate((void **)&d_in,
                                          sizeof(CustomInputT) * num_items));

  InitializeTestAccumulatorTypes<<<grid_size, block_size>>>(num_items,
                                                            d_ok_count,
                                                            d_in,
                                                            d_out);

  std::uint8_t *d_temp_storage{};
  std::size_t temp_storage_bytes{};

  CubDebugExit(cub::DeviceScan::ExclusiveScan(d_temp_storage,
                                              temp_storage_bytes,
                                              d_in,
                                              d_out,
                                              cub::Sum{},
                                              init,
                                              num_items));

  CubDebugExit(
    g_allocator.DeviceAllocate((void **)&d_temp_storage, temp_storage_bytes));
  CubDebugExit(cudaMemset(d_temp_storage, 1, temp_storage_bytes));

  CubDebugExit(cub::DeviceScan::ExclusiveScan(d_temp_storage,
                                              temp_storage_bytes,
                                              d_in,
                                              d_out,
                                              cub::Sum{},
                                              init,
                                              num_items));

  int ok{};
  CubDebugExit(cudaMemcpy(&ok, d_ok_count, sizeof(int), cudaMemcpyDeviceToHost));

  AssertEquals(ok, num_items);

  CubDebugExit(g_allocator.DeviceFree(d_out));
  CubDebugExit(g_allocator.DeviceFree(d_in));
  CubDebugExit(g_allocator.DeviceFree(d_ok_count));
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
    // %PARAM% TEST_VALUE_TYPES types 0:1:2

#if TEST_VALUE_TYPES == 0

    // Test different input+output data types
    TestSize<unsigned char>(num_items, (int)0, (int)99);

    // Test same input+output data types
    TestSize<unsigned char>(num_items, (unsigned char)0, (unsigned char)99);
    TestSize<signed char>(num_items, (char)0, (char)99);
    TestSize<unsigned short>(num_items, (unsigned short)0, (unsigned short)99);
    TestSize<unsigned int>(num_items, (unsigned int)0, (unsigned int)99);
    TestSize<unsigned long long>(num_items,
                                 (unsigned long long)0,
                                 (unsigned long long)99);

#elif TEST_VALUE_TYPES == 1

    TestSize<uchar2>(num_items, make_uchar2(0, 0), make_uchar2(17, 21));
    TestSize<char2>(num_items, make_char2(0, 0), make_char2(17, 21));
    TestSize<ushort2>(num_items, make_ushort2(0, 0), make_ushort2(17, 21));
    TestSize<uint2>(num_items, make_uint2(0, 0), make_uint2(17, 21));
    TestSize<ulonglong2>(num_items,
                         make_ulonglong2(0, 0),
                         make_ulonglong2(17, 21));
    TestSize<uchar4>(num_items,
                     make_uchar4(0, 0, 0, 0),
                     make_uchar4(17, 21, 32, 85));
#elif TEST_VALUE_TYPES == 2
    TestSize<char4>(num_items,
                    make_char4(0, 0, 0, 0),
                    make_char4(17, 21, 32, 85));

    TestSize<ushort4>(num_items,
                      make_ushort4(0, 0, 0, 0),
                      make_ushort4(17, 21, 32, 85));
    TestSize<uint4>(num_items,
                    make_uint4(0, 0, 0, 0),
                    make_uint4(17, 21, 32, 85));
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

    TestAccumulatorTypes();
#endif

    return 0;
}

