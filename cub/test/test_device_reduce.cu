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
 * Test of DeviceReduce utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/iterator/constant_input_iterator.cuh>
#include <cub/iterator/discard_output_iterator.cuh>
#include <cub/iterator/transform_input_iterator.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_allocator.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cstdio>
#include <limits>
#include <type_traits>
#include <typeinfo>

#include "test_util.h"
#include <nv/target>

#define TEST_HALF_T !_NVHPC_CUDA

#define TEST_BF_T !_NVHPC_CUDA

#if TEST_HALF_T
#include <cuda_fp16.h>

// Half support is provided by SM53+. We currently test against a few older architectures. 
// The specializations below can be removed once we drop these architectures.  
namespace cub {

template <>
__host__ __device__ __forceinline__ //
__half Min::operator()(__half& a, __half& b) const 
{
    NV_IF_TARGET(NV_PROVIDES_SM_53, 
                    (return CUB_MIN(a, b);),
                    (return CUB_MIN(__half2float(a), __half2float(b));));
}

template <>
__host__ __device__ __forceinline__ //
KeyValuePair<int, __half> 
ArgMin::operator()(const KeyValuePair<int, __half> &a, 
                   const KeyValuePair<int, __half> &b) const 
{
    const float av = __half2float(a.value);
    const float bv = __half2float(b.value);

    if ((bv < av) || ((av == bv) && (b.key < a.key)))
    {
      return b;
    }

    return a;
}

template <>
__host__ __device__ __forceinline__ //
__half Max::operator()(__half& a, __half& b) const 
{
    NV_IF_TARGET(NV_PROVIDES_SM_53, 
                    (return CUB_MAX(a, b);),
                    (return CUB_MAX(__half2float(a), __half2float(b));));
}

template <>
__host__ __device__ __forceinline__ //
KeyValuePair<int, __half> 
ArgMax::operator()(const KeyValuePair<int, __half> &a, 
                   const KeyValuePair<int, __half> &b) const 
{
    const float av = __half2float(a.value);
    const float bv = __half2float(b.value);

    if ((bv > av) || ((av == bv) && (b.key < a.key)))
    {
      return b;
    }

    return a;
}

} // namespace cub
#endif

#if TEST_BF_T
#include <cuda_bf16.h>
#endif

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

int                     g_ptx_version;
int                     g_sm_count;
double                  g_device_giga_bandwidth;
bool                    g_verbose           = false;
bool                    g_verbose_input     = false;
int                     g_timing_iterations = 0;
CachingDeviceAllocator  g_allocator(true);


// Dispatch types
enum Backend
{
    CUB,            // CUB method
    CUB_SEGMENTED,  // CUB segmented method
    CDP,            // GPU-based (dynamic parallelism) dispatch to CUB method
    CDP_SEGMENTED,  // GPU-based segmented method
};

inline const char* BackendToString(Backend b)
{
  switch (b)
  {
    case CUB:
      return "CUB";
    case CUB_SEGMENTED:
      return "CUB_SEGMENTED";
    case CDP:
      return "CDP";
    case CDP_SEGMENTED:
      return "CDP_SEGMENTED";
    default:
      break;
  }

  return "";
}

// Custom max functor
struct CustomMax
{
    /// Boolean max operator, returns <tt>(a > b) ? a : b</tt>
    template <typename T, typename C>
    __host__ __device__ auto operator()(T&& a, C&& b) 
      -> cub::detail::accumulator_t<cub::Max, T, C>
    {
        return CUB_MAX(a, b);
    }

#if TEST_HALF_T
    __host__ __device__ __half operator()(__half& a, __half& b) 
    {
        return cub::Max{}(a, b);
    }
#endif
};

// Comparing results computed on CPU and GPU for extended floating point types is impossible. 
// For instance, when used with a constant iterator of two, the accumulator in sequential reference 
// computation (CPU) bumps into the 4096 limits, which will never change (`4096 + 2 = 4096`). 
// Meanwhile, per-thread aggregates (`2 * 16 = 32`) are accumulated within and among thread blocks, 
// yielding `inf` as a result. No reasonable epsilon can be selected to compare `inf` with `4096`. 
// To make `__half` and `__nv_bfloat16` arithmetic associative, the function object below raises 
// extended floating points to the area of unsigned short integers. This allows us to test large 
// inputs with few code-path differences in device algorithms. 
struct ExtendedFloatSum
{
    template <class T>
    __host__ __device__ T operator()(T a, T b) const
    {
        T result{};
        result.__x = a.raw() + b.raw();
        return result;
    }

#if TEST_HALF_T
    __host__ __device__ __half operator()(__half a, __half b) const
    {
        uint16_t result = this->operator()(half_t{a}, half_t(b)).raw();
        return reinterpret_cast<__half &>(result);
    }
#endif

#if TEST_BF_T
    __device__ __nv_bfloat16 operator()(__nv_bfloat16 a, __nv_bfloat16 b) const
    {
        uint16_t result = this->operator()(bfloat16_t{a}, bfloat16_t(b)).raw();
        return reinterpret_cast<__nv_bfloat16 &>(result);
    }
#endif
};

//---------------------------------------------------------------------
// Dispatch to different CUB DeviceReduce entrypoints
//---------------------------------------------------------------------

/**
 * Dispatch to reduce entrypoint (custom-max)
 */
template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT, typename ReductionOpT>
CUB_RUNTIME_FUNCTION
cudaError_t Dispatch(
    Int2Type<CUB>       /*dispatch_to*/,
    int                 timing_iterations,
    size_t              */*d_temp_storage_bytes*/,
    cudaError_t         */*d_cdp_error*/,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    int                 num_items,
    int                 /*max_segments*/,
    BeginOffsetIteratorT /*d_segment_begin_offsets*/,
    EndOffsetIteratorT  /*d_segment_end_offsets*/,
    ReductionOpT        reduction_op)
{
    using InputT = cub::detail::value_t<InputIteratorT>;

    // The output value type
    using OutputT = cub::detail::non_void_value_t<OutputIteratorT, InputT>;

    // Max-identity
    OutputT identity = Traits<InputT>::Lowest(); // replace with std::numeric_limits<OutputT>::lowest() when C++ support is more prevalent

    // Invoke kernel to device reduction directly
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_iterations; ++i)
    {
        error = DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes,
            d_in, d_out, num_items, reduction_op, identity);
    }

    return error;
}

/**
 * Dispatch to sum entrypoint
 */
template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
CUB_RUNTIME_FUNCTION
cudaError_t Dispatch(
    Int2Type<CUB>         /*dispatch_to*/,
    int                     timing_iterations,
    size_t *              /*d_temp_storage_bytes*/,
    cudaError_t *         /*d_cdp_error*/,

    void*                   d_temp_storage,
    size_t&                 temp_storage_bytes,
    InputIteratorT          d_in,
    OutputIteratorT         d_out,
    int                     num_items,
    int                   /*max_segments*/,
    BeginOffsetIteratorT  /*d_segment_begin_offsets*/,
    EndOffsetIteratorT    /*d_segment_end_offsets*/,
    cub::Sum              /*reduction_op*/)
{
    // Invoke kernel to device reduction directly
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_iterations; ++i)
    {
        error = DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    }

    return error;
}

/**
 * Dispatch to extended fp sum entrypoint
 */
template <typename InputIteratorT,
          typename OutputIteratorT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT>
CUB_RUNTIME_FUNCTION cudaError_t Dispatch(Int2Type<CUB> /*dispatch_to*/,
                                          int timing_iterations,
                                          size_t * /*d_temp_storage_bytes*/,
                                          cudaError_t * /*d_cdp_error*/,

                                          void *d_temp_storage,
                                          size_t &temp_storage_bytes,
                                          InputIteratorT d_in,
                                          OutputIteratorT d_out,
                                          int num_items,
                                          int /*max_segments*/,
                                          BeginOffsetIteratorT /*d_segment_begin_offsets*/,
                                          EndOffsetIteratorT /*d_segment_end_offsets*/,
                                          ExtendedFloatSum reduction_op)
{
    using InputT  = cub::detail::value_t<InputIteratorT>;
    using OutputT = cub::detail::non_void_value_t<OutputIteratorT, InputT>;

    OutputT identity{};

    // Invoke kernel to device reduction directly
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_iterations; ++i)
    {
        error = DeviceReduce::Reduce(d_temp_storage,
                                     temp_storage_bytes,
                                     d_in,
                                     d_out,
                                     num_items,
                                     reduction_op,
                                     identity);
    }

    return error;
}

/**
 * Dispatch to min entrypoint
 */
template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
CUB_RUNTIME_FUNCTION
cudaError_t Dispatch(
    Int2Type<CUB>       /*dispatch_to*/,
    int                 timing_iterations,
    size_t              */*d_temp_storage_bytes*/,
    cudaError_t         */*d_cdp_error*/,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    int                 num_items,
    int                 /*max_segments*/,
    BeginOffsetIteratorT /*d_segment_begin_offsets*/,
    EndOffsetIteratorT  /*d_segment_end_offsets*/,
    cub::Min            /*reduction_op*/)
{
    // Invoke kernel to device reduction directly
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_iterations; ++i)
    {
        error = DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    }

    return error;
}

/**
 * Dispatch to max entrypoint
 */
template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
CUB_RUNTIME_FUNCTION
cudaError_t Dispatch(
    Int2Type<CUB>       /*dispatch_to*/,
    int                 timing_iterations,
    size_t              */*d_temp_storage_bytes*/,
    cudaError_t         */*d_cdp_error*/,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    int                 num_items,
    int                 /*max_segments*/,
    BeginOffsetIteratorT /*d_segment_begin_offsets*/,
    EndOffsetIteratorT  /*d_segment_end_offsets*/,
    cub::Max            /*reduction_op*/)
{
    // Invoke kernel to device reduction directly
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_iterations; ++i)
    {
        error = DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    }

    return error;
}

/**
 * Dispatch to argmin entrypoint
 */
template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
CUB_RUNTIME_FUNCTION
cudaError_t Dispatch(
    Int2Type<CUB>       /*dispatch_to*/,
    int                 timing_iterations,
    size_t              */*d_temp_storage_bytes*/,
    cudaError_t         */*d_cdp_error*/,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    int                 num_items,
    int                 /*max_segments*/,
    BeginOffsetIteratorT /*d_segment_begin_offsets*/,
    EndOffsetIteratorT  /*d_segment_end_offsets*/,
    cub::ArgMin         /*reduction_op*/)
{
    // Invoke kernel to device reduction directly
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_iterations; ++i)
    {
        error = DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    }

    return error;
}

/**
 * Dispatch to argmax entrypoint
 */
template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
CUB_RUNTIME_FUNCTION
cudaError_t Dispatch(
    Int2Type<CUB>       /*dispatch_to*/,
    int                 timing_iterations,
    size_t              */*d_temp_storage_bytes*/,
    cudaError_t         */*d_cdp_error*/,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    int                 num_items,
    int                 /*max_segments*/,
    BeginOffsetIteratorT /*d_segment_begin_offsets*/,
    EndOffsetIteratorT  /*d_segment_end_offsets*/,
    cub::ArgMax         /*reduction_op*/)
{
    // Invoke kernel to device reduction directly
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_iterations; ++i)
    {
        error = DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    }

    return error;
}


//---------------------------------------------------------------------
// Dispatch to different CUB DeviceSegmentedReduce entrypoints
//---------------------------------------------------------------------

/**
 * Dispatch to reduce entrypoint (custom-max)
 */
template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT, typename ReductionOpT>
CUB_RUNTIME_FUNCTION
cudaError_t Dispatch(
    Int2Type<CUB_SEGMENTED>       /*dispatch_to*/,
    int                 timing_iterations,
    size_t              */*d_temp_storage_bytes*/,
    cudaError_t         */*d_cdp_error*/,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    int                 /*num_items*/,
    int                 max_segments,
    BeginOffsetIteratorT d_segment_begin_offsets,
    EndOffsetIteratorT  d_segment_end_offsets,
    ReductionOpT        reduction_op)
{
    // The input value type
    using InputT = cub::detail::value_t<InputIteratorT>;

    // The output value type
    using OutputT = cub::detail::non_void_value_t<OutputIteratorT, InputT>;

    // Max-identity
    OutputT identity = Traits<InputT>::Lowest(); // replace with std::numeric_limits<OutputT>::lowest() when C++ support is more prevalent

    // Invoke kernel to device reduction directly
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_iterations; ++i)
    {
        error = DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes,
            d_in, d_out, max_segments, d_segment_begin_offsets, d_segment_end_offsets, reduction_op, identity);
    }
    return error;
}

/**
 * Dispatch to sum entrypoint
 */
template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
CUB_RUNTIME_FUNCTION
cudaError_t Dispatch(
    Int2Type<CUB_SEGMENTED>       /*dispatch_to*/,
    int                 timing_iterations,
    size_t              */*d_temp_storage_bytes*/,
    cudaError_t         */*d_cdp_error*/,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    int                 /*num_items*/,
    int                 max_segments,
    BeginOffsetIteratorT d_segment_begin_offsets,
    EndOffsetIteratorT  d_segment_end_offsets,
    cub::Sum            /*reduction_op*/)
{
    // Invoke kernel to device reduction directly
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_iterations; ++i)
    {
        error = DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes,
            d_in, d_out, max_segments, d_segment_begin_offsets, d_segment_end_offsets);
    }
    return error;
}

/**
 * Dispatch to extended fp sum entrypoint
 */
template <typename InputIteratorT,
          typename OutputIteratorT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT>
CUB_RUNTIME_FUNCTION cudaError_t Dispatch(Int2Type<CUB_SEGMENTED> /*dispatch_to*/,
                                          int timing_iterations,
                                          size_t * /*d_temp_storage_bytes*/,
                                          cudaError_t * /*d_cdp_error*/,

                                          void *d_temp_storage,
                                          size_t &temp_storage_bytes,
                                          InputIteratorT d_in,
                                          OutputIteratorT d_out,
                                          int /*num_items*/,
                                          int max_segments,
                                          BeginOffsetIteratorT d_segment_begin_offsets,
                                          EndOffsetIteratorT d_segment_end_offsets,
                                          ExtendedFloatSum reduction_op)
{
    using InputT  = cub::detail::value_t<InputIteratorT>;
    using OutputT = cub::detail::non_void_value_t<OutputIteratorT, InputT>;

    OutputT identity{};

    // Invoke kernel to device reduction directly
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_iterations; ++i)
    {
        error = DeviceSegmentedReduce::Reduce(d_temp_storage,
                                              temp_storage_bytes,
                                              d_in,
                                              d_out,
                                              max_segments,
                                              d_segment_begin_offsets,
                                              d_segment_end_offsets,
                                              reduction_op,
                                              identity);
    }

    return error;
}

/**
 * Dispatch to min entrypoint
 */
template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
CUB_RUNTIME_FUNCTION
cudaError_t Dispatch(
    Int2Type<CUB_SEGMENTED>       /*dispatch_to*/,
    int                 timing_iterations,
    size_t              */*d_temp_storage_bytes*/,
    cudaError_t         */*d_cdp_error*/,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    int                 /*num_items*/,
    int                 max_segments,
    BeginOffsetIteratorT d_segment_begin_offsets,
    EndOffsetIteratorT  d_segment_end_offsets,
    cub::Min            /*reduction_op*/)
{
    // Invoke kernel to device reduction directly
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_iterations; ++i)
    {
        error = DeviceSegmentedReduce::Min(d_temp_storage, temp_storage_bytes,
            d_in, d_out, max_segments, d_segment_begin_offsets, d_segment_end_offsets);
    }
    return error;
}

/**
 * Dispatch to max entrypoint
 */
template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
CUB_RUNTIME_FUNCTION
cudaError_t Dispatch(
    Int2Type<CUB_SEGMENTED>       /*dispatch_to*/,
    int                 timing_iterations,
    size_t              */*d_temp_storage_bytes*/,
    cudaError_t         */*d_cdp_error*/,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    int                 /*num_items*/,
    int                 max_segments,
    BeginOffsetIteratorT d_segment_begin_offsets,
    EndOffsetIteratorT  d_segment_end_offsets,
    cub::Max            /*reduction_op*/)
{
    // Invoke kernel to device reduction directly
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_iterations; ++i)
    {
        error = DeviceSegmentedReduce::Max(d_temp_storage, temp_storage_bytes,
            d_in, d_out, max_segments, d_segment_begin_offsets, d_segment_end_offsets);
    }
    return error;
}

/**
 * Dispatch to argmin entrypoint
 */
template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
CUB_RUNTIME_FUNCTION
cudaError_t Dispatch(
    Int2Type<CUB_SEGMENTED>       /*dispatch_to*/,
    int                 timing_iterations,
    size_t              */*d_temp_storage_bytes*/,
    cudaError_t         */*d_cdp_error*/,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    int                 /*num_items*/,
    int                 max_segments,
    BeginOffsetIteratorT d_segment_begin_offsets,
    EndOffsetIteratorT  d_segment_end_offsets,
    cub::ArgMin         /*reduction_op*/)
{
    // Invoke kernel to device reduction directly
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_iterations; ++i)
    {
        error = DeviceSegmentedReduce::ArgMin(d_temp_storage, temp_storage_bytes,
            d_in, d_out, max_segments, d_segment_begin_offsets, d_segment_end_offsets);
    }
    return error;
}

/**
 * Dispatch to argmax entrypoint
 */
template <typename InputIteratorT, typename OutputIteratorT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
CUB_RUNTIME_FUNCTION
cudaError_t Dispatch(
    Int2Type<CUB_SEGMENTED>       /*dispatch_to*/,
    int                 timing_iterations,
    size_t              */*d_temp_storage_bytes*/,
    cudaError_t         */*d_cdp_error*/,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    int                 /*num_items*/,
    int                 max_segments,
    BeginOffsetIteratorT d_segment_begin_offsets,
    EndOffsetIteratorT  d_segment_end_offsets,
    cub::ArgMax         /*reduction_op*/)
{
    // Invoke kernel to device reduction directly
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_iterations; ++i)
    {
        error = DeviceSegmentedReduce::ArgMax(d_temp_storage, temp_storage_bytes,
            d_in, d_out, max_segments, d_segment_begin_offsets, d_segment_end_offsets);
    }
    return error;
}


//---------------------------------------------------------------------
// CUDA nested-parallelism test kernel
//---------------------------------------------------------------------

#if TEST_CDP == 1

/**
 * Simple wrapper kernel to invoke DeviceReduce
 */
template <int CubBackend,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename ReductionOpT>
__global__ void CDPDispatchKernel(Int2Type<CubBackend> cub_backend,
                                  int                  timing_iterations,
                                  size_t              *d_temp_storage_bytes,
                                  cudaError_t         *d_cdp_error,

                                  void                *d_temp_storage,
                                  size_t               temp_storage_bytes,
                                  InputIteratorT       d_in,
                                  OutputIteratorT      d_out,
                                  int                  num_items,
                                  int                  max_segments,
                                  BeginOffsetIteratorT d_segment_begin_offsets,
                                  EndOffsetIteratorT   d_segment_end_offsets,
                                  ReductionOpT         reduction_op)
{
  *d_cdp_error = Dispatch(cub_backend,
                          timing_iterations,
                          d_temp_storage_bytes,
                          d_cdp_error,
                          d_temp_storage,
                          temp_storage_bytes,
                          d_in,
                          d_out,
                          num_items,
                          max_segments,
                          d_segment_begin_offsets,
                          d_segment_end_offsets,
                          reduction_op);

  *d_temp_storage_bytes = temp_storage_bytes;
}

/**
 * Launch kernel and dispatch on device. Should only be called from host code.
 * The CubBackend should be one of the non-CDP CUB backends to invoke from the
 * device.
 */
template <int CubBackend,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename ReductionOpT>
cudaError_t LaunchCDPKernel(Int2Type<CubBackend> cub_backend,
                            int                  timing_iterations,
                            size_t              *d_temp_storage_bytes,
                            cudaError_t         *d_cdp_error,

                            void                *d_temp_storage,
                            size_t              &temp_storage_bytes,
                            InputIteratorT       d_in,
                            OutputIteratorT      d_out,
                            int                  num_items,
                            int                  max_segments,
                            BeginOffsetIteratorT d_segment_begin_offsets,
                            EndOffsetIteratorT   d_segment_end_offsets,
                            ReductionOpT         reduction_op)
{
  cudaError_t retval =
    thrust::cuda_cub::launcher::triple_chevron(1, 1, 0, 0)
      .doit(CDPDispatchKernel<CubBackend,
                              InputIteratorT,
                              OutputIteratorT,
                              BeginOffsetIteratorT,
                              EndOffsetIteratorT,
                              ReductionOpT>,
            cub_backend,
            timing_iterations,
            d_temp_storage_bytes,
            d_cdp_error,
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            num_items,
            max_segments,
            d_segment_begin_offsets,
            d_segment_end_offsets,
            reduction_op);
  CubDebugExit(retval);
  CubDebugExit(cub::detail::device_synchronize());

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

// Specializations of Dispatch that translate the CDP backend to the appropriate
// CUB backend, and uses the CUB backend to launch the CDP kernel.
#define DEFINE_CDP_DISPATCHER(CdpBackend, CubBackend)                          \
  template <typename InputIteratorT,                                           \
            typename OutputIteratorT,                                          \
            typename BeginOffsetIteratorT,                                     \
            typename EndOffsetIteratorT,                                       \
            typename ReductionOpT>                                             \
  cudaError_t Dispatch(Int2Type<CdpBackend>,                                   \
                       int          timing_iterations,                         \
                       size_t      *d_temp_storage_bytes,                      \
                       cudaError_t *d_cdp_error,                               \
                                                                               \
                       void                *d_temp_storage,                    \
                       size_t              &temp_storage_bytes,                \
                       InputIteratorT       d_in,                              \
                       OutputIteratorT      d_out,                             \
                       int                  num_items,                         \
                       int                  max_segments,                      \
                       BeginOffsetIteratorT d_segment_begin_offsets,           \
                       EndOffsetIteratorT   d_segment_end_offsets,             \
                       ReductionOpT         reduction_op)                      \
  {                                                                            \
    Int2Type<CubBackend> cub_backend{};                                        \
    return LaunchCDPKernel(cub_backend,                                        \
                           timing_iterations,                                  \
                           d_temp_storage_bytes,                               \
                           d_cdp_error,                                        \
                           d_temp_storage,                                     \
                           temp_storage_bytes,                                 \
                           d_in,                                               \
                           d_out,                                              \
                           num_items,                                          \
                           max_segments,                                       \
                           d_segment_begin_offsets,                            \
                           d_segment_end_offsets,                              \
                           reduction_op);                                      \
  }

DEFINE_CDP_DISPATCHER(CDP, CUB)
DEFINE_CDP_DISPATCHER(CDP_SEGMENTED, CUB_SEGMENTED)

#undef DEFINE_CDP_DISPATCHER

#endif // TEST_CDP

//---------------------------------------------------------------------
// Problem generation
//---------------------------------------------------------------------
/// Initialize problem
template <typename InputT>
void Initialize(
    GenMode         gen_mode,
    InputT          *h_in,
    int             num_items)
{
    for (int i = 0; i < num_items; ++i)
    {
        InitValue(gen_mode, h_in[i], i);
    }

    if (g_verbose_input)
    {
        printf("Input:\n");
        DisplayResults(h_in, num_items);
        printf("\n\n");
    }
}


/// Solve problem (max/custom-max functor)
template <typename ReductionOpT, typename InputT, typename _OutputT>
struct Solution
{
    using OutputT = _OutputT;
    using InitT = OutputT;
    using AccumT = cub::detail::accumulator_t<ReductionOpT, InitT, InputT>;

    template <typename HostInputIteratorT, typename OffsetT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    static void Solve(HostInputIteratorT h_in, OutputT *h_reference, OffsetT num_segments,
        BeginOffsetIteratorT h_segment_begin_offsets, EndOffsetIteratorT h_segment_end_offsets, ReductionOpT reduction_op)
    {
        for (int i = 0; i < num_segments; ++i)
        {
            AccumT aggregate = Traits<InputT>::Lowest(); // replace with std::numeric_limits<OutputT>::lowest() when C++ support is more prevalent
            for (int j = h_segment_begin_offsets[i]; j < h_segment_end_offsets[i]; ++j)
                aggregate = reduction_op(aggregate, OutputT(h_in[j]));
            h_reference[i] = aggregate;
        }
    }
};

/// Solve problem (min functor)
template <typename InputT, typename _OutputT>
struct Solution<cub::Min, InputT, _OutputT>
{
    using OutputT = _OutputT;
    using InitT = OutputT;
    using AccumT = cub::detail::accumulator_t<cub::Min, InitT, InputT>;

    template <typename HostInputIteratorT, typename OffsetT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    static void Solve(HostInputIteratorT h_in, OutputT *h_reference, OffsetT num_segments,
        BeginOffsetIteratorT h_segment_begin_offsets, EndOffsetIteratorT h_segment_end_offsets, cub::Min reduction_op)
    {
        for (int i = 0; i < num_segments; ++i)
        {
            AccumT aggregate = Traits<InputT>::Max();    // replace with std::numeric_limits<OutputT>::max() when C++ support is more prevalent
            for (int j = h_segment_begin_offsets[i]; j < h_segment_end_offsets[i]; ++j)
                aggregate = reduction_op(aggregate, OutputT(h_in[j]));
            h_reference[i] = aggregate;
        }
    }
};


/// Solve problem (sum functor)
template <typename InputT, typename _OutputT>
struct Solution<cub::Sum, InputT, _OutputT>
{
    using OutputT = _OutputT;
    using InitT = OutputT;
    using AccumT = cub::detail::accumulator_t<cub::Sum, InitT, InputT>;

    template <typename HostInputIteratorT, typename OffsetT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT, typename ReductionOpT>
    static void Solve(HostInputIteratorT h_in, OutputT *h_reference, OffsetT num_segments,
        BeginOffsetIteratorT h_segment_begin_offsets, EndOffsetIteratorT h_segment_end_offsets, ReductionOpT reduction_op)
    {
        for (int i = 0; i < num_segments; ++i)
        {
            AccumT aggregate{};
            for (int j = h_segment_begin_offsets[i]; j < h_segment_end_offsets[i]; ++j)
                aggregate = reduction_op(aggregate, h_in[j]);
            h_reference[i] = static_cast<OutputT>(aggregate);
        }
    }
};

template <typename InputT, typename _OutputT>
struct Solution<ExtendedFloatSum, InputT, _OutputT>
{
    using OutputT = _OutputT;
    using InitT   = OutputT;
    using AccumT  = cub::detail::accumulator_t<cub::Sum, InitT, InputT>;

    template <typename HostInputIteratorT,
              typename OffsetT,
              typename BeginOffsetIteratorT,
              typename EndOffsetIteratorT,
              typename ReductionOpT>
    static void Solve(HostInputIteratorT h_in,
                      OutputT *h_reference,
                      OffsetT num_segments,
                      BeginOffsetIteratorT h_segment_begin_offsets,
                      EndOffsetIteratorT h_segment_end_offsets,
                      ReductionOpT reduction_op)
    {
        for (int i = 0; i < num_segments; ++i)
        {
            AccumT aggregate{};
            for (int j = h_segment_begin_offsets[i]; j < h_segment_end_offsets[i]; ++j)
                aggregate = reduction_op(aggregate, h_in[j]);
            h_reference[i] = static_cast<OutputT>(aggregate);
        }
    }
};

/// Solve problem (argmin functor)
template <typename InputValueT, typename OutputValueT>
struct Solution<cub::ArgMin, InputValueT, OutputValueT>
{
    typedef KeyValuePair<int, OutputValueT> OutputT;

    template <typename HostInputIteratorT, typename OffsetT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    static void Solve(HostInputIteratorT h_in, OutputT *h_reference, OffsetT num_segments,
        BeginOffsetIteratorT h_segment_begin_offsets, EndOffsetIteratorT h_segment_end_offsets, cub::ArgMin reduction_op)
    {
        for (int i = 0; i < num_segments; ++i)
        {
            const auto segment_begin = h_segment_begin_offsets[i];
            const auto segment_end = h_segment_end_offsets[i];

            if (segment_begin < segment_end) 
            {
                OutputT aggregate(0, OutputValueT(h_in[segment_begin]));
                for (int j = segment_begin + 1; j < segment_end; ++j)
                {
                    OutputT item(j - segment_begin, OutputValueT(h_in[j]));
                    aggregate = reduction_op(aggregate, item);
                }
                h_reference[i] = aggregate;
            }
            else 
            {
                // Guaranteed output for empty segments
                OutputT aggregate(1, Traits<InputValueT>::Max()); // replace with std::numeric_limits<OutputT>::max() when C++ support is more prevalent
                h_reference[i] = aggregate;
            }
        }
    }
};


/// Solve problem (argmax functor)
template <typename InputValueT, typename OutputValueT>
struct Solution<cub::ArgMax, InputValueT, OutputValueT>
{
    typedef KeyValuePair<int, OutputValueT> OutputT;

    template <typename HostInputIteratorT, typename OffsetT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT>
    static void Solve(HostInputIteratorT h_in, OutputT *h_reference, OffsetT num_segments,
        BeginOffsetIteratorT h_segment_begin_offsets, EndOffsetIteratorT h_segment_end_offsets, cub::ArgMax reduction_op)
    {
        for (int i = 0; i < num_segments; ++i)
        {
            const auto segment_begin = h_segment_begin_offsets[i];
            const auto segment_end = h_segment_end_offsets[i];

            if (segment_begin < segment_end) 
            {
                OutputT aggregate(0, OutputValueT(h_in[segment_begin])); 
                for (int j = segment_begin + 1; j < segment_end; ++j)
                {
                    OutputT item(j - segment_begin, OutputValueT(h_in[j]));
                    aggregate = reduction_op(aggregate, item);
                }
                h_reference[i] = aggregate;
            }
            else 
            {
                OutputT aggregate(1, Traits<InputValueT>::Lowest());
                h_reference[i] = aggregate;
            }
        }
    }
};


//---------------------------------------------------------------------
// Problem generation
//---------------------------------------------------------------------

template <class It>
It unwrap_it(It it) {
    return it;
}

#if TEST_HALF_T
__half* unwrap_it(half_t *it) {
    return reinterpret_cast<__half*>(it);
}

template <class OffsetT>
ConstantInputIterator<__half, OffsetT> unwrap_it(ConstantInputIterator<half_t, OffsetT> it) {
    half_t wrapped_val = *it;
    __half val = wrapped_val.operator __half();
    return ConstantInputIterator<__half, OffsetT>(val);
}
#endif

#if TEST_BF_T
__nv_bfloat16* unwrap_it(bfloat16_t* it) {
    return reinterpret_cast<__nv_bfloat16*>(it);
}

template <class OffsetT>
ConstantInputIterator<__nv_bfloat16, OffsetT> unwrap_it(ConstantInputIterator<bfloat16_t, OffsetT> it) {
    bfloat16_t wrapped_val = *it;
    __nv_bfloat16 val = wrapped_val.operator __nv_bfloat16();
    return ConstantInputIterator<__nv_bfloat16, OffsetT>(val);
}
#endif

template <class WrappedItT, //
          class ItT = decltype(unwrap_it(std::declval<WrappedItT>()))>
std::integral_constant<bool, !std::is_same<WrappedItT, ItT>::value> //
reference_extended_fp(WrappedItT)
{
    return {};
}

ExtendedFloatSum unwrap_op(std::true_type /* extended float */, cub::Sum) //
{
    return {};
}

template <bool V, class OpT>
OpT unwrap_op(std::integral_constant<bool, V> /* base case */, OpT op)
{
    return op;
}

/// Test DeviceReduce for a given problem input
template <
    typename                BackendT,
    typename                DeviceWrappedInputIteratorT,
    typename                DeviceWrappedOutputIteratorT,
    typename                HostReferenceIteratorT,
    typename                OffsetT,
    typename                BeginOffsetIteratorT,
    typename                EndOffsetIteratorT,
    typename                ReductionOpT>
void Test(
    BackendT                        backend,
    DeviceWrappedInputIteratorT     d_wrapped_in,
    DeviceWrappedOutputIteratorT    d_wrapped_out,
    OffsetT                         num_items,
    OffsetT                         num_segments,
    BeginOffsetIteratorT            d_segment_begin_offsets,
    EndOffsetIteratorT              d_segment_end_offsets,
    ReductionOpT                    reduction_op,
    HostReferenceIteratorT          h_reference)
{
    // Input data types
    auto d_in = unwrap_it(d_wrapped_in);
    auto d_out = unwrap_it(d_wrapped_out);

    // Allocate CDP device arrays for temp storage size and error
    size_t          *d_temp_storage_bytes = NULL;
    cudaError_t     *d_cdp_error = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage_bytes,  sizeof(size_t) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cdp_error,           sizeof(cudaError_t) * 1));

    // Inquire temp device storage
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;
    CubDebugExit(Dispatch(backend, 1,
        d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes,
        d_in, d_out, num_items, num_segments, d_segment_begin_offsets, d_segment_end_offsets,
        reduction_op));

    // Allocate temp device storage
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Run warmup/correctness iteration
    CubDebugExit(Dispatch(backend, 1,
        d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes,
        d_in, d_out, num_items, num_segments, d_segment_begin_offsets, d_segment_end_offsets,
        reduction_op));

    // Check for correctness (and display results, if specified)
    int compare = CompareDeviceResults(h_reference, d_wrapped_out, num_segments, g_verbose, g_verbose);

    printf("\t%s", compare ? "FAIL" : "PASS");

    // Flush any stdout/stderr
    fflush(stdout);
    fflush(stderr);

    // Performance
    if (g_timing_iterations > 0)
    {
        GpuTimer gpu_timer;
        gpu_timer.Start();

        CubDebugExit(Dispatch(backend, g_timing_iterations,
            d_temp_storage_bytes, d_cdp_error, d_temp_storage, temp_storage_bytes,
            d_in, d_out, num_items, num_segments, d_segment_begin_offsets, d_segment_end_offsets,
            reduction_op));

        gpu_timer.Stop();
        float elapsed_millis = gpu_timer.ElapsedMillis();

        // Display performance
        float avg_millis = elapsed_millis / g_timing_iterations;
        float giga_rate = float(num_items) / avg_millis / 1000.0f / 1000.0f;
        float giga_bandwidth = giga_rate * sizeof(h_reference[0]);
        printf(", %.3f avg ms, %.3f billion items/s, %.3f logical GB/s, %.1f%% peak",
            avg_millis, giga_rate, giga_bandwidth, giga_bandwidth / g_device_giga_bandwidth * 100.0);

    }

    if (d_temp_storage_bytes) CubDebugExit(g_allocator.DeviceFree(d_temp_storage_bytes));
    if (d_cdp_error) CubDebugExit(g_allocator.DeviceFree(d_cdp_error));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    // Correctness asserts
    AssertEquals(0, compare);
}

template <Backend BACKEND,
          typename OutputValueT,
          typename HostInputIteratorT,
          typename DeviceInputIteratorT,
          typename OffsetT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename ReductionOpT>
void SolveAndTest(HostInputIteratorT h_in,
                  DeviceInputIteratorT d_in,
                  OffsetT num_items,
                  OffsetT num_segments,
                  BeginOffsetIteratorT h_segment_begin_offsets,
                  EndOffsetIteratorT h_segment_end_offsets,
                  BeginOffsetIteratorT d_segment_begin_offsets,
                  EndOffsetIteratorT d_segment_end_offsets,
                  ReductionOpT wrapped_reduction_op)
{
    auto reduction_op = unwrap_op(reference_extended_fp(d_in), wrapped_reduction_op);

    using InputValueT = cub::detail::value_t<DeviceInputIteratorT>;
    using SolutionT = Solution<decltype(reduction_op), InputValueT, OutputValueT>;
    using OutputT = typename SolutionT::OutputT;

    printf("\n\n%s cub::DeviceReduce<%s> %d items (%s), %d segments\n",
           BackendToString(BACKEND),
           typeid(ReductionOpT).name(),
           num_items,
           typeid(HostInputIteratorT).name(),
           num_segments);
    fflush(stdout);

    // Allocate and solve solution
    OutputT *h_reference = new OutputT[num_segments];
    SolutionT::Solve(h_in, h_reference, num_segments, h_segment_begin_offsets, h_segment_end_offsets, reduction_op);

    // Run with discard iterator
    DiscardOutputIterator<OffsetT> discard_itr;
    Test(Int2Type<BACKEND>(), d_in, discard_itr, num_items, num_segments, d_segment_begin_offsets, d_segment_end_offsets, reduction_op, h_reference);

    // Run with output data
    OutputT *d_out = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(OutputT) * num_segments));
    CubDebugExit(cudaMemset(d_out, 0, sizeof(OutputT) * num_segments));
    Test(Int2Type<BACKEND>(), d_in, d_out, num_items, num_segments, d_segment_begin_offsets, d_segment_end_offsets, reduction_op, h_reference);

    // Cleanup
    if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
    if (h_reference) delete[] h_reference;
}

/// Test specific problem type
template <
    Backend         BACKEND,
    typename        InputT,
    typename        OutputT,
    typename        OffsetT,
    typename        ReductionOpT>
void TestProblem(
    OffsetT         num_items,
    OffsetT         num_segments,
    GenMode         gen_mode,
    ReductionOpT    reduction_op)
{
    printf("\n\nInitializing %d %s->%s (gen mode %d)... ", num_items, typeid(InputT).name(), typeid(OutputT).name(), gen_mode); fflush(stdout);
    fflush(stdout);

    // Initialize value data
    InputT* h_in = new InputT[num_items];
    Initialize(gen_mode, h_in, num_items);

    // Initialize segment data
    OffsetT *h_segment_offsets = new OffsetT[num_segments + 1];
    InitializeSegments(num_items, num_segments, h_segment_offsets, g_verbose_input);

    // Initialize device data
    OffsetT *d_segment_offsets      = NULL;
    InputT  *d_in                   = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in,              sizeof(InputT) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_segment_offsets, sizeof(OffsetT) * (num_segments + 1)));
    CubDebugExit(cudaMemcpy(d_in,               h_in,                   sizeof(InputT) * num_items, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_segment_offsets,  h_segment_offsets,      sizeof(OffsetT) * (num_segments + 1), cudaMemcpyHostToDevice));

    SolveAndTest<BACKEND, OutputT>(h_in, d_in, num_items, num_segments,
        h_segment_offsets, h_segment_offsets + 1, d_segment_offsets, d_segment_offsets + 1, reduction_op);

    if (h_segment_offsets)  delete[] h_segment_offsets;
    if (d_segment_offsets)  CubDebugExit(g_allocator.DeviceFree(d_segment_offsets));
    if (h_in)               delete[] h_in;
    if (d_in)               CubDebugExit(g_allocator.DeviceFree(d_in));
}


/// Test different operators
template <
    Backend             BACKEND,
    typename            OutputT,
    typename            HostInputIteratorT,
    typename            DeviceInputIteratorT,
    typename            OffsetT,
    typename            BeginOffsetIteratorT,
    typename            EndOffsetIteratorT>
void TestByOp(
    HostInputIteratorT      h_in,
    DeviceInputIteratorT    d_in,
    OffsetT                 num_items,
    OffsetT                 num_segments,
    BeginOffsetIteratorT    h_segment_begin_offsets,
    EndOffsetIteratorT      h_segment_end_offsets,
    BeginOffsetIteratorT    d_segment_begin_offsets,
    EndOffsetIteratorT      d_segment_end_offsets)
{
    SolveAndTest<BACKEND, OutputT>(h_in, d_in, num_items, num_segments,
        h_segment_begin_offsets, h_segment_end_offsets, d_segment_begin_offsets, d_segment_end_offsets, CustomMax());
    SolveAndTest<BACKEND, OutputT>(h_in, d_in, num_items, num_segments,
        h_segment_begin_offsets, h_segment_end_offsets, d_segment_begin_offsets, d_segment_end_offsets, Sum());
    SolveAndTest<BACKEND, OutputT>(h_in, d_in, num_items, num_segments,
        h_segment_begin_offsets, h_segment_end_offsets, d_segment_begin_offsets, d_segment_end_offsets, Min());
    SolveAndTest<BACKEND, OutputT>(h_in, d_in, num_items, num_segments,
        h_segment_begin_offsets, h_segment_end_offsets, d_segment_begin_offsets, d_segment_end_offsets, ArgMin());
    SolveAndTest<BACKEND, OutputT>(h_in, d_in, num_items, num_segments,
        h_segment_begin_offsets, h_segment_end_offsets, d_segment_begin_offsets, d_segment_end_offsets, Max());
    SolveAndTest<BACKEND, OutputT>(h_in, d_in, num_items, num_segments,
        h_segment_begin_offsets, h_segment_end_offsets, d_segment_begin_offsets, d_segment_end_offsets, ArgMax());
}

template<typename OffsetT>
struct TransformFunctor1
{
    __host__ __device__ __forceinline__ OffsetT operator()(OffsetT offset) const
    {
        return offset;
    }
};

template<typename OffsetT>
struct TransformFunctor2
{
    __host__ __device__ __forceinline__ OffsetT operator()(OffsetT offset) const
    {
        return offset;
    }
};

/// Test different backends
template <
    typename    InputT,
    typename    OutputT,
    typename    OffsetT>
void TestByBackend(
    OffsetT     num_items,
    OffsetT     max_segments,
    GenMode     gen_mode)
{
#if TEST_CDP == 0
  constexpr auto NonSegmentedBackend   = CUB;
  constexpr auto SegmentedBackend      = CUB_SEGMENTED;
#else  // TEST_CDP
  constexpr auto NonSegmentedBackend   = CDP;
  constexpr auto SegmentedBackend      = CDP_SEGMENTED;
#endif // TEST_CDP

    // Initialize host data
    printf("\n\nInitializing %d %s -> %s (gen mode %d)... ",
        num_items, typeid(InputT).name(), typeid(OutputT).name(), gen_mode); fflush(stdout);

    InputT  *h_in               = new InputT[num_items];
    OffsetT *h_segment_offsets  = new OffsetT[max_segments + 1];
    Initialize(gen_mode, h_in, num_items);

    // Initialize device data
    InputT  *d_in               = NULL;
    OffsetT *d_segment_offsets  = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(InputT) * num_items));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_segment_offsets, sizeof(OffsetT) * (max_segments + 1)));
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(InputT) * num_items, cudaMemcpyHostToDevice));

    //
    // Test single-segment implementations
    //

    InitializeSegments(num_items, 1, h_segment_offsets, g_verbose_input);

    // Page-aligned-input tests
    TestByOp<NonSegmentedBackend, OutputT>(h_in, d_in, num_items, 1,
        h_segment_offsets, h_segment_offsets + 1, (OffsetT*) NULL, (OffsetT*)NULL);

    // Non-page-aligned-input tests
    if (num_items > 1)
    {
        InitializeSegments(num_items - 1, 1, h_segment_offsets, g_verbose_input);
        TestByOp<NonSegmentedBackend, OutputT>(h_in + 1, d_in + 1, num_items - 1, 1,
            h_segment_offsets, h_segment_offsets + 1, (OffsetT*) NULL, (OffsetT*)NULL);
    }

    //
    // Test segmented implementation
    //

    // Right now we assign a single thread block to each segment, so lets keep it to under 128K items per segment
    int max_items_per_segment = 128000;

    for (int num_segments = cub::DivideAndRoundUp(num_items, max_items_per_segment);
        num_segments < max_segments;
        num_segments = (num_segments * 32) + 1)
    {
        // Test with segment pointer
        InitializeSegments(num_items, num_segments, h_segment_offsets, g_verbose_input);
        CubDebugExit(cudaMemcpy(d_segment_offsets, h_segment_offsets, sizeof(OffsetT) * (num_segments + 1), cudaMemcpyHostToDevice));
        TestByOp<SegmentedBackend, OutputT>(h_in, d_in, num_items, num_segments,
            h_segment_offsets, h_segment_offsets + 1, d_segment_offsets, d_segment_offsets + 1);

        // Test with segment iterator
        typedef CastOp<OffsetT> IdentityOpT;
        IdentityOpT identity_op;
        TransformInputIterator<OffsetT, IdentityOpT, OffsetT*, OffsetT> h_segment_offsets_itr(
            h_segment_offsets,
            identity_op);
        TransformInputIterator<OffsetT, IdentityOpT, OffsetT*, OffsetT> d_segment_offsets_itr(
            d_segment_offsets,
            identity_op);

        TestByOp<SegmentedBackend, OutputT>(h_in, d_in, num_items, num_segments,
            h_segment_offsets_itr, h_segment_offsets_itr + 1, d_segment_offsets_itr, d_segment_offsets_itr + 1);

        // Test with transform iterators of different types

        typedef TransformFunctor1<OffsetT> TransformFunctor1T;
        typedef TransformFunctor2<OffsetT> TransformFunctor2T;

        TransformInputIterator<OffsetT, TransformFunctor1T, OffsetT*, OffsetT> h_segment_begin_offsets_itr(h_segment_offsets, TransformFunctor1T());
        TransformInputIterator<OffsetT, TransformFunctor2T, OffsetT*, OffsetT> h_segment_end_offsets_itr(h_segment_offsets + 1, TransformFunctor2T());

        TransformInputIterator<OffsetT, TransformFunctor1T, OffsetT*, OffsetT> d_segment_begin_offsets_itr(d_segment_offsets, TransformFunctor1T());
        TransformInputIterator<OffsetT, TransformFunctor2T, OffsetT*, OffsetT> d_segment_end_offsets_itr(d_segment_offsets + 1, TransformFunctor2T());

        TestByOp<SegmentedBackend, OutputT>(h_in, d_in, num_items, num_segments,
            h_segment_begin_offsets_itr, h_segment_end_offsets_itr,
            d_segment_begin_offsets_itr, d_segment_end_offsets_itr);
    }

    if (h_in)               delete[] h_in;
    if (h_segment_offsets)  delete[] h_segment_offsets;
    if (d_in)               CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_segment_offsets)  CubDebugExit(g_allocator.DeviceFree(d_segment_offsets));
}


/// Test different input-generation modes
template <
    typename InputT,
    typename OutputT,
    typename OffsetT>
void TestByGenMode(
    OffsetT num_items,
    OffsetT max_segments)
{
    //
    // Test pointer support using different input-generation modes
    //

    TestByBackend<InputT, OutputT>(num_items, max_segments, UNIFORM);
    TestByBackend<InputT, OutputT>(num_items, max_segments, INTEGER_SEED);
    TestByBackend<InputT, OutputT>(num_items, max_segments, RANDOM);

    //
    // Test iterator support using a constant-iterator and SUM
    //

    InputT val;
    InitValue(UNIFORM, val, 0);
    ConstantInputIterator<InputT, OffsetT> in(val);

    OffsetT *h_segment_offsets = new OffsetT[1 + 1];
    InitializeSegments(num_items, 1, h_segment_offsets, g_verbose_input);

#if TEST_CDP == 0
    constexpr auto Backend   = CUB;
#else  // TEST_CDP
    constexpr auto Backend   = CDP;
#endif // TEST_CDP

    SolveAndTest<Backend, OutputT>(in, in, num_items, 1,
        h_segment_offsets, h_segment_offsets + 1, (OffsetT*) NULL, (OffsetT*)NULL, Sum());

    if (h_segment_offsets) delete[] h_segment_offsets;
}

/// Test different problem sizes
template <typename InputT, typename OutputT, typename OffsetT>
void TestBySize(OffsetT max_items, OffsetT max_segments, OffsetT tile_size)
{
  // Test 0, 1, many
  TestByGenMode<InputT, OutputT>(0, max_segments);
  TestByGenMode<InputT, OutputT>(1, max_segments);
  TestByGenMode<InputT, OutputT>(max_items, max_segments);

  // Test random problem sizes from a log-distribution [8, max_items-ish)
  int    num_iterations = 8;
  double max_exp        = log(double(max_items)) / log(double(2.0));
  for (int i = 0; i < num_iterations; ++i)
  {
    OffsetT num_items = (OffsetT)pow(2.0, RandomValue(max_exp - 3.0) + 3.0);
    TestByGenMode<InputT, OutputT>(num_items, max_segments);
  }

  //
  // White-box testing of single-segment problems around specific sizes
  //

#if TEST_CDP == 0
  constexpr auto Backend   = CUB;
#else  // TEST_CDP
  constexpr auto Backend   = CDP;
#endif // TEST_CDP

  // Tile-boundaries: multiple blocks, one tile per block
  TestProblem<Backend, InputT, OutputT>(tile_size * 4, 1, RANDOM, Sum());
  TestProblem<Backend, InputT, OutputT>(tile_size * 4 + 1, 1, RANDOM, Sum());
  TestProblem<Backend, InputT, OutputT>(tile_size * 4 - 1, 1, RANDOM, Sum());

  // Tile-boundaries: multiple blocks, multiple tiles per block
  OffsetT sm_occupancy = 32;
  OffsetT occupancy    = tile_size * sm_occupancy * g_sm_count;
  TestProblem<Backend, InputT, OutputT>(occupancy, 1, RANDOM, Sum());
  TestProblem<Backend, InputT, OutputT>(occupancy + 1, 1, RANDOM, Sum());
  TestProblem<Backend, InputT, OutputT>(occupancy - 1, 1, RANDOM, Sum());
};

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
  bool *m_d_flag{};
  int m_expected{};

public:
  __host__ __device__ CustomOutputT(bool *d_flag, int expected)
      : m_d_flag(d_flag)
      , m_expected(expected)
  {}

  __host__ __device__ void operator=(const CustomAccumulatorT &accum) const
  {
    *m_d_flag = accum.is_valid() && (accum.get() == m_expected);
  }
};

__global__ void InitializeTestAccumulatorTypes(int num_items,
                                               int expected,
                                               bool *d_flag,
                                               CustomInputT *d_in,
                                               CustomOutputT *d_out)
{
  const int idx = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);

  if (idx < num_items)
  {
    d_in[idx] = CustomInputT(1);
  }

  if (idx == 0)
  {
    *d_out = CustomOutputT{d_flag, expected};
  }
}

template <typename T, 
          typename OffsetT>
void TestBigIndicesHelper(OffsetT num_items)
{
  thrust::constant_iterator<T> const_iter(T{1});
  thrust::device_vector<std::size_t> out(1);
  std::size_t* d_out = thrust::raw_pointer_cast(out.data());

  std::uint8_t *d_temp_storage{};
  std::size_t temp_storage_bytes{};

  CubDebugExit(
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, const_iter, d_out, num_items));

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  CubDebugExit(
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, const_iter, d_out, num_items));
  std::size_t result = out[0];

  AssertEquals(result, num_items);
}

template <typename T>
void TestBigIndices()
{
  TestBigIndicesHelper<T, std::uint32_t>(1ull << 30);
  TestBigIndicesHelper<T, std::uint32_t>(1ull << 31);
  TestBigIndicesHelper<T, std::uint32_t>((1ull << 32) - 1);
  TestBigIndicesHelper<T, std::uint64_t>(1ull << 33);
}

#if TEST_TYPES == 3
void TestAccumulatorTypes()
{
  const int num_items  = 2 * 1024 * 1024;
  const int expected   = num_items;
  const int block_size = 256;
  const int grid_size  = (num_items + block_size - 1) / block_size;

  CustomInputT *d_in{};
  CustomOutputT *d_out{};
  CustomAccumulatorT init{};
  bool *d_flag{};

  CubDebugExit(
    g_allocator.DeviceAllocate((void **)&d_out, sizeof(CustomOutputT)));
  CubDebugExit(g_allocator.DeviceAllocate((void **)&d_flag, sizeof(bool)));
  CubDebugExit(g_allocator.DeviceAllocate((void **)&d_in,
                                          sizeof(CustomInputT) * num_items));

  InitializeTestAccumulatorTypes<<<grid_size, block_size>>>(num_items,
                                                            expected,
                                                            d_flag,
                                                            d_in,
                                                            d_out);

  std::uint8_t *d_temp_storage{};
  std::size_t temp_storage_bytes{};

  CubDebugExit(cub::DeviceReduce::Reduce(d_temp_storage,
                                         temp_storage_bytes,
                                         d_in,
                                         d_out,
                                         num_items,
                                         cub::Sum{},
                                         init));

  CubDebugExit(
    g_allocator.DeviceAllocate((void **)&d_temp_storage, temp_storage_bytes));
  CubDebugExit(cudaMemset(d_temp_storage, 1,  temp_storage_bytes));

  CubDebugExit(cub::DeviceReduce::Reduce(d_temp_storage,
                                         temp_storage_bytes,
                                         d_in,
                                         d_out,
                                         num_items,
                                         cub::Sum{},
                                         init));

  bool ok{};
  CubDebugExit(cudaMemcpy(&ok, d_flag, sizeof(bool), cudaMemcpyDeviceToHost));

  AssertTrue(ok);

  CubDebugExit(g_allocator.DeviceFree(d_out));
  CubDebugExit(g_allocator.DeviceFree(d_in));
}

/**
 * ArgMin should return max value for empty input. This interferes with
 * input data containing infinity values. This test checks that ArgMin
 * works correctly with infinity values.
 */
void TestFloatInfInArgMin()
{
  using in_t     = float;
  using offset_t = int;
  using out_t    = cub::KeyValuePair<offset_t, in_t>;

  const int n     = 10;
  const float inf = ::cuda::std::numeric_limits<float>::infinity();

  thrust::device_vector<in_t> in(n, inf);
  thrust::device_vector<out_t> out(1);

  const in_t *d_in = thrust::raw_pointer_cast(in.data());
  out_t *d_out     = thrust::raw_pointer_cast(out.data());

  std::uint8_t *d_temp_storage{};
  std::size_t temp_storage_bytes{};

  CubDebugExit(cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_out, n));
  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());
  CubDebugExit(cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_out, n));

  const out_t result = out[0];
  AssertEquals(result.key, 0);
  AssertEquals(result.value, inf);
}

/**
 * ArgMax should return lowest value for empty input. This interferes with
 * input data containing infinity values. This test checks that ArgMax
 * works correctly with infinity values.
 */
void TestFloatInfInArgMax()
{
  using in_t = float;
  using offset_t = int;
  using out_t = cub::KeyValuePair<offset_t, in_t>;

  const int n = 10;
  const float inf = ::cuda::std::numeric_limits<float>::infinity();
  
  thrust::device_vector<in_t> in(n, -inf);
  thrust::device_vector<out_t> out(1);

  const in_t *d_in = thrust::raw_pointer_cast(in.data());
  out_t *d_out = thrust::raw_pointer_cast(out.data());

  std::uint8_t *d_temp_storage{};
  std::size_t temp_storage_bytes{};  

  CubDebugExit(cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, d_out, n));
  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());
  CubDebugExit(cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, d_out, n));

  const out_t result = out[0];
  AssertEquals(result.key, 0);
  AssertEquals(result.value, -inf);
}

void TestFloatInfInArg()
{
  TestFloatInfInArgMin();
  TestFloatInfInArgMax();
}
#endif

template <typename InputT, typename OutputT, typename OffsetT>
struct GetTileSize
{
  OffsetT max_items{};
  OffsetT max_segments{};
  OffsetT tile_size{};

  GetTileSize(OffsetT max_items, OffsetT max_segments)
      : max_items(max_items)
      , max_segments(max_segments)
  {}

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION cudaError_t Invoke()
  {
    this->tile_size = ActivePolicyT::ReducePolicy::BLOCK_THREADS *
                      ActivePolicyT::ReducePolicy::ITEMS_PER_THREAD;
    return cudaSuccess;
  }
};

/// Test problem type
template <typename InputT, typename OutputT, typename OffsetT>
void TestType(OffsetT max_items, OffsetT max_segments)
{
  // Inspect the tuning policies to determine this arch's tile size:
  using MaxPolicyT =
    typename DeviceReducePolicy<InputT, OffsetT, cub::Sum>::MaxPolicy;
  GetTileSize<InputT, OutputT, OffsetT> dispatch(max_items, max_segments);
  CubDebugExit(MaxPolicyT::Invoke(g_ptx_version, dispatch));

  TestBySize<InputT, OutputT>(max_items, max_segments, dispatch.tile_size);
}

//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------


/**
 * Main
 */
int main(int argc, char** argv)
{
    typedef int OffsetT;

    OffsetT max_items       = 27000000;
    OffsetT max_segments    = 34000;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    g_verbose_input = args.CheckCmdLineFlag("v2");
    args.GetCmdLineArgument("n", max_items);
    args.GetCmdLineArgument("s", max_segments);
    args.GetCmdLineArgument("i", g_timing_iterations);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--n=<input items> "
            "[--s=<num segments> "
            "[--i=<timing iterations> "
            "[--device=<device-id>] "
            "[--v] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());
    g_device_giga_bandwidth = args.device_giga_bandwidth;

    // Get ptx version
    CubDebugExit(PtxVersion(g_ptx_version));

    // Get SM count
    g_sm_count = args.deviceProp.multiProcessorCount;

    // %PARAM% TEST_CDP cdp 0:1
    // %PARAM% TEST_TYPES types 0:1:2:3:4

#if TEST_TYPES == 0
    TestType<signed char, signed char>(max_items, max_segments);
    TestType<unsigned char, unsigned char>(max_items, max_segments);
    TestType<signed char, int>(max_items, max_segments);
#elif TEST_TYPES == 1
    TestType<short, short>(max_items, max_segments);
    TestType<int, int>(max_items, max_segments);
    TestType<long, long>(max_items, max_segments);
    TestType<long long, long long>(max_items, max_segments);
#elif TEST_TYPES == 2
    TestType<uchar2, uchar2>(max_items, max_segments);
    TestType<uint2, uint2>(max_items, max_segments);
    TestType<ulonglong2, ulonglong2>(max_items, max_segments);
    TestType<ulonglong4, ulonglong4>(max_items, max_segments);
#elif TEST_TYPES == 3
    TestType<TestFoo, TestFoo>(max_items, max_segments);
    TestAccumulatorTypes();
    TestFloatInfInArg();

#if TEST_HALF_T
    TestType<half_t, half_t>(max_items, max_segments);
#endif
#else // TEST_TYPES == 4
    TestType<TestBar, TestBar>(max_items, max_segments);
    TestBigIndices<std::size_t>();

#if TEST_BF_T
    TestType<bfloat16_t, bfloat16_t>(max_items, max_segments);
#endif
#endif
    printf("\n");
    return 0;
}



