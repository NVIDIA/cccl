/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
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
 * Test of DeviceHistogram utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/device/device_histogram.cuh>
#include <cub/iterator/constant_input_iterator.cuh>
#include <cub/util_allocator.cuh>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cuda/std/type_traits>

#include <algorithm>
#include <limits>
#include <typeinfo>

#include "test_util.h"

#define TEST_HALF_T !_NVHPC_CUDA

#if TEST_HALF_T 
#include <cuda_fp16.h>
#endif

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------


// Dispatch types
enum Backend
{
    CUB,        // CUB method
    CDP,        // GPU-based (dynamic parallelism) dispatch to CUB method
};


bool                    g_verbose_input     = false;
bool                    g_verbose           = false;
int                     g_timing_iterations = 0;
CachingDeviceAllocator  g_allocator(true);


//---------------------------------------------------------------------
// Dispatch to different DeviceHistogram entrypoints
//---------------------------------------------------------------------

template <int NUM_ACTIVE_CHANNELS, int NUM_CHANNELS, int BACKEND>
struct Dispatch;

template <int NUM_ACTIVE_CHANNELS, int NUM_CHANNELS>
struct Dispatch<NUM_ACTIVE_CHANNELS, NUM_CHANNELS, CUB>
{
    /**
     * Dispatch to CUB multi histogram-range entrypoint
     */
    template <typename SampleIteratorT, typename CounterT, typename LevelT, typename OffsetT>
    //CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Range(
        int                     timing_timing_iterations,
        size_t                  * /*d_temp_storage_bytes*/,
        cudaError_t             * /*d_cdp_error*/,

        void*               d_temp_storage,
        size_t&             temp_storage_bytes,
        SampleIteratorT     d_samples,                                  ///< [in] The pointer to the multi-channel input sequence of data samples. The samples from different channels are assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
        CounterT            *(&d_histogram)[NUM_ACTIVE_CHANNELS],       ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_levels[i]</tt> - 1.
        int                 *num_levels,                                ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
        LevelT              *(&d_levels)[NUM_ACTIVE_CHANNELS],          ///< [in] The pointers to the arrays of boundaries (levels), one for each active channel.  Bin ranges are defined by consecutive boundary pairings: lower sample value boundaries are inclusive and upper sample value boundaries are exclusive.
        OffsetT             num_row_pixels,                             ///< [in] The number of multi-channel pixels per row in the region of interest
        OffsetT             num_rows,                                   ///< [in] The number of rows in the region of interest
        OffsetT             row_stride_bytes)                           ///< [in] The number of bytes between starts of consecutive rows in the region of interest
    {
        cudaError_t error = cudaSuccess;

        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            error = DeviceHistogram::MultiHistogramRange<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
                d_temp_storage,
                temp_storage_bytes,
                d_samples,
                d_histogram,
                num_levels,
                d_levels,
                num_row_pixels,
                num_rows,
                row_stride_bytes);
        }
        return error;
    }

#if TEST_HALF_T
    /**
     * Dispatch to CUB multi histogram-range entrypoint
     */
    template <typename CounterT, typename OffsetT>
    //CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Range(
        int                     timing_timing_iterations,
        size_t                  * /*d_temp_storage_bytes*/,
        cudaError_t             * /*d_cdp_error*/,

        void*               d_temp_storage,
        size_t&             temp_storage_bytes,
        half_t              *d_samples,                                 ///< [in] The pointer to the multi-channel input sequence of data samples. The samples from different channels are assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
        CounterT            *(&d_histogram)[NUM_ACTIVE_CHANNELS],       ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_levels[i]</tt> - 1.
        int                 *num_levels,                                ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
        half_t              *(&d_levels)[NUM_ACTIVE_CHANNELS],          ///< [in] The pointers to the arrays of boundaries (levels), one for each active channel.  Bin ranges are defined by consecutive boundary pairings: lower sample value boundaries are inclusive and upper sample value boundaries are exclusive.
        OffsetT             num_row_pixels,                             ///< [in] The number of multi-channel pixels per row in the region of interest
        OffsetT             num_rows,                                   ///< [in] The number of rows in the region of interest
        OffsetT             row_stride_bytes)                           ///< [in] The number of bytes between starts of consecutive rows in the region of interest
    {
        cudaError_t error = cudaSuccess;

        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            error = DeviceHistogram::MultiHistogramRange<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
                d_temp_storage,
                temp_storage_bytes,
                reinterpret_cast<__half*>(d_samples),
                d_histogram,
                num_levels,
                reinterpret_cast<__half *(&)[NUM_ACTIVE_CHANNELS]>(d_levels),
                num_row_pixels,
                num_rows,
                row_stride_bytes);
        }
        return error;
    }
#endif


    /**
     * Dispatch to CUB multi histogram-even entrypoint
     */
    template <typename SampleIteratorT, typename CounterT, typename LevelT, typename OffsetT>
    //CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Even(
        int                     timing_timing_iterations,
        size_t                  * /*d_temp_storage_bytes*/,
        cudaError_t             * /*d_cdp_error*/,

        void*               d_temp_storage,
        size_t&             temp_storage_bytes,
        SampleIteratorT     d_samples,                                  ///< [in] The pointer to the multi-channel input sequence of data samples. The samples from different channels are assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
        CounterT            *(&d_histogram)[NUM_ACTIVE_CHANNELS],          ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_levels[i]</tt> - 1.
        int                 *num_levels,            ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
        LevelT              *lower_level,           ///< [in] The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
        LevelT              *upper_level,           ///< [in] The upper sample value bound (exclusive) for the highest histogram bin in each active channel.
        OffsetT             num_row_pixels,                             ///< [in] The number of multi-channel pixels per row in the region of interest
        OffsetT             num_rows,                                   ///< [in] The number of rows in the region of interest
        OffsetT             row_stride_bytes)                                 ///< [in] The number of bytes between starts of consecutive rows in the region of interest
    {
        cudaError_t error = cudaSuccess;
        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            error = DeviceHistogram::MultiHistogramEven<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
                d_temp_storage,
                temp_storage_bytes,
                d_samples,
                d_histogram,
                num_levels,
                lower_level,
                upper_level,
                num_row_pixels,
                num_rows,
                row_stride_bytes);
        }
        return error;
    }

#if TEST_HALF_T 
    /**
     * Dispatch to CUB multi histogram-even entrypoint
     */
    template <typename CounterT, typename OffsetT>
    //CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Even(
        int                     timing_timing_iterations,
        size_t                  * /*d_temp_storage_bytes*/,
        cudaError_t             * /*d_cdp_error*/,

        void*               d_temp_storage,
        size_t&             temp_storage_bytes,
        half_t              *d_samples,                                  ///< [in] The pointer to the multi-channel input sequence of data samples. The samples from different channels are assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
        CounterT            *(&d_histogram)[NUM_ACTIVE_CHANNELS],          ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_levels[i]</tt> - 1.
        int                 *num_levels,            ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
        half_t              *lower_level,           ///< [in] The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
        half_t              *upper_level,           ///< [in] The upper sample value bound (exclusive) for the highest histogram bin in each active channel.
        OffsetT             num_row_pixels,                             ///< [in] The number of multi-channel pixels per row in the region of interest
        OffsetT             num_rows,                                   ///< [in] The number of rows in the region of interest
        OffsetT             row_stride_bytes)                                 ///< [in] The number of bytes between starts of consecutive rows in the region of interest
    {
        cudaError_t error = cudaSuccess;
        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            error = DeviceHistogram::MultiHistogramEven<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
                d_temp_storage,
                temp_storage_bytes,
                reinterpret_cast<__half*>(d_samples),
                d_histogram,
                num_levels,
                reinterpret_cast<__half*>(lower_level),
                reinterpret_cast<__half*>(upper_level),
                num_row_pixels,
                num_rows,
                row_stride_bytes);
        }
        return error;
    }
#endif
};


template <>
struct Dispatch<1, 1, CUB>
{

    /**
     * Dispatch to CUB single histogram-range entrypoint
     */
    template <typename SampleIteratorT, typename CounterT, typename LevelT, typename OffsetT>
    //CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Range(
        int                     timing_timing_iterations,
        size_t                  */*d_temp_storage_bytes*/,
        cudaError_t             */*d_cdp_error*/,

        void*               d_temp_storage,
        size_t&             temp_storage_bytes,
        SampleIteratorT     d_samples,                              ///< [in] The pointer to the multi-channel input sequence of data samples. The samples from different channels are assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
        CounterT*           (&d_histogram)[1],                      ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_levels[i]</tt> - 1.
        int                 *num_levels,                            ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
        LevelT              (&d_levels)[1],                         ///< [in] The pointers to the arrays of boundaries (levels), one for each active channel.  Bin ranges are defined by consecutive boundary pairings: lower sample value boundaries are inclusive and upper sample value boundaries are exclusive.
        OffsetT             num_row_pixels,                         ///< [in] The number of multi-channel pixels per row in the region of interest
        OffsetT             num_rows,                               ///< [in] The number of rows in the region of interest
        OffsetT             row_stride_bytes)                       ///< [in] The number of bytes between starts of consecutive rows in the region of interest
    {
        cudaError_t error = cudaSuccess;
        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            error = DeviceHistogram::HistogramRange(
                d_temp_storage,
                temp_storage_bytes,
                d_samples,
                d_histogram[0],
                num_levels[0],
                d_levels[0],
                num_row_pixels,
                num_rows,
                row_stride_bytes);
        }
        return error;
    }

#if TEST_HALF_T
    template <typename CounterT, typename OffsetT>
    //CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Range(
        int                     timing_timing_iterations,
        size_t                  */*d_temp_storage_bytes*/,
        cudaError_t             */*d_cdp_error*/,

        void*               d_temp_storage,
        size_t&             temp_storage_bytes,
        half_t              *d_samples,                              ///< [in] The pointer to the multi-channel input sequence of data samples. The samples from different channels are assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
        CounterT*           (&d_histogram)[1],                      ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_levels[i]</tt> - 1.
        int                 *num_levels,                            ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
        half_t              (&d_levels)[1],                         ///< [in] The pointers to the arrays of boundaries (levels), one for each active channel.  Bin ranges are defined by consecutive boundary pairings: lower sample value boundaries are inclusive and upper sample value boundaries are exclusive.
        OffsetT             num_row_pixels,                         ///< [in] The number of multi-channel pixels per row in the region of interest
        OffsetT             num_rows,                               ///< [in] The number of rows in the region of interest
        OffsetT             row_stride_bytes)                       ///< [in] The number of bytes between starts of consecutive rows in the region of interest
    {
        cudaError_t error = cudaSuccess;
        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            error = DeviceHistogram::HistogramRange(
                d_temp_storage,
                temp_storage_bytes,
                reinterpret_cast<__half*>(d_samples),
                d_histogram[0],
                num_levels[0],
                d_levels[0].operator __half(),
                num_row_pixels,
                num_rows,
                row_stride_bytes);
        }
        return error;
    }
#endif


    /**
     * Dispatch to CUB single histogram-even entrypoint
     */
    template <typename SampleIteratorT, typename CounterT, typename LevelT, typename OffsetT>
    //CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Even(
        int                     timing_timing_iterations,
        size_t                  */*d_temp_storage_bytes*/,
        cudaError_t             */*d_cdp_error*/,

        void*               d_temp_storage,
        size_t&             temp_storage_bytes,
        SampleIteratorT     d_samples,                                  ///< [in] The pointer to the multi-channel input sequence of data samples. The samples from different channels are assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
        CounterT*           (&d_histogram)[1],                      ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_levels[i]</tt> - 1.
        int                 *num_levels,                              ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
        LevelT              *lower_level,                             ///< [in] The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
        LevelT              *upper_level,                             ///< [in] The upper sample value bound (exclusive) for the highest histogram bin in each active channel.
        OffsetT             num_row_pixels,                             ///< [in] The number of multi-channel pixels per row in the region of interest
        OffsetT             num_rows,                                   ///< [in] The number of rows in the region of interest
        OffsetT             row_stride_bytes)                                 ///< [in] The number of bytes between starts of consecutive rows in the region of interest
    {
        cudaError_t error = cudaSuccess;
        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            error = DeviceHistogram::HistogramEven(
                d_temp_storage,
                temp_storage_bytes,
                d_samples,
                d_histogram[0],
                num_levels[0],
                lower_level[0],
                upper_level[0],
                num_row_pixels,
                num_rows,
                row_stride_bytes);
        }
        return error;
    }

#if TEST_HALF_T
    template <typename CounterT, typename OffsetT>
    //CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Even(
        int                     timing_timing_iterations,
        size_t                  */*d_temp_storage_bytes*/,
        cudaError_t             */*d_cdp_error*/,

        void*               d_temp_storage,
        size_t&             temp_storage_bytes,
        half_t              *d_samples,                                  ///< [in] The pointer to the multi-channel input sequence of data samples. The samples from different channels are assumed to be interleaved (e.g., an array of 32-bit pixels where each pixel consists of four RGBA 8-bit samples).
        CounterT*           (&d_histogram)[1],                      ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_levels[i]</tt> - 1.
        int                 *num_levels,                              ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
        half_t              *lower_level,                             ///< [in] The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
        half_t              *upper_level,                             ///< [in] The upper sample value bound (exclusive) for the highest histogram bin in each active channel.
        OffsetT             num_row_pixels,                             ///< [in] The number of multi-channel pixels per row in the region of interest
        OffsetT             num_rows,                                   ///< [in] The number of rows in the region of interest
        OffsetT             row_stride_bytes)                                 ///< [in] The number of bytes between starts of consecutive rows in the region of interest
    {
        cudaError_t error = cudaSuccess;
        for (int i = 0; i < timing_timing_iterations; ++i)
        {
            error = DeviceHistogram::HistogramEven(
                d_temp_storage,
                temp_storage_bytes,
                reinterpret_cast<__half*>(d_samples),
                d_histogram[0],
                num_levels[0],
                lower_level[0].operator __half(),
                upper_level[0].operator __half(),
                num_row_pixels,
                num_rows,
                row_stride_bytes);
        }
        return error;
    }
#endif

};


//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

// Searches for bin given a list of bin-boundary levels
template <typename LevelT>
struct SearchTransform
{
    LevelT          *levels;      // Pointer to levels array
    int             num_levels;   // Number of levels in array

    // Functor for converting samples to bin-ids (num_levels is returned if sample is out of range)
    template <typename SampleT>
    int operator()(SampleT sample)
    {
        int bin = int(std::upper_bound(levels, levels + num_levels, (LevelT) sample) - levels - 1);
        if (bin < 0)
        {
            // Sample out of range
            return num_levels;
        }
        return bin;
    }
};

// Template to scale samples to evenly-spaced bins
template <typename LevelT, typename = void>
struct ScaleTransform;

// [Integral types] Scales samples to evenly-spaced bins
template <typename LevelT>
struct ScaleTransform<LevelT,
                      typename ::cuda::std::enable_if<::cuda::std::is_integral<LevelT>::value>::type>
{
  int num_levels; // Number of levels in array
  LevelT max;     // Max sample level (exclusive)
  LevelT min;     // Min sample level (inclusive)

  void Init(int num_levels_, // Number of levels in array
            LevelT max_,     // Max sample level (exclusive)
            LevelT min_)     // Min sample level (inclusive)
  {
    this->num_levels = num_levels_;
    this->max        = max_;
    this->min        = min_;
  }

  // Functor for converting samples to bin-ids  (num_levels is returned if sample is out of range)
  template <typename SampleT>
  int operator()(SampleT sample)
  {
    if ((sample < min) || (sample >= max))
    {
      // Sample out of range
      return num_levels;
    }

    // Accurate bin computation following the arithmetic we guarantee in the HistoEven docs
    return static_cast<int>(
      (static_cast<uint64_t>(sample - min) * static_cast<uint64_t>(num_levels - 1)) /
      static_cast<uint64_t>(max - min));
  }
};

// [[Extended] floating point types] Scales samples to evenly-spaced bins
template <typename LevelT>
struct ScaleTransform<LevelT,
                      typename ::cuda::std::enable_if<::cuda::std::is_floating_point<LevelT>::value
#if TEST_HALF_T
                                                      || ::cuda::std::is_same<LevelT, half_t>::value
#endif
                                                      >::type>
{
  int num_levels; // Number of levels in array
  LevelT max;     // Max sample level (exclusive)
  LevelT min;     // Min sample level (inclusive)
  LevelT scale;   // Bin scaling factor

  void Init(int _num_levels, // Number of levels in array
            LevelT _max,     // Max sample level (exclusive)
            LevelT _min)     // Min sample level (inclusive)
  {
    this->num_levels = _num_levels;
    this->max        = _max;
    this->min        = _min;
    this->scale      = LevelT{1.0f} /
                  static_cast<LevelT>((max - min) / static_cast<LevelT>(num_levels - 1));
  }

  // Functor for converting samples to bin-ids  (num_levels is returned if sample is out of range)
  template <typename SampleT>
  int operator()(SampleT sample)
  {
    if ((sample < min) || (sample >= max))
    {
      // Sample out of range
      return num_levels;
    }

    return (int)((((float)sample) - min) * scale);
  }
};

/**
 * Generate sample
 */
template <typename T, typename LevelT>
void Sample(T &datum, LevelT max_level, int entropy_reduction)
{
    unsigned int max = (unsigned int) -1;
    unsigned int bits;
    RandomBits(bits, entropy_reduction);
    float fraction = (float(bits) / max);

    datum = (T) (fraction * max_level);
}


/**
 * Initialize histogram samples
 */
template <
    int             NUM_CHANNELS,
    int             NUM_ACTIVE_CHANNELS,
    typename        LevelT,
    typename        SampleT,
    typename        OffsetT>
void InitializeSamples(
    LevelT          max_level,
    int             entropy_reduction,
    SampleT         *h_samples,
    OffsetT         num_row_pixels,         ///< [in] The number of multi-channel pixels per row in the region of interest
    OffsetT         num_rows,               ///< [in] The number of rows in the region of interest
    OffsetT         row_stride_bytes)       ///< [in] The number of bytes between starts of consecutive rows in the region of interest
{
    // Initialize samples
    for (OffsetT row = 0; row < num_rows; ++row)
    {
        for (OffsetT pixel = 0; pixel < num_row_pixels; ++pixel)
        {
            for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
            {
                // Sample offset
                OffsetT offset = (row * (row_stride_bytes / sizeof(SampleT))) + (pixel * NUM_CHANNELS) + channel;

                // Init sample value
                Sample(h_samples[offset], max_level, entropy_reduction);
                if (g_verbose_input)
                {
                    if (channel > 0) printf(", ");
                    std::cout << CoutCast(h_samples[offset]);
                }
            }
        }
    }
}


/**
 * Initialize histogram solutions
 */
template <
    int             NUM_CHANNELS,
    int             NUM_ACTIVE_CHANNELS,
    typename        CounterT,
    typename        SampleIteratorT,
    typename        TransformOp,
    typename        OffsetT>
void InitializeBins(
    SampleIteratorT h_samples,
    int             num_levels[NUM_ACTIVE_CHANNELS],        ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
    TransformOp     transform_op[NUM_ACTIVE_CHANNELS],      ///< [in] The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
    CounterT        *h_histogram[NUM_ACTIVE_CHANNELS],      ///< [out] The pointers to the histogram counter output arrays, one for each active channel.  For channel<sub><em>i</em></sub>, the allocation length of <tt>d_histograms[i]</tt> should be <tt>num_levels[i]</tt> - 1.
    OffsetT         num_row_pixels,                         ///< [in] The number of multi-channel pixels per row in the region of interest
    OffsetT         num_rows,                               ///< [in] The number of rows in the region of interest
    OffsetT         row_stride_bytes)                       ///< [in] The number of bytes between starts of consecutive rows in the region of interest
{
    using SampleT = cub::detail::value_t<SampleIteratorT>;

    // Init bins
    for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
    {
        for (int bin = 0; bin < num_levels[CHANNEL] - 1; ++bin)
        {
            h_histogram[CHANNEL][bin] = 0;
        }
    }

    // Initialize samples
    if (g_verbose_input) printf("Samples: \n");
    for (OffsetT row = 0; row < num_rows; ++row)
    {
        for (OffsetT pixel = 0; pixel < num_row_pixels; ++pixel)
        {
            if (g_verbose_input) printf("[");
            for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
            {
                // Sample offset
                OffsetT offset = (row * (row_stride_bytes / sizeof(SampleT))) + (pixel * NUM_CHANNELS) + channel;

                // Update sample bin
                int bin = transform_op[channel](h_samples[offset]);
                if (g_verbose_input) printf(" (%d)", bin); fflush(stdout);
                if ((bin >= 0) && (bin < num_levels[channel] - 1))
                {
                    // valid bin
                    h_histogram[channel][bin]++;
                }
            }
            if (g_verbose_input) printf("]");
        }
        if (g_verbose_input) printf("\n\n");
    }
}



/**
 * Test histogram-even
 */
template <
    Backend         BACKEND,
    int             NUM_CHANNELS,
    int             NUM_ACTIVE_CHANNELS,
    typename        SampleT,
    typename        CounterT,
    typename        LevelT,
    typename        OffsetT,
    typename        SampleIteratorT>
void TestEven(
    LevelT          max_level,
    int             entropy_reduction,
    int             num_levels[NUM_ACTIVE_CHANNELS],            ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
    LevelT          lower_level[NUM_ACTIVE_CHANNELS],           ///< [in] The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
    LevelT          upper_level[NUM_ACTIVE_CHANNELS],           ///< [in] The upper sample value bound (exclusive) for the highest histogram bin in each active channel.
    OffsetT         num_row_pixels,                             ///< [in] The number of multi-channel pixels per row in the region of interest
    OffsetT         num_rows,                                   ///< [in] The number of rows in the region of interest
    OffsetT         row_stride_bytes,                           ///< [in] The number of bytes between starts of consecutive rows in the region of interest
    SampleIteratorT h_samples,
    SampleIteratorT d_samples)
{
    OffsetT total_samples = num_rows * (row_stride_bytes / sizeof(SampleT));

    printf("\n----------------------------\n");
    printf("%s cub::DeviceHistogram::Even (%s) "
           "%d pixels (%d height, %d width, %d-byte row stride), "
           "%d %d-byte %s samples (entropy reduction %d), "
           "%s levels, %s counters, %d/%d channels, max sample ",
        (BACKEND == CDP) ? "CDP CUB" : "CUB",
        (std::is_pointer<SampleIteratorT>::value) ? "pointer" : "iterator",
        (int) (num_row_pixels * num_rows),
        (int) num_rows,
        (int) num_row_pixels,
        (int) row_stride_bytes,
        (int) total_samples,
        (int) sizeof(SampleT),
        typeid(SampleT).name(),
        entropy_reduction,
        typeid(LevelT).name(),
        typeid(CounterT).name(),
        NUM_ACTIVE_CHANNELS,
        NUM_CHANNELS);
    std::cout << CoutCast(max_level) << "\n";
    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        std::cout << "\tChannel " << channel << ": "
                  << num_levels[channel] - 1 << " bins "
                  << "[" << lower_level[channel] << ", "
                  << upper_level[channel] << ")\n";
    }
    fflush(stdout);

    // Allocate and initialize host and device data

    typedef SampleT Foo;        // rename type to quelch gcc warnings (bug?)
    CounterT*                   h_histogram[NUM_ACTIVE_CHANNELS];
    ScaleTransform<LevelT>      transform_op[NUM_ACTIVE_CHANNELS];

    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
      int bins             = num_levels[channel] - 1;
      h_histogram[channel] = new CounterT[bins];

      transform_op[channel].Init(num_levels[channel], upper_level[channel], lower_level[channel]);
    }

    InitializeBins<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
        h_samples, num_levels, transform_op, h_histogram, num_row_pixels, num_rows, row_stride_bytes);

    // Allocate and initialize device data

    CounterT* d_histogram[NUM_ACTIVE_CHANNELS];
    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_histogram[channel], sizeof(CounterT) * (num_levels[channel] - 1)));
        CubDebugExit(cudaMemset(d_histogram[channel], 0, sizeof(CounterT) * (num_levels[channel] - 1)));
    }

    // Allocate CDP device arrays
    size_t          *d_temp_storage_bytes = NULL;
    cudaError_t     *d_cdp_error = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage_bytes,  sizeof(size_t) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cdp_error,           sizeof(cudaError_t) * 1));

    // Allocate temporary storage
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;

    Dispatch<NUM_ACTIVE_CHANNELS, NUM_CHANNELS, BACKEND>::Even(
        1, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes,
        d_samples, d_histogram, num_levels, lower_level, upper_level,
        num_row_pixels, num_rows, row_stride_bytes);

    // Allocate temporary storage with "canary" zones
    int     canary_bytes    = 256;
    char    canary_token    = 8;
    char*   canary_zone     = new char[canary_bytes];

    memset(canary_zone, canary_token, canary_bytes);
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes + (canary_bytes * 2)));
    CubDebugExit(cudaMemset(d_temp_storage, canary_token, temp_storage_bytes + (canary_bytes * 2)));

    // Run warmup/correctness iteration
    Dispatch<NUM_ACTIVE_CHANNELS, NUM_CHANNELS, BACKEND>::Even(
        1, d_temp_storage_bytes, d_cdp_error,
        ((char *) d_temp_storage) + canary_bytes, temp_storage_bytes,
        d_samples, d_histogram, num_levels, lower_level, upper_level,
        num_row_pixels, num_rows, row_stride_bytes);

    // Check canary zones
    if (g_verbose)
    {
        printf("Checking leading temp_storage canary zone (token = %d)\n"
               "------------------------------------------------------\n",
               static_cast<int>(canary_token));
    }
    int error = CompareDeviceResults(canary_zone, (char *) d_temp_storage, canary_bytes, true, g_verbose);
    AssertEquals(0, error);
    if (g_verbose)
    {
        printf("Checking trailing temp_storage canary zone (token = %d)\n"
               "-------------------------------------------------------\n",
               static_cast<int>(canary_token));
    }
    error = CompareDeviceResults(canary_zone, ((char *) d_temp_storage) + canary_bytes + temp_storage_bytes, canary_bytes, true, g_verbose);
    AssertEquals(0, error);

    // Flush any stdout/stderr
    CubDebugExit(cudaPeekAtLastError());
    CubDebugExit(cudaDeviceSynchronize());
    fflush(stdout);
    fflush(stderr);

    // Check for correctness (and display results, if specified)
    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        if (g_verbose)
        {
            printf("Checking histogram result (channel = %d)\n"
                   "----------------------------------------\n",
                   channel);
        }
        int channel_error = CompareDeviceResults(h_histogram[channel], d_histogram[channel], num_levels[channel] - 1, true, g_verbose);
        printf("\tChannel %d %s", channel, channel_error ? "FAIL" : "PASS\n");
        error |= channel_error;
    }

    // Performance
    GpuTimer gpu_timer;
    gpu_timer.Start();

    Dispatch<NUM_ACTIVE_CHANNELS, NUM_CHANNELS, BACKEND>::Even(
        g_timing_iterations, d_temp_storage_bytes, d_cdp_error,
        ((char *) d_temp_storage) + canary_bytes, temp_storage_bytes,
        d_samples, d_histogram, num_levels, lower_level, upper_level,
        num_row_pixels, num_rows, row_stride_bytes);

    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    // Display performance
    if (g_timing_iterations > 0)
    {
        float avg_millis = elapsed_millis / g_timing_iterations;
        float giga_rate = float(total_samples) / avg_millis / 1000.0f / 1000.0f;
        float giga_bandwidth = giga_rate * sizeof(SampleT);
        printf("\t%.3f avg ms, %.3f billion samples/s, %.3f billion bins/s, %.3f billion pixels/s, %.3f logical GB/s",
            avg_millis,
            giga_rate,
            giga_rate * NUM_ACTIVE_CHANNELS / NUM_CHANNELS,
            giga_rate / NUM_CHANNELS,
            giga_bandwidth);
    }

    printf("\n\n");

    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        if (h_histogram[channel])
            delete[] h_histogram[channel];

        if (d_histogram[channel])
            CubDebugExit(g_allocator.DeviceFree(d_histogram[channel]));
    }

    if (d_temp_storage_bytes) CubDebugExit(g_allocator.DeviceFree(d_temp_storage_bytes));
    if (d_cdp_error) CubDebugExit(g_allocator.DeviceFree(d_cdp_error));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    // Correctness asserts
    AssertEquals(0, error);
}


/**
 * Test histogram-even (native pointer input)
 */
template <
    Backend         BACKEND,
    int             NUM_CHANNELS,
    int             NUM_ACTIVE_CHANNELS,
    typename        SampleT,
    typename        CounterT,
    typename        LevelT,
    typename        OffsetT>
void TestEvenNative(
    LevelT          max_level,
    int             entropy_reduction,
    int             num_levels[NUM_ACTIVE_CHANNELS],            ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
    LevelT          lower_level[NUM_ACTIVE_CHANNELS],           ///< [in] The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
    LevelT          upper_level[NUM_ACTIVE_CHANNELS],           ///< [in] The upper sample value bound (exclusive) for the highest histogram bin in each active channel.
    OffsetT         num_row_pixels,                             ///< [in] The number of multi-channel pixels per row in the region of interest
    OffsetT         num_rows,                                   ///< [in] The number of rows in the region of interest
    OffsetT         row_stride_bytes)                                 ///< [in] The number of bytes between starts of consecutive rows in the region of interest
{
    OffsetT total_samples = num_rows * (row_stride_bytes / sizeof(SampleT));

    // Allocate and initialize host sample data
    typedef SampleT Foo;        // rename type to quelch gcc warnings (bug?)
    SampleT*                    h_samples = new Foo[total_samples];

    InitializeSamples<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
        max_level, entropy_reduction, h_samples, num_row_pixels, num_rows, row_stride_bytes);

    // Allocate and initialize device data
    SampleT* d_samples = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_samples, sizeof(SampleT) * total_samples));
    CubDebugExit(cudaMemcpy(d_samples, h_samples, sizeof(SampleT) * total_samples, cudaMemcpyHostToDevice));

    TestEven<BACKEND, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleT, CounterT, LevelT, OffsetT>(
        max_level, entropy_reduction, num_levels, lower_level, upper_level,
        num_row_pixels, num_rows, row_stride_bytes,
        h_samples, d_samples);

    // Cleanup
    if (h_samples) delete[] h_samples;
    if (d_samples) CubDebugExit(g_allocator.DeviceFree(d_samples));
}


/**
 * Test histogram-even (iterator input)
 */
template <
    Backend         BACKEND,
    int             NUM_CHANNELS,
    int             NUM_ACTIVE_CHANNELS,
    typename        SampleT,
    typename        CounterT,
    typename        LevelT,
    typename        OffsetT>
void TestEvenIterator(
    Int2Type<false> /*is_half*/,
    LevelT          max_level,
    int             entropy_reduction,
    int             num_levels[NUM_ACTIVE_CHANNELS],            ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
    LevelT          lower_level[NUM_ACTIVE_CHANNELS],           ///< [in] The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
    LevelT          upper_level[NUM_ACTIVE_CHANNELS],           ///< [in] The upper sample value bound (exclusive) for the highest histogram bin in each active channel.
    OffsetT         num_row_pixels,                             ///< [in] The number of multi-channel pixels per row in the region of interest
    OffsetT         num_rows,                                   ///< [in] The number of rows in the region of interest
    OffsetT         row_stride_bytes)                                 ///< [in] The number of bytes between starts of consecutive rows in the region of interest
{
    SampleT sample = (SampleT) lower_level[0];
    ConstantInputIterator<SampleT> sample_itr(sample);

    TestEven<BACKEND, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleT, CounterT, LevelT, OffsetT>(
        max_level, entropy_reduction, num_levels, lower_level, upper_level,
        num_row_pixels, num_rows, row_stride_bytes,
        sample_itr, sample_itr);
}

template <Backend,
          int,
          int NUM_ACTIVE_CHANNELS,
          typename,
          typename,
          typename LevelT,
          typename OffsetT>
void TestEvenIterator(Int2Type<true> /*is_half*/,
                      LevelT,
                      int,
                      int[NUM_ACTIVE_CHANNELS],
                      LevelT[NUM_ACTIVE_CHANNELS],
                      LevelT[NUM_ACTIVE_CHANNELS],
                      OffsetT,
                      OffsetT,
                      OffsetT)
{
  // We have to reinterpret cast `half_t *` pointer to `__half *` in this test. 
  // Hence, iterators testing is not supported.
}

/**
 * Test histogram-range
 */
template <
    Backend         BACKEND,
    int             NUM_CHANNELS,
    int             NUM_ACTIVE_CHANNELS,
    typename        SampleT,
    typename        CounterT,
    typename        LevelT,
    typename        OffsetT>
void TestRange(
    LevelT          max_level,
    int             entropy_reduction,
    int             num_levels[NUM_ACTIVE_CHANNELS],            ///< [in] The number of boundaries (levels) for delineating histogram samples in each active channel.  Implies that the number of bins for channel<sub><em>i</em></sub> is <tt>num_levels[i]</tt> - 1.
    LevelT*         levels[NUM_ACTIVE_CHANNELS],                ///< [in] The lower sample value bound (inclusive) for the lowest histogram bin in each active channel.
    OffsetT         num_row_pixels,                             ///< [in] The number of multi-channel pixels per row in the region of interest
    OffsetT         num_rows,                                   ///< [in] The number of rows in the region of interest
    OffsetT         row_stride_bytes)                                 ///< [in] The number of bytes between starts of consecutive rows in the region of interest
{
    OffsetT total_samples = num_rows * (row_stride_bytes / sizeof(SampleT));

    printf("\n----------------------------\n");
    printf("%s cub::DeviceHistogram::Range %d pixels "
           "(%d height, %d width, %d-byte row stride), "
           "%d %d-byte %s samples (entropy reduction %d), "
           "%s levels, %s counters, %d/%d channels, max sample ",
           (BACKEND == CDP) ? "CDP CUB" : "CUB",
           (int)(num_row_pixels * num_rows),
           (int)num_rows,
           (int)num_row_pixels,
           (int)row_stride_bytes,
           (int)total_samples,
           (int)sizeof(SampleT),
           typeid(SampleT).name(),
           entropy_reduction,
           typeid(LevelT).name(),
           typeid(CounterT).name(),
           NUM_ACTIVE_CHANNELS,
           NUM_CHANNELS);
    std::cout << CoutCast(max_level) << "\n";
    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        printf("Channel %d: %d bins", channel, num_levels[channel] - 1);
        if (g_verbose)
        {
            std::cout << "[ " << levels[channel][0];
            for (int level = 1; level < num_levels[channel]; ++level)
            {
                std::cout << ", " << levels[channel][level];
            }
            printf("]");
        }
        printf("\n");
    }
    fflush(stdout);

    // Allocate and initialize host and device data
    typedef SampleT Foo;        // rename type to quelch gcc warnings (bug?)
    SampleT*                    h_samples = new Foo[total_samples];
    CounterT*                   h_histogram[NUM_ACTIVE_CHANNELS];
    SearchTransform<LevelT>     transform_op[NUM_ACTIVE_CHANNELS];

    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        transform_op[channel].levels = levels[channel];
        transform_op[channel].num_levels = num_levels[channel];

        int bins = num_levels[channel] - 1;
        h_histogram[channel] = new CounterT[bins];
    }

    InitializeSamples<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
        max_level, entropy_reduction, h_samples, num_row_pixels, num_rows, row_stride_bytes);

    InitializeBins<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
        h_samples, num_levels, transform_op, h_histogram, num_row_pixels, num_rows, row_stride_bytes);

    // Allocate and initialize device data
    SampleT*        d_samples = NULL;
    LevelT*         d_levels[NUM_ACTIVE_CHANNELS];
    CounterT*       d_histogram[NUM_ACTIVE_CHANNELS];

    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_samples, sizeof(SampleT) * total_samples));
    CubDebugExit(cudaMemcpy(d_samples, h_samples, sizeof(SampleT) * total_samples, cudaMemcpyHostToDevice));

    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_levels[channel], sizeof(LevelT) * num_levels[channel]));
        CubDebugExit(cudaMemcpy(d_levels[channel], levels[channel],         sizeof(LevelT) * num_levels[channel], cudaMemcpyHostToDevice));

        int bins = num_levels[channel] - 1;
        CubDebugExit(g_allocator.DeviceAllocate((void**)&d_histogram[channel],  sizeof(CounterT) * bins));
        CubDebugExit(cudaMemset(d_histogram[channel], 0,                        sizeof(CounterT) * bins));
    }

    // Allocate CDP device arrays
    size_t          *d_temp_storage_bytes = NULL;
    cudaError_t     *d_cdp_error = NULL;

    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_temp_storage_bytes,  sizeof(size_t) * 1));
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_cdp_error,           sizeof(cudaError_t) * 1));

    // Allocate temporary storage
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;

    Dispatch<NUM_ACTIVE_CHANNELS, NUM_CHANNELS, BACKEND>::Range(
        1, d_temp_storage_bytes, d_cdp_error,
        d_temp_storage, temp_storage_bytes,
        d_samples,
        d_histogram,
        num_levels, d_levels,
        num_row_pixels, num_rows, row_stride_bytes);

    // Allocate temporary storage with "canary" zones
    int     canary_bytes    = 256;
    char    canary_token    = 9;
    char*   canary_zone     = new char[canary_bytes];

    memset(canary_zone, canary_token, canary_bytes);
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes + (canary_bytes * 2)));
    CubDebugExit(cudaMemset(d_temp_storage, canary_token, temp_storage_bytes + (canary_bytes * 2)));

    // Run warmup/correctness iteration
    Dispatch<NUM_ACTIVE_CHANNELS, NUM_CHANNELS, BACKEND>::Range(
        1, d_temp_storage_bytes, d_cdp_error,
        ((char *) d_temp_storage) + canary_bytes, temp_storage_bytes,
        d_samples,
        d_histogram,
        num_levels, d_levels,
        num_row_pixels, num_rows, row_stride_bytes);

    // Check canary zones
    int error = CompareDeviceResults(canary_zone, (char *) d_temp_storage, canary_bytes, true, g_verbose);
    AssertEquals(0, error);
    error = CompareDeviceResults(canary_zone, ((char *) d_temp_storage) + canary_bytes + temp_storage_bytes, canary_bytes, true, g_verbose);
    AssertEquals(0, error);

    // Flush any stdout/stderr
    CubDebugExit(cudaPeekAtLastError());
    CubDebugExit(cudaDeviceSynchronize());
    fflush(stdout);
    fflush(stderr);

    // Check for correctness (and display results, if specified)
    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        int channel_error = CompareDeviceResults(h_histogram[channel], d_histogram[channel], num_levels[channel] - 1, true, g_verbose);
        printf("\tChannel %d %s", channel, channel_error ? "FAIL" : "PASS\n");
        error |= channel_error;
    }

    // Performance
    GpuTimer gpu_timer;
    gpu_timer.Start();

    Dispatch<NUM_ACTIVE_CHANNELS, NUM_CHANNELS, BACKEND>::Range(
        g_timing_iterations, d_temp_storage_bytes, d_cdp_error,
        ((char *) d_temp_storage) + canary_bytes, temp_storage_bytes,
        d_samples,
        d_histogram,
        num_levels, d_levels,
        num_row_pixels, num_rows, row_stride_bytes);

    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    // Display performance
    if (g_timing_iterations > 0)
    {
        float avg_millis = elapsed_millis / g_timing_iterations;
        float giga_rate = float(total_samples) / avg_millis / 1000.0f / 1000.0f;
        float giga_bandwidth = giga_rate * sizeof(SampleT);
        printf("\t%.3f avg ms, %.3f billion samples/s, %.3f billion bins/s, %.3f billion pixels/s, %.3f logical GB/s",
            avg_millis,
            giga_rate,
            giga_rate * NUM_ACTIVE_CHANNELS / NUM_CHANNELS,
            giga_rate / NUM_CHANNELS,
            giga_bandwidth);
    }

    printf("\n\n");

    // Cleanup
    if (h_samples) delete[] h_samples;

    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        if (h_histogram[channel])
            delete[] h_histogram[channel];

        if (d_histogram[channel])
            CubDebugExit(g_allocator.DeviceFree(d_histogram[channel]));

        if (d_levels[channel])
            CubDebugExit(g_allocator.DeviceFree(d_levels[channel]));
    }

    if (d_samples) CubDebugExit(g_allocator.DeviceFree(d_samples));
    if (d_temp_storage_bytes) CubDebugExit(g_allocator.DeviceFree(d_temp_storage_bytes));
    if (d_cdp_error) CubDebugExit(g_allocator.DeviceFree(d_cdp_error));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    // Correctness asserts
    AssertEquals(0, error);
}


/**
 * Test histogram-even
 */
template <
    Backend         BACKEND,
    typename        SampleT,
    int             NUM_CHANNELS,
    int             NUM_ACTIVE_CHANNELS,
    typename        CounterT,
    typename        LevelT,
    typename        OffsetT>
void TestEven(
    OffsetT         num_row_pixels,
    OffsetT         num_rows,
    OffsetT         row_stride_bytes,
    int             entropy_reduction,
    int             num_levels[NUM_ACTIVE_CHANNELS],
    LevelT          max_level,
    int             max_num_levels)
{
    LevelT lower_level[NUM_ACTIVE_CHANNELS];
    LevelT upper_level[NUM_ACTIVE_CHANNELS];

    // Find smallest level increment
    int max_bins = max_num_levels - 1;
    LevelT min_level_increment = max_level / static_cast<LevelT>(max_bins);

    // Set upper and lower levels for each channel
    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        int num_bins = num_levels[channel] - 1;
        lower_level[channel] = static_cast<LevelT>((max_level - (static_cast<LevelT>(num_bins) * min_level_increment)) / static_cast<LevelT>(2));
        upper_level[channel] = static_cast<LevelT>((max_level + (static_cast<LevelT>(num_bins) * min_level_increment)) / static_cast<LevelT>(2));
    }

    // Test pointer-based samples
    TestEvenNative<BACKEND, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleT, CounterT, LevelT, OffsetT>(
        max_level, entropy_reduction, num_levels, lower_level, upper_level, num_row_pixels, num_rows, row_stride_bytes);

    // Test iterator-based samples (CUB-only)
    TestEvenIterator<CUB, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleT, CounterT, LevelT, OffsetT>(
        Int2Type<std::is_same<SampleT, half_t>::value>{}, max_level, entropy_reduction, num_levels, lower_level, upper_level, num_row_pixels, num_rows, row_stride_bytes);
}



/**
 * Test histogram-range
 */
template <
    Backend         BACKEND,
    typename        SampleT,
    int             NUM_CHANNELS,
    int             NUM_ACTIVE_CHANNELS,
    typename        CounterT,
    typename        LevelT,
    typename        OffsetT>
void TestRange(
    OffsetT         num_row_pixels,
    OffsetT         num_rows,
    OffsetT         row_stride_bytes,
    int             entropy_reduction,
    int             num_levels[NUM_ACTIVE_CHANNELS],
    LevelT          max_level,
    int             max_num_levels)
{
    // Find smallest level increment
    int max_bins = max_num_levels - 1;
    LevelT min_level_increment = max_level / static_cast<LevelT>(max_bins);

    LevelT* levels[NUM_ACTIVE_CHANNELS];
    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        levels[channel] = new LevelT[num_levels[channel]];

        int num_bins = num_levels[channel] - 1;
        LevelT lower_level = (max_level - static_cast<LevelT>(num_bins * min_level_increment)) / static_cast<LevelT>(2);

        for (int level = 0; level < num_levels[channel]; ++level)
            levels[channel][level] = lower_level + static_cast<LevelT>(level * min_level_increment);
    }

    TestRange<BACKEND, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, SampleT, CounterT, LevelT, OffsetT>(
        max_level, entropy_reduction, num_levels, levels, num_row_pixels, num_rows, row_stride_bytes);

    for (int channel = 0; channel < NUM_ACTIVE_CHANNELS; ++channel)
        delete[] levels[channel];

}



/**
 * Test different entrypoints
 */
template <
    typename        SampleT,
    int             NUM_CHANNELS,
    int             NUM_ACTIVE_CHANNELS,
    typename        CounterT,
    typename        LevelT,
    typename        OffsetT>
void Test(
    OffsetT         num_row_pixels,
    OffsetT         num_rows,
    OffsetT         row_stride_bytes,
    int             entropy_reduction,
    int             num_levels[NUM_ACTIVE_CHANNELS],
    LevelT          max_level,
    int             max_num_levels)
{
    TestEven<CUB, SampleT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT, LevelT, OffsetT>(
        num_row_pixels, num_rows, row_stride_bytes, entropy_reduction, num_levels, max_level, max_num_levels);

    TestRange<CUB, SampleT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT, LevelT, OffsetT>(
        num_row_pixels, num_rows, row_stride_bytes, entropy_reduction, num_levels, max_level, max_num_levels);
}


/**
 * Test different number of levels
 */
template <
    typename        SampleT,
    int             NUM_CHANNELS,
    int             NUM_ACTIVE_CHANNELS,
    typename        CounterT,
    typename        LevelT,
    typename        OffsetT>
void Test(
    OffsetT         num_row_pixels,
    OffsetT         num_rows,
    OffsetT         row_stride_bytes,
    int             entropy_reduction,
    LevelT          max_level,
    int             max_num_levels)
{
    int num_levels[NUM_ACTIVE_CHANNELS];

    // All different levels
    num_levels[0] = max_num_levels;
    for (int channel = 1; channel < NUM_ACTIVE_CHANNELS; ++channel)
    {
        num_levels[channel] = (num_levels[channel - 1] / 2) + 1;
    }
    Test<SampleT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT, LevelT, OffsetT>(
        num_row_pixels, num_rows, row_stride_bytes, entropy_reduction, num_levels, max_level, max_num_levels);
}



/**
 * Test different entropy-levels
 */
template <
    typename        SampleT,
    int             NUM_CHANNELS,
    int             NUM_ACTIVE_CHANNELS,
    typename        CounterT,
    typename        LevelT,
    typename        OffsetT>
void Test(
    OffsetT         num_row_pixels,
    OffsetT         num_rows,
    OffsetT         row_stride_bytes,
    LevelT          max_level,
    int             max_num_levels)
{
    // entropy_reduction = -1 -> all samples == 0
    Test<SampleT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT, LevelT, OffsetT>(
        num_row_pixels, num_rows, row_stride_bytes, -1,  max_level, max_num_levels);

    Test<SampleT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT, LevelT, OffsetT>(
        num_row_pixels, num_rows, row_stride_bytes, 0,  max_level, max_num_levels);

    Test<SampleT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT, LevelT, OffsetT>(
        num_row_pixels, num_rows, row_stride_bytes, 5,   max_level, max_num_levels);
}


/**
 * Test different row strides
 */
template <
    typename        SampleT,
    int             NUM_CHANNELS,
    int             NUM_ACTIVE_CHANNELS,
    typename        CounterT,
    typename        LevelT,
    typename        OffsetT>
void Test(
    OffsetT         num_row_pixels,
    OffsetT         num_rows,
    LevelT          max_level,
    int             max_num_levels)
{
    OffsetT row_stride_bytes = num_row_pixels * NUM_CHANNELS * sizeof(SampleT);

    // No padding
    Test<SampleT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT, LevelT, OffsetT>(
        num_row_pixels, num_rows, row_stride_bytes, max_level, max_num_levels);

    // 13 samples padding
    Test<SampleT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT, LevelT, OffsetT>(
        num_row_pixels, num_rows, row_stride_bytes + (13 * sizeof(SampleT)), max_level, max_num_levels);
}


/**
 * Test different problem sizes
 */
template <
    typename        SampleT,
    int             NUM_CHANNELS,
    int             NUM_ACTIVE_CHANNELS,
    typename        CounterT,
    typename        LevelT,
    typename        OffsetT>
void Test(
    LevelT          max_level,
    int             max_num_levels)
{
    // 0 row/col images
    Test<SampleT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT, LevelT, OffsetT>(
        OffsetT(1920), OffsetT(0), max_level, max_num_levels);
    Test<SampleT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT, LevelT, OffsetT>(
        OffsetT(0), OffsetT(0), max_level, max_num_levels);

    // Small inputs
    Test<SampleT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT, LevelT, OffsetT>(
      OffsetT(15), OffsetT(1), max_level, max_num_levels);

    // 1080 image
    Test<SampleT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT, LevelT, OffsetT>(
        OffsetT(1920), OffsetT(1080), max_level, max_num_levels);

    // Sample different aspect ratios sizes
    for (OffsetT rows = 1; rows < 1000000; rows *= 1000)
    {
        for (OffsetT cols = 1; cols < (1000000 / rows); cols *= 1000)
        {
            Test<SampleT, NUM_CHANNELS, NUM_ACTIVE_CHANNELS, CounterT, LevelT, OffsetT>(
                cols, rows, max_level, max_num_levels);
        }
    }
}



/**
 * Test different channel interleavings (valid specialiation)
 */
template <typename SampleT, typename CounterT, typename LevelT, typename OffsetT>
void TestChannels(LevelT max_level,
                  int max_num_levels,
                  Int2Type<true> /*is_valid_tag*/,
                  Int2Type<false> /*test_extra_channels*/)
{
  Test<SampleT, 1, 1, CounterT, LevelT, OffsetT>(max_level, max_num_levels);
  Test<SampleT, 4, 3, CounterT, LevelT, OffsetT>(max_level, max_num_levels);
}

template <typename SampleT, typename CounterT, typename LevelT, typename OffsetT>
void TestChannels(LevelT max_level,
                  int max_num_levels,
                  Int2Type<true> /*is_valid_tag*/,
                  Int2Type<true> /*test_extra_channels*/)
{
  Test<SampleT, 1, 1, CounterT, LevelT, OffsetT>(max_level, max_num_levels);
  Test<SampleT, 4, 3, CounterT, LevelT, OffsetT>(max_level, max_num_levels);
  Test<SampleT, 3, 3, CounterT, LevelT, OffsetT>(max_level, max_num_levels);
  Test<SampleT, 4, 4, CounterT, LevelT, OffsetT>(max_level, max_num_levels);
}
template <typename SampleT,
          typename CounterT,
          typename LevelT,
          typename OffsetT,
          typename TestExtraChannels>
void TestChannels(LevelT /*max_level*/,
                  int /*max_num_levels*/,
                  Int2Type<false> /*is_valid_tag*/,
                  TestExtraChannels)
{}

void TestLevelsAliasing()
{
  constexpr int num_levels = 7;

  int h_histogram[num_levels - 1]{};
  int h_samples[]{
    0,  2,  4,  6,  8,  10, 12, // levels
    1,                          // bin 0
    3,  3,                      // bin 1
    5,  5,  5,                  // bin 2
    7,  7,  7,  7,              // bin 3
    9,  9,  9,  9,  9,          // bin 4
    11, 11, 11, 11, 11, 11      // bin 5
  };

  constexpr int num_samples = sizeof(h_samples) / sizeof(h_samples[0]);

  int *d_histogram{};
  int *d_samples{};

  CubDebugExit(
    g_allocator.DeviceAllocate((void **)&d_histogram, sizeof(h_histogram)));

  CubDebugExit(
    g_allocator.DeviceAllocate((void **)&d_samples, sizeof(h_samples)));

  CubDebugExit(
    cudaMemcpy(d_samples, h_samples, sizeof(h_samples), cudaMemcpyHostToDevice));

  // Alias levels with samples (fancy way to `d_histogram[bin]++`).
  int *d_levels = d_samples;

  std::uint8_t *d_temp_storage{};
  std::size_t temp_storage_bytes{};

  CubDebugExit(cub::DeviceHistogram::HistogramRange(d_temp_storage,
                                                    temp_storage_bytes,
                                                    d_samples,
                                                    d_histogram,
                                                    num_levels,
                                                    d_levels,
                                                    num_samples));

  CubDebugExit(
    g_allocator.DeviceAllocate((void **)&d_temp_storage, temp_storage_bytes));

  CubDebugExit(cub::DeviceHistogram::HistogramRange(d_temp_storage,
                                                    temp_storage_bytes,
                                                    d_samples,
                                                    d_histogram,
                                                    num_levels,
                                                    d_levels,
                                                    num_samples));

  CubDebugExit(cudaMemcpy(h_histogram,
                          d_histogram,
                          sizeof(h_histogram),
                          cudaMemcpyDeviceToHost));

  for (int bin = 0; bin < num_levels - 1; bin++)
  {
    // Each bin should contain `bin + 1` samples. Since samples also contain
    // levels, they contribute one extra item to each bin.
    AssertEquals(bin + 2, h_histogram[bin]);
  }

  CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
  CubDebugExit(g_allocator.DeviceFree(d_histogram));
  CubDebugExit(g_allocator.DeviceFree(d_levels));
}

// Regression test for NVIDIA/cub#489: integer rounding errors lead to incorrect
// bin detection:
void TestIntegerBinCalcs()
{
  constexpr int num_levels = 8;
  constexpr int num_bins = num_levels - 1;

  int h_histogram[num_bins]{};
  constexpr int h_histogram_ref[num_bins]{1, 5, 0, 2, 1, 0, 0};
  constexpr int h_samples[]{2, 6, 7, 2, 3, 0, 2, 2, 6, 999};
  constexpr int lower_level = 0;
  constexpr int upper_level = 12;

  constexpr int num_samples = sizeof(h_samples) / sizeof(h_samples[0]);

  int *d_histogram{};
  int *d_samples{};

  CubDebugExit(
    g_allocator.DeviceAllocate((void **)&d_histogram, sizeof(h_histogram)));

  CubDebugExit(
    g_allocator.DeviceAllocate((void **)&d_samples, sizeof(h_samples)));

  CubDebugExit(
    cudaMemcpy(d_samples, h_samples, sizeof(h_samples), cudaMemcpyHostToDevice));

  std::uint8_t *d_temp_storage{};
  std::size_t temp_storage_bytes{};

  CubDebugExit(cub::DeviceHistogram::HistogramEven(d_temp_storage,
                                                   temp_storage_bytes,
                                                   d_samples,
                                                   d_histogram,
                                                   num_levels,
                                                   lower_level,
                                                   upper_level,
                                                   num_samples));

  CubDebugExit(
    g_allocator.DeviceAllocate((void **)&d_temp_storage, temp_storage_bytes));

  CubDebugExit(cub::DeviceHistogram::HistogramEven(d_temp_storage,
                                                    temp_storage_bytes,
                                                    d_samples,
                                                    d_histogram,
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    num_samples));

  CubDebugExit(cudaMemcpy(h_histogram,
                          d_histogram,
                          sizeof(h_histogram),
                          cudaMemcpyDeviceToHost));

  for (int bin = 0; bin < num_bins; ++bin)
  {
    AssertEquals(h_histogram_ref[bin], h_histogram[bin]);
  }

  CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
  CubDebugExit(g_allocator.DeviceFree(d_histogram));
  CubDebugExit(g_allocator.DeviceFree(d_samples));
}

/**
 * @brief Our bin computation for HistogramEven is guaranteed only for when (max_level - min_level)
 * * num_bins does not overflow when using uint64_t arithmetic. In case bin computation could
 * overflow, we expect cudaErrorInvalidValue to be returned.
 */
template<typename SampleT>
void TestOverflow()
{
  using CounterT                   = uint32_t;
  constexpr std::size_t test_cases = 2;

  // Test data common across tests
  SampleT lower_level = 0;
  SampleT upper_level = ::cuda::std::numeric_limits<SampleT>::max();
  thrust::counting_iterator<SampleT> d_samples{0UL};
  thrust::device_vector<CounterT> d_histo_out(1024);
  CounterT *d_histogram = thrust::raw_pointer_cast(d_histo_out.data());
  int num_samples       = 1000;

  // Prepare per-test specific data
  constexpr std::size_t canary_bytes = 3;
  std::array<std::size_t, test_cases> temp_storage_bytes{canary_bytes, canary_bytes};
  std::array<int, test_cases> num_bins{1, 2};
  // Since test #1 is just a single bin, we expect it to succeed
  // Since we promote up to 64-bit integer arithmetic we expect tests to not overflow for types of
  // up to 4 bytes. For 64-bit and wider types, we do not perform further promotion to even wider
  // types, hence we expect cudaErrorInvalidValue to be returned to indicate of a potential overflow
  std::array<cudaError_t, test_cases> expected_status{
    cudaSuccess, 
    sizeof(SampleT) <= 4UL ? cudaSuccess : cudaErrorInvalidValue};

  // Verify we always initializes temp_storage_bytes
  cudaError_t error{cudaSuccess};
  for (std::size_t i = 0; i < test_cases; i++)
  {
    error = cub::DeviceHistogram::HistogramEven(nullptr,
                                                temp_storage_bytes[i],
                                                d_samples,
                                                d_histogram,
                                                num_bins[i] + 1,
                                                lower_level,
                                                upper_level,
                                                num_samples);

    // Ensure that temp_storage_bytes has been initialized even in the presence of error
    AssertTrue(temp_storage_bytes[i] != canary_bytes);
  }

  // Allocate sufficient temporary storage
  thrust::device_vector<std::uint8_t> temp_storage(
    std::max(temp_storage_bytes[0], temp_storage_bytes[1]));

  for (std::size_t i = 0; i < test_cases; i++)
  {
    error = cub::DeviceHistogram::HistogramEven(thrust::raw_pointer_cast(temp_storage.data()),
                                                temp_storage_bytes[i],
                                                d_samples,
                                                d_histogram,
                                                num_bins[i] + 1,
                                                lower_level,
                                                upper_level,
                                                num_samples);

    // Ensure we do not return an error on querying temporary storage requirements
    AssertEquals(error, expected_status[i]);
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
    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    g_verbose_input = args.CheckCmdLineFlag("v2");

    args.GetCmdLineArgument("i", g_timing_iterations);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--i=<timing iterations>] "
            "[--device=<device-id>] "
            "[--v] "
            "[--v2] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    TestOverflow<uint8_t>();
    TestOverflow<uint16_t>();
    TestOverflow<uint32_t>();
    TestOverflow<uint64_t>();
    using true_t = Int2Type<true>;
    using false_t = Int2Type<false>;

    TestLevelsAliasing();
    TestIntegerBinCalcs(); // regression test for NVIDIA/cub#489

#if TEST_HALF_T
    TestChannels<half_t, int, half_t, int>(256, 256 + 1, true_t{}, true_t{});
#endif

    TestChannels <signed char,      int, int,   int>(256,   256 + 1,  true_t{}, true_t{});
    TestChannels <unsigned short,   int, int,   int>(8192,  8192 + 1, true_t{}, false_t{});

    // Make sure bin computation works fine when using int32 arithmetic
    TestChannels <unsigned short,   int, unsigned short,   int>(std::numeric_limits<unsigned short>::max(),  std::numeric_limits<unsigned short>::max() + 1, true_t{}, false_t{});
    // Make sure bin computation works fine when requiring int64 arithmetic
    TestChannels <unsigned int,   int, unsigned int,   int>(std::numeric_limits<unsigned int>::max(),  8192 + 1, true_t{}, false_t{});
#if !defined(__ICC)
    // Fails with ICC for unknown reasons, see #332.
    TestChannels <float,            int, float, int>(1.0,   256 + 1,  true_t{}, false_t{});
#endif

    // float samples, int levels, regression test for NVIDIA/cub#479.
    TestChannels <float,            int, int,   int>(12,    7,        true_t{}, true_t{});

    // Test down-conversion of size_t offsets to int
    TestChannels <unsigned char,    int, int,   long long>(256, 256 + 1, Int2Type<(sizeof(size_t) != sizeof(int))>{}, false_t{});

    return 0;
}
