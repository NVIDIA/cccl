// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/device/device_memcpy.cuh>

// %RANGE% TUNE_THREADS tpb 128:1024:32
// %RANGE% TUNE_BUFFERS_PER_THREAD bpt 1:18:1
// %RANGE% TUNE_TLEV_BYTES_PER_THREAD tlevbpt 2:16:2
// %RANGE% TUNE_LARGE_THREADS ltpb 128:1024:32
// %RANGE% TUNE_LARGE_BUFFER_BYTES_PER_THREAD lbbpt 4:128:4
// %RANGE% TUNE_PREFER_POW2_BITS ppb 0:1:1
// %RANGE% TUNE_GROUP4_LEVEL_THRESHOLD g4lt 0:512:32
// %RANGE% TUNE_GROUP8_LEVEL_THRESHOLD g8lt 0:512:32
// %RANGE% TUNE_GROUP16_LEVEL_THRESHOLD g16lt 0:512:32
// %RANGE% TUNE_WARP_LEVEL_THRESHOLD wlt 0:512:32
// %RANGE% TUNE_BLOCK_LEVEL_THRESHOLD blt 1024:16384:512
// %RANGE% TUNE_BLOCK_MAGIC_NS blns 0:2048:4
// %RANGE% TUNE_BLOCK_DELAY_CONSTRUCTOR_ID bldcid 0:7:1
// %RANGE% TUNE_BLOCK_L2_WRITE_LATENCY_NS bll2w 0:1200:5
// %RANGE% TUNE_BUFF_MAGIC_NS buns 0:2048:4
// %RANGE% TUNE_BUFF_DELAY_CONSTRUCTOR_ID budcid 0:7:1
// %RANGE% TUNE_BUFF_L2_WRITE_LATENCY_NS bul2w 0:1200:5

#include <thrust/gather.h>
#include <thrust/random.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/tabulate.h>

#include <cuda/cmath>

#include <nvbench_helper.cuh>

template <class T, class OffsetT>
struct offset_to_ptr_t
{
  T* d_ptr;
  OffsetT* d_offsets;

  __device__ T* operator()(OffsetT i) const
  {
    return d_ptr + d_offsets[i];
  }
};

template <class T, class OffsetT>
struct offset_to_bytes_t
{
  OffsetT* d_offsets;

  __device__ OffsetT operator()(OffsetT i) const
  {
    return (d_offsets[i + 1] - d_offsets[i]) * sizeof(T);
  }
};

template <class OffsetT>
struct offset_to_padded_size_t
{
  OffsetT* d_offsets;
  OffsetT alignment;

  __device__ OffsetT operator()(OffsetT i) const
  {
    return cuda::round_up(d_offsets[i + 1] - d_offsets[i], alignment);
  }
};

#if !TUNE_BASE

struct policy_selector_t
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(cuda::compute_capability) const -> cub::BatchedCopyPolicy
  {
    return {
      cub::BatchedCopyAlgorithm::lookback,
      {
        {
          TUNE_THREADS,
          TUNE_BUFFERS_PER_THREAD,
          TUNE_TLEV_BYTES_PER_THREAD,
          bool{TUNE_PREFER_POW2_BITS},
          TUNE_LARGE_THREADS * TUNE_LARGE_BUFFER_BYTES_PER_THREAD,
          TUNE_GROUP4_LEVEL_THRESHOLD,
          TUNE_GROUP8_LEVEL_THRESHOLD,
          TUNE_GROUP16_LEVEL_THRESHOLD,
          TUNE_WARP_LEVEL_THRESHOLD,
          TUNE_BLOCK_LEVEL_THRESHOLD,
          cub::LookbackDelayPolicy{static_cast<cub::LookbackDelayAlgorithm>(TUNE_BUFF_DELAY_CONSTRUCTOR_ID),
                                   TUNE_BUFF_MAGIC_NS,
                                   TUNE_BUFF_L2_WRITE_LATENCY_NS},
          cub::LookbackDelayPolicy{static_cast<cub::LookbackDelayAlgorithm>(TUNE_BLOCK_DELAY_CONSTRUCTOR_ID),
                                   TUNE_BLOCK_MAGIC_NS,
                                   TUNE_BLOCK_L2_WRITE_LATENCY_NS},

        },
        {TUNE_LARGE_THREADS, TUNE_LARGE_BUFFER_BYTES_PER_THREAD},
      },
    };
  }
};
#endif

// Lays out the buffers back to back (in shuffled order if `randomize` is set), starting each buffer at a multiple of
// `alignment` elements. Returns the start of every buffer, followed by the total number of elements of the layout.
template <class OffsetT>
thrust::device_vector<std::size_t> gen_layout(
  thrust::device_vector<OffsetT>& offsets, bool randomize, std::size_t alignment, thrust::default_random_engine& rne)
{
  const auto buffers = offsets.size() - 1;

  // A single padded size fits in OffsetT, but their sum may not, so the sizes are scanned as std::size_t. The trailing
  // zero size makes the exclusive scans also produce the total number of elements.
  thrust::device_vector<std::size_t> sizes(buffers + 1);
  thrust::tabulate(
    sizes.begin(),
    sizes.end() - 1,
    offset_to_padded_size_t<OffsetT>{thrust::raw_pointer_cast(offsets.data()), static_cast<OffsetT>(alignment)});

  thrust::device_vector<std::size_t> starts(buffers + 1);
  if (randomize)
  {
    // Buffer i is placed at slot map[i]
    thrust::device_vector<OffsetT> map(buffers);
    thrust::sequence(map.begin(), map.end());
    thrust::shuffle(map.begin(), map.end(), rne);
    thrust::device_vector<std::size_t> slots(buffers + 1);
    thrust::scatter(sizes.begin(), sizes.end() - 1, map.begin(), slots.begin());
    thrust::exclusive_scan(slots.begin(), slots.end(), slots.begin());
    thrust::gather(map.begin(), map.end(), slots.begin(), starts.begin());
    starts.back() = slots.back();
  }
  else
  {
    thrust::exclusive_scan(sizes.begin(), sizes.end(), starts.begin());
  }
  return starts;
}

template <class T, class OffsetT>
void copy(nvbench::state& state,
          nvbench::type_list<T, OffsetT>,
          std::size_t elements,
          std::size_t min_buffer_size,
          std::size_t max_buffer_size,
          bool randomize_input,
          bool randomize_output,
          bool align_input)
{
  using offset_t           = OffsetT;
  using it_t               = T*;
  using input_buffer_it_t  = it_t*;
  using output_buffer_it_t = it_t*;
  using buffer_size_it_t   = offset_t*;

  thrust::device_vector<offset_t> offsets =
    generate.uniform.segment_offsets(elements, min_buffer_size, max_buffer_size);

  offset_t* d_offsets = thrust::raw_pointer_cast(offsets.data());

  const auto buffers = offsets.size() - 1;

  thrust::device_vector<offset_t> buffer_sizes(buffers);
  thrust::tabulate(buffer_sizes.begin(), buffer_sizes.end(), offset_to_bytes_t<T, offset_t>{d_offsets});

  // Aligned input buffers start at a multiple of 16 bytes, the widest vector access of the kernel
  const std::size_t input_alignment = align_input ? 16 / sizeof(T) : 1;

  thrust::default_random_engine rne;
  thrust::device_vector<std::size_t> input_starts  = gen_layout(offsets, randomize_input, input_alignment, rne);
  thrust::device_vector<std::size_t> output_starts = gen_layout(offsets, randomize_output, 1, rne);

  // Clear the offsets vector to free memory
  offsets.clear();
  offsets.shrink_to_fit();
  d_offsets = nullptr;

  thrust::device_vector<T> input_buffer = generate(static_cast<std::size_t>(input_starts.back()));
  thrust::device_vector<T> output_buffer(static_cast<std::size_t>(output_starts.back()));

  T* d_input_buffer  = thrust::raw_pointer_cast(input_buffer.data());
  T* d_output_buffer = thrust::raw_pointer_cast(output_buffer.data());

  thrust::device_vector<it_t> input_buffers(buffers);
  thrust::device_vector<it_t> output_buffers(buffers);
  thrust::tabulate(input_buffers.begin(),
                   input_buffers.end(),
                   offset_to_ptr_t<T, std::size_t>{d_input_buffer, thrust::raw_pointer_cast(input_starts.data())});
  thrust::tabulate(output_buffers.begin(),
                   output_buffers.end(),
                   offset_to_ptr_t<T, std::size_t>{d_output_buffer, thrust::raw_pointer_cast(output_starts.data())});

  input_buffer_it_t d_input_buffers   = thrust::raw_pointer_cast(input_buffers.data());
  output_buffer_it_t d_output_buffers = thrust::raw_pointer_cast(output_buffers.data());
  buffer_size_it_t d_buffer_sizes     = thrust::raw_pointer_cast(buffer_sizes.data());

  state.add_element_count(elements);
  state.add_global_memory_writes<T>(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_reads<it_t>(buffers);
  state.add_global_memory_reads<it_t>(buffers);
  state.add_global_memory_reads<offset_t>(buffers);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               auto env = cub_bench_env(
                 alloc,
                 launch
#if !TUNE_BASE
                 ,
                 cuda::execution::tune(policy_selector_t{})
#endif
               );
               _CCCL_TRY_RUNTIME_API(
                 cub::DeviceMemcpy::Batched,
                 "Batched failed",
                 d_input_buffers,
                 d_output_buffers,
                 d_buffer_sizes,
                 static_cast<cuda::std::int64_t>(buffers),
                 env);
             });
}

template <class T, class OffsetT>
void uniform(nvbench::state& state, nvbench::type_list<T, OffsetT> tl)
{
  const auto elements              = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const auto max_buffer_size       = static_cast<std::size_t>(state.get_int64("MaxBufferSize"));
  const auto min_buffer_size_ratio = static_cast<std::size_t>(state.get_int64("MinBufferSizeRatio"));
  const auto min_buffer_size       = max_buffer_size * min_buffer_size_ratio / 100;

  copy(state,
       tl,
       elements,
       min_buffer_size,
       max_buffer_size,
       state.get_int64("Randomize"),
       state.get_int64("Randomize"),
       state.get_int64("InputAligned"));
}

template <class T, class OffsetT>
void large(nvbench::state& state, nvbench::type_list<T, OffsetT> tl)
{
  const auto elements                  = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const auto max_buffer_size           = elements;
  constexpr auto min_buffer_size_ratio = 99;
  const auto min_buffer_size =
    static_cast<std::size_t>(static_cast<double>(max_buffer_size) / 100.0) * min_buffer_size_ratio;

  // No need to randomize large buffers
  constexpr bool randomize_input  = false;
  constexpr bool randomize_output = false;
  constexpr bool align_input      = false;

  copy(state, tl, elements, min_buffer_size, max_buffer_size, randomize_input, randomize_output, align_input);
}

using types = nvbench::type_list<nvbench::uint8_t, nvbench::uint32_t>;

NVBENCH_BENCH_TYPES(uniform, NVBENCH_TYPE_AXES(types, offset_types))
  .set_name("uniform")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(25, 29, 2))
  .add_int64_axis("MinBufferSizeRatio", {1, 99, 100})
  .add_int64_axis("MaxBufferSize", {8, 64, 255, 256, 1024, 64 * 1024})
  .add_int64_axis("Randomize", {0, 1})
  .add_int64_axis("InputAligned", {0, 1});

NVBENCH_BENCH_TYPES(large, NVBENCH_TYPE_AXES(types, offset_types))
  .set_name("large")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", {28, 29});
