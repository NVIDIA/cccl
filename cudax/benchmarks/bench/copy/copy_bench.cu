// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/device_vector.h>

#include <cuda/mdspan>
#include <cuda/std/array>
#include <cuda/stream>

#include <cuda/experimental/copy.cuh>

#include <cstdlib>

#include <nvbench/nvbench.cuh>

// GCC -Warray-bounds false positive for high-rank (20+) __raw_tensor instantiations
_CCCL_DIAG_SUPPRESS_GCC("-Warray-bounds")

using data_t = int;

template <size_t Rank>
int compute_alloc(int offset, const cuda::std::array<int, Rank>& shape, const cuda::std::array<int, Rank>& strides)
{
  int max_pos = offset;
  for (size_t i = 0; i < Rank; ++i)
  {
    int delta = (shape[i] - 1) * strides[i];
    if (delta > 0)
    {
      max_pos += delta;
    }
  }
  return max_pos + 1;
}

template <size_t Rank>
void bench_copy(nvbench::state& state,
                int src_offset,
                const cuda::std::array<int, Rank>& shape,
                const cuda::std::array<int, Rank>& src_strides,
                int dst_offset,
                const cuda::std::array<int, Rank>& dst_strides)
{
  const int src_alloc = compute_alloc(src_offset, shape, src_strides);
  const int dst_alloc = compute_alloc(dst_offset, shape, dst_strides);

  thrust::device_vector<data_t> d_src(src_alloc);
  thrust::device_vector<data_t> d_dst(dst_alloc);

  int num_items = 1;
  for (size_t i = 0; i < Rank; ++i)
  {
    num_items *= shape[i];
  }
  state.add_element_count(num_items);
  state.add_global_memory_reads<data_t>(num_items);
  state.add_global_memory_writes<data_t>(num_items);

  using extents_t = cuda::std::dextents<int, Rank>;
  using strides_t = cuda::dstrides<int, Rank>;
  using mapping_t = cuda::layout_stride_relaxed::mapping<extents_t>;

  extents_t ext(shape);
  auto src_ptr = thrust::raw_pointer_cast(d_src.data()) + src_offset;
  auto dst_ptr = thrust::raw_pointer_cast(d_dst.data()) + dst_offset;
  mapping_t src_map(ext, strides_t(src_strides));
  mapping_t dst_map(ext, strides_t(dst_strides));

  cuda::device_mdspan<data_t, extents_t, cuda::layout_stride_relaxed> src(src_ptr, src_map);
  cuda::device_mdspan<data_t, extents_t, cuda::layout_stride_relaxed> dst(dst_ptr, dst_map);

  state.exec([&](nvbench::launch& launch) {
    cuda::stream_ref stream{launch.get_stream()};
    cuda::experimental::copy(src, dst, stream);
  });
}

template <size_t Rank>
void bench_copy(nvbench::state& state,
                int offset,
                const cuda::std::array<int, Rank>& shape,
                const cuda::std::array<int, Rank>& strides)
{
  bench_copy(state, offset, shape, strides, offset, strides);
}

/***********************************************************************************************************************
 * Memcpy benchmarks
 **********************************************************************************************************************/

// src: (70,90,80,80):(576000,6400,80,1)
// dst: (70,90,80,80):(576000,6400,80,1)
void memcpy_layout_0(nvbench::state& state)
{
  bench_copy(state, 0, cuda::std::array<int, 4>{70, 90, 80, 80}, cuda::std::array<int, 4>{576000, 6400, 80, 1});
}
NVBENCH_BENCH(memcpy_layout_0).set_name("memcpy_layout_0");

// src: (70,90,80,80):(90,1,6300,504000)
// dst: (70,90,80,80):(90,1,6300,504000)
void memcpy_layout_1(nvbench::state& state)
{
  bench_copy(state, 0, cuda::std::array<int, 4>{70, 90, 80, 80}, cuda::std::array<int, 4>{90, 1, 6300, 504000});
}
NVBENCH_BENCH(memcpy_layout_1).set_name("memcpy_layout_1");

// src: (1001,1007,3,31):(31217,1,31248217,1007), offset=0
// dst: (1001,1007,3,31):(31217,1,31248217,1007), offset=31248217
void memcpy_layout_2(nvbench::state& state)
{
  cuda::std::array<int, 4> shape{1001, 1007, 3, 31};
  cuda::std::array<int, 4> strides{31217, 1, 31248217, 1007};
  bench_copy(state, 0, shape, strides, 31248217, strides);
}
NVBENCH_BENCH(memcpy_layout_2).set_name("memcpy_layout_2");

// src: (57,71,1,1007,1):(71497,1,12225987,71,4075329), offset=12225987+4075329
// dst: (57,71,1,1007,1):(71497,1,4075329,71,20376645), offset=3*4075329
void memcpy_layout_3(nvbench::state& state)
{
  cuda::std::array<int, 5> shape{57, 71, 1, 1007, 1};
  cuda::std::array<int, 5> src_strides{71497, 1, 12225987, 71, 4075329};
  cuda::std::array<int, 5> dst_strides{71497, 1, 4075329, 71, 20376645};
  bench_copy(state, 12225987 + 4075329, shape, src_strides, 3 * 4075329, dst_strides);
}
NVBENCH_BENCH(memcpy_layout_3).set_name("memcpy_layout_3");

// src: (63,70,1001):(1001,63063,-1), offset=1000
// dst: (63,70,1001):(1001,63063,-1), offset=1000
void memcpy_neg(nvbench::state& state)
{
  bench_copy(state, 1000, cuda::std::array<int, 3>{63, 70, 1001}, cuda::std::array<int, 3>{1001, 63063, -1});
}
NVBENCH_BENCH(memcpy_neg).set_name("memcpy_neg");

/***********************************************************************************************************************
 * Reorder strides benchmark
 **********************************************************************************************************************/

// src: (8,100019,4):(1100209,11,3)
// dst: (8,100019,4):(1,8,800152)
void reorder_strides(nvbench::state& state)
{
  cuda::std::array<int, 3> shape{8, 100019, 4};
  bench_copy(state, 0, shape, cuda::std::array<int, 3>{1100209, 11, 3}, 0, cuda::std::array<int, 3>{1, 8, 800152});
}
NVBENCH_BENCH(reorder_strides).set_name("reorder_strides");

/***********************************************************************************************************************
 * Negative strides benchmarks
 **********************************************************************************************************************/

// src: (70,90,80,80):(-576000,-6400,-80,-1), offset=alloc-1
// dst: (70,90,80,80):(-576000,-6400,-80,-1), offset=alloc-1
void negative_strides_0(nvbench::state& state)
{
  constexpr int offset = 70 * 90 * 80 * 80 - 1;
  bench_copy(state, offset, cuda::std::array<int, 4>{70, 90, 80, 80}, cuda::std::array<int, 4>{-576000, -6400, -80, -1});
}
NVBENCH_BENCH(negative_strides_0).set_name("Negative Strides (optimized)");

// src: (63,70,1001):(-1001,-63063,-1), offset=alloc-1
// dst: (63,70,1001):(70070,1001,1), offset=0
void src_neg_stride(nvbench::state& state)
{
  constexpr int src_offset = 63 * 70 * 1001 - 1;
  cuda::std::array<int, 3> shape{63, 70, 1001};
  bench_copy(
    state, src_offset, shape, cuda::std::array<int, 3>{-1001, -63063, -1}, 0, cuda::std::array<int, 3>{70070, 1001, 1});
}
NVBENCH_BENCH(src_neg_stride).set_name("src_neg_stride");

/***********************************************************************************************************************
 * Squeezing and flattening benchmarks
 **********************************************************************************************************************/

// src: (2,)^23, bit-permuted strides
// dst: (2,)^23, different bit-permuted strides (dims 20-22 differ)
void flatten_common(nvbench::state& state)
{
  cuda::std::array<int, 23> shape{};
  for (auto& s : shape)
  {
    s = 2;
  }
  // clang-format off
  cuda::std::array<int, 23> src_strides{
    1 << 15, 1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21, 1 << 22,
    1 << 14, 1 << 13, 1 << 12, 1 << 11, 1 << 10, 1 <<  9, 1 <<  8, 1 <<  7,
    1 <<  6, 1 <<  5, 1 <<  4, 1 <<  3, 1 <<  2, 1 <<  0, 1 <<  1};
  cuda::std::array<int, 23> dst_strides{
    1 << 15, 1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21, 1 << 22,
    1 << 14, 1 << 13, 1 << 12, 1 << 11, 1 << 10, 1 <<  9, 1 <<  8, 1 <<  7,
    1 <<  6, 1 <<  5, 1 <<  4, 1 <<  3, 1 <<  1, 1 <<  2, 1 <<  0};
  // clang-format on
  bench_copy(state, 0, shape, src_strides, 0, dst_strides);
}
NVBENCH_BENCH(flatten_common).set_name("flatten_common");

// src: (4,2,...,2):(5,2^4,...,2^22), alloc=2^23
// dst: (4,2,...,2):(2^19,2^18,...,2^0), alloc=2^21
void flatten_one(nvbench::state& state)
{
  // clang-format off
  cuda::std::array<int, 20> shape{4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  cuda::std::array<int, 20> src_strides{
    5,       1 << 4,  1 << 5,  1 << 6,  1 << 7,  1 << 8,  1 << 9,  1 << 10,
    1 << 11, 1 << 12, 1 << 13, 1 << 14, 1 << 15, 1 << 16, 1 << 17, 1 << 18,
    1 << 19, 1 << 20, 1 << 21, 1 << 22};
  cuda::std::array<int, 20> dst_strides{
    1 << 19, 1 << 18, 1 << 17, 1 << 16, 1 << 15, 1 << 14, 1 << 13, 1 << 12,
    1 << 11, 1 << 10, 1 <<  9, 1 <<  8, 1 <<  7, 1 <<  6, 1 <<  5, 1 <<  4,
    1 <<  3, 1 <<  2, 1 <<  1, 1 <<  0};
  // clang-format on
  bench_copy(state, 0, shape, src_strides, 0, dst_strides);
}
NVBENCH_BENCH(flatten_one).set_name("flatten_one");

/***********************************************************************************************************************
 * Vectorize benchmarks
 **********************************************************************************************************************/

// src: (35,255,10,24):(61440,240,24,1), offset=240
// dst: (35,255,10,24):(61200,240,24,1), offset=0
void sliced_vec(nvbench::state& state)
{
  cuda::std::array<int, 4> shape{35, 255, 10, 24};
  bench_copy(
    state, 240, shape, cuda::std::array<int, 4>{61440, 240, 24, 1}, 0, cuda::std::array<int, 4>{61200, 240, 24, 1});
}
NVBENCH_BENCH(sliced_vec).set_name("sliced_vec");

// src: (355,255,4,3):(3072,12,3,1), offset=12
// dst: (355,255,4,3):(3060,12,3,1), offset=0
void sliced_vec_2(nvbench::state& state)
{
  cuda::std::array<int, 4> shape{355, 255, 4, 3};
  bench_copy(state, 12, shape, cuda::std::array<int, 4>{3072, 12, 3, 1}, 0, cuda::std::array<int, 4>{3060, 12, 3, 1});
}
NVBENCH_BENCH(sliced_vec_2).set_name("sliced_vec_2");

// src: (35,255,5,10):(153000,600,20,1), offset=205
// dst: (35,255,5,10):(12750,50,10,1), offset=0
void sliced_unaligned_ptr(nvbench::state& state)
{
  cuda::std::array<int, 4> shape{35, 255, 5, 10};
  bench_copy(
    state, 205, shape, cuda::std::array<int, 4>{153000, 600, 20, 1}, 0, cuda::std::array<int, 4>{12750, 50, 10, 1});
}
NVBENCH_BENCH(sliced_unaligned_ptr).set_name("sliced_unaligned_ptr");

/***********************************************************************************************************************
 * Transpose benchmark
 **********************************************************************************************************************/

// src: (8192,8192):(8192,1)
// dst: (8192,8192):(1,8192)
void transpose(nvbench::state& state)
{
  cuda::std::array<int, 2> shape{8192, 8192};
  bench_copy(state, 0, shape, cuda::std::array<int, 2>{8192, 1}, 0, cuda::std::array<int, 2>{1, 8192});
}
NVBENCH_BENCH(transpose).set_name("transpose");
