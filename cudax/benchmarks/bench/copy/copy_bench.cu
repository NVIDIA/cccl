// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/device_vector.h>

#include <cuda/mdspan>
#include <cuda/std/array>
#include <cuda/stream>

#include <cuda/experimental/copy.cuh>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <nvbench/nvbench.cuh>

// GCC -Warray-bounds false positive for high-rank (20+) __raw_tensor instantiations
_CCCL_DIAG_SUPPRESS_GCC("-Warray-bounds")

template <size_t Rank, typename idx_t>
size_t
compute_alloc(size_t offset, const cuda::std::array<idx_t, Rank>& shape, const cuda::std::array<idx_t, Rank>& strides)
{
  int64_t max_pos = static_cast<int64_t>(offset);
  for (size_t i = 0; i < Rank; ++i)
  {
    auto delta = static_cast<ptrdiff_t>(shape[i] - 1) * strides[i];
    if (delta > 0)
    {
      max_pos += delta;
    }
  }
  return max_pos + 1;
}

template <typename data_t = int, typename idx_t = int, size_t Rank>
void bench_copy(nvbench::state& state,
                size_t src_offset,
                const cuda::std::array<idx_t, Rank>& shape,
                const cuda::std::array<idx_t, Rank>& src_strides,
                size_t dst_offset,
                const cuda::std::array<idx_t, Rank>& dst_strides)
{
  const auto src_alloc = compute_alloc(src_offset, shape, src_strides);
  const auto dst_alloc = compute_alloc(dst_offset, shape, dst_strides);

  thrust::device_vector<data_t> d_src(src_alloc);
  thrust::device_vector<data_t> d_dst(dst_alloc);

  size_t num_items = 1;
  for (size_t i = 0; i < Rank; ++i)
  {
    num_items *= shape[i];
  }
  state.add_element_count(num_items);
  state.add_global_memory_reads<data_t>(num_items);
  state.add_global_memory_writes<data_t>(num_items);

  using extents_t = cuda::std::dextents<idx_t, Rank>;
  using strides_t = cuda::dstrides<idx_t, Rank>;
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

template <typename data_t = int, typename idx_t = int, size_t Rank>
void bench_copy(nvbench::state& state,
                size_t offset,
                const cuda::std::array<idx_t, Rank>& shape,
                const cuda::std::array<idx_t, Rank>& strides)
{
  bench_copy<data_t>(state, offset, shape, strides, offset, strides);
}

/***********************************************************************************************************************
 * Memcpy benchmarks
 **********************************************************************************************************************/

// src: (25, 70, 90, 80, 80):(40320000, 576000, 6400, 80, 1)
// dst: (25, 70, 90, 80, 80):(40320000, 576000, 6400, 80, 1)
void memcpy_layout_0(nvbench::state& state)
{
  cuda::std::array<int, 5> shape{25, 70, 90, 80, 80};
  cuda::std::array<int, 5> strides{40320000, 576000, 6400, 80, 1};
  bench_copy(state, 0, shape, strides);
}
NVBENCH_BENCH(memcpy_layout_0).set_name("contiguous (5D, int, 4GB)");

// src: (25, 80, 70, 80, 90):(40320000, 1, 576000, 80, 6400)
// dst: (25, 80, 70, 80, 90):(40320000, 1, 576000, 80, 6400)
void memcpy_layout_1(nvbench::state& state)
{
  cuda::std::array<int, 5> shape{25, 80, 70, 80, 90};
  cuda::std::array<int, 5> strides{40320000, 1, 576000, 80, 6400};
  bench_copy(state, 0, shape, strides);
}
NVBENCH_BENCH(memcpy_layout_1).set_name("contiguous-perm (5D, int, 4GB)");

// src: (1, 25, 1, 80, 1, 70, 1, 80, 1, 90):(1, 40320000, 1, 1, 1, 576000, 1, 80, 1, 6400)
// dst: (1, 25, 1, 80, 1, 70, 1, 80, 1, 90):(1, 40320000, 1, 1, 1, 576000, 1, 80, 1, 6400)
void memcpy_layout_1b(nvbench::state& state)
{
  cuda::std::array<int, 10> shape{1, 25, 1, 80, 1, 70, 1, 80, 1, 90};
  cuda::std::array<int, 10> strides{1, 40320000, 1, 1, 1, 576000, 1, 80, 1, 6400};
  bench_copy(state, 0, shape, strides);
}
NVBENCH_BENCH(memcpy_layout_1b).set_name("contiguous-1-sized (10D, int, 4GB)");

// src: (25, 70, 90, 80, 80):(40320000, 576000, 6400, 80, 1)
// dst: (25, 70, 90, 80, 80):(40320000, 576000, 6400, 80, 1)
void memcpy_layout_2(nvbench::state& state)
{
  cuda::std::array<int, 5> shape{25, 70, 90, 80, 80};
  cuda::std::array<int, 5> strides{40320000, 576000, 6400, 80, 1};
  bench_copy(state, 1, shape, strides);
}
NVBENCH_BENCH(memcpy_layout_2).set_name("contiguous-not-aligned (5D, int, 4GB)");

// src: (100, 70, 90, 80, 80):(40320000, 576000, 6400, 80, 1)
// dst: (100, 70, 90, 80, 80):(40320000, 576000, 6400, 80, 1)
void memcpy_layout_3(nvbench::state& state)
{
  cuda::std::array<int64_t, 5> shape{100, 70, 90, 80, 80};
  cuda::std::array<int64_t, 5> strides{40320000, 576000, 6400, 80, 1};
  bench_copy<char, int64_t>(state, 0, shape, strides);
}
NVBENCH_BENCH(memcpy_layout_3).set_name("contiguous-small (5D, char, 4GB)");

// src: (100, 70, 90, 80, 80):(40320000, 576000, 6400, 80, 1)
// dst: (100, 70, 90, 80, 80):(40320000, 576000, 6400, 80, 1)
void memcpy_layout_4(nvbench::state& state)
{
  cuda::std::array<int64_t, 5> shape{100, 70, 90, 80, 80};
  cuda::std::array<int64_t, 5> strides{40320000, 576000, 6400, 80, 1};
  bench_copy<char, int64_t>(state, 1, shape, strides);
}
NVBENCH_BENCH(memcpy_layout_4).set_name("contiguous-small-not-aligned (5D, char, 4GB)");

// src: (25, 70, 90, 80, 80):(40320000, 576000, 6400, 80, -1), offset=80
// dst: (25, 70, 90, 80, 80):(40320000, 576000, 6400, 80, -1), offset=80
void memcpy_neg(nvbench::state& state)
{
  cuda::std::array<int, 5> shape{25, 70, 90, 80, 80};
  cuda::std::array<int, 5> strides{40320000, 576000, 6400, 80, -1};
  bench_copy(state, 80, shape, strides);
}
NVBENCH_BENCH(memcpy_neg).set_name("contiguous-negative-stride (5D, int, 4GB)");

// src: (134217600, 32):(128, 1), offset=32
// dst: (134217600, 32):(128, 1), offset=32
// Copies 4GB while allocating 16GB per tensor because of the padded outer stride.
void vectorization(nvbench::state& state)
{
  cuda::std::array<int64_t, 2> shape{134217600, 32};
  cuda::std::array<int64_t, 2> strides{128, 1};
  bench_copy<char, int64_t>(state, 32, shape, strides);
}
NVBENCH_BENCH(vectorization).set_name("vectorization (2D, char, 4GB copy, 16GB alloc)");

// src: (32767, (128 * 1024) / sizeof(int)):(128 * 1024, 1)
// dst: (32767, (128 * 1024) / sizeof(int)):(128 * 1024, 1)
// Copies 4GB while allocating 16GB per tensor because each row is padded to 128K elements.
void block_contiguous(nvbench::state& state)
{
  cuda::std::array<int, 2> shape{32767, (128 * 1024) / sizeof(int)};
  cuda::std::array<int, 2> strides{128 * 1024, 1};
  bench_copy(state, 0, shape, strides);
}
NVBENCH_BENCH(block_contiguous).set_name("block-contiguous (2D, int, 4GB copy, 16GB alloc)");
// (non-vectorizable)

void several_dimensions(nvbench::state& state)
{
  cuda::std::array<int, 5> shape{64, 64, 64, 64, 64};
  cuda::std::array<int, 5> strides{17043520 + 1, 266304 + 1, 4160 + 1, 64 + 1, 1};
  bench_copy(state, 0, shape, strides);
}
NVBENCH_BENCH(several_dimensions).set_name("several_dimensions (5D, int, 4GB)");

void several_dimensions_non_square(nvbench::state& state)
{
  cuda::std::array<int, 5> shape{63, 65, 67, 69, 57};
  cuda::std::array<int, 5> strides{17433131, 268202, 4003, 58, 1};
  bench_copy(state, 0, shape, strides);
}
NVBENCH_BENCH(several_dimensions_non_square).set_name("several_dimensions_non_square (5D, int, 4GB)");

/***********************************************************************************************************************
 * Transpose benchmark
 **********************************************************************************************************************/

// src: (32768,32768):(1,32768)
// dst: (32768,32768):(32768,1)
void transpose_2D_col_row(nvbench::state& state)
{
  cuda::std::array<int, 2> shape{32768, 32768};
  cuda::std::array<int, 2> src_strides{1, 32768};
  cuda::std::array<int, 2> dst_strides{32768, 1};
  bench_copy(state, 0, shape, src_strides, 0, dst_strides);
}
NVBENCH_BENCH(transpose_2D_col_row).set_name("transpose_2D_col_row (2D, int, 4GB)");

void transpose_2D_row_col(nvbench::state& state)
{
  cuda::std::array<int, 2> shape{32768, 32768};
  cuda::std::array<int, 2> src_strides{32768, 1};
  cuda::std::array<int, 2> dst_strides{1, 32768};
  bench_copy(state, 0, shape, src_strides, 0, dst_strides);
}
NVBENCH_BENCH(transpose_2D_row_col).set_name("transpose_2D_row_col (2D, int, 4GB)");

void transpose_2D_char(nvbench::state& state)
{
  cuda::std::array<int64_t, 2> shape{65536, 65536};
  cuda::std::array<int64_t, 2> src_strides{1, 65536};
  cuda::std::array<int64_t, 2> dst_strides{65536, 1};
  bench_copy<char, int64_t>(state, 0, shape, src_strides, 0, dst_strides);
}
NVBENCH_BENCH(transpose_2D_char).set_name("transpose_2D_char (2D, char, 4GB)");

void transpose_2D_short(nvbench::state& state)
{
  cuda::std::array<int, 2> shape{32760, 32768 * 2};
  cuda::std::array<int, 2> src_strides{1, 32760};
  cuda::std::array<int, 2> dst_strides{32768 * 2, 1};
  bench_copy<short>(state, 0, shape, src_strides, 0, dst_strides);
}
NVBENCH_BENCH(transpose_2D_short).set_name("transpose_2D_short (2D, short, 4GB)");

void transpose_2D_double(nvbench::state& state)
{
  cuda::std::array<int64_t, 2> shape{32768, 16384};
  cuda::std::array<int64_t, 2> src_strides{1, 32768};
  cuda::std::array<int64_t, 2> dst_strides{16384, 1};
  bench_copy<double, int64_t>(state, 0, shape, src_strides, 0, dst_strides);
}
NVBENCH_BENCH(transpose_2D_double).set_name("transpose_2D_double (2D, double, 4GB)");

void transpose_2D_odd_both(nvbench::state& state)
{
  cuda::std::array<int, 2> shape{32767, 32769};
  cuda::std::array<int, 2> src_strides{1, 32767};
  cuda::std::array<int, 2> dst_strides{32769, 1};
  bench_copy(state, 0, shape, src_strides, 0, dst_strides);
}
NVBENCH_BENCH(transpose_2D_odd_both).set_name("transpose_2D_odd_both (2D, int, 4GB)");

void transpose_3D(nvbench::state& state)
{
  cuda::std::array<int, 3> shape{1024, 1024, 1024};
  cuda::std::array<int, 3> src_strides{1, 1024, 1024 * 1024};
  cuda::std::array<int, 3> dst_strides{1024 * 1024, 1024, 1};
  bench_copy(state, 0, shape, src_strides, 0, dst_strides);
}
NVBENCH_BENCH(transpose_3D).set_name("transpose_3D (3D, int, 4GB)");

void transpose_3D_odd_edges(nvbench::state& state)
{
  cuda::std::array<int, 3> shape{1023, 1025, 1024};
  cuda::std::array<int, 3> src_strides{1, 1023, 1023 * 1025};
  cuda::std::array<int, 3> dst_strides{1025 * 1024, 1024, 1};
  bench_copy(state, 0, shape, src_strides, 0, dst_strides);
}
NVBENCH_BENCH(transpose_3D_odd_edges).set_name("transpose_3D_odd_edges (3D, int, 4GB)");

void transpose_src_small_15(nvbench::state& state)
{
  cuda::std::array<int, 3> shape{15, 2236962, 32};
  cuda::std::array<int, 3> src_strides{1, 15 * 32, 15};
  cuda::std::array<int, 3> dst_strides{2236962 * 32, 32, 1};
  bench_copy(state, 0, shape, src_strides, 0, dst_strides);
}
NVBENCH_BENCH(transpose_src_small_15).set_name("transpose_src_small_15 (3D, int, 4GB)");

void transpose_src_small_16(nvbench::state& state)
{
  cuda::std::array<int, 3> shape{16, 2097152, 32};
  cuda::std::array<int, 3> src_strides{1, 16 * 32, 16};
  cuda::std::array<int, 3> dst_strides{2097152 * 32, 32, 1};
  bench_copy(state, 0, shape, src_strides, 0, dst_strides);
}
NVBENCH_BENCH(transpose_src_small_16).set_name("transpose_src_small_16 (3D, int, 4GB)");

void transpose_src_small_17(nvbench::state& state)
{
  cuda::std::array<int, 3> shape{17, 1973790, 32};
  cuda::std::array<int, 3> src_strides{1, 17 * 32, 17};
  cuda::std::array<int, 3> dst_strides{1973790 * 32, 32, 1};
  bench_copy(state, 0, shape, src_strides, 0, dst_strides);
}
NVBENCH_BENCH(transpose_src_small_17).set_name("transpose_src_small_17 (3D, int, 4GB)");

void transpose_dst_small_8_padded(nvbench::state& state)
{
  cuda::std::array<int64_t, 3> shape{32, 4194304, 8};
  cuda::std::array<int64_t, 3> src_strides{1, 32 * 8, 32};
  cuda::std::array<int64_t, 3> dst_strides{4194304 * 16, 16, 1};
  bench_copy<int, int64_t>(state, 0, shape, src_strides, 0, dst_strides);
}
NVBENCH_BENCH(transpose_dst_small_8_padded).set_name("transpose_dst_small_8_padded (3D, int, 4GB)");

void transpose_dst_small_16_padded(nvbench::state& state)
{
  cuda::std::array<int64_t, 3> shape{32, 2097152, 16};
  cuda::std::array<int64_t, 3> src_strides{1, 32 * 16, 32};
  cuda::std::array<int64_t, 3> dst_strides{2097152 * 32, 32, 1};
  bench_copy<int, int64_t>(state, 0, shape, src_strides, 0, dst_strides);
}
NVBENCH_BENCH(transpose_dst_small_16_padded).set_name("transpose_dst_small_16_padded (3D, int, 4GB)");

void transpose_src_small_16_4D(nvbench::state& state)
{
  cuda::std::array<int, 4> shape{16, 1024, 2048, 32};
  cuda::std::array<int, 4> src_strides{1, 16 * 32, 16 * 32 * 1024, 16};
  cuda::std::array<int, 4> dst_strides{1024 * 2048 * 32, 32, 1024 * 32, 1};
  bench_copy(state, 0, shape, src_strides, 0, dst_strides);
}
NVBENCH_BENCH(transpose_src_small_16_4D).set_name("transpose_src_small_16_4D (4D, int, 4GB)");
