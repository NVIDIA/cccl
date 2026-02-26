// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/device_vector.h>

#include <cuda/experimental/__copy_bytes/copy_bytes_naive.cuh>
#include <cuda/experimental/__copy_bytes/copy_bytes_registers.cuh>

#include <cute/layout.hpp>
#include <nvbench/nvbench.cuh>

using data_t = int;

template <typename SrcLayout, typename DstLayout>
void bench_copy(nvbench::state& state,
                const std::string& impl,
                int src_offset,
                const SrcLayout& src_layout,
                int dst_offset,
                const DstLayout& dst_layout)
{
  namespace cudax = cuda::experimental;
  thrust::device_vector<data_t> d_src(src_offset + static_cast<int>(cute::cosize(src_layout)));
  thrust::device_vector<data_t> d_dst(dst_offset + static_cast<int>(cute::cosize(dst_layout)));
  auto* src_ptr       = thrust::raw_pointer_cast(d_src.data()) + src_offset;
  auto* dst_ptr       = thrust::raw_pointer_cast(d_dst.data()) + dst_offset;
  const int num_items = static_cast<int>(cute::size(src_layout));
  state.add_element_count(num_items);
  state.add_global_memory_reads<data_t>(num_items);
  state.add_global_memory_writes<data_t>(num_items);

  if (impl == "naive")
  {
    state.exec([&](nvbench::launch& launch) {
      cuda::stream_ref stream{launch.get_stream()};
      cudax::copy_bytes_naive(src_ptr, src_layout, dst_ptr, dst_layout, stream);
    });
  }
  else
  {
    state.exec([&](nvbench::launch& launch) {
      cuda::stream_ref stream{launch.get_stream()};
      cudax::copy_bytes_registers(src_ptr, src_layout, dst_ptr, dst_layout, stream);
    });
  }
}

/***********************************************************************************************************************
 * Memcpy benchmarks
 **********************************************************************************************************************/

// memcpy_layout_0: simple C-order, shape (70,90,80,80)
void memcpy_layout_0(nvbench::state& state)
{
  using namespace cute;
  auto layout = make_layout(make_shape(70, 90, 80, 80), make_stride(576000, 6400, 80, 1));
  bench_copy(state, state.get_string("Impl"), 0, layout, 0, layout);
}
NVBENCH_BENCH(memcpy_layout_0).set_name("memcpy_layout_0").add_string_axis("Impl", {"naive", "registers"});

// memcpy_layout_1: stride_order (3,2,0,1), shape (70,90,80,80)
void memcpy_layout_1(nvbench::state& state)
{
  using namespace cute;
  auto layout = make_layout(make_shape(70, 90, 80, 80), make_stride(90, 1, 6300, 504000));
  bench_copy(state, state.get_string("Impl"), 0, layout, 0, layout);
}
NVBENCH_BENCH(memcpy_layout_1).set_name("memcpy_layout_1").add_string_axis("Impl", {"naive", "registers"});

// memcpy_layout_2: sliced contiguous, copy shape (1001,1007,3,31)
void memcpy_layout_2(nvbench::state& state)
{
  using namespace cute;
  constexpr int dst_offset = 31248217;
  auto copy_layout         = make_layout(make_shape(1001, 1007, 3, 31), make_stride(31217, 1, 31248217, 1007));
  bench_copy(state, state.get_string("Impl"), 0, copy_layout, dst_offset, copy_layout);
}
NVBENCH_BENCH(memcpy_layout_2).set_name("memcpy_layout_2").add_string_axis("Impl", {"naive", "registers"});

// memcpy_layout_3: two most-strided extents sliced to 1, copy shape (57,71,1,1007,1)
void memcpy_layout_3(nvbench::state& state)
{
  using namespace cute;
  constexpr int src_offset = 12225987 + 4075329;
  auto src_layout          = make_layout(make_shape(57, 71, 1, 1007, 1), make_stride(71497, 1, 12225987, 71, 4075329));
  constexpr int dst_offset = 3 * 4075329;
  auto dst_layout          = make_layout(make_shape(57, 71, 1, 1007, 1), make_stride(71497, 1, 4075329, 71, 20376645));
  bench_copy(state, state.get_string("Impl"), src_offset, src_layout, dst_offset, dst_layout);
}
NVBENCH_BENCH(memcpy_layout_3).set_name("memcpy_layout_3").add_string_axis("Impl", {"naive", "registers"});

// memcpy_neg: negative stride on last dim, shape (63,70,1001)
void memcpy_neg(nvbench::state& state)
{
  using namespace cute;
  constexpr int offset = 1000;
  auto layout          = make_layout(make_shape(63, 70, 1001), make_stride(1001, 63063, -1));
  bench_copy(state, state.get_string("Impl"), offset, layout, offset, layout);
}
NVBENCH_BENCH(memcpy_neg).set_name("memcpy_neg").add_string_axis("Impl", {"naive", "registers"});

/***********************************************************************************************************************
 * Reorder strides benchmark
 **********************************************************************************************************************/

void reorder_strides(nvbench::state& state)
{
  using namespace cute;
  auto dst_layout = make_layout(make_shape(8, 100019, 4), make_stride(1, 8, 800152));
  auto src_layout = make_layout(make_shape(8, 100019, 4), make_stride(1100209, 11, 3));
  bench_copy(state, state.get_string("Impl"), 0, src_layout, 0, dst_layout);
}
NVBENCH_BENCH(reorder_strides).set_name("reorder_strides").add_string_axis("Impl", {"naive", "registers"});

/***********************************************************************************************************************
 * Negative strides benchmark
 **********************************************************************************************************************/

void src_neg_stride(nvbench::state& state)
{
  using namespace cute;
  constexpr int src_offset = 63 * 70 * 1001 - 1;
  auto src_layout          = make_layout(make_shape(63, 70, 1001), make_stride(-1001, -63063, -1));
  auto dst_layout          = make_layout(make_shape(63, 70, 1001), make_stride(70070, 1001, 1));
  bench_copy(state, state.get_string("Impl"), src_offset, src_layout, 0, dst_layout);
}
NVBENCH_BENCH(src_neg_stride).set_name("src_neg_stride").add_string_axis("Impl", {"naive", "registers"});

/***********************************************************************************************************************
 * Squeezing and flattening benchmarks
 **********************************************************************************************************************/

// flatten_common: 23-dim (2,)*23, stride orders differ only in dims 20-22
void flatten_common(nvbench::state& state)
{
  using namespace cute;
  // clang-format off
  auto shape = make_shape(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
  auto dst_layout = make_layout(shape,
    make_stride(1 << 15, 1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21, 1 << 22,
                1 << 14, 1 << 13, 1 << 12, 1 << 11, 1 << 10, 1 <<  9, 1 <<  8, 1 <<  7,
                1 <<  6, 1 <<  5, 1 <<  4, 1 <<  3,
                1 <<  1, 1 <<  2, 1 <<  0));
  auto src_layout = make_layout(shape,
    make_stride(1 << 15, 1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21, 1 << 22,
                1 << 14, 1 << 13, 1 << 12, 1 << 11, 1 << 10, 1 <<  9, 1 <<  8, 1 <<  7,
                1 <<  6, 1 <<  5, 1 <<  4, 1 <<  3,
                1 <<  2, 1 <<  0, 1 <<  1));
  // clang-format on
  bench_copy(state, state.get_string("Impl"), 0, src_layout, 0, dst_layout);
}
NVBENCH_BENCH(flatten_common).set_name("flatten_common").add_string_axis("Impl", {"naive", "registers"});

// flatten_one: 20-dim, dst C-order, src F-order with slice [::5]
void flatten_one(nvbench::state& state)
{
  using namespace cute;
  // clang-format off
  auto dst_layout = make_layout(
    make_shape(4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2),
    make_stride(1 << 19, 1 << 18, 1 << 17, 1 << 16, 1 << 15, 1 << 14, 1 << 13, 1 << 12,
                1 << 11, 1 << 10, 1 <<  9, 1 <<  8, 1 <<  7, 1 <<  6, 1 <<  5, 1 <<  4,
                1 <<  3, 1 <<  2, 1 <<  1, 1 <<  0));
  auto src_layout = make_layout(
    make_shape(4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2),
    make_stride(5, 1 << 4, 1 << 5, 1 << 6, 1 << 7, 1 << 8, 1 << 9, 1 << 10,
                1 << 11, 1 << 12, 1 << 13, 1 << 14, 1 << 15, 1 << 16, 1 << 17, 1 << 18,
                1 << 19, 1 << 20, 1 << 21, 1 << 22));
  // clang-format on
  bench_copy(state, state.get_string("Impl"), 0, src_layout, 0, dst_layout);
}
NVBENCH_BENCH(flatten_one).set_name("flatten_one").add_string_axis("Impl", {"naive", "registers"});

/***********************************************************************************************************************
 * Vectorize benchmarks
 **********************************************************************************************************************/

// sliced_vec: src sliced along dim 1, vectorizable
void sliced_vec(nvbench::state& state)
{
  using namespace cute;
  constexpr int src_offset = 240;
  auto dst_layout          = make_layout(make_shape(35, 255, 10, 24), make_stride(61200, 240, 24, 1));
  auto src_layout          = make_layout(make_shape(35, 255, 10, 24), make_stride(61440, 240, 24, 1));
  bench_copy(state, state.get_string("Impl"), src_offset, src_layout, 0, dst_layout);
}
NVBENCH_BENCH(sliced_vec).set_name("sliced_vec").add_string_axis("Impl", {"naive", "registers"});

// sliced_vec_2: odd least-strided extent, flattened inner dims even -> vectorizable
void sliced_vec_2(nvbench::state& state)
{
  using namespace cute;
  constexpr int src_offset = 12;
  auto dst_layout          = make_layout(make_shape(355, 255, 4, 3), make_stride(3060, 12, 3, 1));
  auto src_layout          = make_layout(make_shape(355, 255, 4, 3), make_stride(3072, 12, 3, 1));
  bench_copy(state, state.get_string("Impl"), src_offset, src_layout, 0, dst_layout);
}
NVBENCH_BENCH(sliced_vec_2).set_name("sliced_vec_2").add_string_axis("Impl", {"naive", "registers"});

// sliced_unaligned_ptr: misaligned pointer due to slicing (offset=205)
void sliced_unaligned_ptr(nvbench::state& state)
{
  using namespace cute;
  constexpr int src_offset = 10 * 20 + 5;
  auto dst_layout          = make_layout(make_shape(35, 255, 5, 10), make_stride(12750, 50, 10, 1));
  auto src_layout          = make_layout(make_shape(35, 255, 5, 10), make_stride(153000, 600, 20, 1));
  bench_copy(state, state.get_string("Impl"), src_offset, src_layout, 0, dst_layout);
}
NVBENCH_BENCH(sliced_unaligned_ptr).set_name("sliced_unaligned_ptr").add_string_axis("Impl", {"naive", "registers"});
