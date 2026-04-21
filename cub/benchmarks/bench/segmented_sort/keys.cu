// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/device/dispatch/dispatch_segmented_sort.cuh>

#include <nvbench_helper.cuh>

// %RANGE% TUNE_L_ITEMS ipt 7:24:1
// %RANGE% TUNE_M_ITEMS ipmw 1:17:1
// %RANGE% TUNE_S_ITEMS ipsw 1:17:1
// %RANGE% TUNE_THREADS tpb 128:1024:32
// %RANGE% TUNE_SW_THREADS_POW2 tpsw 1:4:1
// %RANGE% TUNE_MW_THREADS_POW2 tpmw 1:5:1
// %RANGE% TUNE_RADIX_BITS bits 4:8:1
// %RANGE% TUNE_PARTITIONING_THRESHOLD pt 100:800:50
// %RANGE% TUNE_RANK_ALGORITHM ra 0:4:1
// %RANGE% TUNE_LOAD ld 0:2:1
// %RANGE% TUNE_TRANSPOSE trp 0:1:1
// %RANGE% TUNE_S_LOAD sld 0:2:1
// %RANGE% TUNE_S_TRANSPOSE strp 0:1:1
// %RANGE% TUNE_M_LOAD mld 0:2:1
// %RANGE% TUNE_M_TRANSPOSE mtrp 0:1:1

#if !TUNE_BASE
template <class KeyT>
struct device_seg_sort_policy_selector
{
  _CCCL_API constexpr auto operator()(::cuda::arch_id /*arch*/) const
  {
    constexpr int tune_sw_threads     = 1 << TUNE_SW_THREADS_POW2;
    constexpr int tune_mw_threads     = 1 << TUNE_MW_THREADS_POW2;
    constexpr int small_segment_size  = TUNE_S_ITEMS * tune_sw_threads;
    constexpr int medium_segment_size = TUNE_M_ITEMS * tune_mw_threads;
    constexpr int large_segment_size  = TUNE_L_ITEMS * TUNE_THREADS;

    static_assert((large_segment_size > small_segment_size) && (large_segment_size > medium_segment_size),
                  "Large segment size must be larger than small and medium segment sizes");
    static_assert(medium_segment_size > small_segment_size, "Medium segment size must be larger than small one");

    using namespace cub::detail::segmented_sort;

    return segmented_sort_policy{
      segmented_radix_sort_policy{
        TUNE_THREADS,
        TUNE_L_ITEMS,
        (TUNE_TRANSPOSE == 0) ? cub::BLOCK_LOAD_DIRECT : cub::BLOCK_LOAD_WARP_TRANSPOSE,
        (TUNE_LOAD == 0)   ? cub::LOAD_DEFAULT
        : (TUNE_LOAD == 1) ? cub::LOAD_LDG
                           : cub::LOAD_CA,
        static_cast<cub::RadixRankAlgorithm>(TUNE_RANK_ALGORITHM),
        cub::BLOCK_SCAN_WARP_SCANS,
        TUNE_RADIX_BITS,
      },
      sub_warp_merge_sort_policy{
        TUNE_THREADS,
        tune_sw_threads,
        TUNE_S_ITEMS,
        (TUNE_S_TRANSPOSE == 0) ? cub::WarpLoadAlgorithm::WARP_LOAD_DIRECT : cub::WarpLoadAlgorithm::WARP_LOAD_TRANSPOSE,
        (TUNE_S_LOAD == 0)   ? cub::LOAD_DEFAULT
        : (TUNE_S_LOAD == 1) ? cub::LOAD_LDG
                             : cub::LOAD_CA,
        cub::WARP_STORE_DIRECT,
      },
      sub_warp_merge_sort_policy{
        TUNE_THREADS,
        tune_mw_threads,
        TUNE_M_ITEMS,
        (TUNE_M_TRANSPOSE == 0) ? cub::WarpLoadAlgorithm::WARP_LOAD_DIRECT : cub::WarpLoadAlgorithm::WARP_LOAD_TRANSPOSE,
        (TUNE_M_LOAD == 0)   ? cub::LOAD_DEFAULT
        : (TUNE_M_LOAD == 1) ? cub::LOAD_LDG
                             : cub::LOAD_CA,
        cub::WARP_STORE_DIRECT,
      },
      TUNE_PARTITIONING_THRESHOLD,
    };
  }
};
#endif // !TUNE_BASE

template <class T, typename OffsetT>
void seg_sort(nvbench::state& state,
              nvbench::type_list<T, OffsetT> ts,
              const thrust::device_vector<OffsetT>& offsets,
              bit_entropy entropy)
{
  constexpr cub::SortOrder sort_order = cub::SortOrder::Ascending;
  constexpr bool is_overwrite_ok      = false;

  using offset_t          = OffsetT;
  using begin_offset_it_t = const offset_t*;
  using end_offset_it_t   = const offset_t*;
  using key_t             = T;
  using value_t           = cub::NullType;

  const auto elements = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const auto segments = offsets.size() - 1;

  thrust::device_vector<key_t> buffer_1 = generate(elements, entropy);
  thrust::device_vector<key_t> buffer_2(elements);

  key_t* d_buffer_1 = thrust::raw_pointer_cast(buffer_1.data());
  key_t* d_buffer_2 = thrust::raw_pointer_cast(buffer_2.data());

  cub::DoubleBuffer<key_t> d_keys(d_buffer_1, d_buffer_2);
  cub::DoubleBuffer<value_t> d_values;

  begin_offset_it_t d_begin_offsets = thrust::raw_pointer_cast(offsets.data());
  end_offset_it_t d_end_offsets     = d_begin_offsets + 1;

  state.add_element_count(elements);
  state.add_global_memory_reads<key_t>(elements);
  state.add_global_memory_writes<key_t>(elements);
  state.add_global_memory_reads<offset_t>(segments + 1);

  std::size_t temp_storage_bytes{};
  std::uint8_t* d_temp_storage{};
  cub::detail::segmented_sort::dispatch<sort_order, offset_t>(
    d_temp_storage,
    temp_storage_bytes,
    d_keys,
    d_values,
    elements,
    segments,
    d_begin_offsets,
    d_end_offsets,
    is_overwrite_ok,
    nullptr
#if !TUNE_BASE
    ,
    device_seg_sort_policy_selector<key_t>{}
#endif // !TUNE_BASE
  );

  thrust::device_vector<nvbench::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               cub::DoubleBuffer<key_t> keys     = d_keys;
               cub::DoubleBuffer<value_t> values = d_values;

               cub::detail::segmented_sort::dispatch<sort_order, offset_t>(
                 d_temp_storage,
                 temp_storage_bytes,
                 keys,
                 values,
                 elements,
                 segments,
                 d_begin_offsets,
                 d_end_offsets,
                 is_overwrite_ok,
                 launch.get_stream()
#if !TUNE_BASE
                   ,
                 device_seg_sort_policy_selector<key_t>{}
#endif // !TUNE_BASE
               );
             });
}

using some_offset_types = nvbench::type_list<int32_t>;

template <class T, typename OffsetT>
void power_law(nvbench::state& state, nvbench::type_list<T, OffsetT> ts)
{
  const auto elements                    = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const auto segments                    = static_cast<std::size_t>(state.get_int64("Segments{io}"));
  const bit_entropy entropy              = str_to_entropy(state.get_string("Entropy"));
  thrust::device_vector<OffsetT> offsets = generate.power_law.segment_offsets(elements, segments);

  seg_sort(state, ts, offsets, entropy);
}

NVBENCH_BENCH_TYPES(power_law, NVBENCH_TYPE_AXES(fundamental_types, some_offset_types))
  .set_name("power")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(22, 30, 4))
  .add_int64_power_of_two_axis("Segments{io}", nvbench::range(12, 20, 4))
  .add_string_axis("Entropy", {"1.000", "0.201"});

template <class T, typename OffsetT>
void uniform(nvbench::state& state, nvbench::type_list<T, OffsetT> ts)
{
  const auto elements         = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const auto max_segment_size = static_cast<std::size_t>(state.get_int64("MaxSegmentSize"));

  const auto max_segment_size_log = static_cast<OffsetT>(std::log2(max_segment_size));
  const auto min_segment_size     = 1 << (max_segment_size_log - 1);

  thrust::device_vector<OffsetT> offsets =
    generate.uniform.segment_offsets(elements, min_segment_size, max_segment_size);

  seg_sort(state, ts, offsets, bit_entropy::_1_000);
}

NVBENCH_BENCH_TYPES(uniform, NVBENCH_TYPE_AXES(fundamental_types, some_offset_types))
  .set_name("small")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(22, 30, 4))
  .add_int64_power_of_two_axis("MaxSegmentSize", nvbench::range(1, 8, 1));

NVBENCH_BENCH_TYPES(uniform, NVBENCH_TYPE_AXES(fundamental_types, some_offset_types))
  .set_name("large")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(22, 30, 4))
  .add_int64_power_of_two_axis("MaxSegmentSize", nvbench::range(10, 18, 2));
