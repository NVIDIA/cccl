// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>

#include <nvbench_helper.cuh>

template <class T, typename OffsetT>
void seg_radix_sort(nvbench::state& state,
                    nvbench::type_list<T, OffsetT> ts,
                    const thrust::device_vector<OffsetT>& offsets,
                    bit_entropy entropy)
{
  constexpr bool is_overwrite_ok = false;

  using offset_t          = OffsetT;
  using begin_offset_it_t = const offset_t*;
  using end_offset_it_t   = const offset_t*;
  using key_t             = T;
  using value_t           = cub::NullType;
  using dispatch_t        = cub::
    DispatchSegmentedRadixSort<cub::SortOrder::Ascending, key_t, value_t, begin_offset_it_t, end_offset_it_t, offset_t>;

  constexpr int begin_bit = 0;
  constexpr int end_bit   = sizeof(key_t) * 8;

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
  dispatch_t::Dispatch(
    d_temp_storage,
    temp_storage_bytes,
    d_keys,
    d_values,
    elements,
    segments,
    d_begin_offsets,
    d_end_offsets,
    begin_bit,
    end_bit,
    is_overwrite_ok,
    0);

  thrust::device_vector<nvbench::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               cub::DoubleBuffer<key_t> keys     = d_keys;
               cub::DoubleBuffer<value_t> values = d_values;

               dispatch_t::Dispatch(
                 d_temp_storage,
                 temp_storage_bytes,
                 keys,
                 values,
                 elements,
                 segments,
                 d_begin_offsets,
                 d_end_offsets,
                 begin_bit,
                 end_bit,
                 is_overwrite_ok,
                 launch.get_stream());
             });
}

#ifdef TUNE_OffsetT
using some_offset_types = nvbench::type_list<TUNE_OffsetT>;
#else
using some_offset_types = nvbench::type_list<int32_t, int64_t>;
#endif

template <class T, typename OffsetT>
void power_law(nvbench::state& state, nvbench::type_list<T, OffsetT> ts)
{
  const auto elements                    = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const auto segments                    = static_cast<std::size_t>(state.get_int64("Segments{io}"));
  const bit_entropy entropy              = str_to_entropy(state.get_string("Entropy"));
  thrust::device_vector<OffsetT> offsets = generate.power_law.segment_offsets(elements, segments);

  seg_radix_sort(state, ts, offsets, entropy);
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

  seg_radix_sort(state, ts, offsets, bit_entropy::_1_000);
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
