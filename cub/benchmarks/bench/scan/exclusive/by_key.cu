// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/detail/choose_offset.cuh>
#include <cub/device/device_scan.cuh>

#include <look_back_helper.cuh>
#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS ipt 7:24:1
// %RANGE% TUNE_THREADS tpb 128:1024:32
// %RANGE% TUNE_MAGIC_NS ns 0:2048:4
// %RANGE% TUNE_DELAY_CONSTRUCTOR_ID dcid 0:7:1
// %RANGE% TUNE_L2_WRITE_LATENCY_NS l2w 0:1200:5
// %RANGE% TUNE_TRANSPOSE trp 0:1:1
// %RANGE% TUNE_LOAD ld 0:1:1

#if !TUNE_BASE
struct bench_scan_by_key_policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(cuda::compute_capability) const
    -> cub::detail::scan_by_key::scan_by_key_policy
  {
    return {TUNE_THREADS,
            TUNE_ITEMS,
            TUNE_TRANSPOSE == 0 ? cub::BLOCK_LOAD_DIRECT : cub::BLOCK_LOAD_WARP_TRANSPOSE,
            TUNE_LOAD == 0 ? cub::LOAD_DEFAULT : cub::LOAD_CA,
            TUNE_TRANSPOSE == 0 ? cub::BLOCK_STORE_DIRECT : cub::BLOCK_STORE_WARP_TRANSPOSE,
            cub::BLOCK_SCAN_WARP_SCANS,
            delay_constructor_policy};
  }
};
#endif // !TUNE_BASE

template <typename KeyT, typename ValueT, typename OffsetT>
static void scan(nvbench::state& state, nvbench::type_list<KeyT, ValueT, OffsetT>)
{
  using init_value_t    = ValueT;
  using op_t            = ::cuda::std::plus<>;
  using accum_t         = ::cuda::std::__accumulator_t<op_t, ValueT, init_value_t>;
  using key_input_it_t  = const KeyT*;
  using val_input_it_t  = const ValueT*;
  using val_output_it_t = ValueT*;
  using equality_op_t   = ::cuda::std::equal_to<>;
  using offset_t        = cub::detail::choose_offset_t<OffsetT>;

  const auto elements = static_cast<std::size_t>(state.get_int64("Elements{io}"));

  thrust::device_vector<ValueT> in_vals(elements);
  thrust::device_vector<ValueT> out_vals(elements);
  thrust::device_vector<KeyT> keys = generate.uniform.key_segments(elements, 0, 5200);

  const KeyT* d_keys       = thrust::raw_pointer_cast(keys.data());
  const ValueT* d_in_vals  = thrust::raw_pointer_cast(in_vals.data());
  ValueT* d_out_vals       = thrust::raw_pointer_cast(out_vals.data());
  const offset_t num_items = static_cast<offset_t>(elements);

  state.add_element_count(elements);
  state.add_global_memory_reads<KeyT>(elements);
  state.add_global_memory_reads<ValueT>(elements);
  state.add_global_memory_writes<ValueT>(elements);

  size_t tmp_size{};
  nvbench::uint8_t* d_tmp = nullptr;
  auto dispatch_on_stream = [&](cudaStream_t stream) {
    return cub::detail::scan_by_key::
      dispatch<key_input_it_t, val_input_it_t, val_output_it_t, equality_op_t, op_t, init_value_t, offset_t, accum_t>(
        d_tmp,
        tmp_size,
        d_keys,
        d_in_vals,
        d_out_vals,
        equality_op_t{},
        op_t{},
        init_value_t{},
        num_items,
        stream
#if !TUNE_BASE
        ,
        bench_scan_by_key_policy_selector{}
#endif
      );
  };

  dispatch_on_stream(nullptr /* stream */);

  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size, thrust::no_init);
  d_tmp = thrust::raw_pointer_cast(tmp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_on_stream(launch.get_stream());
  });
}

using some_offset_types = nvbench::type_list<nvbench::int32_t>;

#ifdef TUNE_KeyT
using key_types = nvbench::type_list<TUNE_KeyT>;
#else // !defined(TUNE_KeyT)
using key_types = all_types;
#endif // TUNE_KeyT

#ifdef TUNE_ValueT
using value_types = nvbench::type_list<TUNE_ValueT>;
#else // !defined(TUNE_ValueT)
using value_types =
  nvbench::type_list<int8_t,
                     int16_t,
                     int32_t,
                     int64_t
#  if _CCCL_HAS_INT128()
                     ,
                     int128_t
#  endif
                     >;
#endif // TUNE_ValueT

NVBENCH_BENCH_TYPES(scan, NVBENCH_TYPE_AXES(key_types, value_types, some_offset_types))
  .set_name("base")
  .set_type_axes_names({"KeyT{ct}", "ValueT{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
