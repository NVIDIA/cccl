// SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_scan.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/detail/delay_constructor.cuh>
#include <cub/detail/warpspeed/allocators/smem_allocator.cuh>
#include <cub/detail/warpspeed/resource/smem_resource_raw.cuh>
#include <cub/detail/warpspeed/squad/squad_desc.cuh>
#include <cub/detail/warpspeed/sync_handler.cuh>
#include <cub/device/dispatch/tuning/common.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

#include <thrust/type_traits/is_contiguous_iterator.h>

#include <cuda/__device/compute_capability.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__host_stdlib/ostream>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_trivially_copy_constructible.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/optional>

CUB_NAMESPACE_BEGIN

namespace detail::scan
{
// TODO(bgruber): remove this in CCCL 4.0 when we remove the public scan dispatcher
enum class keep_rejects
{
  no,
  yes
};
// TODO(bgruber): remove this in CCCL 4.0 when we remove the public scan dispatcher
enum class primitive_accum
{
  no,
  yes
};
// TODO(bgruber): remove this in CCCL 4.0 when we remove the public scan dispatcher
enum class primitive_op
{
  no,
  yes
};
// TODO(bgruber): remove this in CCCL 4.0 when we remove the public scan dispatcher
enum class offset_size
{
  _4,
  _8,
  unknown
};
// TODO(bgruber): remove this in CCCL 4.0 when we remove the public scan dispatcher
enum class value_size
{
  _1,
  _2,
  _4,
  _8,
  _16,
  unknown
};
// TODO(bgruber): remove this in CCCL 4.0 when we remove the public scan dispatcher
enum class accum_size
{
  _1,
  _2,
  _4,
  _8,
  _16,
  unknown
};

// TODO(bgruber): remove this in CCCL 4.0 when we remove the public scan dispatcher
template <class AccumT>
constexpr _CCCL_HOST_DEVICE primitive_accum is_primitive_accum()
{
  return is_primitive<AccumT>::value ? primitive_accum::yes : primitive_accum::no;
}

// TODO(bgruber): remove this in CCCL 4.0 when we remove the public scan dispatcher
template <class ScanOpT>
constexpr _CCCL_HOST_DEVICE primitive_op is_primitive_op()
{
  return basic_binary_op_t<ScanOpT>::value ? primitive_op::yes : primitive_op::no;
}

// TODO(bgruber): remove this in CCCL 4.0 when we remove the public scan dispatcher
template <class ValueT>
constexpr _CCCL_HOST_DEVICE value_size classify_value_size()
{
  return sizeof(ValueT) == 1 ? value_size::_1
       : sizeof(ValueT) == 2 ? value_size::_2
       : sizeof(ValueT) == 4 ? value_size::_4
       : sizeof(ValueT) == 8 ? value_size::_8
       : sizeof(ValueT) == 16
         ? value_size::_16
         : value_size::unknown;
}

// TODO(bgruber): remove this in CCCL 4.0 when we remove the public scan dispatcher
template <class AccumT>
constexpr _CCCL_HOST_DEVICE accum_size classify_accum_size()
{
  return sizeof(AccumT) == 1 ? accum_size::_1
       : sizeof(AccumT) == 2 ? accum_size::_2
       : sizeof(AccumT) == 4 ? accum_size::_4
       : sizeof(AccumT) == 8 ? accum_size::_8
       : sizeof(AccumT) == 16
         ? accum_size::_16
         : accum_size::unknown;
}

// TODO(bgruber): remove this in CCCL 4.0 when we remove the public scan dispatcher
template <class OffsetT>
constexpr _CCCL_HOST_DEVICE offset_size classify_offset_size()
{
  return sizeof(OffsetT) == 4 ? offset_size::_4 : sizeof(OffsetT) == 8 ? offset_size::_8 : offset_size::unknown;
}

// TODO(bgruber): remove this in CCCL 4.0 when we remove the public scan dispatcher
template <class ValueT,
          class AccumT,
          class OffsetT,
          op_kind_t OpTypeT,
          primitive_accum PrimitiveAccumulator = is_primitive_accum<AccumT>(),
          offset_size OffsetSize               = classify_offset_size<OffsetT>(),
          value_size ValueSize                 = classify_value_size<ValueT>()>
struct sm75_tuning;

template <class ValueT, class AccumT, class OffsetT>
struct sm75_tuning<ValueT, AccumT, OffsetT, op_kind_t::plus, primitive_accum::yes, offset_size::_8, value_size::_4>
{
  // ipt_7.tpb_128.ns_628.dcid_1.l2w_520.trp_1.ld_0
  static constexpr int threads                         = 128;
  static constexpr int items                           = 7;
  using delay_constructor                              = fixed_delay_constructor_t<628, 520>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier     = LOAD_DEFAULT;
};

// Add sm89 tuning and verify it

// TODO(bgruber): remove this in CCCL 4.0 when we remove the public scan dispatcher
template <type_t AccumT, primitive_op PrimitiveOp, primitive_accum PrimitiveAccumulator, accum_size AccumSize>
struct sm80_tuning;

template <type_t T>
struct sm80_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_1>
{
  static constexpr int threads                         = 320;
  static constexpr int items                           = 14;
  using delay_constructor                              = fixed_delay_constructor_t<368, 725>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
};

template <type_t T>
struct sm80_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_2>
{
  static constexpr int threads                         = 352;
  static constexpr int items                           = 16;
  using delay_constructor                              = fixed_delay_constructor_t<488, 1040>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
};

template <type_t T>
struct sm80_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_4>
{
  static constexpr int threads                         = 320;
  static constexpr int items                           = 12;
  using delay_constructor                              = fixed_delay_constructor_t<268, 1180>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
};

template <type_t T>
struct sm80_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_8>
{
  static constexpr int threads                         = 288;
  static constexpr int items                           = 22;
  using delay_constructor                              = fixed_delay_constructor_t<716, 785>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
};

template <>
struct sm80_tuning<type_t::float32, primitive_op::yes, primitive_accum::yes, accum_size::_4>
{
  static constexpr int threads                         = 288;
  static constexpr int items                           = 8;
  using delay_constructor                              = fixed_delay_constructor_t<724, 1050>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
};

template <>
struct sm80_tuning<type_t::float64, primitive_op::yes, primitive_accum::yes, accum_size::_8>
{
  static constexpr int threads                         = 384;
  static constexpr int items                           = 12;
  using delay_constructor                              = fixed_delay_constructor_t<388, 1100>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
};

#if _CCCL_HAS_INT128()
template <>
struct sm80_tuning<type_t::int128, primitive_op::yes, primitive_accum::no, accum_size::_16>
{
  static constexpr int threads                         = 640;
  static constexpr int items                           = 24;
  using delay_constructor                              = no_delay_constructor_t<1200>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_DIRECT;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_DIRECT;
};

template <>
struct sm80_tuning<type_t::uint128, primitive_op::yes, primitive_accum::no, accum_size::_16>
    : sm80_tuning<type_t::int128, primitive_op::yes, primitive_accum::no, accum_size::_16>
{};
#endif

// TODO(griwes): remove for CCCL 4.0 when we drop the public scan dispatcher
template <class AccumT,
          primitive_op PrimitiveOp,
          primitive_accum PrimitiveAccumulator = is_primitive_accum<AccumT>(),
          accum_size AccumSize                 = classify_accum_size<AccumT>()>
struct sm90_tuning;

// TODO(bgruber): remove this in CCCL 4.0 when we remove the public scan dispatcher
template <class AccumT, int Threads, int Items, int L2B, int L2W>
struct sm90_tuning_vals
{
  static constexpr int threads = Threads;
  static constexpr int items   = Items;
  using delay_constructor      = fixed_delay_constructor_t<L2B, L2W>;
  // same logic as default policy:
  static constexpr bool large_values = sizeof(AccumT) > 128;
  static constexpr BlockLoadAlgorithm load_algorithm =
    large_values ? BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED : BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm =
    large_values ? BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED : BLOCK_STORE_WARP_TRANSPOSE;
};

// clang-format off
template <class T> struct sm90_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_1> : sm90_tuning_vals<T, 192, 22, 168, 1140> {};
template <class T> struct sm90_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_2> : sm90_tuning_vals<T, 512, 12, 376, 1125> {};
template <class T> struct sm90_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_4> : sm90_tuning_vals<T, 128, 24, 648, 1245> {};
template <class T> struct sm90_tuning<T, primitive_op::yes, primitive_accum::yes, accum_size::_8> : sm90_tuning_vals<T, 224, 24, 632, 1290> {};

template <> struct sm90_tuning<float,  primitive_op::yes, primitive_accum::yes, accum_size::_4> : sm90_tuning_vals<float,  128, 24, 688, 1140> {};
template <> struct sm90_tuning<double, primitive_op::yes, primitive_accum::yes, accum_size::_8> : sm90_tuning_vals<double, 224, 24, 576, 1215> {};

#if _CCCL_HAS_INT128()
template <> struct sm90_tuning<__int128_t, primitive_op::yes, primitive_accum::no, accum_size::_16> : sm90_tuning_vals<__int128_t, 576, 21, 860, 630> {};
template <>
struct sm90_tuning<__uint128_t, primitive_op::yes, primitive_accum::no, accum_size::_16>
    : sm90_tuning<__int128_t, primitive_op::yes, primitive_accum::no, accum_size::_16>
{};
#endif
// clang-format on

// TODO(griwes): remove for CCCL 4.0 when we drop the public scan dispatcher
template <class ValueT,
          class AccumT,
          class OffsetT,
          op_kind_t OpTypeT,
          primitive_accum PrimitiveAccumulator = is_primitive_accum<AccumT>(),
          offset_size OffsetSize               = classify_offset_size<OffsetT>(),
          value_size ValueSize                 = classify_value_size<ValueT>()>
struct sm100_tuning;

// sum
template <class ValueT, class AccumT, class OffsetT>
struct sm100_tuning<ValueT, AccumT, OffsetT, op_kind_t::plus, primitive_accum::yes, offset_size::_4, value_size::_1>
{
  // ipt_18.tpb_512.ns_768.dcid_7.l2w_820.trp_1.ld_0 1.188818  1.005682  1.173041  1.305288
  static constexpr int items                           = 18;
  static constexpr int threads                         = 512;
  using delay_constructor                              = exponential_backon_constructor_t<768, 820>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier     = LOAD_DEFAULT;
};

template <class ValueT, class AccumT, class OffsetT>
struct sm100_tuning<ValueT, AccumT, OffsetT, op_kind_t::plus, primitive_accum::yes, offset_size::_8, value_size::_1>
{
  // ipt_14.tpb_384.ns_228.dcid_7.l2w_775.trp_1.ld_1 1.107210  1.000000  1.100637  1.307692
  static constexpr int items                           = 14;
  static constexpr int threads                         = 384;
  using delay_constructor                              = exponential_backon_constructor_t<228, 775>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier     = LOAD_CA;
};

template <class ValueT, class AccumT, class OffsetT>
struct sm100_tuning<ValueT, AccumT, OffsetT, op_kind_t::plus, primitive_accum::yes, offset_size::_4, value_size::_2>
{
  // ipt_13.tpb_512.ns_1384.dcid_7.l2w_720.trp_1.ld_0 1.128443  1.002841  1.119688  1.307692
  static constexpr int items                           = 13;
  static constexpr int threads                         = 512;
  using delay_constructor                              = exponential_backon_constructor_t<1384, 720>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier     = LOAD_DEFAULT;
};

// todo(gonidelis): Regresses for large inputs. Find better tuning.
// template <class ValueT, class AccumT, class OffsetT>
// struct sm100_tuning<ValueT,
//                     AccumT,
//                     OffsetT,
//                     op_kind_t::plus,
//                     primitive_value::yes,
//                     primitive_accum::yes,
//                     offset_size::_8,
//                     value_size::_2>
// {
//   // ipt_13.tpb_288.ns_1520.dcid_5.l2w_895.trp_1.ld_1 1.080934  0.983509  1.077724  1.305288
//   static constexpr int items                           = 13;
//   static constexpr int threads                         = 288;
//   using delay_constructor                              = exponential_backon_jitter_window_constructor_t<1520, 895>;
//   static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
//   static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
//   static constexpr CacheLoadModifier load_modifier     = LOAD_CA;
// };

template <class ValueT, class AccumT, class OffsetT>
struct sm100_tuning<ValueT, AccumT, OffsetT, op_kind_t::plus, primitive_accum::yes, offset_size::_4, value_size::_4>
{
  // ipt_22.tpb_384.ns_1904.dcid_6.l2w_830.trp_1.ld_0 1.148442  0.997167  1.139902  1.462651
  static constexpr int items                           = 22;
  static constexpr int threads                         = 384;
  using delay_constructor                              = exponential_backon_jitter_constructor_t<1904, 830>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier     = LOAD_DEFAULT;
};

template <class ValueT, class AccumT, class OffsetT>
struct sm100_tuning<ValueT, AccumT, OffsetT, op_kind_t::plus, primitive_accum::yes, offset_size::_8, value_size::_4>
{
  // ipt_19.tpb_416.ns_956.dcid_7.l2w_550.trp_1.ld_1 1.146142  0.994350  1.137459  1.455636
  static constexpr int items                           = 19;
  static constexpr int threads                         = 416;
  using delay_constructor                              = exponential_backon_constructor_t<956, 550>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier     = LOAD_CA;
};

template <class ValueT, class AccumT, class OffsetT>
struct sm100_tuning<ValueT, AccumT, OffsetT, op_kind_t::plus, primitive_accum::yes, offset_size::_4, value_size::_8>
{
  // ipt_23.tpb_416.ns_772.dcid_5.l2w_710.trp_1.ld_0 1.089468  1.015581  1.085630  1.264583
  static constexpr int items                           = 23;
  static constexpr int threads                         = 416;
  using delay_constructor                              = exponential_backon_jitter_window_constructor_t<772, 710>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier     = LOAD_DEFAULT;
};

template <class ValueT, class AccumT, class OffsetT>
struct sm100_tuning<ValueT, AccumT, OffsetT, op_kind_t::plus, primitive_accum::yes, offset_size::_8, value_size::_8>
{
  // ipt_22.tpb_320.ns_328.dcid_2.l2w_965.trp_1.ld_0 1.080133  1.000000  1.075577  1.248963
  static constexpr int items                           = 22;
  static constexpr int threads                         = 320;
  using delay_constructor                              = exponential_backoff_constructor_t<328, 965>;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier     = LOAD_DEFAULT;
};

// todo(gonidelis): Add tunings for i128, float and double.
// template <class OffsetT> struct sm100_tuning<float, OffsetT, op_kind_t::plus, primitive_accum::yes, offset_size::_8,
// accum_size::_4>;
// Default explicitly so it doesn't pick up the sm100<I64, I64> tuning.
template <class AccumT, class OffsetT>
struct sm100_tuning<double, AccumT, OffsetT, op_kind_t::plus, primitive_accum::yes, offset_size::_8, value_size::_8>
    : sm90_tuning<double, primitive_op::yes, primitive_accum::yes, accum_size::_8>
{};

#if _CCCL_HAS_INT128()
// template <class OffsetT> struct sm100_tuning<__int128_t, OffsetT, op_kind_t::plus, primitive_accum::no,
// offset_size::_8, accum_size::_16> : tuning<576, 21, 860, 630> {}; template <class OffsetT> struct
// sm100_tuning<__uint128_t, OffsetT, op_kind_t::plus, primitive_accum::no, offset_size::_8, accum_size::_16>
//     : sm100_tuning<__int128_t, OffsetT, op_kind_t::plus, primitive_accum::no, offset_size::_8, accum_size::_16>
// {};
#endif

// TODO(griwes): remove this in CCCL 4.0 when we remove the public scan dispatcher
template <typename InputValueT, typename OutputValueT, typename AccumT, typename OffsetT, typename ScanOpT>
struct policy_hub
{
  // For large values, use timesliced loads/stores to fit shared memory.
  static constexpr bool large_values = sizeof(AccumT) > 128;
  static constexpr BlockLoadAlgorithm scan_transposed_load =
    large_values ? BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED : BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm scan_transposed_store =
    large_values ? BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED : BLOCK_STORE_WARP_TRANSPOSE;

  struct Policy500 : ChainedPolicy<500, Policy500, Policy500>
  {
    // GTX Titan: 29.5B items/s (232.4 GB/s) @ 48M 32-bit T
    using ScanPolicyT =
      AgentScanPolicy<128, 12, AccumT, BLOCK_LOAD_DIRECT, LOAD_CA, BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED, BLOCK_SCAN_RAKING>;
  };
  struct Policy520 : ChainedPolicy<520, Policy520, Policy500>
  {
    // Titan X: 32.47B items/s @ 48M 32-bit T
    using ScanPolicyT =
      AgentScanPolicy<128, 12, AccumT, BLOCK_LOAD_DIRECT, LOAD_CA, scan_transposed_store, BLOCK_SCAN_WARP_SCANS>;
  };

  struct DefaultPolicy
  {
    using ScanPolicyT =
      AgentScanPolicy<128, 15, AccumT, scan_transposed_load, LOAD_DEFAULT, scan_transposed_store, BLOCK_SCAN_WARP_SCANS>;
  };

  struct Policy600
      : DefaultPolicy
      , ChainedPolicy<600, Policy600, Policy520>
  {};

  // Use values from tuning if a specialization exists, otherwise pick DefaultPolicy
  template <typename Tuning>
  _CCCL_HOST_DEVICE static auto select_agent_policy(int)
    -> AgentScanPolicy<Tuning::threads,
                       Tuning::items,
                       AccumT,
                       Tuning::load_algorithm,
                       LOAD_DEFAULT,
                       Tuning::store_algorithm,
                       BLOCK_SCAN_WARP_SCANS,
                       cub::detail::MemBoundScaling<Tuning::threads, Tuning::items, AccumT>,
                       typename Tuning::delay_constructor>;
  template <typename Tuning>
  _CCCL_HOST_DEVICE static auto select_agent_policy(long) -> typename DefaultPolicy::ScanPolicyT;

  struct Policy750 : ChainedPolicy<750, Policy750, Policy600>
  {
    // Use values from tuning if a specialization exists that matches a benchmark, otherwise pick Policy600
    template <typename Tuning,
              typename IVT,
              // In the tuning benchmarks the Initial-, Input- and OutputType are the same. Let's check that the
              // accumulator type's size matches what we used during the benchmark since that has an impact (The
              // tunings also check later that it's a primitive type, so arithmetic impact is also comparable to the
              // benchmark). Input- and OutputType only impact loading and storing data (all arithmetic is done in the
              // accumulator type), so let's check that they are the same size and dispatch the size in the tunings.
              ::cuda::std::enable_if_t<sizeof(AccumT) == sizeof(::cuda::std::__accumulator_t<ScanOpT, IVT, IVT>)
                                         && sizeof(IVT) == sizeof(OutputValueT),
                                       int> = 0>
    _CCCL_HOST_DEVICE static auto select_agent_policy750(int)
      -> AgentScanPolicy<Tuning::threads,
                         Tuning::items,
                         AccumT,
                         Tuning::load_algorithm,
                         Tuning::load_modifier,
                         Tuning::store_algorithm,
                         BLOCK_SCAN_WARP_SCANS,
                         MemBoundScaling<Tuning::threads, Tuning::items, AccumT>,
                         typename Tuning::delay_constructor>;
    template <typename Tuning, typename IVT>
    _CCCL_HOST_DEVICE static auto select_agent_policy750(long) -> typename Policy600::ScanPolicyT;

    using ScanPolicyT =
      decltype(select_agent_policy750<sm75_tuning<InputValueT, AccumT, OffsetT, classify_op<ScanOpT>>, InputValueT>(0));
  };

  struct Policy800 : ChainedPolicy<800, Policy800, Policy750>
  {
    using ScanPolicyT =
      decltype(select_agent_policy<sm80_tuning<classify_type<AccumT>,
                                               is_primitive_op<ScanOpT>(),
                                               is_primitive_accum<AccumT>(),
                                               classify_accum_size<AccumT>()>>(0));
  };

  struct Policy860
      : DefaultPolicy
      , ChainedPolicy<860, Policy860, Policy800>
  {};

  struct Policy900 : ChainedPolicy<900, Policy900, Policy860>
  {
    using ScanPolicyT = decltype(select_agent_policy<sm90_tuning<AccumT, is_primitive_op<ScanOpT>()>>(0));
  };

  struct Policy1000 : ChainedPolicy<1000, Policy1000, Policy900>
  {
    // Use values from tuning if a specialization exists that matches a benchmark, otherwise pick Policy900
    template <typename Tuning,
              typename IVT,
              // In the tuning benchmarks the Initial-, Input- and OutputType are the same. Let's check that the
              // accumulator type's size matches what we used during the benchmark since that has an impact (The
              // tunings also check later that it's a primitive type, so arithmetic impact is also comparable to the
              // benchmark). Input- and OutputType only impact loading and storing data (all arithmetic is done in the
              // accumulator type), so let's check that they are the same size and dispatch the size in the tunings.
              ::cuda::std::enable_if_t<sizeof(AccumT) == sizeof(::cuda::std::__accumulator_t<ScanOpT, IVT, IVT>)
                                         && sizeof(IVT) == sizeof(OutputValueT),
                                       int> = 0>
    _CCCL_HOST_DEVICE static auto select_agent_policy100(int)
      -> AgentScanPolicy<Tuning::threads,
                         Tuning::items,
                         AccumT,
                         Tuning::load_algorithm,
                         Tuning::load_modifier,
                         Tuning::store_algorithm,
                         BLOCK_SCAN_WARP_SCANS,
                         MemBoundScaling<Tuning::threads, Tuning::items, AccumT>,
                         typename Tuning::delay_constructor>;
    template <typename Tuning, typename IVT>
    _CCCL_HOST_DEVICE static auto select_agent_policy100(long) -> typename Policy900::ScanPolicyT;

    using ScanPolicyT =
      decltype(select_agent_policy100<sm100_tuning<InputValueT, AccumT, OffsetT, classify_op<ScanOpT>>, InputValueT>(0));
  };

  using MaxPolicy = Policy1000;
};

struct scan_lookback_policy
{
  int block_threads;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockStoreAlgorithm store_algorithm;
  BlockScanAlgorithm scan_algorithm;
  delay_constructor_policy delay_constructor;

  _CCCL_API constexpr friend bool operator==(const scan_lookback_policy& lhs, const scan_lookback_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.items_per_thread == rhs.items_per_thread
        && lhs.load_algorithm == rhs.load_algorithm && lhs.load_modifier == rhs.load_modifier
        && lhs.store_algorithm == rhs.store_algorithm && lhs.scan_algorithm == rhs.scan_algorithm
        && lhs.delay_constructor == rhs.delay_constructor;
  }

  _CCCL_API constexpr friend bool operator!=(const scan_lookback_policy& lhs, const scan_lookback_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const scan_lookback_policy& p)
  {
    return os
        << "scan_lookback_policy { .block_threads = " << p.block_threads
        << ", .items_per_thread = " << p.items_per_thread << ", .load_algorithm = " << p.load_algorithm
        << ", .load_modifier = " << p.load_modifier << ", .store_algorithm = " << p.store_algorithm
        << ", .scan_algorithm = " << p.scan_algorithm << ", .delay_constructor = " << p.delay_constructor << " }";
  }
#endif // _CCCL_HOSTED()
};

struct scan_warpspeed_policy
{
  int num_reduce_and_scan_warps;
  int look_ahead_items_per_thread;
  int items_per_thread;

  _CCCL_API constexpr int tile_size() const noexcept
  {
    return items_per_thread * num_reduce_and_scan_warps * warp_threads;
  }

  _CCCL_API constexpr friend bool operator==(const scan_warpspeed_policy& lhs, const scan_warpspeed_policy& rhs)
  {
    return lhs.num_reduce_and_scan_warps == rhs.num_reduce_and_scan_warps
        && lhs.look_ahead_items_per_thread == rhs.look_ahead_items_per_thread
        && lhs.items_per_thread == rhs.items_per_thread;
  }

  _CCCL_API constexpr friend bool operator!=(const scan_warpspeed_policy& lhs, const scan_warpspeed_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const scan_warpspeed_policy& p)
  {
    return os << "scan_warpspeed_policy { .num_reduce_and_scan_warps = " << p.num_reduce_and_scan_warps
              << ", .look_ahead_items_per_thread = " << p.look_ahead_items_per_thread
              << ", .items_per_thread = " << p.items_per_thread << " }";
  }
#endif // _CCCL_HOSTED()
};

enum class scan_algorithm
{
  lookback,
  warpspeed
};

#if _CCCL_HOSTED()
inline ::std::ostream& operator<<(::std::ostream& os, scan_algorithm algorithm)
{
  switch (algorithm)
  {
    case scan_algorithm::lookback:
      return os << "scan_algorithm::lookback";
    case scan_algorithm::warpspeed:
      return os << "scan_algorithm::warpspeed";
    default:
      return os << "scan_algorithm::<unknown>";
  }
}
#endif // _CCCL_HOSTED()

struct scan_policy
{
  scan_algorithm algorithm;
  scan_lookback_policy lookback;
  scan_warpspeed_policy warpspeed;

  _CCCL_API constexpr friend bool operator==(const scan_policy& lhs, const scan_policy& rhs)
  {
    return lhs.lookback == rhs.lookback && lhs.warpspeed == rhs.warpspeed && lhs.algorithm == rhs.algorithm;
  }

  _CCCL_API constexpr friend bool operator!=(const scan_policy& lhs, const scan_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const scan_policy& p)
  {
    return os << "scan_policy { .algorithm = " << p.algorithm << ", .lookback = " << p.lookback
              << ", .warpspeed = " << p.warpspeed << " }";
  }
#endif // _CCCL_HOSTED()
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept scan_policy_selector = policy_selector<T, scan_policy>;
#endif // _CCCL_HAS_CONCEPTS()

_CCCL_API constexpr auto make_mem_scaled_lookback_scan_policy(
  int nominal_4b_block_threads,
  int nominal_4b_items_per_thread,
  int compute_t_size,
  BlockLoadAlgorithm load_algorithm,
  CacheLoadModifier load_modifier,
  BlockStoreAlgorithm store_algorithm,
  BlockScanAlgorithm scan_algorithm,
  delay_constructor_policy delay_constructor = {delay_constructor_kind::fixed_delay, 350, 450}) -> scan_policy
{
  const auto scaled = scale_mem_bound(nominal_4b_block_threads, nominal_4b_items_per_thread, compute_t_size);
  return scan_policy{
    scan_algorithm::lookback,
    scan_lookback_policy{
      scaled.block_threads,
      scaled.items_per_thread,
      load_algorithm,
      load_modifier,
      store_algorithm,
      scan_algorithm,
      delay_constructor},
    scan_warpspeed_policy{}};
}

_CCCL_API constexpr warpspeed::SquadDesc squad_reduce(const scan_warpspeed_policy& policy)
{
  return warpspeed::SquadDesc{0, policy.num_reduce_and_scan_warps};
}

_CCCL_API constexpr warpspeed::SquadDesc squad_scan_store(const scan_warpspeed_policy& policy)
{
  return warpspeed::SquadDesc{1, policy.num_reduce_and_scan_warps};
}

_CCCL_API constexpr warpspeed::SquadDesc squad_load(const scan_warpspeed_policy&)
{
  return warpspeed::SquadDesc{2, 1}; // no point in being more than 1 warp
}

_CCCL_API constexpr warpspeed::SquadDesc squad_sched(const scan_warpspeed_policy&)
{
  return warpspeed::SquadDesc{3, 1}; // no point in being more than 1 warp
}

_CCCL_API constexpr warpspeed::SquadDesc squad_lookback(const scan_warpspeed_policy&)
{
  return warpspeed::SquadDesc{4, 1}; // must have 1 warp
}

// TODO(bgruber): put this somewhere else
constexpr _CCCL_API bool is_arithmetic_type(type_t type)
{
  switch (type)
  {
    case type_t::boolean:
    case type_t::int8:
    case type_t::int16:
    case type_t::int32:
    case type_t::int64:
    case type_t::int128:
    case type_t::uint8:
    case type_t::uint16:
    case type_t::uint32:
    case type_t::uint64:
    case type_t::uint128:
    case type_t::float32:
    case type_t::float64:
      return true;
    case type_t::other:
      return false;
  }

  return false;
}

struct scan_stage_counts
{
  int num_block_idx_stages;
  int num_sum_exclusive_cta_stages;
};

_CCCL_API constexpr scan_stage_counts make_scan_stage_counts(int num_stages)
{
  // If numBlockIdxStages is one less than the number of stages, we find a small speedup compared to setting it equal to
  // num_stages. Not sure why. TODO(bgruber): make this tunable
  const int num_block_idx_stages = ::cuda::std::max(1, num_stages - 1);

  // We do not need too many sumExclusiveCta stages. The lookback warp is the bottleneck. As soon as it produces a new
  // value, it will be consumed by the scanStore squad, releasing the stage.
  return {num_block_idx_stages, 2};
}

struct ScanResourcesRaw
{
  warpspeed::SmemResourceRaw smemInOut;
  warpspeed::SmemResourceRaw smemNextBlockIdx;
  warpspeed::SmemResourceRaw smemSumExclusiveCta;
  warpspeed::SmemResourceRaw smemSumThreadAndWarp;
};

template <typename SmemInOutT, typename SmemNextBlockIdxT, typename SmemSumExclusiveCtaT, typename SmemSumThreadAndWarpT>
_CCCL_API constexpr void setup_scan_resources(
  const scan_warpspeed_policy& policy,
  warpspeed::SyncHandler& syncHandler,
  warpspeed::SmemAllocator& smemAllocator,
  SmemInOutT& smemInOut,
  SmemNextBlockIdxT& smemNextBlockIdx,
  SmemSumExclusiveCtaT& smemSumExclusiveCta,
  SmemSumThreadAndWarpT& smemSumThreadAndWarp)
{
  const warpspeed::SquadDesc scanSquads[] = {
    squad_reduce(policy),
    squad_scan_store(policy),
    squad_load(policy),
    squad_sched(policy),
    squad_lookback(policy),
  };

  smemInOut.addPhase(syncHandler, smemAllocator, squad_load(policy));
  smemInOut.addPhase(syncHandler, smemAllocator, {squad_reduce(policy), squad_scan_store(policy)});

  smemNextBlockIdx.addPhase(syncHandler, smemAllocator, squad_sched(policy));
  smemNextBlockIdx.addPhase(syncHandler, smemAllocator, scanSquads);

  smemSumExclusiveCta.addPhase(syncHandler, smemAllocator, squad_lookback(policy));
  smemSumExclusiveCta.addPhase(syncHandler, smemAllocator, squad_scan_store(policy));

  smemSumThreadAndWarp.addPhase(syncHandler, smemAllocator, squad_reduce(policy));
  smemSumThreadAndWarp.addPhase(syncHandler, smemAllocator, squad_scan_store(policy));
}

_CCCL_API constexpr auto smem_for_stages(
  const scan_warpspeed_policy& policy,
  int num_stages,
  int input_size,
  int input_align,
  int output_align,
  int accum_size,
  int accum_align) -> int
{
  warpspeed::SyncHandler syncHandler{};
  warpspeed::SmemAllocator smemAllocator{};
  const auto counts = make_scan_stage_counts(num_stages);

  const int align_inout = ::cuda::std::max({16, input_align, output_align});
  const int inout_bytes = policy.tile_size() * input_size + 16;
  // Match sizeof(InOutT): round up to the alignment so each stage matches SmemResource<InOutT>.
  const int inout_stride    = (inout_bytes + align_inout - 1) & ~(align_inout - 1);
  const auto reduce_squad   = squad_reduce(policy);
  const int sum_thread_warp = (reduce_squad.threadCount() + reduce_squad.warpCount()) * accum_size;

  void* inout_base = smemAllocator.alloc(static_cast<::cuda::std::uint32_t>(inout_stride * num_stages), align_inout);
  void* next_block_idx_base = smemAllocator.alloc(
    static_cast<::cuda::std::uint32_t>(sizeof(uint4) * counts.num_block_idx_stages), alignof(uint4));
  void* sum_exclusive_base = smemAllocator.alloc(
    static_cast<::cuda::std::uint32_t>(accum_size * counts.num_sum_exclusive_cta_stages), accum_align);
  void* sum_thread_warp_base =
    smemAllocator.alloc(static_cast<::cuda::std::uint32_t>(sum_thread_warp * num_stages), accum_align);

  ScanResourcesRaw res = {
    warpspeed::SmemResourceRaw{syncHandler, inout_base, inout_stride, inout_stride, num_stages},
    warpspeed::SmemResourceRaw{
      syncHandler,
      next_block_idx_base,
      static_cast<int>(sizeof(uint4)),
      static_cast<int>(sizeof(uint4)),
      counts.num_block_idx_stages},
    warpspeed::SmemResourceRaw{
      syncHandler, sum_exclusive_base, accum_size, accum_size, counts.num_sum_exclusive_cta_stages},
    warpspeed::SmemResourceRaw{syncHandler, sum_thread_warp_base, sum_thread_warp, sum_thread_warp, num_stages},
  };

  setup_scan_resources(
    policy,
    syncHandler,
    smemAllocator,
    res.smemInOut,
    res.smemNextBlockIdx,
    res.smemSumExclusiveCta,
    res.smemSumThreadAndWarp);
  syncHandler.mHasInitialized = true; // avoid assertion in destructor
  return static_cast<int>(smemAllocator.sizeBytes());
}

struct policy_selector
{
  int input_value_size;
  int input_value_alignment;
  int output_value_size; // TODO(bgruber): unused at the moment
  int output_value_alignment;
  int accum_size;
  int accum_alignment;
  int offset_size;
  type_t input_type;
  type_t accum_type;
  op_kind_t operation_t;
  bool input_contiguous;
  bool output_contiguous;
  bool input_trivially_copyable;
  bool output_trivially_copyable;
  bool output_default_constructible;
  bool accum_is_primitive_or_trivially_copy_constructible;
  // TODO(griwes): remove this field before policy_selector is publicly exposed
  bool benchmark_match;

  _CCCL_API constexpr auto get_sm100_fallback_warpspeed_policy() const -> scan_warpspeed_policy
  {
    scan_warpspeed_policy warpspeed_policy{};

    // TODO(bgruber): tune this
#if _CCCL_COMPILER(NVHPC)
    // need to reduce the number of threads to <= 256, so each thread can use up to 255 registers. This avoids an
    // error in ptxas, see also: https://github.com/NVIDIA/cccl/issues/7700.
    warpspeed_policy.num_reduce_and_scan_warps = 2;
#else // _CCCL_COMPILER(NVHPC)
    warpspeed_policy.num_reduce_and_scan_warps = 4;
#endif // _CCCL_COMPILER(NVHPC)

    // TODO(bgruber): 5 is a bit better for complex<float>
    warpspeed_policy.look_ahead_items_per_thread = accum_size == 2 ? 3 : 4;

    // manual tuning based on cub.bench.scan.exclusive.sum.base
    // 256 / sizeof(InputValueT) - 1 should minimize bank conflicts (and fits into 48KiB SMEM)
    // 2-byte types and double needed special handling
    warpspeed_policy.items_per_thread = ::cuda::std::max(256 / (input_value_size == 2 ? 2 : accum_size) - 1, 1);
    // TODO(bgruber): the special handling of double below is a LOT faster on B200, but exceeds 48KiB SMEM
    // clang-format off
      // |   F64   |      I64      |     72576      |  12.301 us |       8.18% |  12.987 us |       5.75% |     0.686 us |   5.58% |   SAME   |
      // |   F64   |      I64      |    1056384     |  16.775 us |       5.70% |  16.091 us |       6.14% |    -0.684 us |  -4.08% |   SAME   |
      // |   F64   |      I64      |    16781184    |  66.970 us |       1.41% |  58.024 us |       3.17% |    -8.946 us | -13.36% |   FAST   |
      // |   F64   |      I64      |   268442496    | 863.826 us |       0.23% | 676.465 us |       0.98% |  -187.360 us | -21.69% |   FAST   |
      // |   F64   |      I64      |   1073745792   |   3.419 ms |       0.11% |   2.664 ms |       0.48% |  -755.409 us | -22.09% |   FAST   |
      // |   F64   |      I64      |   4294975104   |  13.641 ms |       0.05% |  10.575 ms |       0.24% | -3065.815 us | -22.48% |   FAST   |
    // clang-format on
    // (256 / (sizeof(InputValueT) == 2 ? 2 : (::cuda::std::is_same_v<InputValueT, double> ? 4 : sizeof(AccumT))) -
    // 1);

    return warpspeed_policy;
  }

  _CCCL_API constexpr auto get_sm120_fallback_warpspeed_policy() const -> scan_warpspeed_policy
  {
    auto policy = get_sm100_fallback_warpspeed_policy();
    if (operation_t == op_kind_t::other && is_arithmetic_type(input_type))
    {
      if (input_value_size == 4 || input_value_size == 8)
      {
        policy.items_per_thread = 127;
      }
      else
      {
        policy.items_per_thread = ::cuda::std::min(policy.items_per_thread, input_value_size <= 2 ? 63 : 127);
      }
    }
    return policy;
  }

  _CCCL_API constexpr auto get_warpspeed_policy(::cuda::compute_capability cc) const
    -> ::cuda::std::optional<scan_warpspeed_policy>
  {
    if (cc >= ::cuda::compute_capability{12, 0})
    {
      return get_sm120_fallback_warpspeed_policy();
    }
    if (cc >= ::cuda::compute_capability{10, 0})
    {
      // tunings from cub/benchmarks/bench/scan/exclusive/sum.warpspeed.cu
      if (operation_t == op_kind_t::plus && accum_is_primitive_or_trivially_copy_constructible)
      {
        switch (input_value_size)
        {
          case 1:
            // wrps_4.lbi_8.ipt_160 ()  1.264254  1.264254  1.264254  1.264254
            return scan_warpspeed_policy{4, 8, 160 - 1};
            // TODO(gonidelis): we found this tuning but it regressed:
            // wrps_3.lbi_4.ipt_96 ()  1.454824  1.247212  1.450590  1.560418
            // return scan_warpspeed_policy{3, 4, 96 - 1};
          case 2:
            // TODO(gonidelis): we found this tuning but it regresses large problems, we should revisit this
            // // wrps_4.lbi_2.ipt_96 ()  1.082511  0.929516  1.091523  1.264033
            // return scan_warpspeed_policy{4, 2, 96 - 1};
            // clang-format off
            //|   I16   |      I64      |      2^16      |  17.304 us |       1.07% |  15.244 us |       0.77% |    -2.060 us | -11.91% |   FAST   |
            //|   I16   |      I64      |      2^20      |  19.466 us |       1.21% |  17.266 us |       2.93% |    -2.200 us | -11.30% |   FAST   |
            //|   I16   |      I64      |      2^24      |  39.565 us |       2.46% |  35.835 us |       4.25% |    -3.730 us |  -9.43% |   FAST   |
            //|   I16   |      I64      |      2^28      | 224.318 us |       0.37% | 233.381 us |       0.46% |     9.063 us |   4.04% |   SLOW   |
            //|   I16   |      I64      |      2^32      |   3.238 ms |       0.53% |   3.429 ms |       0.53% |   191.299 us |   5.91% |   SLOW   |
            // clang-format on
            // wrps_6.lbi_2.ipt_96 ()  1.167633  1.167633  1.167633  1.167633
            return scan_warpspeed_policy{6, 2, 96 - 1};
          case 4:
            if (input_type == type_t::float32)
            {
              // wrps_4.lbi_3.ipt_88 ()  1.047200  1.002119  1.042654  1.081102
              return scan_warpspeed_policy{4, 3, 88 - 1};
            }
            // wrps_4.lbi_3.ipt_80 ()  1.019078  0.999708  1.017346  1.052592
            return scan_warpspeed_policy{4, 3, 80 - 1};
          case 8:
            // wrps_2.lbi_5.ipt_88 ()  1.085781   1.0  1.079245  1.103545
            return scan_warpspeed_policy{2, 5, 88 - 1};
          case 16:
            // wrps_5.lbi_8.ipt_16 ()  1.159883  1.000000  1.143709  1.275821
            return scan_warpspeed_policy{5, 8, 16 - 1};
            // TODO(bgruber): tune for more data types
          default:
            break;
        }
      }

      return get_sm100_fallback_warpspeed_policy();
    }
    return {};
  }

  _CCCL_API constexpr bool can_use_warpspeed([[maybe_unused]] const scan_warpspeed_policy& warpspeed_policy) const
  {
    // We need `cuda::std::is_constant_evaluated` for the compile-time SMEM computation. And we need PTX ISA 8.6.
    // MSVC + nvcc < 13.1 just fails to compile `cub.test.device.scan.lid_1.types_0` with `Internal error` and nothing
    // else.
    // The macro `CCCL_DISABLE_WARPSPEED_SCAN` will be left in as a kill-switch for users in case they find any bugs
    // after we shipped the implementation. TODO(bgruber): remove CCCL_DISABLE_WARPSPEED_SCAN in CCCL 4.0
#if __cccl_ptx_isa < 860 || !defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED) \
  || ((_CCCL_COMPILER(MSVC) && _CCCL_CUDA_COMPILER(NVCC, <, 13, 1))) || defined(CCCL_DISABLE_WARPSPEED_SCAN)
    return false;
#else
    if (!input_contiguous || !output_contiguous || !input_trivially_copyable || !output_trivially_copyable
        || !output_default_constructible)
    {
      return false;
    }

    if (smem_for_stages(
          warpspeed_policy,
          /* num_stages */ 1,
          input_value_size,
          input_value_alignment,
          output_value_alignment,
          accum_size,
          accum_alignment)
        > static_cast<int>(max_smem_per_block))
    {
      return false;
    }
    return true;
#endif
  }

  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::compute_capability cc) const -> scan_policy
  {
    // we first try to get the valid warpspeed implementation. if we can't run it, fall back to the old scan impl.
    {
      const auto warpspeed_policy_opt = get_warpspeed_policy(cc);
      if (warpspeed_policy_opt && can_use_warpspeed(*warpspeed_policy_opt))
      {
        return {scan_algorithm::warpspeed, scan_lookback_policy{}, *warpspeed_policy_opt};
      }
    }

    const primitive_accum primitive_accum_t =
      accum_type != type_t::other && accum_type != type_t::int128 ? primitive_accum::yes : primitive_accum::no;
    const primitive_op primitive_op_t = operation_t != op_kind_t::other ? primitive_op::yes : primitive_op::no;

    const bool large_values = accum_size > 128;
    const BlockLoadAlgorithm scan_transposed_load =
      large_values ? BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED : BLOCK_LOAD_WARP_TRANSPOSE;
    const BlockStoreAlgorithm scan_transposed_store =
      large_values ? BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED : BLOCK_STORE_WARP_TRANSPOSE;
    const auto default_delay = default_delay_constructor_policy(accum_is_primitive_or_trivially_copy_constructible);

    if (cc >= ::cuda::compute_capability{10, 0})
    {
      if (benchmark_match && operation_t == op_kind_t::plus && primitive_accum_t == primitive_accum::yes)
      {
        if (offset_size == 4)
        {
          switch (input_value_size)
          {
            case 1:
              // ipt_18.tpb_512.ns_768.dcid_7.l2w_820.trp_1.ld_0 1.188818  1.005682  1.173041  1.305288
              return make_mem_scaled_lookback_scan_policy(
                512,
                18,
                accum_size,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_STORE_WARP_TRANSPOSE,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::exponential_backon, 768, 820});
            case 2:
              // ipt_13.tpb_512.ns_1384.dcid_7.l2w_720.trp_1.ld_0 1.128443  1.002841  1.119688  1.307692
              return make_mem_scaled_lookback_scan_policy(
                512,
                13,
                accum_size,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_STORE_WARP_TRANSPOSE,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::exponential_backon, 1384, 720});
            case 4:
              // ipt_22.tpb_384.ns_1904.dcid_6.l2w_830.trp_1.ld_0 1.148442  0.997167  1.139902  1.462651
              return make_mem_scaled_lookback_scan_policy(
                384,
                22,
                accum_size,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_STORE_WARP_TRANSPOSE,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::exponential_backon_jitter, 1904, 830});
            case 8:
              // ipt_23.tpb_416.ns_772.dcid_5.l2w_710.trp_1.ld_0 1.089468  1.015581  1.085630  1.264583
              return make_mem_scaled_lookback_scan_policy(
                416,
                23,
                accum_size,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_STORE_WARP_TRANSPOSE,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::exponential_backon_jitter_window, 772, 710});
            default:
              break;
          }
        }
        else if (offset_size == 8)
        {
          switch (input_value_size)
          {
            case 1:
              // ipt_14.tpb_384.ns_228.dcid_7.l2w_775.trp_1.ld_1 1.107210  1.000000  1.100637  1.307692
              return make_mem_scaled_lookback_scan_policy(
                384,
                14,
                accum_size,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_CA,
                BLOCK_STORE_WARP_TRANSPOSE,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::exponential_backon, 228, 775});
            case 2:
              // todo(gonidelis): Regresses for large inputs. Find better tuning.
              // ipt_13.tpb_288.ns_1520.dcid_5.l2w_895.trp_1.ld_1 1.080934  0.983509  1.077724  1.305288
              break;
            case 4:
              // ipt_19.tpb_416.ns_956.dcid_7.l2w_550.trp_1.ld_1 1.146142  0.994350  1.137459  1.455636
              return make_mem_scaled_lookback_scan_policy(
                416,
                19,
                accum_size,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_CA,
                BLOCK_STORE_WARP_TRANSPOSE,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::exponential_backon, 956, 550});
            case 8:
              if (accum_type == type_t::float64)
              {
                break;
              }
              // ipt_22.tpb_320.ns_328.dcid_2.l2w_965.trp_1.ld_0 1.080133  1.000000  1.075577  1.248963
              return make_mem_scaled_lookback_scan_policy(
                320,
                22,
                accum_size,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_STORE_WARP_TRANSPOSE,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::exponential_backoff, 328, 965});
            default:
              break;
          }
        }
      }
    }

    if (cc >= ::cuda::compute_capability{9, 0})
    {
      if (primitive_op_t == primitive_op::yes)
      {
        if (primitive_accum_t == primitive_accum::yes)
        {
          switch (accum_size)
          {
            case 1:
              return make_mem_scaled_lookback_scan_policy(
                192,
                22,
                accum_size,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_STORE_WARP_TRANSPOSE,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::fixed_delay, 168, 1140});
            case 2:
              return make_mem_scaled_lookback_scan_policy(
                512,
                12,
                accum_size,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_STORE_WARP_TRANSPOSE,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::fixed_delay, 376, 1125});
            case 4:
              if (accum_type == type_t::float32)
              {
                return make_mem_scaled_lookback_scan_policy(
                  128,
                  24,
                  accum_size,
                  BLOCK_LOAD_WARP_TRANSPOSE,
                  LOAD_DEFAULT,
                  BLOCK_STORE_WARP_TRANSPOSE,
                  BLOCK_SCAN_WARP_SCANS,
                  delay_constructor_policy{delay_constructor_kind::fixed_delay, 688, 1140});
              }
              return make_mem_scaled_lookback_scan_policy(
                128,
                24,
                accum_size,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_STORE_WARP_TRANSPOSE,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::fixed_delay, 648, 1245});
            case 8:
              if (accum_type == type_t::float64)
              {
                return make_mem_scaled_lookback_scan_policy(
                  224,
                  24,
                  accum_size,
                  BLOCK_LOAD_WARP_TRANSPOSE,
                  LOAD_DEFAULT,
                  BLOCK_STORE_WARP_TRANSPOSE,
                  BLOCK_SCAN_WARP_SCANS,
                  delay_constructor_policy{delay_constructor_kind::fixed_delay, 576, 1215});
              }
              return make_mem_scaled_lookback_scan_policy(
                224,
                24,
                accum_size,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_STORE_WARP_TRANSPOSE,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::fixed_delay, 632, 1290});
            default:
              break;
          }
        }

#if _CCCL_HAS_INT128()
        if (primitive_accum_t == primitive_accum::no && accum_size == 16
            && (accum_type == type_t::int128 || accum_type == type_t::uint128))
        {
          return make_mem_scaled_lookback_scan_policy(
            576,
            21,
            accum_size,
            BLOCK_LOAD_WARP_TRANSPOSE,
            LOAD_DEFAULT,
            BLOCK_STORE_WARP_TRANSPOSE,
            BLOCK_SCAN_WARP_SCANS,
            delay_constructor_policy{delay_constructor_kind::fixed_delay, 860, 630});
        }
#endif
      }
    }

    // Keep sm_86 aligned with legacy policy_hub behavior: policy_hub resets to default policy for 86.
    if (cc >= ::cuda::compute_capability{8, 6})
    {
      return make_mem_scaled_lookback_scan_policy(
        128,
        15,
        accum_size,
        scan_transposed_load,
        LOAD_DEFAULT,
        scan_transposed_store,
        BLOCK_SCAN_WARP_SCANS,
        default_delay);
    }

    if (cc >= ::cuda::compute_capability{8, 0})
    {
      if (primitive_op_t == primitive_op::yes)
      {
        if (primitive_accum_t == primitive_accum::yes)
        {
          switch (accum_size)
          {
            case 1:
              return make_mem_scaled_lookback_scan_policy(
                320,
                14,
                accum_size,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_STORE_WARP_TRANSPOSE,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::fixed_delay, 368, 725});
            case 2:
              return make_mem_scaled_lookback_scan_policy(
                352,
                16,
                accum_size,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_STORE_WARP_TRANSPOSE,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::fixed_delay, 488, 1040});
            case 4:
              if (accum_type == type_t::float32)
              {
                return make_mem_scaled_lookback_scan_policy(
                  288,
                  8,
                  accum_size,
                  BLOCK_LOAD_WARP_TRANSPOSE,
                  LOAD_DEFAULT,
                  BLOCK_STORE_WARP_TRANSPOSE,
                  BLOCK_SCAN_WARP_SCANS,
                  delay_constructor_policy{delay_constructor_kind::fixed_delay, 724, 1050});
              }
              return make_mem_scaled_lookback_scan_policy(
                320,
                12,
                accum_size,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_STORE_WARP_TRANSPOSE,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::fixed_delay, 268, 1180});
            case 8:
              if (accum_type == type_t::float64)
              {
                return make_mem_scaled_lookback_scan_policy(
                  384,
                  12,
                  accum_size,
                  BLOCK_LOAD_WARP_TRANSPOSE,
                  LOAD_DEFAULT,
                  BLOCK_STORE_WARP_TRANSPOSE,
                  BLOCK_SCAN_WARP_SCANS,
                  delay_constructor_policy{delay_constructor_kind::fixed_delay, 388, 1100});
              }
              return make_mem_scaled_lookback_scan_policy(
                288,
                22,
                accum_size,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_STORE_WARP_TRANSPOSE,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::fixed_delay, 716, 785});
            default:
              break;
          }
        }

#if _CCCL_HAS_INT128()
        if (primitive_accum_t == primitive_accum::no && accum_size == 16
            && (accum_type == type_t::int128 || accum_type == type_t::uint128))
        {
          return make_mem_scaled_lookback_scan_policy(
            640,
            24,
            accum_size,
            BLOCK_LOAD_DIRECT,
            LOAD_DEFAULT,
            BLOCK_STORE_DIRECT,
            BLOCK_SCAN_WARP_SCANS,
            delay_constructor_policy{delay_constructor_kind::no_delay, 0, 1200});
        }
#endif
      }
    }

    if (cc >= ::cuda::compute_capability{7, 5})
    {
      if (benchmark_match && operation_t == op_kind_t::plus && primitive_accum_t == primitive_accum::yes
          && offset_size == 8 && input_value_size == 4)
      {
        // ipt_7.tpb_128.ns_628.dcid_1.l2w_520.trp_1.ld_0
        return make_mem_scaled_lookback_scan_policy(
          128,
          7,
          accum_size,
          BLOCK_LOAD_WARP_TRANSPOSE,
          LOAD_DEFAULT,
          BLOCK_STORE_WARP_TRANSPOSE,
          BLOCK_SCAN_WARP_SCANS,
          delay_constructor_policy{delay_constructor_kind::fixed_delay, 628, 520});
      }

      return make_mem_scaled_lookback_scan_policy(
        128,
        15,
        accum_size,
        scan_transposed_load,
        LOAD_DEFAULT,
        scan_transposed_store,
        BLOCK_SCAN_WARP_SCANS,
        default_delay);
    }

    if (cc >= ::cuda::compute_capability{6, 0})
    {
      return make_mem_scaled_lookback_scan_policy(
        128,
        15,
        accum_size,
        scan_transposed_load,
        LOAD_DEFAULT,
        scan_transposed_store,
        BLOCK_SCAN_WARP_SCANS,
        default_delay);
    }

    return make_mem_scaled_lookback_scan_policy(
      128,
      12,
      accum_size,
      BLOCK_LOAD_DIRECT,
      LOAD_CA,
      BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED,
      BLOCK_SCAN_RAKING,
      default_delay);
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(scan_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

template <typename ScanOpT, typename InputValueT, typename OutputValueT, typename AccumT, typename = void>
struct benchmark_match_for_policy_selector
{
  static constexpr bool value = false;
};

template <typename ScanOpT, typename InputValueT, typename OutputValueT, typename AccumT>
struct benchmark_match_for_policy_selector<
  ScanOpT,
  InputValueT,
  OutputValueT,
  AccumT,
  ::cuda::std::void_t<::cuda::std::__accumulator_t<ScanOpT, InputValueT, InputValueT>>>
{
  static constexpr bool value =
    sizeof(AccumT) == sizeof(::cuda::std::__accumulator_t<ScanOpT, InputValueT, InputValueT>)
    && sizeof(InputValueT) == sizeof(OutputValueT);
};

// stateless version which can be passed to kernels
template <typename InputIteratorT, typename OutputIteratorT, typename AccumT, typename OffsetT, typename ScanOpT>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::compute_capability cc) const -> scan_policy
  {
    using InputValueT  = it_value_t<InputIteratorT>;
    using OutputValueT = it_value_t<OutputIteratorT>;

    constexpr bool benchmark_match =
      benchmark_match_for_policy_selector<ScanOpT, InputValueT, OutputValueT, AccumT>::value;

    constexpr bool accum_is_primitive_or_trivially_copy_constructible =
      is_primitive<AccumT>::value || ::cuda::std::is_trivially_copy_constructible_v<AccumT>;

    constexpr auto policies = policy_selector{
      int{sizeof(InputValueT)},
      int{alignof(InputValueT)},
      int{sizeof(OutputValueT)},
      int{alignof(OutputValueT)},
      int{sizeof(AccumT)},
      int{alignof(AccumT)},
      int{sizeof(OffsetT)},
      classify_type<InputValueT>,
      classify_type<AccumT>,
      classify_op<ScanOpT>,
      THRUST_NS_QUALIFIER::is_contiguous_iterator_v<InputIteratorT>,
      THRUST_NS_QUALIFIER::is_contiguous_iterator_v<OutputIteratorT>,
      ::cuda::std::is_trivially_copyable_v<InputValueT>,
      ::cuda::std::is_trivially_copyable_v<OutputValueT>,
      ::cuda::std::is_default_constructible_v<OutputValueT>,
      accum_is_primitive_or_trivially_copy_constructible,
      benchmark_match};
    return policies(cc);
  }
};
} // namespace detail::scan

CUB_NAMESPACE_END
