// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
#include <cub/detail/warpspeed/squad/squad_desc.h>
#include <cub/device/dispatch/tuning/common.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/void_t.h>

CUB_NAMESPACE_BEGIN

namespace detail::scan
{
enum class keep_rejects
{
  no,
  yes
};
enum class primitive_accum
{
  no,
  yes
};
enum class primitive_op
{
  no,
  yes
};
enum class offset_size
{
  _4,
  _8,
  unknown
};
enum class value_size
{
  _1,
  _2,
  _4,
  _8,
  _16,
  unknown
};
enum class accum_size
{
  _1,
  _2,
  _4,
  _8,
  _16,
  unknown
};

template <class AccumT>
constexpr _CCCL_HOST_DEVICE primitive_accum is_primitive_accum()
{
  return is_primitive<AccumT>::value ? primitive_accum::yes : primitive_accum::no;
}

template <class ScanOpT>
constexpr _CCCL_HOST_DEVICE primitive_op is_primitive_op()
{
  return basic_binary_op_t<ScanOpT>::value ? primitive_op::yes : primitive_op::no;
}

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

template <class OffsetT>
constexpr _CCCL_HOST_DEVICE offset_size classify_offset_size()
{
  return sizeof(OffsetT) == 4 ? offset_size::_4 : sizeof(OffsetT) == 8 ? offset_size::_8 : offset_size::unknown;
}

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

template <class AccumT,
          primitive_op PrimitiveOp,
          primitive_accum PrimitiveAccumulator = is_primitive_accum<AccumT>(),
          accum_size AccumSize                 = classify_accum_size<AccumT>()>
struct sm90_tuning;

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

template <typename PolicyT, typename = void, typename = void>
struct ScanPolicyWrapper : PolicyT
{
  _CCCL_HOST_DEVICE ScanPolicyWrapper(PolicyT base)
      : PolicyT(base)
  {}
};

template <typename StaticPolicyT>
struct ScanPolicyWrapper<StaticPolicyT, ::cuda::std::void_t<decltype(StaticPolicyT::ScanPolicyT::LOAD_MODIFIER)>>
    : StaticPolicyT
{
  _CCCL_HOST_DEVICE ScanPolicyWrapper(StaticPolicyT base)
      : StaticPolicyT(base)
  {}

  _CCCL_HOST_DEVICE static constexpr auto Scan()
  {
    return cub::detail::MakePolicyWrapper(typename StaticPolicyT::ScanPolicyT());
  }

  _CCCL_HOST_DEVICE static constexpr CacheLoadModifier LoadModifier()
  {
    return StaticPolicyT::ScanPolicyT::LOAD_MODIFIER;
  }

  _CCCL_HOST_DEVICE constexpr void CheckLoadModifier()
  {
    static_assert(LoadModifier() != CacheLoadModifier::LOAD_LDG,
                  "The memory consistency model does not apply to texture "
                  "accesses");
  }

#if defined(CUB_ENABLE_POLICY_PTX_JSON)
  _CCCL_DEVICE static constexpr auto EncodedPolicy()
  {
    using namespace ptx_json;
    return object<key<"ScanPolicyT">() = Scan().EncodedPolicy(),
                  key<"DelayConstructor">() =
                    StaticPolicyT::ScanPolicyT::detail::delay_constructor_t::EncodedConstructor()>();
  }
#endif
};

template <typename PolicyT>
_CCCL_HOST_DEVICE ScanPolicyWrapper<PolicyT> MakeScanPolicyWrapper(PolicyT policy)
{
  return ScanPolicyWrapper<PolicyT>{policy};
}

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

#if __cccl_ptx_isa >= 860
    struct WarpspeedPolicy
    {
      // Squad definitions
      static constexpr int num_squads           = 5;
      static constexpr int num_threads_per_warp = 32;

      // TODO(bgruber): tune this
      static constexpr int num_reduce_warps     = 4;
      static constexpr int num_scan_stor_warps  = 4;
      static constexpr int num_load_warps       = 1;
      static constexpr int num_sched_warps      = 1;
      static constexpr int num_look_ahead_warps = 1;

      // TODO(bgruber): 5 is a bit better for complex<float>
      static constexpr int num_look_ahead_items = sizeof(AccumT) == 2 ? 3 : 4;

      // Deduced definitions
      static constexpr int num_total_warps =
        num_reduce_warps + num_scan_stor_warps + num_load_warps + num_sched_warps + num_look_ahead_warps;
      static constexpr int num_total_threads = num_total_warps * num_threads_per_warp;

      static constexpr int squad_reduce_thread_count = num_reduce_warps * num_threads_per_warp;

      // 256 / sizeof(InputValueT) - 1 should minimize bank conflicts (and fits into 48KiB SMEM)
      // 2-byte types and double needed special handling
      static constexpr int items_per_thread =
        ::cuda::std::max(256 / (sizeof(InputValueT) == 2 ? 2 : int{sizeof(AccumT)}) - 1, 1);
      // TODO(bgruber): the special handling of double below is a LOT faster, but exceeds 48KiB SMEM
      // clang-format off
      // |   F64   |      I32      |     72576      |  11.295 us |       2.44% |  11.917 us |       8.02% |     0.622 us |   5.50% |   SLOW   |
      // |   F64   |      I32      |    1056384     |  16.162 us |       6.24% |  15.847 us |       5.57% |    -0.315 us |  -1.95% |   SAME   |
      // |   F64   |      I32      |    16781184    |  65.696 us |       1.64% |  60.650 us |       3.37% |    -5.046 us |  -7.68% |   FAST   |
      // |   F64   |      I32      |   268442496    | 863.896 us |       0.22% | 679.100 us |       0.93% |  -184.796 us | -21.39% |   FAST   |
      // |   F64   |      I32      |   1073745792   |   3.418 ms |       0.12% |   2.662 ms |       0.46% |  -755.740 us | -22.11% |   FAST   |
      // |   F64   |      I64      |     72576      |  12.301 us |       8.18% |  12.987 us |       5.75% |     0.686 us |   5.58% |   SAME   |
      // |   F64   |      I64      |    1056384     |  16.775 us |       5.70% |  16.091 us |       6.14% |    -0.684 us |  -4.08% |   SAME   |
      // |   F64   |      I64      |    16781184    |  66.970 us |       1.41% |  58.024 us |       3.17% |    -8.946 us | -13.36% |   FAST   |
      // |   F64   |      I64      |   268442496    | 863.826 us |       0.23% | 676.465 us |       0.98% |  -187.360 us | -21.69% |   FAST   |
      // |   F64   |      I64      |   1073745792   |   3.419 ms |       0.11% |   2.664 ms |       0.48% |  -755.409 us | -22.09% |   FAST   |
      // |   F64   |      I64      |   4294975104   |  13.641 ms |       0.05% |  10.575 ms |       0.24% | -3065.815 us | -22.48% |   FAST   |
      // clang-format on
      // (256 / (sizeof(InputValueT) == 2 ? 2 : (::cuda::std::is_same_v<InputValueT, double> ? 4 : sizeof(AccumT))) -
      // 1);

      static constexpr int tile_size = items_per_thread * squad_reduce_thread_count;

      // The squads cannot be static constexpr variables, as those are not device accessible
      [[nodiscard]] _CCCL_API _CCCL_FORCEINLINE static constexpr SquadDesc squadReduce() noexcept
      {
        return SquadDesc{0, num_reduce_warps};
      }
      [[nodiscard]] _CCCL_API _CCCL_FORCEINLINE static constexpr SquadDesc squadScanStore() noexcept
      {
        return SquadDesc{1, num_scan_stor_warps};
      }
      [[nodiscard]] _CCCL_API _CCCL_FORCEINLINE static constexpr SquadDesc squadLoad() noexcept
      {
        return SquadDesc{2, num_load_warps};
      }
      [[nodiscard]] _CCCL_API _CCCL_FORCEINLINE static constexpr SquadDesc squadSched() noexcept
      {
        return SquadDesc{3, num_sched_warps};
      }
      [[nodiscard]] _CCCL_API _CCCL_FORCEINLINE static constexpr SquadDesc squadLookback() noexcept
      {
        return SquadDesc{4, num_look_ahead_warps};
      }
    };
#endif // __cccl_ptx_isa >= 860
  };

  using MaxPolicy = Policy1000;
};

template <class Policy, class = void>
inline constexpr bool scan_use_warpspeed = false;

// detect the use via CCCL.C (pre-compiled dispatch and JIT pass) and disable the new kernel.
// See https://github.com/NVIDIA/cccl/issues/6821 for more details.
#if !defined(CUB_ENABLE_POLICY_PTX_JSON) && !defined(CUB_DEFINE_RUNTIME_POLICIES)
template <class Policy>
inline constexpr bool scan_use_warpspeed<Policy, ::cuda::std::void_t<typename Policy::WarpspeedPolicy>> = true;
#endif // !defined(CUB_ENABLE_POLICY_PTX_JSON) && !defined(CUB_DEFINE_RUNTIME_POLICIES)
} // namespace detail::scan

CUB_NAMESPACE_END
