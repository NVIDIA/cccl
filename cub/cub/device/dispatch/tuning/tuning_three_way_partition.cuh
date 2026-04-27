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

#include <cub/agent/agent_three_way_partition.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/detail/delay_constructor.cuh>
#include <cub/device/dispatch/tuning/common.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>

#include <cuda/__device/arch_id.h>
#include <cuda/std/__host_stdlib/ostream>
#include <cuda/std/concepts>

CUB_NAMESPACE_BEGIN

namespace detail::three_way_partition
{
// TODO(bgruber): drop in CCCL 4.0
template <typename PolicyT, typename = void>
struct ThreeWayPartitionPolicyWrapper : PolicyT
{
  _CCCL_HOST_DEVICE ThreeWayPartitionPolicyWrapper(PolicyT base)
      : PolicyT(base)
  {}
};

// TODO(bgruber): drop in CCCL 4.0
template <typename StaticPolicyT>
struct ThreeWayPartitionPolicyWrapper<StaticPolicyT, ::cuda::std::void_t<typename StaticPolicyT::ThreeWayPartitionPolicy>>
    : StaticPolicyT
{
  _CCCL_HOST_DEVICE ThreeWayPartitionPolicyWrapper(StaticPolicyT base)
      : StaticPolicyT(base)
  {}

  CUB_DEFINE_SUB_POLICY_GETTER(ThreeWayPartition)
};

// TODO(bgruber): drop in CCCL 4.0
template <typename PolicyT>
_CCCL_HOST_DEVICE ThreeWayPartitionPolicyWrapper<PolicyT> MakeThreeWayPartitionPolicyWrapper(PolicyT policy)
{
  return ThreeWayPartitionPolicyWrapper<PolicyT>{policy};
}

enum class input_size
{
  _1,
  _2,
  _4,
  _8,
  _16,
  unknown
};

enum class offset_size
{
  _4,
  _8,
  unknown
};

template <class InputT>
_CCCL_HOST_DEVICE constexpr input_size classify_input_size()
{
  return sizeof(InputT) == 1 ? input_size::_1
       : sizeof(InputT) == 2 ? input_size::_2
       : sizeof(InputT) == 4 ? input_size::_4
       : sizeof(InputT) == 8 ? input_size::_8
       : sizeof(InputT) == 16
         ? input_size::_16
         : input_size::unknown;
}

template <class OffsetT>
_CCCL_HOST_DEVICE constexpr offset_size classify_offset_size()
{
  return sizeof(OffsetT) == 4 ? offset_size::_4 : sizeof(OffsetT) == 8 ? offset_size::_8 : offset_size::unknown;
}

template <class InputT,
          class OffsetT,
          input_size InputSize   = classify_input_size<InputT>(),
          offset_size OffsetSize = classify_offset_size<OffsetT>()>
struct sm80_tuning;

template <class Input, class OffsetT>
struct sm80_tuning<Input, OffsetT, input_size::_2, offset_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = no_delay_constructor_t<910>;
};

template <class Input, class OffsetT>
struct sm80_tuning<Input, OffsetT, input_size::_4, offset_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = no_delay_constructor_t<1120>;
};

template <class Input, class OffsetT>
struct sm80_tuning<Input, OffsetT, input_size::_8, offset_size::_4>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = fixed_delay_constructor_t<264, 1080>;
};

template <class Input, class OffsetT>
struct sm80_tuning<Input, OffsetT, input_size::_16, offset_size::_4>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = fixed_delay_constructor_t<672, 1120>;
};

template <class InputT,
          class OffsetT,
          input_size InputSize   = classify_input_size<InputT>(),
          offset_size OffsetSize = classify_offset_size<OffsetT>()>
struct sm90_tuning;

template <class Input, class OffsetT>
struct sm90_tuning<Input, OffsetT, input_size::_1, offset_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = no_delay_constructor_t<445>;
};

template <class Input, class OffsetT>
struct sm90_tuning<Input, OffsetT, input_size::_2, offset_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = fixed_delay_constructor_t<104, 512>;
};

template <class Input, class OffsetT>
struct sm90_tuning<Input, OffsetT, input_size::_4, offset_size::_4>
{
  static constexpr int threads                       = 320;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = no_delay_constructor_t<1105>;
};

template <class Input, class OffsetT>
struct sm90_tuning<Input, OffsetT, input_size::_8, offset_size::_4>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = fixed_delay_constructor_t<464, 1165>;
};

template <class Input, class OffsetT>
struct sm90_tuning<Input, OffsetT, input_size::_16, offset_size::_4>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = no_delay_constructor_t<1040>;
};

template <class Input, class OffsetT>
struct sm90_tuning<Input, OffsetT, input_size::_1, offset_size::_8>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 24;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = fixed_delay_constructor_t<4, 285>;
};

template <class Input, class OffsetT>
struct sm90_tuning<Input, OffsetT, input_size::_2, offset_size::_8>
{
  static constexpr int threads                       = 640;
  static constexpr int items                         = 24;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = no_delay_constructor_t<245>;
};

template <class Input, class OffsetT>
struct sm90_tuning<Input, OffsetT, input_size::_4, offset_size::_8>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 23;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = no_delay_constructor_t<910>;
};

template <class Input, class OffsetT>
struct sm90_tuning<Input, OffsetT, input_size::_8, offset_size::_8>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 18;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = no_delay_constructor_t<1145>;
};

template <class Input, class OffsetT>
struct sm90_tuning<Input, OffsetT, input_size::_16, offset_size::_8>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = no_delay_constructor_t<1050>;
};

template <class InputT,
          class OffsetT,
          input_size InputSize   = classify_input_size<InputT>(),
          offset_size OffsetSize = classify_offset_size<OffsetT>()>
struct sm100_tuning;

// This tuning regressed during validation, so we disabled it and fall back to the SM90 tuning
// template <class Input, class OffsetT>
// struct sm100_tuning<Input, OffsetT, input_size::_1, offset_size::_4>
// {
//   // trp_0.ipt_12.tpb_256.ns_792.dcid_6.l2w_365 1.063960  0.978016  1.072833  1.301435
//   static constexpr int items                         = 12;
//   static constexpr int threads                       = 256;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
//   using delay_constructor                            = exponential_backon_jitter_constructor_t<792, 365>;
// };

// This tuning regressed during validation, so we disabled it and fall back to the SM90 tuning
// template <class Input, class OffsetT>
// struct sm100_tuning<Input, OffsetT, input_size::_2, offset_size::_4>
// {
//   // trp_1.ipt_14.tpb_288.ns_496.dcid_6.l2w_400 1.170449  1.123515  1.170428  1.252066
//   static constexpr int items                         = 14;
//   static constexpr int threads                       = 288;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
//   using delay_constructor                            = exponential_backon_jitter_constructor_t<496, 400>;
// };

template <class Input, class OffsetT>
struct sm100_tuning<Input, OffsetT, input_size::_4, offset_size::_4>
{
  // trp_0.ipt_11.tpb_512.ns_72.dcid_6.l2w_840 1.261035  1.069054  1.243873  1.394013
  static constexpr int items                         = 11;
  static constexpr int threads                       = 512;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = exponential_backon_jitter_constructor_t<72, 840>;
};

template <class Input, class OffsetT>
struct sm100_tuning<Input, OffsetT, input_size::_8, offset_size::_4>
{
  // trp_1.ipt_10.tpb_256.ns_8.dcid_6.l2w_845 1.137286  1.105647  1.140905  1.194373
  static constexpr int items                         = 10;
  static constexpr int threads                       = 256;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = exponential_backon_jitter_constructor_t<8, 845>;
};

// todo(gonidelis): Add tunings for I128.
// template <class Input, class OffsetT>
// struct sm90_tuning<Input, OffsetT, input_size::_16, offset_size::_4>
// {
//   static constexpr int threads                       = 128;
//   static constexpr int items                         = 7;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
//   using delay_constructor                            = no_delay_constructor_t<1040>;
// };

// template <class Input, class OffsetT>
// struct sm100_tuning<Input, OffsetT, input_size::_1, offset_size::_8>
// {
//   // trp_1.ipt_20.tpb_768.ns_444.dcid_5.l2w_330 1.510085  0.887070  1.446621  1.982442
//   static constexpr int items                         = 20;
//   static constexpr int threads                       = 768;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
//   using delay_constructor                            = exponential_backon_jitter_window_constructor_t<444, 330>;
// };

template <class Input, class OffsetT>
struct sm100_tuning<Input, OffsetT, input_size::_2, offset_size::_8>
{
  // trp_1.ipt_20.tpb_768.ns_544.dcid_5.l2w_500 1.064438  1.000000  1.069149  1.200658
  static constexpr int items                         = 20;
  static constexpr int threads                       = 768;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<544, 500>;
};

template <class Input, class OffsetT>
struct sm100_tuning<Input, OffsetT, input_size::_4, offset_size::_8>
{
  // trp_1.ipt_15.tpb_768.ns_144.dcid_6.l2w_280 1.099504  1.002083  1.095122  1.352941
  static constexpr int items                         = 15;
  static constexpr int threads                       = 768;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = exponential_backon_jitter_constructor_t<144, 280>;
};

template <class Input, class OffsetT>
struct sm100_tuning<Input, OffsetT, input_size::_8, offset_size::_8>
{
  // trp_1.ipt_14.tpb_320.ns_872.dcid_7.l2w_620 1.083194  1.000000  1.078944  1.315789
  static constexpr int items                         = 14;
  static constexpr int threads                       = 320;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = exponential_backon_constructor_t<872, 620>;
};

// todo(gonidelis): Add tunings for I128.
// template <class Input, class OffsetT>
// struct sm90_tuning<Input, OffsetT, input_size::_16, offset_size::_8>
// {
//   static constexpr int threads                       = 128;
//   static constexpr int items                         = 7;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
//   using delay_constructor                            = no_delay_constructor_t<1040>;
// };

template <class InputT, class OffsetT>
struct policy_hub
{
  template <typename DelayConstructor>
  struct DefaultPolicy
  {
    using ThreeWayPartitionPolicy =
      AgentThreeWayPartitionPolicy<256,
                                   Nominal4BItemsToItems<InputT>(9),
                                   BLOCK_LOAD_DIRECT,
                                   LOAD_DEFAULT,
                                   BLOCK_SCAN_WARP_SCANS,
                                   DelayConstructor>;
  };

  // nvbug5935129: GCC-11.2 cannot directly use DefaultPolicy inside Policy500
  using DefaultPolicy500 = DefaultPolicy<fixed_delay_constructor_t<350, 450>>;

  struct Policy500
      : DefaultPolicy500
      , ChainedPolicy<500, Policy500, Policy500>
  {};

  // Use values from tuning if a specialization exists, otherwise pick DefaultPolicy
  template <typename Tuning>
  static _CCCL_HOST_DEVICE auto select_agent_policy(int)
    -> AgentThreeWayPartitionPolicy<Tuning::threads,
                                    Tuning::items,
                                    Tuning::load_algorithm,
                                    LOAD_DEFAULT,
                                    BLOCK_SCAN_WARP_SCANS,
                                    typename Tuning::delay_constructor>;

  template <typename Tuning>
  static _CCCL_HOST_DEVICE auto select_agent_policy(long) -> typename DefaultPolicy<
    default_delay_constructor_t<typename accumulator_pack_t<OffsetT>::pack_t>>::ThreeWayPartitionPolicy;

  struct Policy800 : ChainedPolicy<800, Policy800, Policy500>
  {
    using ThreeWayPartitionPolicy = decltype(select_agent_policy<sm80_tuning<InputT, OffsetT>>(0));
  };

  // nvbug5935129: GCC-11.2 cannot directly use DefaultPolicy inside Policy860
  using DefaultPolicy860 = DefaultPolicy<fixed_delay_constructor_t<350, 450>>;

  struct Policy860
      : DefaultPolicy860
      , ChainedPolicy<860, Policy860, Policy800>
  {};

  struct Policy900 : ChainedPolicy<900, Policy900, Policy860>
  {
    using ThreeWayPartitionPolicy = decltype(select_agent_policy<sm90_tuning<InputT, OffsetT>>(0));
  };

  struct Policy1000 : ChainedPolicy<1000, Policy1000, Policy900>
  {
    // Use values from tuning if a specialization exists, otherwise pick Policy900
    template <typename Tuning>
    static _CCCL_HOST_DEVICE auto select_agent_policy100(int)
      -> AgentThreeWayPartitionPolicy<Tuning::threads,
                                      Tuning::items,
                                      Tuning::load_algorithm,
                                      LOAD_DEFAULT,
                                      BLOCK_SCAN_WARP_SCANS,
                                      typename Tuning::delay_constructor>;

    template <typename Tuning>
    static _CCCL_HOST_DEVICE auto select_agent_policy100(long) -> typename Policy900::ThreeWayPartitionPolicy;

    using ThreeWayPartitionPolicy = decltype(select_agent_policy100<sm100_tuning<InputT, OffsetT>>(0));
  };

  using MaxPolicy = Policy1000;
};

struct three_way_partition_policy
{
  int block_threads;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockScanAlgorithm block_scan_algorithm;
  delay_constructor_policy delay_constructor;

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator==(const three_way_partition_policy& lhs, const three_way_partition_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.items_per_thread == rhs.items_per_thread
        && lhs.load_algorithm == rhs.load_algorithm && lhs.load_modifier == rhs.load_modifier
        && lhs.block_scan_algorithm == rhs.block_scan_algorithm && lhs.delay_constructor == rhs.delay_constructor;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator!=(const three_way_partition_policy& lhs, const three_way_partition_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const three_way_partition_policy& policy)
  {
    return os
        << "three_way_partition_policy { .block_threads = " << policy.block_threads
        << ", .items_per_thread = " << policy.items_per_thread << ", .load_algorithm = " << policy.load_algorithm
        << ", .load_modifier = " << policy.load_modifier << ", .block_scan_algorithm = " << policy.block_scan_algorithm
        << ", .delay_constructor = " << policy.delay_constructor << " }";
  }
#endif // _CCCL_HOSTED()
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept three_way_partition_policy_selector = policy_selector<T, three_way_partition_policy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  type_t input_type;
  int input_size;
  int offset_size;

  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> three_way_partition_policy
  {
    const auto default_policy = three_way_partition_policy{
      256,
      nominal_4B_items_to_items(9, input_size),
      BLOCK_LOAD_DIRECT,
      LOAD_DEFAULT,
      BLOCK_SCAN_WARP_SCANS,
      default_delay_constructor_policy(true)}; // we assume that the OffsetT is trivially copyable

    if (arch >= ::cuda::arch_id::sm_100)
    {
      // offset_size == 4 && input_size == 1
      // trp_0.ipt_12.tpb_256.ns_792.dcid_6.l2w_365 1.063960  0.978016  1.072833  1.301435
      // This tuning regressed during validation, so we disabled it and fall back to the SM90 tuning

      // offset_size == 4 && input_size == 2
      // trp_1.ipt_14.tpb_288.ns_496.dcid_6.l2w_400 1.170449  1.123515  1.170428  1.252066
      // This tuning regressed during validation, so we disabled it and fall back to the SM90 tuning

      if (offset_size == 4 && input_size == 4)
      {
        return three_way_partition_policy{
          512,
          11,
          BLOCK_LOAD_DIRECT,
          LOAD_DEFAULT,
          BLOCK_SCAN_WARP_SCANS,
          delay_constructor_policy{delay_constructor_kind::exponential_backon_jitter, 72, 840}};
      }
      if (offset_size == 4 && input_size == 8)
      {
        return three_way_partition_policy{
          256,
          10,
          BLOCK_LOAD_WARP_TRANSPOSE,
          LOAD_DEFAULT,
          BLOCK_SCAN_WARP_SCANS,
          delay_constructor_policy{delay_constructor_kind::exponential_backon_jitter, 8, 845}};
      }

      // TODO(gonidelis): Add tunings for I128.

      if (offset_size == 8 && input_size == 2)
      {
        // trp_1.ipt_20.tpb_768.ns_544.dcid_5.l2w_500 1.064438  1.000000  1.069149  1.200658
        return three_way_partition_policy{
          768,
          20,
          BLOCK_LOAD_WARP_TRANSPOSE,
          LOAD_DEFAULT,
          BLOCK_SCAN_WARP_SCANS,
          delay_constructor_policy{delay_constructor_kind::exponential_backon_jitter_window, 544, 500}};
      }

      if (offset_size == 8 && input_size == 4)
      {
        // trp_1.ipt_15.tpb_768.ns_144.dcid_6.l2w_280 1.099504  1.002083  1.095122  1.352941
        return three_way_partition_policy{
          768,
          15,
          BLOCK_LOAD_WARP_TRANSPOSE,
          LOAD_DEFAULT,
          BLOCK_SCAN_WARP_SCANS,
          delay_constructor_policy{delay_constructor_kind::exponential_backon_jitter, 144, 280}};
      }

      if (offset_size == 8 && input_size == 8)
      {
        // trp_1.ipt_14.tpb_320.ns_872.dcid_7.l2w_620 1.083194  1.000000  1.078944  1.315789
        return three_way_partition_policy{
          320,
          14,
          BLOCK_LOAD_WARP_TRANSPOSE,
          LOAD_DEFAULT,
          BLOCK_SCAN_WARP_SCANS,
          delay_constructor_policy{delay_constructor_kind::exponential_backon, 872, 620}};
      }

      // TODO(gonidelis): Add tunings for I128.

      // fall through to SM90
    }

    if (arch >= ::cuda::arch_id::sm_90)
    {
      if (offset_size == 4 && input_size == 1)
      {
        return three_way_partition_policy{
          256,
          12,
          BLOCK_LOAD_DIRECT,
          LOAD_DEFAULT,
          BLOCK_SCAN_WARP_SCANS,
          delay_constructor_policy{delay_constructor_kind::no_delay, 0, 445}};
      }
      if (offset_size == 4 && input_size == 2)
      {
        return three_way_partition_policy{
          256,
          12,
          BLOCK_LOAD_DIRECT,
          LOAD_DEFAULT,
          BLOCK_SCAN_WARP_SCANS,
          delay_constructor_policy{delay_constructor_kind::fixed_delay, 104, 512}};
      }
      if (offset_size == 4 && input_size == 4)
      {
        return three_way_partition_policy{
          320,
          12,
          BLOCK_LOAD_DIRECT,
          LOAD_DEFAULT,
          BLOCK_SCAN_WARP_SCANS,
          delay_constructor_policy{delay_constructor_kind::no_delay, 0, 1105}};
      }
      if (offset_size == 4 && input_size == 8)
      {
        return three_way_partition_policy{
          384,
          7,
          BLOCK_LOAD_WARP_TRANSPOSE,
          LOAD_DEFAULT,
          BLOCK_SCAN_WARP_SCANS,
          delay_constructor_policy{delay_constructor_kind::fixed_delay, 464, 1165}};
      }
      if (offset_size == 4 && input_size == 16)
      {
        return three_way_partition_policy{
          128,
          7,
          BLOCK_LOAD_WARP_TRANSPOSE,
          LOAD_DEFAULT,
          BLOCK_SCAN_WARP_SCANS,
          delay_constructor_policy{delay_constructor_kind::no_delay, 0, 1040}};
      }
      if (offset_size == 8 && input_size == 1)
      {
        return three_way_partition_policy{
          256,
          24,
          BLOCK_LOAD_DIRECT,
          LOAD_DEFAULT,
          BLOCK_SCAN_WARP_SCANS,
          delay_constructor_policy{delay_constructor_kind::fixed_delay, 4, 285}};
      }
      if (offset_size == 8 && input_size == 2)
      {
        return three_way_partition_policy{
          640,
          24,
          BLOCK_LOAD_WARP_TRANSPOSE,
          LOAD_DEFAULT,
          BLOCK_SCAN_WARP_SCANS,
          delay_constructor_policy{delay_constructor_kind::no_delay, 0, 245}};
      }
      if (offset_size == 8 && input_size == 4)
      {
        return three_way_partition_policy{
          256,
          23,
          BLOCK_LOAD_WARP_TRANSPOSE,
          LOAD_DEFAULT,
          BLOCK_SCAN_WARP_SCANS,
          delay_constructor_policy{delay_constructor_kind::no_delay, 0, 910}};
      }
      if (offset_size == 8 && input_size == 8)
      {
        return three_way_partition_policy{
          256,
          18,
          BLOCK_LOAD_WARP_TRANSPOSE,
          LOAD_DEFAULT,
          BLOCK_SCAN_WARP_SCANS,
          delay_constructor_policy{delay_constructor_kind::no_delay, 0, 1145}};
      }
      if (offset_size == 8 && input_size == 16)
      {
        return three_way_partition_policy{
          256,
          11,
          BLOCK_LOAD_WARP_TRANSPOSE,
          LOAD_DEFAULT,
          BLOCK_SCAN_WARP_SCANS,
          delay_constructor_policy{delay_constructor_kind::no_delay, 0, 1050}};
      }
      return default_policy;
    }

    if (arch >= ::cuda::arch_id::sm_86)
    {
      return default_policy;
    }

    if (arch >= ::cuda::arch_id::sm_80)
    {
      if (offset_size == 4 && input_size == 2)
      {
        return three_way_partition_policy{
          256,
          12,
          BLOCK_LOAD_WARP_TRANSPOSE,
          LOAD_DEFAULT,
          BLOCK_SCAN_WARP_SCANS,
          delay_constructor_policy{delay_constructor_kind::no_delay, 0, 910}};
      }
      if (offset_size == 4 && input_size == 4)
      {
        return three_way_partition_policy{
          256,
          11,
          BLOCK_LOAD_WARP_TRANSPOSE,
          LOAD_DEFAULT,
          BLOCK_SCAN_WARP_SCANS,
          delay_constructor_policy{delay_constructor_kind::no_delay, 0, 1120}};
      }
      if (offset_size == 4 && input_size == 8)
      {
        return three_way_partition_policy{
          224,
          11,
          BLOCK_LOAD_WARP_TRANSPOSE,
          LOAD_DEFAULT,
          BLOCK_SCAN_WARP_SCANS,
          delay_constructor_policy{delay_constructor_kind::fixed_delay, 264, 1080}};
      }
      if (offset_size == 4 && input_size == 16)
      {
        return three_way_partition_policy{
          128,
          10,
          BLOCK_LOAD_WARP_TRANSPOSE,
          LOAD_DEFAULT,
          BLOCK_SCAN_WARP_SCANS,
          delay_constructor_policy{delay_constructor_kind::fixed_delay, 672, 1120}};
      }
      return default_policy;
    }

    // from SM50
    return default_policy;
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(three_way_partition_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

template <typename InputT, typename OffsetT>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> three_way_partition_policy
  {
    constexpr auto selector = policy_selector{classify_type<InputT>, int{sizeof(InputT)}, int{sizeof(OffsetT)}};
    return selector(arch);
  }
};
} // namespace detail::three_way_partition

CUB_NAMESPACE_END
