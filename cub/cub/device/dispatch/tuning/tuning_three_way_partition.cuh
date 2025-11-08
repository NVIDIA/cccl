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
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::three_way_partition
{
template <typename PolicyT, typename = void>
struct ThreeWayPartitionPolicyWrapper : PolicyT
{
  _CCCL_HOST_DEVICE ThreeWayPartitionPolicyWrapper(PolicyT base)
      : PolicyT(base)
  {}
};

template <typename StaticPolicyT>
struct ThreeWayPartitionPolicyWrapper<StaticPolicyT, ::cuda::std::void_t<typename StaticPolicyT::ThreeWayPartitionPolicy>>
    : StaticPolicyT
{
  _CCCL_HOST_DEVICE ThreeWayPartitionPolicyWrapper(StaticPolicyT base)
      : StaticPolicyT(base)
  {}

  CUB_DEFINE_SUB_POLICY_GETTER(ThreeWayPartition)

#if defined(CUB_ENABLE_POLICY_PTX_JSON)
  _CCCL_DEVICE static constexpr auto EncodedPolicy()
  {
    using namespace ptx_json;
    return object<key<"ThreeWayPartitionPolicy">() = ThreeWayPartition().EncodedPolicy(),
                  key<"DelayConstructor">() =
                    StaticPolicyT::ThreeWayPartitionPolicy::detail::delay_constructor_t::EncodedConstructor()>();
  }
#endif
};

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

  struct Policy500
      : DefaultPolicy<fixed_delay_constructor_t<350, 450>>
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

  struct Policy860
      : DefaultPolicy<fixed_delay_constructor_t<350, 450>>
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
} // namespace detail::three_way_partition

CUB_NAMESPACE_END
