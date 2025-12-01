// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include <cub/agent/agent_reduce.cuh>
#include <cub/util_device.cuh>
#include <cub/util_macro.cuh>

#include <cuda/__device/arch_id.h>
#include <cuda/std/optional>

#if _CCCL_HAS_CONCEPTS()
#  include <cuda/std/concepts>
#endif // _CCCL_HAS_CONCEPTS()

#if !_CCCL_COMPILER(NVRTC)
#  include <ostream>
#endif

CUB_NAMESPACE_BEGIN
namespace detail
{
namespace reduce
{
// TODO(bgruber): bikeshed name before we make the tuning API public
struct agent_reduce_policy // equivalent of AgentReducePolicy
{
  int block_threads;
  int items_per_thread;
  int vector_load_length;
  BlockReduceAlgorithm block_algorithm;
  CacheLoadModifier load_modifier;

  _CCCL_API constexpr friend bool operator==(const agent_reduce_policy& lhs, const agent_reduce_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.items_per_thread == rhs.items_per_thread
        && lhs.vector_load_length == rhs.vector_load_length && lhs.block_algorithm == rhs.block_algorithm
        && lhs.load_modifier == rhs.load_modifier;
  }

  _CCCL_API constexpr friend bool operator!=(const agent_reduce_policy& lhs, const agent_reduce_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const agent_reduce_policy& p)
  {
    return os << "agent_reduce_policy { .block_threads = " << p.block_threads
              << ", .items_per_thread = " << p.items_per_thread << ", .vector_load_length = " << p.vector_load_length
              << ", .block_algorithm = " << p.block_algorithm << ", .load_modifier = " << p.load_modifier << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

// TODO(bgruber): bikeshed name before we make the tuning API public
struct reduce_arch_policy // equivalent of a policy for a single CUDA architecture
{
  agent_reduce_policy reduce_policy;
  agent_reduce_policy single_tile_policy;
  agent_reduce_policy segmented_reduce_policy;
  agent_reduce_policy reduce_nondeterministic_policy;

  _CCCL_API constexpr friend bool operator==(const reduce_arch_policy& lhs, const reduce_arch_policy& rhs)
  {
    return lhs.reduce_policy == rhs.reduce_policy && lhs.single_tile_policy == rhs.single_tile_policy
        && lhs.segmented_reduce_policy == rhs.segmented_reduce_policy
        && lhs.reduce_nondeterministic_policy == rhs.reduce_nondeterministic_policy;
  }

  _CCCL_API constexpr friend bool operator!=(const reduce_arch_policy& lhs, const reduce_arch_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const reduce_arch_policy& p)
  {
    return os << "reduce_arch_policy { .reduce_policy = " << p.reduce_policy << ", .single_tile_policy = "
              << p.single_tile_policy << ", .segmented_reduce_policy = " << p.segmented_reduce_policy
              << ", .reduce_nondeterministic_policy = " << p.reduce_nondeterministic_policy << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

#if _CCCL_HAS_CONCEPTS()
_CCCL_API consteval void __needs_a_constexpr_value(auto) {}

// TODO(bgruber): bikeshed name before we make the tuning API public
template <typename T>
concept reduce_policy_hub = requires(T hub, ::cuda::arch_id arch) {
  { hub(arch) } -> _CCCL_CONCEPT_VSTD::same_as<reduce_arch_policy>;
  { __needs_a_constexpr_value(hub(arch)) };
};
#endif // _CCCL_HAS_CONCEPTS()

template <typename PolicyT, typename = void>
struct ReducePolicyWrapper : PolicyT
{
  _CCCL_HOST_DEVICE ReducePolicyWrapper(PolicyT base)
      : PolicyT(base)
  {}
};

template <typename StaticPolicyT>
struct ReducePolicyWrapper<StaticPolicyT,
                           ::cuda::std::void_t<typename StaticPolicyT::ReducePolicy,
                                               typename StaticPolicyT::SingleTilePolicy,
                                               typename StaticPolicyT::SegmentedReducePolicy>> : StaticPolicyT
{
  _CCCL_HOST_DEVICE ReducePolicyWrapper(StaticPolicyT base)
      : StaticPolicyT(base)
  {}

  CUB_DEFINE_SUB_POLICY_GETTER(Reduce)
  CUB_DEFINE_SUB_POLICY_GETTER(SingleTile)
  CUB_DEFINE_SUB_POLICY_GETTER(SegmentedReduce)
  CUB_DEFINE_SUB_POLICY_GETTER(ReduceNondeterministic)

  // TODO(bgruber): no longer needed by CCCL.C for reduce, but still needed for segmented_reduce
#if defined(CUB_ENABLE_POLICY_PTX_JSON)
  _CCCL_DEVICE static constexpr auto EncodedPolicy()
  {
    using namespace ptx_json;
    return object<key<"ReducePolicy">()                 = Reduce().EncodedPolicy(),
                  key<"SingleTilePolicy">()             = SingleTile().EncodedPolicy(),
                  key<"SegmentedReducePolicy">()        = SegmentedReduce().EncodedPolicy(),
                  key<"ReduceNondeterministicPolicy">() = ReduceNondeterministic().EncodedPolicy()>();
  }
#endif
};

template <typename PolicyT>
_CCCL_HOST_DEVICE ReducePolicyWrapper<PolicyT> MakeReducePolicyWrapper(PolicyT policy)
{
  return ReducePolicyWrapper<PolicyT>{policy};
}

enum class offset_size
{
  _4,
  _8,
  unknown
};
enum class op_type
{
  plus,
  min_or_max,
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
_CCCL_HOST_DEVICE constexpr accum_size classify_accum_size()
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
_CCCL_HOST_DEVICE constexpr offset_size classify_offset_size()
{
  return sizeof(OffsetT) == 4 ? offset_size::_4 : sizeof(OffsetT) == 8 ? offset_size::_8 : offset_size::unknown;
}

template <typename Op>
struct is_plus
{
  static constexpr bool value = false;
};

template <typename T>
struct is_plus<::cuda::std::plus<T>>
{
  static constexpr bool value = true;
};
template <typename Op>
struct is_min_or_max
{
  static constexpr bool value = false;
};
template <typename T>
struct is_min_or_max<::cuda::minimum<T>>
{
  static constexpr bool value = true;
};
template <typename T>
struct is_min_or_max<::cuda::maximum<T>>
{
  static constexpr bool value = true;
};

template <class ScanOpT>
_CCCL_HOST_DEVICE constexpr op_type classify_op()
{
  return is_plus<ScanOpT>::value
         ? op_type::plus
         : (is_min_or_max<ScanOpT>::value ? op_type::min_or_max : op_type::unknown);
}

template <class AccumT,
          class OffsetT,
          op_type OpTypeT        = classify_op<OffsetT>(),
          offset_size OffsetSize = classify_offset_size<OffsetT>(),
          accum_size AccumSize   = classify_accum_size<AccumT>()>
struct sm100_tuning;

// sum

// Tunings for offset size 4/8 and accum size 1/2/4 all showed no significant improvement during verification

template <class T, class OffsetT>
struct sm100_tuning<T, OffsetT, op_type::plus, offset_size::_4, accum_size::_8>
{
  // ipt_15.tpb_512.ipv_2 1.019887   1.0  1.017636  1.058036
  static constexpr int items              = 15;
  static constexpr int threads            = 512;
  static constexpr int items_per_vec_load = 2;
};

template <class T, class OffsetT>
struct sm100_tuning<T, OffsetT, op_type::plus, offset_size::_8, accum_size::_8>
{
  // ipt_15.tpb_512.ipv_1 1.019414  1.000000  1.017218  1.057143
  static constexpr int items              = 15;
  static constexpr int threads            = 512;
  static constexpr int items_per_vec_load = 1;
};

template <class OffsetT>
struct sm100_tuning<float, OffsetT, op_type::plus, offset_size::_4, accum_size::_4>
{
  // ipt_16.tpb_512.ipv_2 1.061295  1.000000  1.065478  1.167139
  static constexpr int items              = 16;
  static constexpr int threads            = 512;
  static constexpr int items_per_vec_load = 2;
};

template <class OffsetT>
struct sm100_tuning<double, OffsetT, op_type::plus, offset_size::_4, accum_size::_8>
{
  // ipt_16.tpb_640.ipv_1 1.017834  1.000000  1.015835  1.057092
  static constexpr int items              = 16;
  static constexpr int threads            = 640;
  static constexpr int items_per_vec_load = 1;
};

// TODO(bgruber): we should have a more central enum for types, like cccl_type_enum in CCCL.C
enum class accum_type
{
  float32,
  float64,
  other,
};

template <typename AccumT>
_CCCL_HOST_DEVICE constexpr accum_type classify_accum_type()
{
  return ::cuda::std::is_same_v<AccumT, float> ? accum_type::float32
       : ::cuda::std::is_same_v<AccumT, double>
         ? accum_type::float64
         : accum_type::other;
}

struct sm100_tuning_values
{
  int items;
  int threads;
  int items_per_vec_load;
};

_CCCL_API constexpr auto get_sm100_tuning(accum_type accum_t, op_type operation_t, int offset_size, int accum_size)
  -> ::cuda::std::optional<sm100_tuning_values>
{
  if (operation_t != op_type::plus)
  {
    return {};
  }

  if (accum_t == accum_type::float32 && offset_size == 4 && accum_size == 4)
  {
    return sm100_tuning_values{16, 512, 2};
  }
  if (accum_t == accum_type::float64 && offset_size == 4 && accum_size == 8)
  {
    return sm100_tuning_values{16, 640, 1};
  }
  if (offset_size == 4 && accum_size == 8)
  {
    return sm100_tuning_values{15, 512, 2};
  }
  if (offset_size == 8 && accum_size == 8)
  {
    return sm100_tuning_values{15, 512, 1};
  }

  return {};
}

// For min or max, verification showed the benefits were too small (within noise)

// TODO(bgruber): drop after migrating DispatchSegmentedReduce to the new tuning API
template <typename AccumT, typename OffsetT, typename ReductionOpT>
struct policy_hub
{
  struct Policy500 : ChainedPolicy<500, Policy500, Policy500>
  {
    static constexpr int threads_per_block  = 256;
    static constexpr int items_per_thread   = 20;
    static constexpr int items_per_vec_load = 4;

    // ReducePolicy (GTX Titan: 255.1 GB/s @ 48M 4B items; 228.7 GB/s @ 192M 1B items)
    using ReducePolicy =
      AgentReducePolicy<threads_per_block,
                        items_per_thread,
                        AccumT,
                        items_per_vec_load,
                        BLOCK_REDUCE_WARP_REDUCTIONS,
                        LOAD_LDG>;

    using SingleTilePolicy      = ReducePolicy;
    using SegmentedReducePolicy = ReducePolicy;

    using ReduceNondeterministicPolicy =
      AgentReducePolicy<ReducePolicy::BLOCK_THREADS,
                        ReducePolicy::ITEMS_PER_THREAD,
                        AccumT,
                        ReducePolicy::VECTOR_LOAD_LENGTH,
                        BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC,
                        ReducePolicy::LOAD_MODIFIER,
                        NoScaling<ReducePolicy::BLOCK_THREADS, ReducePolicy::ITEMS_PER_THREAD>>;
  };

  struct Policy600 : ChainedPolicy<600, Policy600, Policy500>
  {
    static constexpr int threads_per_block  = 256;
    static constexpr int items_per_thread   = 16;
    static constexpr int items_per_vec_load = 4;

    // ReducePolicy (P100: 591 GB/s @ 64M 4B items; 583 GB/s @ 256M 1B items)
    using ReducePolicy =
      AgentReducePolicy<threads_per_block,
                        items_per_thread,
                        AccumT,
                        items_per_vec_load,
                        BLOCK_REDUCE_WARP_REDUCTIONS,
                        LOAD_LDG>;

    using SingleTilePolicy      = ReducePolicy;
    using SegmentedReducePolicy = ReducePolicy;

    using ReduceNondeterministicPolicy =
      AgentReducePolicy<ReducePolicy::BLOCK_THREADS,
                        ReducePolicy::ITEMS_PER_THREAD,
                        AccumT,
                        ReducePolicy::VECTOR_LOAD_LENGTH,
                        BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC,
                        ReducePolicy::LOAD_MODIFIER,
                        NoScaling<ReducePolicy::BLOCK_THREADS, ReducePolicy::ITEMS_PER_THREAD>>;
  };

  struct Policy1000 : ChainedPolicy<1000, Policy1000, Policy600>
  {
    // Use values from tuning if a specialization exists, otherwise pick Policy600
    template <typename Tuning>
    static _CCCL_HOST_DEVICE auto select_agent_policy(int)
      -> AgentReducePolicy<Tuning::threads,
                           Tuning::items,
                           AccumT,
                           Tuning::items_per_vec_load,
                           BLOCK_REDUCE_WARP_REDUCTIONS,
                           LOAD_LDG>;
    // use Policy600 as DefaultPolicy
    template <typename Tuning>
    static _CCCL_HOST_DEVICE auto select_agent_policy(long) -> typename Policy600::ReducePolicy;

    using ReducePolicy =
      decltype(select_agent_policy<sm100_tuning<AccumT,
                                                OffsetT,
                                                classify_op<ReductionOpT>(),
                                                classify_offset_size<OffsetT>(),
                                                classify_accum_size<AccumT>()>>(0));

    using SingleTilePolicy      = ReducePolicy;
    using SegmentedReducePolicy = ReducePolicy;

    using ReduceNondeterministicPolicy =
      AgentReducePolicy<ReducePolicy::BLOCK_THREADS,
                        ReducePolicy::ITEMS_PER_THREAD,
                        AccumT,
                        ReducePolicy::VECTOR_LOAD_LENGTH,
                        BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC,
                        ReducePolicy::LOAD_MODIFIER,
                        NoScaling<ReducePolicy::BLOCK_THREADS, ReducePolicy::ITEMS_PER_THREAD>>;
  };

  using MaxPolicy = Policy1000;
};

struct arch_policies // equivalent to the policy_hub, holds policies for a bunch of CUDA architectures
{
  accum_type accum_t; // TODO(bgruber): accum_type should become some CCCL global enum
  op_type operation_t; // TODO(bgruber): op_type should become some CCCL global enum
  int offset_size;
  int accum_size;

  // IDEA(bgruber): instead of the constexpr function, we could also provide a map<int, reduce_arch_policy> and move the
  // selection mechanism elsewhere

  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> reduce_arch_policy
  {
    // if we don't have a tuning for sm100, fall through
    auto sm100_tuning = get_sm100_tuning(accum_t, operation_t, offset_size, accum_size);
    if (arch >= ::cuda::arch_id::sm_100 && sm100_tuning)
    {
      agent_reduce_policy rp{};
      auto [scaled_items, scaled_threads] = scale_mem_bound(sm100_tuning->threads, sm100_tuning->items, accum_size);
      rp                                  = agent_reduce_policy{
        scaled_threads, scaled_items, sm100_tuning->items_per_vec_load, BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};

      auto rp_nondet            = rp;
      rp_nondet.block_algorithm = BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC;
      return {rp, rp, rp, rp_nondet};
    }

    if (arch >= ::cuda::arch_id::sm_60)
    {
      constexpr int threads_per_block  = 256;
      constexpr int items_per_thread   = 16;
      constexpr int items_per_vec_load = 4;

      // ReducePolicy (P100: 591 GB/s @ 64M 4B items; 583 GB/s @ 256M 1B items)
      auto [scaled_items, scaled_threads] = scale_mem_bound(threads_per_block, items_per_thread, accum_size);
      const auto rp =
        agent_reduce_policy{scaled_threads, scaled_items, items_per_vec_load, BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};

      auto rp_nondet            = rp;
      rp_nondet.block_algorithm = BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC;
      return {rp, rp, rp, rp_nondet};
    }

    // base policy is for 500
    // GTX Titan: 255.1 GB/s @ 48M 4B items; 228.7 GB/s @ 192M 1B items
    constexpr int threads_per_block  = 256;
    constexpr int items_per_thread   = 20;
    constexpr int items_per_vec_load = 4;

    auto [scaled_items, scaled_threads] = scale_mem_bound(threads_per_block, items_per_thread, accum_size);
    const auto rp =
      agent_reduce_policy{scaled_threads, scaled_items, items_per_vec_load, BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};

    auto rp_nondet            = rp;
    rp_nondet.block_algorithm = BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC;
    return {rp, rp, rp, rp_nondet};
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(reduce_policy_hub<arch_policies>);
#endif // _CCCL_HAS_CONCEPTS()

// stateless version which can be passed to kernels
template <typename AccumT, typename OffsetT, typename ReductionOpT>
struct arch_policies_from_types
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> reduce_arch_policy
  {
    constexpr auto policies = arch_policies{
      classify_accum_type<AccumT>(), classify_op<ReductionOpT>(), int{sizeof(OffsetT)}, int{sizeof(AccumT)}};
    return policies(arch);
  }
};
} // namespace reduce

namespace rfa
{
template <class AccumT>
struct sm90_tuning;

template <>
struct sm90_tuning<float>
{
  // ipt_13.tpb_224  1.107188  1.009709  1.097114  1.316820
  static constexpr int items   = 13;
  static constexpr int threads = 224;
};

template <class AccumT>
struct sm86_tuning;

template <>
struct sm86_tuning<float>
{
  // ipt_6.tpb_224  1.034383  1.000000  1.032097  1.090909
  static constexpr int items   = 6;
  static constexpr int threads = 224;
};

template <>
struct sm86_tuning<double>
{
  // ipt_11.tpb_128 ()  1.232089  1.002124  1.245336  1.582279
  static constexpr int items   = 11;
  static constexpr int threads = 128;
};

/**
 * @tparam AccumT
 *   Accumulator data type
 *
 * OffsetT
 *   Signed integer type for global offsets
 *
 * ReductionOpT
 *   Binary reduction functor type having member
 *   `auto operator()(const T &a, const U &b)`
 */
template <typename AccumT, typename OffsetT, typename ReductionOpT>
struct policy_hub
{
  //---------------------------------------------------------------------------
  // Architecture-specific tuning policies
  //---------------------------------------------------------------------------

  /// SM50
  struct Policy500 : ChainedPolicy<500, Policy500, Policy500>
  {
    static constexpr int threads_per_block  = 256;
    static constexpr int items_per_thread   = 20;
    static constexpr int items_per_vec_load = 4;

    // ReducePolicy (GTX Titan: 255.1 GB/s @ 48M 4B items; 228.7 GB/s @ 192M 1B
    // items)
    using ReducePolicy =
      AgentReducePolicy<threads_per_block,
                        items_per_thread,
                        AccumT,
                        items_per_vec_load,
                        BLOCK_REDUCE_WARP_REDUCTIONS,
                        LOAD_LDG>;

    // SingleTilePolicy
    using SingleTilePolicy = ReducePolicy;
  };

  /// SM60
  struct Policy600 : ChainedPolicy<600, Policy600, Policy500>
  {
    static constexpr int threads_per_block  = 256;
    static constexpr int items_per_thread   = 16;
    static constexpr int items_per_vec_load = 4;

    // ReducePolicy (P100: 591 GB/s @ 64M 4B items; 583 GB/s @ 256M 1B items)
    using ReducePolicy =
      AgentReducePolicy<threads_per_block,
                        items_per_thread,
                        AccumT,
                        items_per_vec_load,
                        BLOCK_REDUCE_WARP_REDUCTIONS,
                        LOAD_LDG>;

    // SingleTilePolicy
    using SingleTilePolicy = ReducePolicy;
  };

  /// SM86
  struct Policy860 : ChainedPolicy<860, Policy860, Policy600>
  {
    static constexpr int items_per_vec_load = 4;

    // Use values from tuning if a specialization exists, otherwise pick Policy600
    template <typename Tuning>
    static _CCCL_HOST_DEVICE auto select_agent_policy(int)
      -> AgentReducePolicy<Tuning::threads, Tuning::items, AccumT, items_per_vec_load, BLOCK_REDUCE_RAKING, LOAD_LDG>;

    // use Policy600 as DefaultPolicy
    template <typename Tuning>
    static _CCCL_HOST_DEVICE auto select_agent_policy(long) -> typename Policy600::ReducePolicy;

    using ReducePolicy = decltype(select_agent_policy<sm86_tuning<AccumT>>(0));

    using SingleTilePolicy = ReducePolicy;
  };

  /// SM90
  struct Policy900 : ChainedPolicy<900, Policy900, Policy860>
  {
    static constexpr int items_per_vec_load = 4;

    // Use values from tuning if a specialization exists, otherwise pick Policy860
    template <typename Tuning>
    static _CCCL_HOST_DEVICE auto select_agent_policy(int)
      -> AgentReducePolicy<Tuning::threads, Tuning::items, AccumT, items_per_vec_load, BLOCK_REDUCE_RAKING, LOAD_LDG>;

    // use Policy860 as DefaultPolicy
    template <typename Tuning>
    static _CCCL_HOST_DEVICE auto select_agent_policy(long) -> typename Policy860::ReducePolicy;

    using ReducePolicy = decltype(select_agent_policy<sm90_tuning<AccumT>>(0));

    // SingleTilePolicy
    using SingleTilePolicy = ReducePolicy;
  };

  using MaxPolicy = Policy900;
};
} // namespace rfa

namespace fixed_size_segmented_reduce
{
template <typename AccumT, typename OffsetT, typename ReductionOpT>
struct policy_hub
{
  struct Policy500 : ChainedPolicy<500, Policy500, Policy500>
  {
  private:
    static constexpr int items_per_vec_load = 4;

    static constexpr int small_threads_per_warp  = 1;
    static constexpr int medium_threads_per_warp = 32;

    static constexpr int nominal_4b_large_threads_per_block = 256;

    static constexpr int nominal_4b_small_items_per_thread  = 16;
    static constexpr int nominal_4b_medium_items_per_thread = 16;
    static constexpr int nominal_4b_large_items_per_thread  = 16;

  public:
    using ReducePolicy =
      cub::AgentReducePolicy<nominal_4b_large_threads_per_block,
                             nominal_4b_large_items_per_thread,
                             AccumT,
                             items_per_vec_load,
                             cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                             cub::LOAD_LDG>;

    using SmallReducePolicy =
      cub::AgentWarpReducePolicy<ReducePolicy::BLOCK_THREADS,
                                 small_threads_per_warp,
                                 nominal_4b_small_items_per_thread,
                                 AccumT,
                                 items_per_vec_load,
                                 cub::LOAD_LDG>;

    using MediumReducePolicy =
      cub::AgentWarpReducePolicy<ReducePolicy::BLOCK_THREADS,
                                 medium_threads_per_warp,
                                 nominal_4b_medium_items_per_thread,
                                 AccumT,
                                 items_per_vec_load,
                                 cub::LOAD_LDG>;
  };

  using MaxPolicy = Policy500;
};
} // namespace fixed_size_segmented_reduce
} // namespace detail

CUB_NAMESPACE_END
