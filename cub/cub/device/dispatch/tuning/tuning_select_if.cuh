/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_select_if.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

namespace detail
{
namespace select
{
enum class flagged
{
  no,
  yes
};
enum class keep_rejects
{
  no,
  yes
};
enum class primitive
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
enum class input_size
{
  _1,
  _2,
  _4,
  _8,
  _16,
  unknown
};

template <class InputT, flagged, keep_rejects, offset_size OffsetSize, primitive, input_size InputSize>
struct sm80_tuning;

// select::if
template <class Input>
struct sm80_tuning<Input, flagged::no, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_1>
{
  static constexpr int threads                       = 992;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<395>;
};

template <class Input>
struct sm80_tuning<Input, flagged::no, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_2>
{
  static constexpr int threads                       = 576;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<870>;
};

template <class Input>
struct sm80_tuning<Input, flagged::no, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 18;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1130>;
};

template <class Input>
struct sm80_tuning<Input, flagged::no, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_8>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<832, 1165>;
};

#if _CCCL_HAS_INT128()
template <>
struct sm80_tuning<__int128_t, flagged::no, keep_rejects::no, offset_size::_4, primitive::no, input_size::_16>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 4;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<1140>;
};

template <>
struct sm80_tuning<__uint128_t, flagged::no, keep_rejects::no, offset_size::_4, primitive::no, input_size::_16>
    : sm80_tuning<__int128_t, flagged::no, keep_rejects::no, offset_size::_4, primitive::no, input_size::_16>
{};
#endif

// select::flagged
template <class Input>
struct sm80_tuning<Input, flagged::yes, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_1>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<735>;
};

template <class Input>
struct sm80_tuning<Input, flagged::yes, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_2>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1155>;
};

template <class Input>
struct sm80_tuning<Input, flagged::yes, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_4>
{
  static constexpr int threads                       = 320;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<124, 1115>;
};

template <class Input>
struct sm80_tuning<Input, flagged::yes, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_8>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 6;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<1130>;
};

#if _CCCL_HAS_INT128()
template <>
struct sm80_tuning<__int128_t, flagged::yes, keep_rejects::no, offset_size::_4, primitive::no, input_size::_16>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 5;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<464, 1025>;
};

template <>
struct sm80_tuning<__uint128_t, flagged::yes, keep_rejects::no, offset_size::_4, primitive::no, input_size::_16>
    : sm80_tuning<__int128_t, flagged::yes, keep_rejects::no, offset_size::_4, primitive::no, input_size::_16>
{};
#endif

// partition::if
template <class Input>
struct sm80_tuning<Input, flagged::no, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_1>
{
  static constexpr int threads                       = 512;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<510>;
};

template <class Input>
struct sm80_tuning<Input, flagged::no, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_2>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 18;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1045>;
};

template <class Input>
struct sm80_tuning<Input, flagged::no, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_4>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 15;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<1040>;
};

template <class Input>
struct sm80_tuning<Input, flagged::no, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_8>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<68, 1160>;
};

#if _CCCL_HAS_INT128()
template <>
struct sm80_tuning<__int128_t, flagged::no, keep_rejects::yes, offset_size::_4, primitive::no, input_size::_16>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 5;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<400, 1090>;
};

template <>
struct sm80_tuning<__uint128_t, flagged::no, keep_rejects::yes, offset_size::_4, primitive::no, input_size::_16>
    : sm80_tuning<__int128_t, flagged::no, keep_rejects::yes, offset_size::_4, primitive::no, input_size::_16>
{};
#endif

// partition::flagged
template <class Input>
struct sm80_tuning<Input, flagged::yes, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_1>
{
  static constexpr int threads                       = 512;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<595>;
};

template <class Input>
struct sm80_tuning<Input, flagged::yes, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_2>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 18;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1105>;
};

template <class Input>
struct sm80_tuning<Input, flagged::yes, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_4>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<912, 1025>;
};

template <class Input>
struct sm80_tuning<Input, flagged::yes, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_8>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<884, 1130>;
};

#if _CCCL_HAS_INT128()
template <>
struct sm80_tuning<__int128_t, flagged::yes, keep_rejects::yes, offset_size::_4, primitive::no, input_size::_16>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 5;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<400, 1090>;
};

template <>
struct sm80_tuning<__uint128_t, flagged::yes, keep_rejects::yes, offset_size::_4, primitive::no, input_size::_16>
    : sm80_tuning<__int128_t, flagged::yes, keep_rejects::yes, offset_size::_4, primitive::no, input_size::_16>
{};
#endif

template <class InputT, flagged, keep_rejects, offset_size OffsetSize, primitive, input_size InputSize>
struct sm90_tuning;

// select::if
template <class Input>
struct sm90_tuning<Input, flagged::no, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 22;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<580>;
};

template <class Input>
struct sm90_tuning<Input, flagged::no, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_2>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 22;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<320, 605>;
};

template <class Input>
struct sm90_tuning<Input, flagged::no, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_4>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 17;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<76, 1150>;
};

template <class Input>
struct sm90_tuning<Input, flagged::no, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_8>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<380, 1140>;
};

#if _CCCL_HAS_INT128()
template <>
struct sm90_tuning<__int128_t, flagged::no, keep_rejects::no, offset_size::_4, primitive::no, input_size::_16>
{
  static constexpr int threads                       = 512;
  static constexpr int items                         = 5;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<460, 1145>;
};

template <>
struct sm90_tuning<__uint128_t, flagged::no, keep_rejects::no, offset_size::_4, primitive::no, input_size::_16>
    : sm90_tuning<__int128_t, flagged::no, keep_rejects::no, offset_size::_4, primitive::no, input_size::_16>
{};
#endif

// select::flagged
template <class Input>
struct sm90_tuning<Input, flagged::yes, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_1>
{
  static constexpr int threads                       = 448;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<715>;
};

template <class Input>
struct sm90_tuning<Input, flagged::yes, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_2>
{
  static constexpr int threads                       = 448;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<504, 765>;
};

template <class Input>
struct sm90_tuning<Input, flagged::yes, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_4>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 15;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<415, 1125>;
};

template <class Input>
struct sm90_tuning<Input, flagged::yes, keep_rejects::no, offset_size::_4, primitive::yes, input_size::_8>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<360, 1170>;
};

#if _CCCL_HAS_INT128()
template <>
struct sm90_tuning<__int128_t, flagged::yes, keep_rejects::no, offset_size::_4, primitive::no, input_size::_16>
{
  static constexpr int threads                       = 512;
  static constexpr int items                         = 3;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<284, 1130>;
};

template <>
struct sm90_tuning<__uint128_t, flagged::yes, keep_rejects::no, offset_size::_4, primitive::no, input_size::_16>
    : sm90_tuning<__int128_t, flagged::yes, keep_rejects::no, offset_size::_4, primitive::no, input_size::_16>
{};
#endif

// partition::if
template <class Input>
struct sm90_tuning<Input, flagged::no, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_1>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<908, 995>;
};

template <class Input>
struct sm90_tuning<Input, flagged::no, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_2>
{
  static constexpr int threads                       = 320;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<500, 560>;
};

template <class Input>
struct sm90_tuning<Input, flagged::no, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<536, 1055>;
};

template <class Input>
struct sm90_tuning<Input, flagged::no, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_8>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<512, 1075>;
};

#if _CCCL_HAS_INT128()
template <>
struct sm90_tuning<__int128_t, flagged::no, keep_rejects::yes, offset_size::_4, primitive::no, input_size::_16>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 5;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<1616, 1115>;
};

template <>
struct sm90_tuning<__uint128_t, flagged::no, keep_rejects::yes, offset_size::_4, primitive::no, input_size::_16>
    : sm90_tuning<__int128_t, flagged::no, keep_rejects::yes, offset_size::_4, primitive::no, input_size::_16>
{};
#endif

// partition::flagged
template <class Input>
struct sm90_tuning<Input, flagged::yes, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<580, 850>;
};

template <class Input>
struct sm90_tuning<Input, flagged::yes, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_2>
{
  static constexpr int threads                       = 512;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<388, 1055>;
};

template <class Input>
struct sm90_tuning<Input, flagged::yes, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<72, 1165>;
};

template <class Input>
struct sm90_tuning<Input, flagged::yes, keep_rejects::yes, offset_size::_4, primitive::yes, input_size::_8>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 6;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<532, 1180>;
};

#if _CCCL_HAS_INT128()
template <>
struct sm90_tuning<__int128_t, flagged::yes, keep_rejects::yes, offset_size::_4, primitive::no, input_size::_16>
{
  static constexpr int threads                       = 160;
  static constexpr int items                         = 5;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<720, 1105>;
};

template <>
struct sm90_tuning<__uint128_t, flagged::yes, keep_rejects::yes, offset_size::_4, primitive::no, input_size::_16>
    : sm90_tuning<__int128_t, flagged::yes, keep_rejects::yes, offset_size::_4, primitive::no, input_size::_16>
{};
#endif

template <class InputT>
constexpr primitive is_primitive()
{
  return Traits<InputT>::PRIMITIVE ? primitive::yes : primitive::no;
}

template <class FlagT>
constexpr flagged is_flagged()
{
  return ::cuda::std::is_same<FlagT, NullType>::value ? flagged::no : flagged::yes;
}

template <bool KeepRejects>
constexpr keep_rejects are_rejects_kept()
{
  return KeepRejects ? keep_rejects::yes : keep_rejects::no;
}

template <class InputT>
constexpr input_size classify_input_size()
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
constexpr offset_size classify_offset_size()
{
  return sizeof(OffsetT) == 4 ? offset_size::_4 : sizeof(OffsetT) == 8 ? offset_size::_8 : offset_size::unknown;
}

template <class InputT, class FlagT, class OffsetT, bool MayAlias, bool KeepRejects>
struct policy_hub
{
  template <CacheLoadModifier LoadModifier>
  struct DefaultPolicy
  {
    static constexpr int nominal_4B_items_per_thread = 10;
    // TODO(bgruber): use cuda::std::clamp() in C++14
    static constexpr int items_per_thread =
      CUB_MIN(nominal_4B_items_per_thread, CUB_MAX(1, (nominal_4B_items_per_thread * 4 / sizeof(InputT))));
    using SelectIfPolicyT =
      AgentSelectIfPolicy<128,
                          items_per_thread,
                          BLOCK_LOAD_DIRECT,
                          LoadModifier,
                          BLOCK_SCAN_WARP_SCANS,
                          detail::fixed_delay_constructor_t<350, 450>>;
  };

  struct Policy500
      : DefaultPolicy<MayAlias ? LOAD_CA : LOAD_LDG>
      , ChainedPolicy<500, Policy500, Policy500>
  {};

  struct Policy800 : ChainedPolicy<800, Policy800, Policy500>
  {
    // Use values from tuning if a specialization exists, otherwise pick the default
    template <typename Tuning>
    static auto select_agent_policy(int)
      -> AgentSelectIfPolicy<Tuning::threads,
                             Tuning::items,
                             Tuning::load_algorithm,
                             LOAD_DEFAULT,
                             BLOCK_SCAN_WARP_SCANS,
                             typename Tuning::delay_constructor>;
    template <typename Tuning>
    static auto select_agent_policy(long) -> typename DefaultPolicy<LOAD_DEFAULT>::SelectIfPolicyT;

    using SelectIfPolicyT =
      decltype(select_agent_policy<sm80_tuning<InputT,
                                               is_flagged<FlagT>(),
                                               are_rejects_kept<KeepRejects>(),
                                               classify_offset_size<OffsetT>(),
                                               is_primitive<InputT>(),
                                               classify_input_size<InputT>()>>(0));
  };

  struct Policy860
      : DefaultPolicy<MayAlias ? LOAD_CA : LOAD_LDG>
      , ChainedPolicy<860, Policy860, Policy800>
  {};

  struct Policy900 : ChainedPolicy<900, Policy900, Policy860>
  {
    // Use values from tuning if a specialization exists, otherwise pick the default
    template <typename Tuning>
    static auto select_agent_policy(int)
      -> AgentSelectIfPolicy<Tuning::threads,
                             Tuning::items,
                             Tuning::load_algorithm,
                             LOAD_DEFAULT,
                             BLOCK_SCAN_WARP_SCANS,
                             typename Tuning::delay_constructor>;
    template <typename Tuning>
    static auto select_agent_policy(long) -> typename DefaultPolicy<LOAD_DEFAULT>::SelectIfPolicyT;

    using SelectIfPolicyT =
      decltype(select_agent_policy<sm90_tuning<InputT,
                                               is_flagged<FlagT>(),
                                               are_rejects_kept<KeepRejects>(),
                                               classify_offset_size<OffsetT>(),
                                               is_primitive<InputT>(),
                                               classify_input_size<InputT>()>>(0));
  };

  using MaxPolicy = Policy900;
};
} // namespace select
} // namespace detail

CUB_NAMESPACE_END
