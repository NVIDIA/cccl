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

#include <cub/agent/agent_three_way_partition.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

namespace detail
{
namespace three_way_partition
{
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

  struct Policy350
      : DefaultPolicy<fixed_delay_constructor_t<350, 450>>
      , ChainedPolicy<350, Policy350, Policy350>
  {};

  // Use values from tuning if a specialization exists, otherwise pick DefaultPolicy
  template <typename Tuning>
  static auto select_agent_policy(int)
    -> AgentThreeWayPartitionPolicy<Tuning::threads,
                                    Tuning::items,
                                    Tuning::load_algorithm,
                                    LOAD_DEFAULT,
                                    BLOCK_SCAN_WARP_SCANS,
                                    typename Tuning::delay_constructor>;

  template <typename Tuning>
  static auto select_agent_policy(long) ->
    typename DefaultPolicy<
      default_delay_constructor_t<typename accumulator_pack_t<OffsetT>::pack_t>>::ThreeWayPartitionPolicy;

  struct Policy800 : ChainedPolicy<800, Policy800, Policy350>
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

  using MaxPolicy = Policy900;
};
} // namespace three_way_partition
} // namespace detail

CUB_NAMESPACE_END
