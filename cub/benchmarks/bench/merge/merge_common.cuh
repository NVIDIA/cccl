// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#if !TUNE_BASE
#  define TUNE_THREADS_PER_BLOCK (1 << TUNE_THREADS_PER_BLOCK_POW2)
#  if TUNE_TRANSPOSE == 0
#    define TUNE_LOAD_ALGORITHM  cub::BLOCK_LOAD_DIRECT
#    define TUNE_STORE_ALGORITHM cub::BLOCK_STORE_DIRECT
#  else // TUNE_TRANSPOSE == 1
#    define TUNE_LOAD_ALGORITHM  cub::BLOCK_LOAD_WARP_TRANSPOSE
#    define TUNE_STORE_ALGORITHM cub::BLOCK_STORE_WARP_TRANSPOSE
#  endif // TUNE_TRANSPOSE

#  if TUNE_LOAD == 0
#    define TUNE_LOAD_MODIFIER cub::LOAD_DEFAULT
#  elif TUNE_LOAD == 1
#    define TUNE_LOAD_MODIFIER cub::LOAD_LDG
#  else // TUNE_LOAD == 2
#    define TUNE_LOAD_MODIFIER cub::LOAD_CA
#  endif // TUNE_LOAD

template <typename KeyT>
struct policy_hub_t
{
  struct policy_t : cub::ChainedPolicy<500, policy_t, policy_t>
  {
    using merge_policy =
      cub::agent_policy_t<TUNE_THREADS_PER_BLOCK,
                          cub::Nominal4BItemsToItems<KeyT>(TUNE_ITEMS_PER_THREAD),
                          TUNE_LOAD_ALGORITHM,
                          TUNE_LOAD_MODIFIER,
                          TUNE_STORE_ALGORITHM>;
  };

  using MaxPolicy = policy_t;
};
#endif // TUNE_BASE

struct select_if_less_than_t
{
  bool negate;
  uint8_t threshold;

  __device__ __forceinline__ bool operator()(uint8_t val) const
  {
    return negate ? !(val < threshold) : val < threshold;
  }
};

template <typename OffsetT>
struct write_pivot_point_t
{
  OffsetT threshold;
  OffsetT* pivot_point;

  __device__ void operator()(OffsetT output_index, OffsetT input_index) const
  {
    if (output_index == threshold)
    {
      *pivot_point = input_index;
    }
  }
};
