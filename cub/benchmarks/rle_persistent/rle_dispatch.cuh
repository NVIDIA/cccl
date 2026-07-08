#pragma once

#include <cub/device/device_run_length_encode.cuh>

#include <algorithm>

#include "persistent_rle.cu"

namespace rle_impl
{
// tile count below which stock CUB runs
constexpr int kStockDispatchTiles = 1024;

template <class Config>
inline long long rle_state_tiles(long long n)
{
  return (n + Config::kTileSize - 1) / Config::kTileSize;
}

struct RleTempHeader
{
  unsigned long long magic;
  unsigned launch_gen;
  unsigned rsvd;
};
inline constexpr unsigned long long kRleTempMagic = 0x524c455f54454d50ull; // "RLE_TEMP"

template <class StateT>
__global__ void rle_init_states(RleTempHeader* hdr, StateT* states, long long n_states)
{
  __shared__ unsigned s_gen;
  __shared__ bool s_clear;
  if (threadIdx.x == 0)
  {
    const bool fresh = (hdr->magic != kRleTempMagic) || (hdr->launch_gen >= 0xfffffff0u);
    const unsigned g = fresh ? 1u : hdr->launch_gen + 1u;
    s_gen            = g;
    s_clear          = fresh;
  }
  __syncthreads();
  if (s_clear)
  {
    for (long long i = threadIdx.x; i < n_states; i += blockDim.x)
    {
      states[i] = StateT{}; // tag 0 never matches a live tag (>= 1)
    }
  }
  if (threadIdx.x == 0)
  {
    hdr->magic      = kRleTempMagic;
    hdr->launch_gen = s_gen;
  }
}

template <class Config, class KeyT, class LenT, class NumRunsT, class OffT>
inline cudaError_t persistent_rle_encode(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  const KeyT* d_keys,
  KeyT* d_unique,
  LenT* d_counts,
  NumRunsT* d_num_runs,
  OffT num_items,
  cudaStream_t stream = 0)
{
  // size query must cover BOTH paths (the same allocation may serve either across calls)
  size_t cub_bytes = 0;
  cub::DeviceRunLengthEncode::Encode(nullptr, cub_bytes, d_keys, d_unique, d_counts, d_num_runs, num_items, stream);
  const size_t pers_bytes =
    sizeof(RleTempHeader) + (size_t) rle_state_tiles<Config>((long long) num_items) * sizeof(TilePartialStateT);
  const size_t required = std::max(cub_bytes, pers_bytes);
  if (d_temp_storage == nullptr)
  {
    temp_storage_bytes = required;
    return cudaSuccess;
  }
  if (temp_storage_bytes < required)
  {
    return cudaErrorInvalidValue;
  }
  const long long tiles = rle_state_tiles<Config>((long long) num_items);
  if (tiles < kStockDispatchTiles)
  {
    return cub::DeviceRunLengthEncode::Encode(
      d_temp_storage, temp_storage_bytes, d_keys, d_unique, d_counts, d_num_runs, num_items, stream);
  }
  auto* hdr                 = (RleTempHeader*) d_temp_storage;
  TilePartialStateT* states = (TilePartialStateT*) (hdr + 1);
  rle_init_states<<<1, 256, 0, stream>>>(hdr, states, tiles);
  persistent_rle_launch<KeyT, LenT, NumRunsT, OffT, Config>(
    d_keys, d_unique, d_counts, d_num_runs, states, &hdr->launch_gen, num_items, (int) tiles, stream);
  return cudaGetLastError();
}
} // namespace rle_impl
