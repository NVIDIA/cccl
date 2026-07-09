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

// CUB temp storage is caller scratch with no contents contract between calls, so the states are
// cleared on EVERY launch (same as stock CUB's init kernels)
template <class StateT>
__global__ void rle_init_states(StateT* states, long long n_states)
{
  const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_states)
  {
    states[i] = StateT{}; // tag 0 never matches the published tag (1)
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
  const size_t pers_bytes = (size_t) rle_state_tiles<Config>((long long) num_items) * sizeof(TilePartialStateT);
  const size_t required   = std::max(cub_bytes, pers_bytes);
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
  auto* states          = (TilePartialStateT*) d_temp_storage;
  const int init_blocks = (int) ((tiles + 255) / 256);
  rle_init_states<<<init_blocks, 256, 0, stream>>>(states, tiles);
  persistent_rle_launch<KeyT, LenT, NumRunsT, OffT, Config>(
    d_keys, d_unique, d_counts, d_num_runs, states, num_items, (int) tiles, stream);
  return cudaGetLastError();
}
} // namespace rle_impl
