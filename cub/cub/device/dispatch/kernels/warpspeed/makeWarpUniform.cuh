#pragma once

#include <cub/config.cuh>

#include <cstdint> // uint32_t

// Move register to uniform register

// For int32_t and uint32_t, we can use the new CREDUX instruction, which is
// coupled and has a constant 13 cycle latency.
// For 64-bit types, we still use __shfl_sync
static inline __device__ int makeWarpUniform(int x)
{
  return __reduce_min_sync(~0, x);
}
static inline __device__ uint32_t makeWarpUniform(uint32_t x)
{
  return __reduce_min_sync(~0, x);
}
static inline __device__ uint64_t makeWarpUniform(uint64_t x)
{
  return __shfl_sync(~0, x, 0);
}
