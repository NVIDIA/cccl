#pragma once

#include <cuda/ptx> // cuda::ptx

#include <cstdint> // uint32_t

#include "makeWarpUniform.cuh" // makeWarpUniform.cuh

namespace ptx = cuda::ptx;

// Commonly used special registers that we should cache in registers or uniform
// registers.
struct SpecialRegisters
{
  const uint32_t clusterCtaRank;
  const uint32_t blockIdxX;
  const uint32_t threadIdxX;
  const uint32_t warpIdx;
  const uint32_t laneIdx;
};

static inline __device__ SpecialRegisters getSpecialRegisters()
{
  uint32_t clusterCtaRank = ptx::get_sreg_cluster_ctarank();
  uint32_t threadIdxX     = threadIdx.x;
  uint32_t warpIdx        = makeWarpUniform(threadIdxX / 32);
  return {clusterCtaRank, blockIdx.x, threadIdxX, warpIdx, ptx::get_sreg_laneid()};
}
