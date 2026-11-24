// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/util_device.cuh>

#include <cuda/devices>

#include "cub_test_macros.h"

#if __cccl_ptx_isa >= 800
#  include <cub/detail/warpspeed/special_registers.cuh>
#  include <cub/detail/warpspeed/sync_handler.cuh>

#  include <cuda/ptxs/mbarrier_arrive.h>
#  include <cuda/ptxs/mbarrier_pending_count.h>
#  include <cuda/std/cstdint>
#  include <cuda/std/iterator>

#  include <nv/target>

#  include <c2h/vector.h>

constexpr int block_threads = 32;

[[maybe_unused]] __device__ void check_phase_barriers(int* matches)
{
  using Barrier          = cuda::std::uint64_t;
  constexpr int a_stages = 19;
  constexpr int b_stages = 7;

  struct Barriers
  {
    Barrier first_a[a_stages];
    Barrier guard;
    Barrier second_a[a_stages];
    Barrier b[b_stages];
  };

  __shared__ Barriers storage;

  const int lane = static_cast<int>(threadIdx.x);
  if (lane == 0)
  {
    storage.guard = 0;
  }
  __syncthreads();

  cub::detail::warpspeed::SyncHandler handler;
  const int a     = handler.registerResource(a_stages);
  const int b     = handler.registerResource(b_stages);
  const int empty = handler.registerResource(0);
  // Interleave resources in a different order from storage. The second phase of A crosses a block-stride boundary.
  handler.registerPhase(a, 3, storage.first_a);
  handler.registerPhase(b, 5, storage.b);
  handler.registerPhase(empty, 7, &storage.guard);
  handler.registerPhase(a, 9, storage.second_a);
  handler.clusterInitSync<block_threads>(cub::detail::warpspeed::getSpecialRegisters());

  // One non-completing arrival exposes the original count without waiting for any other threads.
  const auto has_count = [&](auto& barriers, unsigned expected) {
    if (lane < cuda::std::ssize(barriers))
    {
      const auto state = cuda::ptx::mbarrier_arrive_no_complete(&barriers[lane], 1);
      return cuda::ptx::mbarrier_pending_count(state) == expected;
    }
    return true;
  };
  matches[lane] = has_count(storage.first_a, 3) && has_count(storage.b, 5) && has_count(storage.second_a, 9)
               && storage.guard == 0;
}

__global__ void check_phase_barriers_kernel(int* matches)
{
  NV_IF_TARGET(NV_PROVIDES_SM_90, (check_phase_barriers(matches);))
}
#endif // __cccl_ptx_isa >= 800

CUB_TEST("WarpSpeed initializes each registered phase barrier with its own arrival count",
         "[block][warpspeed]",
         CUB_SMALL)
{
#if __cccl_ptx_isa >= 800
  cuda::compute_capability cc{};
  REQUIRE(cub::detail::ptx_compute_cap(cc) == cudaSuccess);
  if (cc < cuda::compute_capability{9, 0})
  {
    SKIP("WarpSpeed initializes shared-memory barriers on SM90 and newer.");
  }
  c2h::device_vector<int> matches(block_threads);
  check_phase_barriers_kernel<<<1, block_threads>>>(matches.data().get());
  REQUIRE(cudaPeekAtLastError() == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  REQUIRE(c2h::host_vector<int>(matches) == c2h::host_vector<int>(block_threads, 1));
#else
  SKIP("WarpSpeed barrier initialization requires PTX ISA 8.0.");
#endif
}
