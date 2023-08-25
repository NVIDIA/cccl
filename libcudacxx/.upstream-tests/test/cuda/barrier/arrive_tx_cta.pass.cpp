//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: pre-sm-70

// <cuda/barrier>

#include <cuda/barrier>

#include <cuda/std/utility>

#include "concurrent_agents.h"
#include "cuda_space_selector.h"
#include "test_macros.h"

// Suppress warning about barrier in shared memory
TEST_NV_DIAG_SUPPRESS(static_var_with_dynamic_init)

enum BlockSize {
    Thread = 2,
    Warp   = 32,
    CTA    = 256
};

using barrier = cuda::barrier<cuda::thread_scope_block>;
inline __device__ void mbarrier_complete_tx(barrier* bar, int transaction_count) {
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_90, (
      asm volatile(
          "mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
          :
          : "r"((unsigned int) __cvta_generic_to_shared(cuda::device::barrier_native_handle(*bar)))
          , "r"(transaction_count)
          : "memory");
    ), NV_IS_DEVICE, (
      // On architectures pre-SM90, we drop the transaction count
      // update. The barriers do not keep track of transaction counts.
    )
  );
}

template<typename Barrier,
    template<typename, typename> typename Selector,
    typename Initializer = constructor_initializer>
__device__
void test()
{
  Selector<Barrier, Initializer> sel;
  SHARED Barrier* b;
  b = sel.construct(blockDim.x);
  auto tok = b->arrive_tx(1, 1);

  // Manually increase the transaction count of the barrier by blockDim.x.
  // This emulates a cp.async.bulk instruction or equivalently, a memcpy_async call.
  if (threadIdx.x == 0) {
    mbarrier_complete_tx(b, blockDim.x);
  }
  b->wait(cuda::std::move(tok));
}

int main(int, char**)
{
    NV_IF_ELSE_TARGET(NV_IS_HOST,(
      //Required by concurrent_agents_launch to know how many we're launching
      cuda_thread_count = BlockSize::CTA;
    ),(
      test<cuda::barrier<cuda::thread_scope_block>, shared_memory_selector>();
      test<cuda::barrier<cuda::thread_scope_block>, global_memory_selector>();
    ))

    return 0;
}
