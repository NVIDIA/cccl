//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef TEST_ARRIVE_TX_H_
#define TEST_ARRIVE_TX_H_

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
inline __host__  __device__ void mbarrier_complete_tx(barrier *bar, int transaction_count) {
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_90, (
        if (__isShared(bar)) {
        asm volatile(
            "mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
            :
            : "r"((unsigned int) __cvta_generic_to_shared(cuda::device::barrier_native_handle(*bar)))
              , "r"(transaction_count)
            : "memory");
        } else {
            // When arriving on non-shared barriers, we drop the transaction count
            // update. The barriers do not keep track of transaction counts.
        }
    ), NV_ANY_TARGET, (
      // On architectures pre-SM90 (and on host), we drop the transaction count
      // update. The barriers do not keep track of transaction counts.
    )
  );
}

template<typename Barrier,
    template<typename, typename> typename Selector,
    typename Initializer = constructor_initializer>
__host__ __device__
void test(BlockSize block_size)
{
  Selector<Barrier, Initializer> sel;
  SHARED Barrier *b;
  b = sel.construct((int) block_size);
  int arrival_count = 1;
  int tx_count = 1;
  auto tok = b->arrive_tx(arrival_count, tx_count);

  // Manually increase the transaction count of the barrier by blockDim.x.
  // This emulates a cp.async.bulk instruction or equivalently, a memcpy_async call.
  mbarrier_complete_tx(b, tx_count);

  b->wait(cuda::std::move(tok));
}

#endif // TEST_ARRIVE_TX_H_
