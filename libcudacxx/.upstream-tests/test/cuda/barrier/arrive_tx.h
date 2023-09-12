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

inline __device__
void mbarrier_complete_tx(
  cuda::barrier<cuda::thread_scope_block> &b, int transaction_count)
{
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_90, (
        if (__isShared(cuda::device::barrier_native_handle(b))) {
            asm volatile(
              "mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
              :
              : "r"((unsigned int) __cvta_generic_to_shared(cuda::device::barrier_native_handle(b))),
                "r"(transaction_count)
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

template<typename Barrier>
__device__
void thread(Barrier& b, int arrives_per_thread)
{
  constexpr int tx_count = 1;
  auto tok = cuda::device::arrive_tx(b, arrives_per_thread, tx_count);
  // Manually increase the transaction count of the barrier.
  mbarrier_complete_tx(b, tx_count);

  b.wait(cuda::std::move(tok));
}

template<typename Barrier,
    template<typename, typename> typename Selector,
    typename Initializer = constructor_initializer>
__device__
void test()
{
  NV_DISPATCH_TARGET(
    NV_IS_DEVICE, (
      // Initialize barrier in main thread
      __shared__ Barrier bar_1;
      __shared__ Barrier bar_2;
      init(&bar_1, (int) blockDim.x);
      init(&bar_2, (int) 2 * blockDim.x);

      // Run all threads, each arriving with arrival count 1
      thread(bar_1, 1);
      // Run all threads, each arriving with arrival count 2
      thread(bar_2, 2);
    )
  );
}

#endif // TEST_ARRIVE_TX_H_
