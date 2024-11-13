//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc, pre-sm-70
// UNSUPPORTED: clang && (!nvcc)

// uncomment for a really verbose output detailing what test steps are being launched
#define DEBUG_TESTERS

#include <cuda/barrier>
#include <cuda/std/cassert>

#include "helpers.h"

template <typename Barrier>
struct barrier_and_token
{
  using barrier_t = Barrier;
  using token_t   = typename barrier_t::arrival_token;

  barrier_t barrier;
  cuda::std::atomic<token_t> token{token_t{}};
  cuda::std::atomic<bool> token_set{false};

  template <typename... Args>
  __host__ __device__ barrier_and_token(Args&&... args)
      : barrier{cuda::std::forward<Args>(args)...}
  {}
};

template <template <typename> class Barrier>
struct barrier_and_token_with_completion
{
  struct completion_t
  {
    cuda::std::atomic<bool>& completed;

    __host__ __device__ void operator()() const
    {
      assert(completed.load() == false);
      completed.store(true);
    }
  };

  using barrier_t = Barrier<completion_t>;
  using token_t   = typename barrier_t::arrival_token;

  barrier_t barrier;
  cuda::std::atomic<token_t> token{token_t{}};
  cuda::std::atomic<bool> token_set{false};
  cuda::std::atomic<bool> completed{false};

  template <typename Arg>
  __host__ __device__ barrier_and_token_with_completion(Arg&& arg)
      : barrier{std::forward<Arg>(arg), completion_t{completed}}
  {}
};

template <int N>
struct barrier_arrive
{
  using async                         = cuda::std::true_type;
  static constexpr size_t threadcount = N;

  template <typename Data>
  __host__ __device__ static void perform(Data& data)
  {
    data.token.store(data.barrier.arrive(), cuda::std::memory_order_release);
    data.token_set.store(true, cuda::std::memory_order_release);
    data.token_set.notify_all();
  }
};

struct barrier_wait
{
  using async = cuda::std::true_type;

  template <typename Data>
  __host__ __device__ static void perform(Data& data)
  {
    while (data.token_set.load(cuda::std::memory_order_acquire) == false)
    {
      data.token_set.wait(false);
    }
    data.barrier.wait(data.token);
  }
};

struct validate_completion_result
{
  template <typename Data>
  __host__ __device__ static void perform(Data& data)
  {
    assert(data.completed.load(cuda::std::memory_order_acquire) == true);
    data.completed.store(false, cuda::std::memory_order_release);
  }
};

struct clear_token
{
  template <typename Data>
  __host__ __device__ static void perform(Data& data)
  {
    data.token_set.store(false, cuda::std::memory_order_release);
  }
};

using a2_w_w = performer_list<barrier_arrive<2>, barrier_wait, barrier_wait, async_tester_fence, clear_token>;

using completion_performers_a =
  performer_list<clear_token, barrier_arrive<2>, barrier_wait, async_tester_fence, validate_completion_result, barrier_wait>;

using completion_performers_b =
  performer_list<clear_token, barrier_arrive<2>, barrier_wait, async_tester_fence, validate_completion_result>;

template <typename Completion>
using cuda_barrier_system = cuda::barrier<cuda::thread_scope_system, Completion>;

void kernel_invoker()
{
  validate_pinned<barrier_and_token<cuda::std::barrier<>>, a2_w_w>(2);
  validate_pinned<barrier_and_token<cuda::barrier<cuda::thread_scope_system>>, a2_w_w>(2);

  validate_pinned<barrier_and_token_with_completion<cuda::std::barrier>, completion_performers_a>(2);
  validate_pinned<barrier_and_token_with_completion<cuda_barrier_system>, completion_performers_a>(2);

  validate_pinned<barrier_and_token_with_completion<cuda::std::barrier>, completion_performers_b>(2);
  validate_pinned<barrier_and_token_with_completion<cuda_barrier_system>, completion_performers_b>(2);
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (kernel_invoker();))

  return 0;
}
