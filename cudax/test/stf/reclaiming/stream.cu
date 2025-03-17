//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

__global__ void kernel(int i, slice<char> buf)
{
  buf[0] = (char) i;
}

static __global__ void cuda_sleep_kernel(long long int clock_cnt)
{
  long long int start_clock  = clock64();
  long long int clock_offset = 0;
  while (clock_offset < clock_cnt)
  {
    clock_offset = clock64() - start_clock;
  }
}

void cuda_sleep(double ms, cudaStream_t stream)
{
  int device;
  cudaGetDevice(&device);

  // cudaDevAttrClockRate: Peak clock frequency in kilohertz;
  int clock_rate;
  cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, device);

  long long int clock_cnt = (long long int) (ms * clock_rate);
  cuda_sleep_kernel<<<1, 1, 0, stream>>>(clock_cnt);
}

// Dummy allocator which only allocates a single block on a single device
//
// The first allocation succeeds, next attempt will fail until buffer is
// deallocated.
class one_block_allocator : public block_allocator_interface
{
public:
  one_block_allocator() = default;

public:
  // Note that this allocates memory immediately, so we just do not modify the event list and ignore it
  void* allocate(backend_ctx_untyped&, const data_place& memory_node, ::std::ptrdiff_t& s, event_list&) override
  {
    if (busy)
    {
      s = -s;
      return nullptr;
    }

    EXPECT(memory_node.is_device());
    if (!base)
    {
      cuda_safe_call(cudaMalloc(&base, s));
    }

    busy = true;

    return base;
  }

  void deallocate(backend_ctx_untyped&, const data_place&, event_list&, void*, size_t) override
  {
    EXPECT(busy);
    busy = false;
  }

  event_list deinit(backend_ctx_untyped&) override
  {
    return event_list();
  }

  std::string to_string() const override
  {
    return "dummy";
  }

private:
  // We have a single block, so we keep its address, and a flag to indicate
  // if it's busy
  void* base = nullptr;
  bool busy  = false;
};

int main(int argc, char** argv)
{
  int nblocks       = 4;
  size_t block_size = 1024 * 1024;

  if (argc > 1)
  {
    nblocks = atoi(argv[1]);
  }

  if (argc > 2)
  {
    block_size = atoi(argv[2]);
  }

  stream_ctx ctx;

  auto dummy_alloc = block_allocator<one_block_allocator>(ctx);
  ctx.set_allocator(dummy_alloc);

  ::std::vector<logical_data<slice<char>>> handles(nblocks);

  char* h_buffer = new char[nblocks * block_size];

  for (int i = 0; i < nblocks; i++)
  {
    handles[i] = ctx.logical_data(make_slice(&h_buffer[i * block_size], block_size));
    handles[i].set_symbol("D_" + std::to_string(i));
  }

  // We only 2 buffers, we are forced to reuse the buffer from D0 for D2
  for (int i = 0; i < nblocks; i++)
  {
    ctx.task(handles[i % nblocks].rw())->*[&](cudaStream_t s, auto buf) {
      // Wait 100ms to have a more stressful asynchronous execution
      cuda_sleep(100, s);
      kernel<<<1, 1, 0, s>>>(i, buf);
    };
  }

  ctx.finalize();

  for (int i = 0; i < nblocks; i++)
  {
    EXPECT(h_buffer[block_size * i] == i);
  }
}
