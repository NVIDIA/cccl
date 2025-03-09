//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief This test ensures that we can apply a reduction on a logical data
 *        described using a shape
 */

#include <cuda/experimental/__stf/stream/reduction.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

using namespace cuda::experimental::stf;

using scalar_t = slice<int>;

template <typename T>
__global__ void set_value(T* addr, T val)
{
  *addr = val;
}

// instance id are put for debugging purpose ...
template <typename T>
__global__ void add(const T* in_addr, T* inout_addr)
{
  *inout_addr += *in_addr;
}

template <typename T>
__global__ void add_val(T* inout_addr, T val)
{
  *inout_addr += val;
}

class scalar_sum_t : public stream_reduction_operator<scalar_t>
{
public:
  void op(const scalar_t& in, scalar_t& inout, const exec_place& e, cudaStream_t s) override
  {
    if (e.affine_data_place() == data_place::host)
    {
      // TODO make a callback when the situation gets better
      cuda_safe_call(cudaStreamSynchronize(s));
      *inout.data_handle() += *in.data_handle();
    }
    else
    {
      // this is not the host, so this has to be a device ... (XXX)
      add<int><<<1, 1, 0, s>>>(in.data_handle(), inout.data_handle());
    }
  }

  void init_op(scalar_t& out, const exec_place& e, cudaStream_t s) override
  {
    if (e.affine_data_place() == data_place::host)
    {
      // TODO make a callback when the situation gets better
      cuda_safe_call(cudaStreamSynchronize(s));
      *out.data_handle() = 0;
    }
    else
    {
      // this is not the host, so this has to be a device ... (XXX)
      set_value<int><<<1, 1, 0, s>>>(out.data_handle(), 0);
    }
  }
};

int main()
{
  stream_ctx ctx;

  int ndevs;
  cuda_safe_call(cudaGetDeviceCount(&ndevs));

  const int N = 4;

  auto var_handle = ctx.logical_data(shape_of<slice<int>>(1));
  var_handle.set_symbol("var");

  auto redux_op = std::make_shared<scalar_sum_t>();

  // We add i twice (total = N(N-1) + initial_value)
  for (int i = 0; i < N; i++)
  {
    // device
    for (int d = 0; d < ndevs; d++)
    {
      ctx.task(exec_place::device(d), var_handle.relaxed(redux_op))->*[&](cudaStream_t s, auto var) {
        add_val<int><<<1, 1, 0, s>>>(var.data_handle(), i);
      };
    }

    // host
    ctx.task(exec_place::host, var_handle.relaxed(redux_op))->*[&](cudaStream_t s, auto var) {
      cuda_safe_call(cudaStreamSynchronize(s));
      *var.data_handle() += i;
    };
  }

  // Check result
  ctx.task(exec_place::host, var_handle.read())->*[&](cudaStream_t s, auto var) {
    cuda_safe_call(cudaStreamSynchronize(s));
    int value    = *var.data_handle();
    int expected = (N * (N - 1)) / 2 * (ndevs + 1);
    EXPECT(value == expected);
  };

  ctx.finalize();
}
