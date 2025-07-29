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
 *
 * @brief JIT version of the AXPY example
 *
 */

#include <cuda/experimental/__stf/nvrtc/jit_utils.cuh>
#include <cuda/experimental/__stf/utility/occupancy.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

double X0(int i)
{
  return sin((double) i);
}

double Y0(int i)
{
  return cos((double) i);
}

constexpr char header_template[] = JITABLE_CODE(
  template <typename Sx, typename Sy, auto alpha, typename T0, typename T1> __device__ void axpy(T0 dynX, T1 dynY) {
    if constexpr (!cuda::experimental::stf::jit_execution)
    {
      assert(!"That's not supposed to happen.");
    }
    Sx X{dynX};
    Sy Y{dynY};

    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int dimx = blockDim.x * gridDim.x;

    for (size_t i = tidx; i < X.extent(0); i += dimx)
    {
      Y(i) += alpha * X(i);
    }
  });

const char* axpy_kernel_template = R"(
#include <cuda/experimental/__stf/nvrtc/slice.cuh>

extern "C"
__global__ void %KERNEL_NAME%(%s dynX, %s dynY)
{
  axpy<%s, %s, %a>(dynX, dynY);
}

)";

int main()
{
  const size_t N = 16;
  double X[N], Y[N];

  for (size_t i = 0; i < N; i++)
  {
    X[i] = X0(i);
    Y[i] = Y0(i);
  }

  context ctx;
  double alpha = 3.14;

  auto lX = ctx.logical_data(X);
  auto lY = ctx.logical_data(Y);

  /* Compute Y = Y + alpha X */
  ctx.cuda_kernel(lX.read(), lY.rw())->*[alpha](auto dX, auto dY) {
    jit_adapter jdX{dX};
    jit_adapter jdY{dY};

    CUfunction axpy_kernel = lazy_jit(
      axpy_kernel_template,
      get_nvrtc_flags(),
      header_template,
      jdX.kernel_param_t_name(),
      jdY.kernel_param_t_name(),
      jdX.kernel_side_t_name(),
      jdY.kernel_side_t_name(),
      alpha);
    auto [grid_size, block_size] = reserved::compute_occupancy(axpy_kernel);
    return cuda_kernel_desc{
      axpy_kernel, (unsigned) grid_size, (unsigned) block_size, 0, jdX.to_kernel_arg(), jdY.to_kernel_arg()};
  };

  ctx.finalize();

  for (size_t i = 0; i < N; i++)
  {
    assert(fabs(Y[i] - (Y0(i) + alpha * X0(i))) < 0.0001);
    assert(fabs(X[i] - X0(i)) < 0.0001);
  }
}
