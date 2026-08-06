//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief The CUDASTF_DISABLE_LOCALIZED_MEMORY A/B knob
 *
 * Sets the environment variable before any locality-domain allocation is
 * made, then checks that domain data places hand out plain (non-localized)
 * device memory while tasks on locality-domain places still run correctly.
 * This knob separates the effect of localized execution from localized
 * memory placement in benchmarks.
 */

#include <cuda/experimental/stf.cuh>

#include <cstdlib>

using namespace cuda::experimental::stf;

__global__ void scale(double a, slice<double> x)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  size_t n = x.extent(0);
  for (size_t ind = tid; ind < n; ind += nthreads)
  {
    x(ind) *= a;
  }
}

int main()
{
#if _CCCL_COMPILER(MSVC)
  EXPECT(_putenv_s("CUDASTF_DISABLE_LOCALIZED_MEMORY", "1") == 0);
#else // ^^^ MSVC ^^^ / vvv POSIX vvv
  EXPECT(setenv("CUDASTF_DISABLE_LOCALIZED_MEMORY", "1", 1) == 0);
#endif // !MSVC

  const int dev               = 0;
  const unsigned int ndomains = locality_domain_count(dev);

  // Never 0: a device without locality-domain support reports a single
  // whole-device domain.
  EXPECT(ndomains >= 1);

  cuda_safe_call(cudaSetDevice(dev));
  cudaStream_t stream = nullptr;
  cuda_safe_call(cudaStreamCreate(&stream));

  const size_t bytes = 1 << 20;

  // With the knob set, allocations from domain data places must not be
  // localized, on every domain.
  for (unsigned int i = 0; i < ndomains; i++)
  {
    data_place dp = data_place::locality_domain(dev, static_cast<int>(i));

    void* ptr = dp.allocate(static_cast<ptrdiff_t>(bytes), stream);
    EXPECT(ptr != nullptr);
    cuda_safe_call(cudaStreamSynchronize(stream));

#if _CCCL_CTK_AT_LEAST(13, 4) && !defined(CUDAX_PLACES_FORCE_LOCALITY_DOMAIN_FALLBACK)
    int ordinal = -2;
    cuda_safe_call(cuPointerGetAttribute(
      &ordinal, CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL, reinterpret_cast<CUdeviceptr>(ptr)));
    EXPECT(ordinal == -1); // not localized
#endif // _CCCL_CTK_AT_LEAST(13, 4) && !defined(CUDAX_PLACES_FORCE_LOCALITY_DOMAIN_FALLBACK)

    dp.deallocate(ptr, bytes, stream);
    cuda_safe_call(cudaStreamSynchronize(stream));
  }

  cuda_safe_call(cudaStreamDestroy(stream));

  // Execution on locality-domain places remains correct with the knob set
  stream_ctx ctx;
  const int n = 1024;

  double X[n];
  for (int ind = 0; ind < n; ind++)
  {
    X[ind] = 1.0 * ind;
  }

  auto handle_X = ctx.logical_data(make_slice(&X[0], n));

  const double a  = 3.0;
  const int NITER = 2 * static_cast<int>(ndomains);
  for (int iter = 0; iter < NITER; iter++)
  {
    const int domain_id = iter % static_cast<int>(ndomains);
    ctx.task(exec_place::locality_domain(dev, domain_id), handle_X.rw())->*[&](cudaStream_t s, auto dX) {
      scale<<<16, 128, 0, s>>>(a, dX);
    };
  }

  ctx.host_launch(handle_X.read())->*[&](auto hX) {
    const double expected_factor = pow(a, NITER);
    for (int ind = 0; ind < n; ind++)
    {
      EXPECT(fabs(hX(ind) - expected_factor * ind) < 0.00001 * expected_factor);
    }
  };

  ctx.finalize();

  return 0;
}
