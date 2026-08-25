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
 * @brief The CUDASTF_FAKE_LOCALITY_DOMAINS topology override
 *
 * Sets the environment variable before any locality-domain query, then
 * checks that exactly the requested number of green-context-backed domains
 * is reported (the override is strict; if this device cannot provide the
 * requested topology the query throws and the test waives), that exec/data
 * places built under the override are consistent green-context places with
 * plain (non-localized) device memory, and that tasks and grids over the
 * fake domains run correctly.
 */

#include <cuda/experimental/stf.cuh>

#include <cstdlib>
#include <stdexcept>

using namespace cuda::experimental::stf;

__global__ void add_one(slice<double> x)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  size_t n = x.extent(0);
  for (size_t ind = tid; ind < n; ind += nthreads)
  {
    x(ind) += 1.0;
  }
}

int main()
{
#if _CCCL_CTK_BELOW(12, 4)
  fprintf(stderr,
          "Green contexts are not supported by this version of CUDA: the fake topology "
          "override is inactive, test waived.\n");
  return 0;
#else // ^^^ _CCCL_CTK_BELOW(12, 4) ^^^ / vvv _CCCL_CTK_AT_LEAST(12, 4) vvv
  const int requested = 2;
#  if _CCCL_COMPILER(MSVC)
  EXPECT(_putenv_s("CUDASTF_FAKE_LOCALITY_DOMAINS", "2") == 0);
#  else // ^^^ MSVC ^^^ / vvv POSIX vvv
  EXPECT(setenv("CUDASTF_FAKE_LOCALITY_DOMAINS", "2", 1) == 0);
#  endif // !MSVC

  const int dev = 0;

  int ndevs = 0;
  if (cudaGetDeviceCount(&ndevs) != cudaSuccess || ndevs == 0)
  {
    fprintf(stderr, "No CUDA device: test waived.\n");
    return 0;
  }

  // The override must take precedence over the compile-time backend, and it
  // is strict: it reports exactly the requested count, or throws when the
  // device cannot provide that many domains.
  unsigned int ndomains = 0;
  try
  {
    ndomains = locality_domain_count(dev);
  }
  catch (const ::std::runtime_error& e)
  {
    fprintf(stderr, "Requested fake topology not achievable on this device (%s): test waived.\n", e.what());
    return 0;
  }
  EXPECT(ndomains == static_cast<unsigned int>(requested));

  // Invalid device ordinals are rejected under the override too
  bool threw_invalid = false;
  try
  {
    locality_domain_count(-1);
  }
  catch (...)
  {
    threw_invalid = true;
  }
  EXPECT(threw_invalid);

  bool threw_oob = false;
  try
  {
    locality_domain_count(ndevs);
  }
  catch (...)
  {
    threw_oob = true;
  }
  EXPECT(threw_oob);

  cuda_safe_call(cudaSetDevice(dev));
  cudaStream_t stream = nullptr;
  cuda_safe_call(cudaStreamCreate(&stream));

  for (unsigned int i = 0; i < ndomains; i++)
  {
    exec_place ep = exec_place::locality_domain(dev, static_cast<int>(i));
    data_place dp = data_place::locality_domain(dev, static_cast<int>(i));

    // Repeated construction yields the same (cached) place
    EXPECT(ep == exec_place::locality_domain(dev, static_cast<int>(i)));
    EXPECT(dp == data_place::locality_domain(dev, static_cast<int>(i)));

    // Exec and data sides agree (use_green_ctx_data_place)
    EXPECT(ep.affine_data_place() == dp);
    EXPECT(device_ordinal(dp) == dev);
    EXPECT(!dp.is_device());

    // Memory from a fake domain is plain device memory, not localized
    const size_t bytes = 1 << 20;
    void* ptr          = dp.allocate(static_cast<ptrdiff_t>(bytes), stream);
    EXPECT(ptr != nullptr);
    cuda_safe_call(cudaStreamSynchronize(stream));
    cuda_safe_call(cudaMemsetAsync(ptr, 0xab, bytes, stream));
    cuda_safe_call(cudaStreamSynchronize(stream));

#  if _CCCL_CTK_AT_LEAST(13, 4)
    int ordinal = -2;
    cuda_safe_call(cuPointerGetAttribute(
      &ordinal, CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL, reinterpret_cast<CUdeviceptr>(ptr)));
    EXPECT(ordinal == -1); // not localized
#  endif // _CCCL_CTK_AT_LEAST(13, 4)

    dp.deallocate(ptr, bytes, stream);
    cuda_safe_call(cudaStreamSynchronize(stream));
  }

  // Distinct fake domains are distinct places (the strict override guarantees
  // that both requested domains exist)
  EXPECT(exec_place::locality_domain(dev, 0) != exec_place::locality_domain(dev, 1));
  EXPECT(data_place::locality_domain(dev, 0) != data_place::locality_domain(dev, 1));

  cuda_safe_call(cudaStreamDestroy(stream));

  // Grid over the fake domains
  exec_place grid = make_locality_domain_grid(dev);
  EXPECT(grid.size() == ndomains);
  for (size_t i = 0; i < grid.size(); i++)
  {
    EXPECT(grid.get_place(i) == exec_place::locality_domain(dev, static_cast<int>(i)));
  }

  // Tasks on every fake domain run correctly
  stream_ctx ctx;
  const int n = 1024;

  double X[n];
  for (int ind = 0; ind < n; ind++)
  {
    X[ind] = 7.0 + ind;
  }

  auto handle_X = ctx.logical_data(make_slice(&X[0], n));

  const int NITER = 3 * static_cast<int>(ndomains);
  for (int iter = 0; iter < NITER; iter++)
  {
    const int domain_id = iter % static_cast<int>(ndomains);
    ctx.task(exec_place::locality_domain(dev, domain_id), handle_X.rw())->*[](cudaStream_t s, auto dX) {
      add_one<<<16, 128, 0, s>>>(dX);
    };
  }

  ctx.host_launch(handle_X.read())->*[&](auto hX) {
    for (int ind = 0; ind < n; ind++)
    {
      EXPECT(fabs(hX(ind) - (7.0 + ind + NITER)) < 0.00001);
    }
  };

  ctx.finalize();

  return 0;
#endif // ^^^ _CCCL_CTK_AT_LEAST(12, 4) ^^^
}
