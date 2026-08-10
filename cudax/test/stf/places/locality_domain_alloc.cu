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
 * @brief Locality domain data place allocation paths
 *
 * For every domain the device reports, exercises the stream-ordered
 * allocation path (and, with the CUDA 13.4+ backend, verifies through
 * `CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL` that the allocation actually
 * lands in the requested domain) as well as the VMM `mem_create` path.
 */

#include <cuda/experimental/stf.cuh>

#include <cstdlib>

using namespace cuda::experimental::stf;

int main()
{
  const int dev               = 0;
  const unsigned int ndomains = locality_domain_count(dev);

  // Never 0: a device without locality-domain support reports a single
  // whole-device domain.
  EXPECT(ndomains >= 1);

  cuda_safe_call(cudaSetDevice(dev));
  cudaStream_t stream = nullptr;
  cuda_safe_call(cudaStreamCreate(&stream));

  const size_t bytes = 1 << 20;

  for (unsigned int i = 0; i < ndomains; i++)
  {
    data_place dp = data_place::locality_domain(dev, static_cast<int>(i));

    // ==== Stream-ordered allocation ====

    void* ptr = dp.allocate(static_cast<ptrdiff_t>(bytes), stream);
    EXPECT(ptr != nullptr);
    cuda_safe_call(cudaStreamSynchronize(stream));

    // The memory must be usable on the device
    cuda_safe_call(cudaMemsetAsync(ptr, 0xab, bytes, stream));
    cuda_safe_call(cudaStreamSynchronize(stream));

#if _CCCL_CTK_AT_LEAST(13, 4) && !defined(CUDAX_PLACES_FORCE_LOCALITY_DOMAIN_FALLBACK)
    // With the native backend, the allocation must report the requested
    // domain ordinal (unless localization is disabled, or the fake topology
    // override is active, in which case it must report "not localized",
    // i.e. -1).
    int ordinal = -2;
    cuda_safe_call(cuPointerGetAttribute(
      &ordinal, CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL, reinterpret_cast<CUdeviceptr>(ptr)));
    if (std::getenv("CUDASTF_DISABLE_LOCALIZED_MEMORY") != nullptr
        || std::getenv("CUDASTF_FAKE_LOCALITY_DOMAINS") != nullptr)
    {
      EXPECT(ordinal == -1);
    }
    else
    {
      EXPECT(ordinal == static_cast<int>(i));
    }
#endif // _CCCL_CTK_AT_LEAST(13, 4) && !defined(CUDAX_PLACES_FORCE_LOCALITY_DOMAIN_FALLBACK)

    dp.deallocate(ptr, bytes, stream);
    cuda_safe_call(cudaStreamSynchronize(stream));

    // ==== VMM physical handle (mem_create) ====

    // Query the allocation granularity for a plain device allocation; the
    // localized backing store obeys the same minimum granularity contract.
    CUmemAllocationProp gran_prop = {};
    gran_prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
    gran_prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;
    gran_prop.location.id         = dev;
    size_t granularity            = 0;
    cuda_safe_call(cuMemGetAllocationGranularity(&granularity, &gran_prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    EXPECT(granularity > 0);

    CUmemGenericAllocationHandle handle{};
    cuda_safe_call(dp.mem_create(&handle, granularity));
    cuda_safe_call(cuMemRelease(handle));
  }

  cuda_safe_call(cudaStreamDestroy(stream));

  return 0;
}
