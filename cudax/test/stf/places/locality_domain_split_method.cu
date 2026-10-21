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
 * @brief SM split methods for locality-domain execution places
 *
 * For every domain the device reports (never a hardcoded count) and for every
 * split method, builds the execution place, checks identity semantics (same
 * method -> same place, the default is `backfill`, distinct methods ->
 * distinct places on the native backend), checks the documented structural
 * properties of the SM partitions adaptively against the queried device
 * resources, and runs tasks on every place.
 */

#include <cuda/__driver/driver_api.h>

#include <cuda/experimental/stf.cuh>

#include <cstdio>
#include <cstdlib>

using namespace cuda::experimental::stf;

__global__ void axpy(double a, slice<const double> x, slice<double> y)
{
  const int tid      = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int nthreads = static_cast<int>(gridDim.x * blockDim.x);

  const size_t n = x.extent(0);
  for (size_t ind = tid; ind < n; ind += nthreads)
  {
    y(ind) += a * x(ind);
  }
}

int main()
{
#if _CCCL_CTK_AT_LEAST(13, 4)
  // The locality-domain driver APIs exercised below need a CUDA 13.4+ driver
  if (::cuda::__driver::__version_below(13, 4))
  {
    fprintf(stderr, "Driver is too old for locality domain tests (requires CUDA 13.4 / R615 or later): test waived.\n");
    return 0;
  }
#endif // _CCCL_CTK_AT_LEAST(13, 4)

  const int dev               = 0;
  const unsigned int ndomains = locality_domain_count(dev);

  // Never 0: a device without locality-domain support reports a single
  // whole-device domain.
  EXPECT(ndomains >= 1);

  const locality_domain_sm_split methods[] = {
    locality_domain_sm_split::backfill, locality_domain_sm_split::aligned, locality_domain_sm_split::fine};

  // ==== Construction, affine data place and identity, for every method ====

  for (auto m : methods)
  {
    for (unsigned int i = 0; i < ndomains; i++)
    {
      exec_place p = exec_place::locality_domain(dev, static_cast<int>(i), m);

      // Scalar place
      EXPECT(p.size() == 1);

      // The split method only affects the execution side: whatever the
      // method, the affine data place is the matching locality-domain data
      // place.
      EXPECT(p.affine_data_place() == data_place::locality_domain(dev, static_cast<int>(i)));

      // Same (device, domain, method) -> same place
      EXPECT(p == exec_place::locality_domain(dev, static_cast<int>(i), m));
    }
  }

  // The default method is backfill
  EXPECT(exec_place::locality_domain(dev, 0)
         == exec_place::locality_domain(dev, 0, locality_domain_sm_split::backfill));

#if _CCCL_CTK_AT_LEAST(13, 4) && !defined(CUDAX_PLACES_FORCE_LOCALITY_DOMAIN_FALLBACK)
  // The fake-topology override deliberately ignores the split method, so its
  // places compare equal across methods and carry no per-domain partition.
  if (std::getenv("CUDASTF_FAKE_LOCALITY_DOMAINS") != nullptr)
  {
    fprintf(stderr, "Fake locality-domain topology active: SM partition checks waived.\n");
  }
  else
  {
    // The SM partition of a place, read through the public API: activate the
    // place and ask the current (green) context for its SM resource.
    auto partition_of = [](const exec_place& p) {
      CUdevResource r;
      auto scope = p.activate();
      cuda_safe_call(cuCtxGetDevResource(cuda_try<cuCtxGetCurrent>(), &r, CU_DEV_RESOURCE_TYPE_SM));
      return r.sm;
    };

    CUdevice device;
    cuda_safe_call(cuDeviceGet(&device, dev));
    CUdevResource total;
    cuda_safe_call(cuDeviceGetDevResource(device, &total, CU_DEV_RESOURCE_TYPE_SM));
    const unsigned int alignment = total.sm.smCoscheduledAlignment;
    EXPECT(alignment >= 1);

    if (ndomains == 1)
    {
      // ==== Single domain: the domain is the device ====
      //
      // Whether the driver reports one domain or cannot answer the query at
      // all, every method resolves to the same whole-device place: the
      // method is documented to be ignored, so the places compare equal and
      // the partition spans every SM. None of the multi-domain structural
      // properties below apply.
      exec_place p_bf = exec_place::locality_domain(dev, 0, locality_domain_sm_split::backfill);
      EXPECT(p_bf == exec_place::locality_domain(dev, 0, locality_domain_sm_split::aligned));
      EXPECT(p_bf == exec_place::locality_domain(dev, 0, locality_domain_sm_split::fine));
      EXPECT(partition_of(p_bf).smCount == total.sm.smCount);
    }
    else
    {
      // ==== Distinct methods select distinct SM partitions ====

      exec_place p_bf = exec_place::locality_domain(dev, 0, locality_domain_sm_split::backfill);
      exec_place p_al = exec_place::locality_domain(dev, 0, locality_domain_sm_split::aligned);
      exec_place p_fi = exec_place::locality_domain(dev, 0, locality_domain_sm_split::fine);
      EXPECT(p_bf != p_al);
      EXPECT(p_bf != p_fi);
      EXPECT(p_al != p_fi);

      // ==== Structural properties of the partitions (adaptive) ====

      // backfill sizes every group to an even share of the device (rounded
      // down to the required multiple of 2).
      const unsigned int share             = total.sm.smCount / ndomains;
      const unsigned int expected_backfill = (share < 2) ? 2 : (share - share % 2);

      unsigned int sum_aligned = 0, sum_fine = 0, sum_backfill = 0;
      for (unsigned int i = 0; i < ndomains; i++)
      {
        const auto bf =
          partition_of(exec_place::locality_domain(dev, static_cast<int>(i), locality_domain_sm_split::backfill));
        const auto al =
          partition_of(exec_place::locality_domain(dev, static_cast<int>(i), locality_domain_sm_split::aligned));
        const auto fi =
          partition_of(exec_place::locality_domain(dev, static_cast<int>(i), locality_domain_sm_split::fine));

        // The driver must have honored the request, not silently handed
        // back an unsplit device: every group echoes the requested domain.
        for (const auto* r : {&bf, &al, &fi})
        {
          EXPECT((r->flags & CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID) != 0);
          EXPECT(r->localityDomainId == i);
        }
        EXPECT((bf.flags & CU_DEV_SM_RESOURCE_GROUP_BACKFILL) != 0);

        EXPECT(bf.smCount == expected_backfill); // even share of the device
        EXPECT(al.smCount % alignment == 0); // complete co-scheduled groups only
        EXPECT(fi.smCount % 2 == 0); // finest granularity: groups of 2
        EXPECT(fi.smCount >= al.smCount); // fine recovers at least the aligned SMs

        sum_aligned += al.smCount;
        sum_fine += fi.smCount;
        sum_backfill += bf.smCount;
      }

      // The strict methods never exceed the device, and together the
      // backfilled groups cover (up to rounding) the whole device.
      EXPECT(sum_aligned <= sum_fine);
      EXPECT(sum_fine <= total.sm.smCount);
      EXPECT(sum_backfill <= total.sm.smCount);
      EXPECT(sum_backfill == expected_backfill * ndomains);
    }
  }
#endif // _CCCL_CTK_AT_LEAST(13, 4) && !defined(CUDAX_PLACES_FORCE_LOCALITY_DOMAIN_FALLBACK)

  // ==== A grid built with an explicit method has one place per domain ====

  exec_place grid = exec_place::locality_domains(dev, locality_domain_sm_split::fine);
  EXPECT(grid.size() == ndomains);

  // ==== Tasks run on every method's places ====

  stream_ctx ctx;
  const double alpha = 2.0;
  const int n        = 1024;

  double X[n], Y[n];
  for (int ind = 0; ind < n; ind++)
  {
    X[ind] = 1.0 * ind;
    Y[ind] = 2.0 * ind - 3.0;
  }

  auto handle_X = ctx.logical_data(make_slice(&X[0], n));
  auto handle_Y = ctx.logical_data(make_slice(&Y[0], n));

  int niter = 0;
  for (auto m : methods)
  {
    for (unsigned int i = 0; i < ndomains; i++)
    {
      ctx.task(exec_place::locality_domain(dev, static_cast<int>(i), m), handle_X.read(), handle_Y.rw())
          ->*[&](cudaStream_t stream, auto dX, auto dY) {
                axpy<<<16, 128, 0, stream>>>(alpha, dX, dY);
              };
      niter++;
    }
  }

  ctx.host_launch(handle_X.read(), handle_Y.read())->*[&](auto hX, auto hY) {
    for (int ind = 0; ind < n; ind++)
    {
      EXPECT(fabs(hX(ind) - 1.0 * ind) < 0.00001);
      EXPECT(fabs(hY(ind) - (2.0 * ind - 3.0) - niter * alpha * hX(ind)) < 0.00001);
    }
  };

  ctx.finalize();

  return 0;
}
