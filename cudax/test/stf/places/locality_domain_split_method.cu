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

#include <cuda/experimental/stf.cuh>

#include <cstdio>

using namespace cuda::experimental::stf;

__global__ void axpy(double a, slice<const double> x, slice<double> y)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  size_t n = x.extent(0);
  for (size_t ind = tid; ind < n; ind += nthreads)
  {
    y(ind) += a * x(ind);
  }
}

int main()
{
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
  namespace places = ::cuda::experimental::places;

  // (Native backend only: the whole-device fallback and the fake-topology
  // override deliberately ignore the split method, so their places compare
  // equal across methods.)
  if (places::locality_domain_fake_count() == 0)
  {
    // ==== Distinct methods select distinct SM partitions ====

    exec_place p_bf = exec_place::locality_domain(dev, 0, locality_domain_sm_split::backfill);
    exec_place p_al = exec_place::locality_domain(dev, 0, locality_domain_sm_split::aligned);
    exec_place p_fi = exec_place::locality_domain(dev, 0, locality_domain_sm_split::fine);
    EXPECT(p_bf != p_al);
    EXPECT(p_bf != p_fi);
    EXPECT(p_al != p_fi);

    // ==== Structural properties of the partitions (adaptive) ====
    //
    // Only meaningful when the driver actually splits by domain (not under
    // the whole-device degrade, where every method spans all SMs).
    // Under the whole-device degrade the split method is documented to be
    // ignored: every method must resolve to the SAME cached whole-device
    // green context (same execution-place identity), not one context per
    // requested method.
    if (places::locality_domain_native_raw_count(dev) == 0)
    {
      const auto& e_bf = places::locality_domain_ctx_cache::instance().get(dev, 0, locality_domain_sm_split::backfill);
      const auto& e_al = places::locality_domain_ctx_cache::instance().get(dev, 0, locality_domain_sm_split::aligned);
      const auto& e_fi = places::locality_domain_ctx_cache::instance().get(dev, 0, locality_domain_sm_split::fine);
      EXPECT(e_bf.g_ctx == e_al.g_ctx);
      EXPECT(e_bf.g_ctx == e_fi.g_ctx);
    }

    if (places::locality_domain_native_raw_count(dev) > 0)
    {
      CUdevice device;
      cuda_safe_call(cuDeviceGet(&device, dev));
      CUdevResource total;
      cuda_safe_call(cuDeviceGetDevResource(device, &total, CU_DEV_RESOURCE_TYPE_SM));
      const unsigned int alignment = total.sm.smCoscheduledAlignment;
      EXPECT(alignment >= 1);

      // backfill sizes every group to an even share of the device (rounded
      // down to the required multiple of 2).
      const unsigned int share             = total.sm.smCount / ndomains;
      const unsigned int expected_backfill = (share < 2) ? 2 : (share - share % 2);

      auto domain_sm_count = [&](unsigned int i, locality_domain_sm_split m) {
        const auto& entry = places::locality_domain_ctx_cache::instance().get(dev, static_cast<int>(i), m);
        CUdevResource r;
        cuda_safe_call(cuGreenCtxGetDevResource(entry.g_ctx, &r, CU_DEV_RESOURCE_TYPE_SM));
        return r.sm.smCount;
      };

      unsigned int sum_aligned = 0, sum_fine = 0, sum_backfill = 0;
      for (unsigned int i = 0; i < ndomains; i++)
      {
        const unsigned int c_bf = domain_sm_count(i, locality_domain_sm_split::backfill);
        const unsigned int c_al = domain_sm_count(i, locality_domain_sm_split::aligned);
        const unsigned int c_fi = domain_sm_count(i, locality_domain_sm_split::fine);

        EXPECT(c_bf == expected_backfill); // even share of the device
        EXPECT(c_al % alignment == 0); // complete co-scheduled groups only
        EXPECT(c_fi % 2 == 0); // finest granularity: groups of 2
        EXPECT(c_fi >= c_al); // fine recovers at least the aligned SMs

        sum_aligned += c_al;
        sum_fine += c_fi;
        sum_backfill += c_bf;
      }

      // The strict methods never exceed the device, and together the
      // backfilled groups cover (up to rounding) the whole device.
      EXPECT(sum_aligned <= sum_fine);
      EXPECT(sum_fine <= total.sm.smCount);
      EXPECT(sum_backfill <= total.sm.smCount);
      EXPECT(sum_backfill == expected_backfill * ndomains);
    }
    else
    {
      fprintf(stderr, "Driver reports no locality domains: structural partition checks waived.\n");
    }
  }
  else
  {
    fprintf(stderr, "Fake locality-domain topology active: method distinction checks waived.\n");
  }
#endif // _CCCL_CTK_AT_LEAST(13, 4) && !defined(CUDAX_PLACES_FORCE_LOCALITY_DOMAIN_FALLBACK)

  // ==== A grid built with an explicit method has one place per domain ====

  exec_place grid = make_locality_domain_grid(dev, locality_domain_sm_split::fine);
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
