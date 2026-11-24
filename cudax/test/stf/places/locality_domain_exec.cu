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
 * @brief Locality domain execution places: creation, identity, activation
 *
 * For every domain the device reports (never a hardcoded count), builds the
 * execution place, checks its affine data place, checks equality semantics,
 * and activates/deactivates it, verifying that the previous execution state
 * is restored.
 */

#include <cuda/experimental/stf.cuh>

#include <cstdio>
#include <unordered_map>

using namespace cuda::experimental::stf;

int main()
{
  const int dev               = 0;
  const unsigned int ndomains = locality_domain_count(dev);

  // Never 0: a device without locality-domain support reports a single
  // whole-device domain.
  EXPECT(ndomains >= 1);

  // ==== Creation and affine data place, for every reported domain ====

  for (unsigned int i = 0; i < ndomains; i++)
  {
    exec_place p = exec_place::locality_domain(dev, static_cast<int>(i));

    // Scalar place
    EXPECT(p.size() == 1);

    // The affine data place is the matching locality-domain data place
    data_place affine = p.affine_data_place();
    EXPECT(affine.is_resolved());
    EXPECT(!affine.is_device());
    EXPECT(device_ordinal(affine) == dev);
    EXPECT(affine == data_place::locality_domain(dev, static_cast<int>(i)));
  }

  // ==== Identity ====

  exec_place p0  = exec_place::locality_domain(dev, 0);
  exec_place p0b = exec_place::locality_domain(dev, 0);
  EXPECT(p0 == p0b);
  EXPECT(p0 != exec_place::device(dev));
  EXPECT(p0 != exec_place::host());

  if (ndomains >= 2)
  {
    exec_place p1 = exec_place::locality_domain(dev, 1);
    EXPECT(p0 != p1);
    EXPECT(p0.affine_data_place() != p1.affine_data_place());
  }
  else
  {
    fprintf(stderr, "Device reports a single locality domain: cross-domain identity checks waived.\n");
  }

  // ==== Exec places as keys in associative containers ====

  {
    std::unordered_map<exec_place, int, hash<exec_place>> umap;
    umap[exec_place::device(dev)] = 1;
    umap[exec_place::host()]      = 2;
    for (unsigned int i = 0; i < ndomains; i++)
    {
      umap[exec_place::locality_domain(dev, static_cast<int>(i))] = 100 + static_cast<int>(i);
    }
    EXPECT(umap.size() == 2 + ndomains);
    for (unsigned int i = 0; i < ndomains; i++)
    {
      EXPECT(umap[exec_place::locality_domain(dev, static_cast<int>(i))] == 100 + static_cast<int>(i));
    }
  }

#if _CCCL_CTK_AT_LEAST(13, 4) && !defined(CUDAX_PLACES_FORCE_LOCALITY_DOMAIN_FALLBACK)
  // ==== Out-of-range ordinals are rejected at use, on every build type ====
  //
  // (Native backend only: the whole-device fallback deliberately treats the
  // ordinal as a pure label and does not validate it.)
  {
    bool threw = false;
    try
    {
      exec_place p_bad = exec_place::locality_domain(dev, static_cast<int>(ndomains) + 7);
      (void) p_bad;
    }
    catch (...)
    {
      threw = true;
    }
    EXPECT(threw);
  }
#endif // _CCCL_CTK_AT_LEAST(13, 4) && !defined(CUDAX_PLACES_FORCE_LOCALITY_DOMAIN_FALLBACK)

  // ==== Activation restores the previous execution state ====

  cuda_safe_call(cudaSetDevice(dev));

  CUcontext ctx_before = nullptr;
  cuda_safe_call(cuCtxGetCurrent(&ctx_before));

  for (unsigned int i = 0; i < ndomains; i++)
  {
    exec_place p = exec_place::locality_domain(dev, static_cast<int>(i));
    {
      auto scope = p.activate();

      // While active, we execute on the place's device
      int cur_dev = -1;
      cuda_safe_call(cudaGetDevice(&cur_dev));
      EXPECT(cur_dev == dev);

#if _CCCL_CTK_AT_LEAST(13, 4) && !defined(CUDAX_PLACES_FORCE_LOCALITY_DOMAIN_FALLBACK)
      // On the native backend, activation must actually select the domain's
      // (green) context rather than being a no-op: cudaSetDevice(dev) above
      // already makes the device check pass trivially, so also verify the
      // current driver context changed under the scope.
      CUcontext ctx_in = nullptr;
      cuda_safe_call(cuCtxGetCurrent(&ctx_in));
      EXPECT(ctx_in != ctx_before);
#endif // _CCCL_CTK_AT_LEAST(13, 4) && !defined(CUDAX_PLACES_FORCE_LOCALITY_DOMAIN_FALLBACK)
    }

    // After deactivation the previous driver context is restored
    CUcontext ctx_after = nullptr;
    cuda_safe_call(cuCtxGetCurrent(&ctx_after));
    EXPECT(ctx_after == ctx_before);
  }

  return 0;
}
