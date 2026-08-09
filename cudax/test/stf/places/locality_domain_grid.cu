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
 * @brief Grid of execution places over all locality domains of a device
 *
 * Checks that `make_locality_domain_grid` builds a grid whose size adapts to
 * the queried domain count and whose sub-places match the scalar factories.
 */

#include <cuda/experimental/stf.cuh>

#include <cstddef>
#include <cstdio>

using namespace cuda::experimental::stf;

int main()
{
  const int dev               = 0;
  const unsigned int ndomains = locality_domain_count(dev);

  // Never 0: a device without locality-domain support reports a single
  // whole-device domain.
  EXPECT(ndomains >= 1);

  exec_place grid = make_locality_domain_grid(dev);

  // The grid adapts to the reported count (which may be 1)
  EXPECT(grid.size() == ndomains);

  for (size_t i = 0; i < grid.size(); i++)
  {
    exec_place sub = grid.get_place(i);

    // Sub-places match the scalar factory
    EXPECT(sub == exec_place::locality_domain(dev, static_cast<int>(i)));

    // Multi-dimensional indexing agrees with linear indexing
    EXPECT(sub == grid.get_place(pos4(i, 0, 0, 0)));

    // Affine data places are the matching domain data places
    data_place affine = sub.affine_data_place();
    EXPECT(affine.is_resolved());
    EXPECT(!affine.is_device());
    EXPECT(device_ordinal(affine) == dev);
    EXPECT(affine == data_place::locality_domain(dev, static_cast<int>(i)));
  }

  // All sub-places are pairwise distinct
  for (size_t i = 0; i < grid.size(); i++)
  {
    for (size_t j = i + 1; j < grid.size(); j++)
    {
      EXPECT(grid.get_place(i) != grid.get_place(j));
    }
  }

  if (ndomains < 2)
  {
    fprintf(stderr, "Device reports a single locality domain: multi-domain grid checks degenerate to size 1.\n");
  }

  return 0;
}
