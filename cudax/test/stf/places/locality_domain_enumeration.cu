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
 * @brief Locality domain enumeration and place identity
 *
 * Checks that the domain count queries are consistent, that the helper
 * enumerates exactly the queried number of domains (never a hardcoded
 * count), and that locality-domain views and data places behave as value
 * types (equality, ordering, hashing, use as map keys). Identity checks do
 * not require locality-domain hardware: (device, domain) pairs are pure
 * identity tokens.
 */

#include <cuda/experimental/stf.cuh>

#include <cstddef>
#include <cstdio>
#include <map>
#include <unordered_map>

using namespace cuda::experimental::stf;

int main()
{
  int ndevs = 0;
  cuda_safe_call(cudaGetDeviceCount(&ndevs));

  // ==== Enumeration: adapts to whatever the device/driver reports ====

  for (int d = 0; d < ndevs; d++)
  {
    const unsigned int cnt = locality_domain_count(d);

    // The helper must agree with the free function
    locality_domain_helper helper(d);
    EXPECT(helper.get_count() == cnt);
    EXPECT(helper.get_device_id() == d);

    // Never 0: a device without locality-domain support reports a single
    // whole-device domain.
    EXPECT(cnt >= 1);

    // Enumerate every domain through the helper
    for (size_t i = 0; i < helper.get_count(); i++)
    {
      locality_domain_view v = helper.get_view(i);
      EXPECT(v.devid == d);
      EXPECT(v.domain_id == static_cast<int>(i));
      EXPECT(v == locality_domain_view(d, static_cast<int>(i)));
    }

    // Out-of-range ordinals are rejected on every build type (the check is
    // an EXPECT, not an assertion that compiles out in release builds)
    bool threw = false;
    try
    {
      auto v = helper.get_view(helper.get_count());
      (void) v;
    }
    catch (...)
    {
      threw = true;
    }
    EXPECT(threw);
  }

  // Out-of-range device ordinals are rejected: the count is never 0
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

  // ==== View identity (no hardware requirement) ====

  locality_domain_view v00(0, 0);
  locality_domain_view v01(0, 1);
  locality_domain_view v10(1, 0);

  EXPECT(v00 == locality_domain_view(0, 0));
  EXPECT(v00 != v01);
  EXPECT(v00 != v10);
  EXPECT(v00 < v01);
  EXPECT(v00 < v10);

  // Equal views hash equal
  EXPECT(hash<locality_domain_view>{}(v00) == hash<locality_domain_view>{}(locality_domain_view(0, 0)));

  // ==== Data place identity (no hardware requirement) ====

  data_place dp = data_place::locality_domain(0, 1);
  EXPECT(dp.is_resolved());
  EXPECT(!dp.is_device());
  EXPECT(device_ordinal(dp) == 0);

  // Equality with an identical place, built from ints or from a view
  EXPECT(dp == data_place::locality_domain(0, 1));
  EXPECT(dp == data_place::locality_domain(locality_domain_view(0, 1)));

  // Inequality with different domain, different device, and other place kinds
  EXPECT(dp != data_place::locality_domain(0, 0));
  EXPECT(dp != data_place::locality_domain(1, 1));
  EXPECT(dp != data_place::device(0));
  EXPECT(dp != data_place::host());

  // to_string distinguishes domains on the same device
  EXPECT(dp.to_string() != data_place::locality_domain(0, 0).to_string());

  // ==== Data places as keys in associative containers ====

  {
    std::unordered_map<data_place, int, hash<data_place>> umap;
    umap[data_place::device(0)]             = 1;
    umap[data_place::host()]                = 2;
    umap[data_place::locality_domain(0, 0)] = 100;
    umap[data_place::locality_domain(0, 1)] = 200;

    EXPECT(umap.size() == 4);
    EXPECT(umap[data_place::device(0)] == 1);
    EXPECT(umap[data_place::host()] == 2);
    EXPECT(umap[data_place::locality_domain(0, 0)] == 100);
    EXPECT(umap[data_place::locality_domain(0, 1)] == 200);
  }

  {
    std::map<data_place, int> omap;
    omap[data_place::device(0)]             = 1;
    omap[data_place::host()]                = 2;
    omap[data_place::locality_domain(0, 0)] = 100;
    omap[data_place::locality_domain(0, 1)] = 200;

    EXPECT(omap.size() == 4);
    EXPECT(omap[data_place::locality_domain(0, 0)] == 100);
    EXPECT(omap[data_place::locality_domain(0, 1)] == 200);
  }

  return 0;
}
