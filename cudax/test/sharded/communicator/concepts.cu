//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Concept conformance of the places communicators (checked with the
 *        `__multi_gpu` concepts themselves) and the `make_communicators`
 *        factory contract: rank = place index, size = group size, native
 *        handle = the place.
 */

#include <cuda/experimental/__multi_gpu/concepts.h>
#include <cuda/experimental/sharded.cuh>

#include <stdexcept>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::place_group;

// Compile-time evidence: both variants model the communicator concept; only
// the full variant exposes all_reduce (each MGMN combine path stays
// reachable).
static_assert(cuda::experimental::__communicator<places_communicator>);
static_assert(cuda::experimental::__communicator<basic_places_communicator>);
static_assert(cuda::experimental::__has_all_reduce<places_communicator, float*, cuda::std::plus<>>);
static_assert(!cuda::experimental::__has_all_reduce<basic_places_communicator, float*, cuda::std::plus<>>);
static_assert(cuda::experimental::__has_all_gather<places_communicator, float*>);
static_assert(cuda::experimental::__has_all_gather<basic_places_communicator, float*>);
static_assert(cuda::experimental::__has_all_gather_v<places_communicator, float*>);
static_assert(cuda::experimental::__has_all_to_all<places_communicator, float*>);
static_assert(cuda::experimental::__has_all_to_all_v<places_communicator, float*>);
static_assert(cuda::experimental::__has_send<places_communicator>);
static_assert(cuda::experimental::__has_recv<places_communicator>);

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  auto group = place_group::by_locality_domains();
  const int n = static_cast<int>(group.size());

  // Factory contract: one communicator per place, rank = place index,
  // size = group size, native handle = the place itself, in group order.
  auto comms = make_communicators(group);
  EXPECT(comms.size() == group.size());
  for (int i = 0; i < n; i++)
  {
    EXPECT(comms[i].rank() == i);
    EXPECT(comms[i].size() == n);
    EXPECT(comms[i].native_handle() == group.place(i));
  }

  // The basic variant shares the same factory and contract.
  auto basic = make_communicators<basic_places_communicator>(group);
  EXPECT(basic.size() == group.size());
  for (int i = 0; i < n; i++)
  {
    EXPECT(basic[i].rank() == i);
    EXPECT(basic[i].size() == n);
    EXPECT(basic[i].native_handle() == group.place(i));
  }

  // Empty place lists are refused.
  bool threw = false;
  try
  {
    make_communicators(::std::vector<cuda::experimental::places::exec_place>{});
  }
  catch (const ::std::invalid_argument&)
  {
    threw = true;
  }
  EXPECT(threw);

  return 0;
}
