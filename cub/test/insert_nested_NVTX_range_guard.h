#pragma once

// Include this file at the top of a unit test for CUB device algorithms to check whether any inserted NVTX ranges nest.

#include <cstdio>
#include <cstdlib>

#include <catch2/catch_test_macros.hpp>

#if defined(__cpp_inline_variables)
inline thread_local bool entered = false;

struct NestedNVTXRangeGuard
{
  NestedNVTXRangeGuard(const char* name)
  {
    UNSCOPED_INFO("Entering NVTX range " << name);
    if (entered)
    {
      FAIL("Nested NVTX range detected");
    }
    entered = true;
  }

  ~NestedNVTXRangeGuard()
  {
    entered = false;
    UNSCOPED_INFO("Leaving NVTX range");
  }
};

// TODO(giannis): Thrust algorithms lead to NVTX nesting, we still want to avoid nested ranges
// on the CUB Device level side. This guard makes sure that when a newly added CUB primitive
// uses another CUB primitive it calls it from the dispatch layer. We can disable the guard
// just for thrust conditionally in the future, since thrust nesting is more often and very
// frequent on the high API level.
#  define _CCCL_BEFORE_NVTX_RANGE_SCOPE(name)
// ::cuda::std::optional<::NestedNVTXRangeGuard> __cub_nvtx3_reentrency_guard;
// NV_IF_TARGET(NV_IS_HOST, __cub_nvtx3_reentrency_guard.emplace(name););
#endif // defined(__cpp_inline_variables)
