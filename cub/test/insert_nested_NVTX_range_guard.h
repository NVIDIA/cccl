#pragma once

// Include this file at the top of a unit test for CUB device algorithms to check whether any inserted NVTX ranges nest.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <catch2/catch_test_macros.hpp>

inline thread_local bool entered = false;

struct NestedNVTXRangeGuard
{
  // complain only about CUB nested NVTX ranges, not Thrust
  bool inside_cub_range;

  explicit NestedNVTXRangeGuard(const char* name)
      : inside_cub_range(strstr(name, "cub::") == name)
  {
    UNSCOPED_INFO("Entering NVTX range " << name);
    if (inside_cub_range)
    {
      if (entered)
      {
        FAIL("Nested NVTX range detected");
      }
      entered = true;
    }
  }

  ~NestedNVTXRangeGuard()
  {
    if (inside_cub_range)
    {
      entered = false;
    }
    UNSCOPED_INFO("Leaving NVTX range");
  }
};

#define _CCCL_BEFORE_NVTX_RANGE_SCOPE(name)                                   \
  ::cuda::std::optional<::NestedNVTXRangeGuard> __cub_nvtx3_reentrency_guard; \
  NV_IF_TARGET(NV_IS_HOST, __cub_nvtx3_reentrency_guard.emplace(name););
