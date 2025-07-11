#pragma once

// Include this file at the top of a unit test for CUB device algorithms to check whether any inserted NVTX ranges nest.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <catch2/catch_test_macros.hpp>

inline thread_local const char* current_nvtx_range_name = nullptr;

struct NestedNVTXRangeGuard
{
  // complain only about CUB nested NVTX ranges, not Thrust
  bool inside_cub_range;

  explicit NestedNVTXRangeGuard(const char* name)
      : inside_cub_range(strstr(name, "cub::") == name)
  {
    if (inside_cub_range)
    {
      if (current_nvtx_range_name)
      {
        FAIL("Nested NVTX range detected. Entered " << current_nvtx_range_name << ". Now entering " << name);
      }
      current_nvtx_range_name = name;
    }
  }

  ~NestedNVTXRangeGuard()
  {
    if (inside_cub_range)
    {
      current_nvtx_range_name = nullptr;
    }
  }
};

#define _CCCL_BEFORE_NVTX_RANGE_SCOPE(name)                                   \
  ::cuda::std::optional<::NestedNVTXRangeGuard> __cub_nvtx3_reentrency_guard; \
  NV_IF_TARGET(NV_IS_HOST, __cub_nvtx3_reentrency_guard.emplace(name););
