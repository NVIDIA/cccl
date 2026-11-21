// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdlib>

#include <nccl.h>

extern "C" ncclResult_t ncclGetVersion(int* version)
{
#if defined(CUDAX_TEST_OLDER_NCCL)
  *version = 22707;
#else
  *version = NCCL_VERSION_CODE;
#endif
  return ncclSuccess;
}

#if defined(CUDAX_TEST_OLDER_NCCL) && NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 2)
// Supply every optional symbol with a deliberately older runtime version. The
// loader must decline it before passing any versioned structure to these APIs.
#  define NCCL_HOSTLIB_ONLY
#  include <nccl_device.h>

#  include <nccl_device/gin_barrier.h>

extern "C" ncclResult_t ncclCommQueryProperties(ncclComm_t, ncclCommProperties_t*)
{
  std::abort();
}
extern "C" ncclResult_t ncclMemAlloc(void**, size_t)
{
  std::abort();
}
extern "C" ncclResult_t ncclMemFree(void*)
{
  std::abort();
}
extern "C" ncclResult_t ncclCommWindowRegister(ncclComm_t, void*, size_t, ncclWindow_t*, int)
{
  std::abort();
}
extern "C" ncclResult_t ncclCommWindowDeregister(ncclComm_t, ncclWindow_t)
{
  std::abort();
}
extern "C" ncclResult_t ncclDevCommCreate(ncclComm_t, const ncclDevCommRequirements_t*, ncclDevComm_t*)
{
  std::abort();
}
extern "C" ncclResult_t ncclDevCommDestroy(ncclComm_t, const ncclDevComm_t*)
{
  std::abort();
}
extern "C" ncclTeam_t ncclTeamLsa(ncclComm_t)
{
  std::abort();
}
extern "C" ncclTeam_t ncclTeamWorld(ncclComm_t)
{
  std::abort();
}
extern "C" ncclResult_t
ncclGinBarrierCreateRequirement(ncclComm_t, ncclTeam_t, int, ncclGinBarrierHandle_t*, ncclDevResourceRequirements_t*)
{
  std::abort();
}
#endif
