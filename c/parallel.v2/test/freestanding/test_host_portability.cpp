//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Host-portability checks for libnvcc that need no LLVM at run time:
//  * the host target identity follows the machine the library was built for;
//  * the PTX ISA table is consistent with what LLVM accepts per SM, including
//    the SM of the GPU this test runs on.
// A failure here reproduces, in one assertion, what otherwise surfaces as an
// LLVM fatal error ("PTX version 8.5 does not support target 'sm_103'") or an
// lld "incompatible with elf_x86_64" link failure deep inside a HostJIT build.

#include <cstdio>

#include <cuda_runtime.h>

#include <libnvcc/host_target.h>
#include <libnvcc/ptx_isa.h>

// ---- Host target identity (compile time) ----
constexpr bool starts_with(const char* s, const char* prefix)
{
  for (; *prefix; ++s, ++prefix)
  {
    if (*s != *prefix)
    {
      return false;
    }
  }
  return true;
}

#if defined(__aarch64__)
static_assert(starts_with(HOSTJIT_HOST_TRIPLE, "aarch64"), "aarch64 host must use an aarch64 host triple");
static_assert(starts_with(HOSTJIT_LLD_EMULATION, "aarch64linux"),
              "aarch64 host must link with the aarch64 lld emulation");
#elif defined(__x86_64__) && !defined(_WIN32)
static_assert(starts_with(HOSTJIT_HOST_TRIPLE, "x86_64"), "x86-64 host must use an x86-64 host triple");
static_assert(starts_with(HOSTJIT_LLD_EMULATION, "elf_x86_64"),
              "x86-64 host must link with the elf_x86_64 lld emulation");
#endif

// ---- PTX ISA table (compile time) ----
// Minimum ISA LLVM's NVPTX backend accepts per SM (llvm/lib/Target/NVPTX/NVPTX.td),
// and the floor libnvcc keeps for generated code.
static_assert(libnvcc_ptx_isa_for_sm(75) == 78, "floor");
static_assert(libnvcc_ptx_isa_for_sm(80) == 78, "floor");
static_assert(libnvcc_ptx_isa_for_sm(86) == 78, "floor");
static_assert(libnvcc_ptx_isa_for_sm(90) >= 80, "sm_90 needs PTX 8.0");
static_assert(libnvcc_ptx_isa_for_sm(100) >= 85, "sm_100 needs PTX 8.5");
static_assert(libnvcc_ptx_isa_for_sm(101) >= 85, "sm_101 needs PTX 8.5");
static_assert(libnvcc_ptx_isa_for_sm(103) >= 88, "sm_103 (GB300) needs PTX 8.8");
static_assert(libnvcc_ptx_isa_for_sm(120) >= 87, "sm_120 needs PTX 8.7");
static_assert(libnvcc_ptx_isa_for_sm(121) >= 88, "sm_121 needs PTX 8.8");
// Monotonic within a family: a newer SM never asks for an older ISA.
static_assert(libnvcc_ptx_isa_for_sm(103) >= libnvcc_ptx_isa_for_sm(100), "monotonic");
static_assert(libnvcc_ptx_isa_for_sm(121) >= libnvcc_ptx_isa_for_sm(120), "monotonic");

// Minimum PTX ISA per SM as LLVM requires it; mirrors the static_asserts for
// the SMs CI runs on, so a new GPU in the matrix fails here first.
static int min_ptx_isa_for_sm(int sm)
{
  if (sm == 103 || sm >= 121)
  {
    return 88;
  }
  if (sm >= 120)
  {
    return 87;
  }
  if (sm >= 100)
  {
    return 85;
  }
  if (sm >= 90)
  {
    return 80;
  }
  return 70;
}

int main()
{
  // Like the other freestanding tests, a CUDA runtime failure is a test
  // failure: the per-GPU ISA check below must not be skipped silently.
  int device = 0;
  cudaDeviceProp prop{};
  cudaError_t err = cudaGetDevice(&device);
  if (err == cudaSuccess)
  {
    err = cudaGetDeviceProperties(&prop, device);
  }
  if (err != cudaSuccess)
  {
    std::fprintf(stderr, "CUDA runtime initialization failed: %s\n", cudaGetErrorString(err));
    return 1;
  }
  const int sm  = prop.major * 10 + prop.minor;
  const int isa = libnvcc_ptx_isa_for_sm(sm);
  std::printf(
    "host triple %s, cpu %s, lld -m %s; device sm_%d -> PTX ISA %d (min %d)\n",
    HOSTJIT_HOST_TRIPLE,
    HOSTJIT_HOST_CPU,
    HOSTJIT_LLD_EMULATION,
    sm,
    isa,
    min_ptx_isa_for_sm(sm));
  // Explicit checks rather than assert(): the test suite builds Release
  // (NDEBUG), and this check must fail there too.
  if (isa < min_ptx_isa_for_sm(sm))
  {
    std::fprintf(stderr, "libnvcc PTX ISA table is too old for sm_%d: PTX %d < %d\n", sm, isa, min_ptx_isa_for_sm(sm));
    return 1;
  }
  if (isa < 78)
  {
    std::fprintf(stderr, "PTX ISA floor violated for sm_%d: PTX %d < 78\n", sm, isa);
    return 1;
  }
  return 0;
}
