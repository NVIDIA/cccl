//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// PTX ISA version libnvcc asks LLVM for (`-target-feature +ptxNN`) when
// compiling device code for a given SM. LLVM's NVPTX backend rejects an SM it
// considers newer than the requested ISA (e.g. "PTX version 8.5 does not
// support target 'sm_103'"), and the toolkit's ptxas must accept the ISA, so
// the value has to grow with the SM. Kept header-only so the table is unit
// tested without loading LLVM (test/freestanding/test_host_portability.cpp).
//
// Floor is 7.8: generated device code uses PTX 7.6 features (e.g. `bmsk`)
// that older ISAs cannot assemble even for sm_75 / sm_80.
constexpr int libnvcc_ptx_isa_for_sm(int sm_version)
{
  if (sm_version == 103 || sm_version >= 121) // GB300 and the sm_121+ family: PTX 8.8 (CUDA 13)
  {
    return 88;
  }
  if (sm_version >= 120) // Blackwell consumer (sm_120/120a/120f): PTX 8.7
  {
    return 87;
  }
  if (sm_version >= 100) // Blackwell datacenter (sm_100/101): PTX 8.5
  {
    return 85;
  }
  if (sm_version >= 90) // Hopper: PTX 8.0
  {
    return 80;
  }
  return 78;
}
