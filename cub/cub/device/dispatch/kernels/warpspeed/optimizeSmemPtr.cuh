/***************************************************************************************************
 * Copyright (c) 2011-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

template <typename T>
static inline __device__ T* optimizeSmemPtr(const T* smemGeneric)
{
  // See https://nvbugspro.nvidia.com/bug/4907996

  // 1. Convert to 32-bit shared memory pointer
  uint32_t smem32 = __cvta_generic_to_shared(smemGeneric);
  // 2. Pretend to NVVM that the 32-bit pointer is modified. This is required to avoid NVVM constant
  // propagation from pulling the smem32 definition into loops and branches in subsequent code.
  asm("" : "+r"(smem32));
  // 3. Make a generic pointer to smem that is constructed using `__cvta_shared_to_generic`. This
  // benefits from an
  //    optimization pass in NVVM that performs the following simplification:
  //    __cvta_generic_to_shared(__cvta_shared_to_generic(x))    => x.
  //    In our case, `x` is smem32, which is exactly what we want.
  return reinterpret_cast<T*>(__cvta_shared_to_generic(smem32));
}
