/***************************************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <cstdint> // uint*_t

#include "../optimizeSmemPtr.cuh" // optimizeSmemPtr.cuh

struct SmemAllocator
{
  uint32_t mPtrSmem32 = 0;
  int mAllocatedSize  = 0;

  __host__ __device__ inline SmemAllocator();

  // SmemAllocator is a non-copyable, non-movable type. It must be passed by
  // (mutable) reference to be useful.
  SmemAllocator(const SmemAllocator&)             = delete; // Delete copy constructor
  SmemAllocator(SmemAllocator&&)                  = delete; // Delete move constructor
  SmemAllocator& operator=(const SmemAllocator&)  = delete; // Delete copy assignment
  SmemAllocator& operator=(const SmemAllocator&&) = delete; // Delete move assignment

  __host__ __device__ inline void* alloc(uint32_t size, uint32_t align = 0);

  __host__ __device__ inline uint32_t sizeBytes() const;
};

__host__ __device__ inline SmemAllocator::SmemAllocator()
{
#ifdef __CUDA_ARCH__
  extern __shared__ char warpSpeedDynamicSmemBase[];
  mPtrSmem32 = __cvta_generic_to_shared(warpSpeedDynamicSmemBase);
#else
  mPtrSmem32 = 0;
#endif
}

__host__ __device__ inline void* SmemAllocator::alloc(uint32_t size, uint32_t align)
{
  // Align mPtrSmem32 to requested alignment (round-up)
  uint32_t ptrAllocation32 = (mPtrSmem32 + (align - 1)) & ~(align - 1);

  // Move base pointer and update allocated size
  mAllocatedSize += size + ptrAllocation32 - mPtrSmem32;
  mPtrSmem32 = ptrAllocation32 + size;

#ifdef __CUDA_ARCH__ /* device */
  // Convert allocated smem address to generic pointer
  void* mPtrAllocation = __cvta_shared_to_generic(ptrAllocation32);
  // Ensure alignment calculation does not move down into rest of kernel code.
  return optimizeSmemPtr(mPtrAllocation);
#else /* host */
  // Host code is not going to use the return values anyway. So might as well
  // return a nullptr.
  return nullptr;
#endif
}

__host__ __device__ inline uint32_t SmemAllocator::sizeBytes() const
{
  return mAllocatedSize;
}
