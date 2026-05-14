// ClangJIT minimal stub for cuda/std/__cstdlib/aligned_alloc.h
//
// Problem: hostjit compiles with _CCCL_ENABLE_FREESTANDING=1 in both device
// and host passes.  The host pass needs ::cuda::std::__aligned_alloc_host, but
// the real header gates that function on _CCCL_HOSTED(), which is 0 in a
// freestanding build.
//
// Solution: replace the entire header with a bare-metal stub that uses only
// compiler builtins (__builtin_malloc, __SIZE_TYPE__) and NO CCCL headers.
// Including CCCL headers from within this stub caused __clang_cuda_device_functions.h
// to be re-processed before __clang_cuda_libdevice_declares.h during device
// compilation, producing "undeclared identifier __nv_ull2float_rz" errors.
//
// __builtin_malloc is a compiler intrinsic — no headers required.
// __SIZE_TYPE__ is a compiler predefined macro equal to the platform size_t type.
//
// Neither path is ever actually called at runtime:
//   - Host pass: CUB dispatch never calls aligned_alloc in our generated source.
//   - Device pass: NV_IF_ELSE_TARGET discards the NV_IS_HOST branch at compile time.

#ifndef _CUDA_STD___CSTDLIB_ALIGNED_ALLOC_H
#define _CUDA_STD___CSTDLIB_ALIGNED_ALLOC_H

#if defined(__CUDA_ARCH__)

// ── Device compilation ────────────────────────────────────────────────────
// Provide cuda::std::aligned_alloc via the CUDA device syscall.
// The NV_IS_HOST branch of the CUB include chain is discarded by Clang's
// "if target" extension, so this function is never actually called.
extern "C" __device__ void* __cuda_syscall_aligned_malloc(__SIZE_TYPE__, __SIZE_TYPE__);

namespace cuda
{
namespace std
{
inline __device__ void* aligned_alloc(__SIZE_TYPE__ __nbytes, __SIZE_TYPE__ __align) noexcept
{
  return ::__cuda_syscall_aligned_malloc(__nbytes, __align);
}
} // namespace std
} // namespace cuda

#else

// ── Host compilation ──────────────────────────────────────────────────────
// Define __aligned_alloc_host unconditionally so the CUB include chain
// compiles even when _CCCL_HOSTED() == 0.  __builtin_malloc needs no headers.
namespace cuda
{
namespace std
{
inline void* __aligned_alloc_host(__SIZE_TYPE__ __nbytes, __SIZE_TYPE__) noexcept
{
  return __builtin_malloc(__nbytes);
}
inline void* aligned_alloc(__SIZE_TYPE__ __nbytes, __SIZE_TYPE__ __align) noexcept
{
  return ::cuda::std::__aligned_alloc_host(__nbytes, __align);
}
} // namespace std
} // namespace cuda

#endif // __CUDA_ARCH__

#endif // _CUDA_STD___CSTDLIB_ALIGNED_ALLOC_H
