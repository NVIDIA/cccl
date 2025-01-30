// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_BARRIER_CLUSTER_ALIGNED_H_
#define _CUDA_PTX_GENERATED_BARRIER_CLUSTER_ALIGNED_H_

/*
// barrier.cluster.arrive.aligned; // PTX ISA 78, SM_90
// .aligned   = { .aligned }
// Marked volatile and as clobbering memory
template <typename = void>
__device__ static inline void barrier_cluster_arrive(
  cuda::ptx::dot_aligned_t);
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void barrier_cluster_arrive(dot_aligned_t)
{
// __aligned == aligned (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.arrive.aligned;" : : : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// barrier.cluster.wait.aligned; // PTX ISA 78, SM_90
// .aligned   = { .aligned }
// Marked volatile and as clobbering memory
template <typename = void>
__device__ static inline void barrier_cluster_wait(
  cuda::ptx::dot_aligned_t);
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_barrier_cluster_wait_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void barrier_cluster_wait(dot_aligned_t)
{
// __aligned == aligned (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.wait.aligned;" : : : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_barrier_cluster_wait_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 780

/*
// barrier.cluster.arrive.sem.aligned; // PTX ISA 80, SM_90
// .sem       = { .release }
// .aligned   = { .aligned }
// Marked volatile and as clobbering memory
template <typename = void>
__device__ static inline void barrier_cluster_arrive(
  cuda::ptx::sem_release_t,
  cuda::ptx::dot_aligned_t);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void barrier_cluster_arrive(sem_release_t, dot_aligned_t)
{
// __sem == sem_release (due to parameter type constraint)
// __aligned == aligned (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.arrive.release.aligned;" : : : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// barrier.cluster.arrive.sem.aligned; // PTX ISA 80, SM_90
// .sem       = { .relaxed }
// .aligned   = { .aligned }
// Marked volatile
template <typename = void>
__device__ static inline void barrier_cluster_arrive(
  cuda::ptx::sem_relaxed_t,
  cuda::ptx::dot_aligned_t);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void barrier_cluster_arrive(sem_relaxed_t, dot_aligned_t)
{
// __sem == sem_relaxed (due to parameter type constraint)
// __aligned == aligned (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" : : :);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// barrier.cluster.wait.sem.aligned; // PTX ISA 80, SM_90
// .sem       = { .acquire }
// .aligned   = { .aligned }
// Marked volatile and as clobbering memory
template <typename = void>
__device__ static inline void barrier_cluster_wait(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::dot_aligned_t);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_barrier_cluster_wait_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void barrier_cluster_wait(sem_acquire_t, dot_aligned_t)
{
// __sem == sem_acquire (due to parameter type constraint)
// __aligned == aligned (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.wait.acquire.aligned;" : : : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_barrier_cluster_wait_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

#endif // _CUDA_PTX_GENERATED_BARRIER_CLUSTER_ALIGNED_H_
