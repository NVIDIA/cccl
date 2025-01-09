// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_BARRIER_CLUSTER_H_
#define _CUDA_PTX_GENERATED_BARRIER_CLUSTER_H_

/*
// barrier.cluster.arrive; // PTX ISA 78, SM_90
// Marked volatile and as clobbering memory
template <typename = void>
__device__ static inline void barrier_cluster_arrive();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void barrier_cluster_arrive()
{
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (asm volatile("barrier.cluster.arrive;" : : : "memory");),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();));
}
#endif // __cccl_ptx_isa >= 780

/*
// barrier.cluster.wait; // PTX ISA 78, SM_90
// Marked volatile and as clobbering memory
template <typename = void>
__device__ static inline void barrier_cluster_wait();
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_barrier_cluster_wait_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void barrier_cluster_wait()
{
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (asm volatile("barrier.cluster.wait;" : : : "memory");),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_barrier_cluster_wait_is_not_supported_before_SM_90__();));
}
#endif // __cccl_ptx_isa >= 780

/*
// barrier.cluster.arrive.sem; // PTX ISA 80, SM_90
// .sem       = { .release }
// Marked volatile and as clobbering memory
template <typename = void>
__device__ static inline void barrier_cluster_arrive(
  cuda::ptx::sem_release_t);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void barrier_cluster_arrive(sem_release_t)
{
  // __sem == sem_release (due to parameter type constraint)
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (asm volatile("barrier.cluster.arrive.release;" : : : "memory");),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();));
}
#endif // __cccl_ptx_isa >= 800

/*
// barrier.cluster.arrive.sem; // PTX ISA 80, SM_90
// .sem       = { .relaxed }
// Marked volatile
template <typename = void>
__device__ static inline void barrier_cluster_arrive(
  cuda::ptx::sem_relaxed_t);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void barrier_cluster_arrive(sem_relaxed_t)
{
  // __sem == sem_relaxed (due to parameter type constraint)
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (asm volatile("barrier.cluster.arrive.relaxed;" : : :);),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_barrier_cluster_arrive_is_not_supported_before_SM_90__();));
}
#endif // __cccl_ptx_isa >= 800

/*
// barrier.cluster.wait.sem; // PTX ISA 80, SM_90
// .sem       = { .acquire }
// Marked volatile and as clobbering memory
template <typename = void>
__device__ static inline void barrier_cluster_wait(
  cuda::ptx::sem_acquire_t);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_barrier_cluster_wait_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void barrier_cluster_wait(sem_acquire_t)
{
  // __sem == sem_acquire (due to parameter type constraint)
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (asm volatile("barrier.cluster.wait.acquire;" : : : "memory");),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_barrier_cluster_wait_is_not_supported_before_SM_90__();));
}
#endif // __cccl_ptx_isa >= 800

#endif // _CUDA_PTX_GENERATED_BARRIER_CLUSTER_H_
