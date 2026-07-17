// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_CP_REDUCE_ASYNC_BULK_BF16_H_
#define _CUDA_PTX_GENERATED_CP_REDUCE_ASYNC_BULK_BF16_H_

/*
// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 4. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .bf16 }
// .op        = { .min }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_min_t,
  __nv_bfloat16* dstMem,
  const __nv_bfloat16* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 800
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_min_t,
  __nv_bfloat16* __dstMem,
  const __nv_bfloat16* __srcMem,
  ::cuda::std::uint32_t __size)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __type == type_bf16 (due to parameter type constraint)
  // __op == op_min (due to parameter type constraint)
  asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.min.bf16  [%0], [%1], %2; // 4."
      :
      : "l"(__as_ptr_gmem(__dstMem)), "r"(__as_ptr_smem(__srcMem)), "r"(__size)
      : "memory");
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.dst.src.bulk_group.op.type  [dstMem], [srcMem], size; // 4. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .bf16 }
// .op        = { .max }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_max_t,
  __nv_bfloat16* dstMem,
  const __nv_bfloat16* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 800
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_max_t,
  __nv_bfloat16* __dstMem,
  const __nv_bfloat16* __srcMem,
  ::cuda::std::uint32_t __size)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __type == type_bf16 (due to parameter type constraint)
  // __op == op_max (due to parameter type constraint)
  asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.max.bf16  [%0], [%1], %2; // 4."
      :
      : "l"(__as_ptr_gmem(__dstMem)), "r"(__as_ptr_smem(__srcMem)), "r"(__size)
      : "memory");
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.dst.src.bulk_group.op.noftz.type  [dstMem], [srcMem], size; // 5. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .type      = { .bf16 }
// .op        = { .add }
template <typename = void>
__device__ static inline void cp_reduce_async_bulk(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_add_t,
  __nv_bfloat16* dstMem,
  const __nv_bfloat16* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 800
template <typename = void>
_CCCL_DEVICE static inline void cp_reduce_async_bulk(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_add_t,
  __nv_bfloat16* __dstMem,
  const __nv_bfloat16* __srcMem,
  ::cuda::std::uint32_t __size)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  // __type == type_bf16 (due to parameter type constraint)
  // __op == op_add (due to parameter type constraint)
  asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16  [%0], [%1], %2; // 5."
      :
      : "l"(__as_ptr_gmem(__dstMem)), "r"(__as_ptr_smem(__srcMem)), "r"(__size)
      : "memory");
}
#endif // __cccl_ptx_isa >= 800

#endif // _CUDA_PTX_GENERATED_CP_REDUCE_ASYNC_BULK_BF16_H_
