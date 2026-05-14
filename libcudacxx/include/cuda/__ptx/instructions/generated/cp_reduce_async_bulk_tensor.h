// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_CP_REDUCE_ASYNC_BULK_TENSOR_H_
#define _CUDA_PTX_GENERATED_CP_REDUCE_ASYNC_BULK_TENSOR_H_

/*
// cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const int32_t (&tensorCoords)[1],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_t<_Op> __op,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec
                  || __op == op_and_op || __op == op_or_op || __op == op_xor_op,
                "");
  if constexpr (__op == op_add)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.add.tile.bulk_group [%0, {%1}], [%2];"
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_min)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.min.tile.bulk_group [%0, {%1}], [%2];"
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.max.tile.bulk_group [%0, {%1}], [%2];"
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_inc)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.inc.tile.bulk_group [%0, {%1}], [%2];"
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_dec)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.dec.tile.bulk_group [%0, {%1}], [%2];"
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_and_op)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.and.tile.bulk_group [%0, {%1}], [%2];"
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_or_op)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.or.tile.bulk_group [%0, {%1}], [%2];"
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_xor_op)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.xor.tile.bulk_group [%0, {%1}], [%2];"
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const int32_t (&tensorCoords)[2],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_t<_Op> __op,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec
                  || __op == op_and_op || __op == op_or_op || __op == op_xor_op,
                "");
  if constexpr (__op == op_add)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2}], [%3];"
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__tensorCoords[1]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_min)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.min.tile.bulk_group [%0, {%1, %2}], [%3];"
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__tensorCoords[1]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.max.tile.bulk_group [%0, {%1, %2}], [%3];"
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__tensorCoords[1]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_inc)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.inc.tile.bulk_group [%0, {%1, %2}], [%3];"
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__tensorCoords[1]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_dec)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.dec.tile.bulk_group [%0, {%1, %2}], [%3];"
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__tensorCoords[1]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_and_op)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.and.tile.bulk_group [%0, {%1, %2}], [%3];"
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__tensorCoords[1]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_or_op)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.or.tile.bulk_group [%0, {%1, %2}], [%3];"
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__tensorCoords[1]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_xor_op)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.xor.tile.bulk_group [%0, {%1, %2}], [%3];"
        :
        : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__tensorCoords[1]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const int32_t (&tensorCoords)[3],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_t<_Op> __op,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec
                  || __op == op_and_op || __op == op_or_op || __op == op_xor_op,
                "");
  if constexpr (__op == op_add)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2, %3}], [%4];"
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_min)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.min.tile.bulk_group [%0, {%1, %2, %3}], [%4];"
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.max.tile.bulk_group [%0, {%1, %2, %3}], [%4];"
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_inc)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.inc.tile.bulk_group [%0, {%1, %2, %3}], [%4];"
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_dec)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.dec.tile.bulk_group [%0, {%1, %2, %3}], [%4];"
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_and_op)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.and.tile.bulk_group [%0, {%1, %2, %3}], [%4];"
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_or_op)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.or.tile.bulk_group [%0, {%1, %2, %3}], [%4];"
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_xor_op)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.xor.tile.bulk_group [%0, {%1, %2, %3}], [%4];"
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const int32_t (&tensorCoords)[4],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_t<_Op> __op,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec
                  || __op == op_and_op || __op == op_or_op || __op == op_xor_op,
                "");
  if constexpr (__op == op_add)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5];"
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_min)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.min.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5];"
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.max.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5];"
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_inc)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.inc.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5];"
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_dec)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.dec.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5];"
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_and_op)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.and.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5];"
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_or_op)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.or.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5];"
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_xor_op)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.xor.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5];"
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const int32_t (&tensorCoords)[5],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_t<_Op> __op,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec
                  || __op == op_and_op || __op == op_or_op || __op == op_xor_op,
                "");
  if constexpr (__op == op_add)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6];"
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_min)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.min.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6];"
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.max.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6];"
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_inc)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.inc.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6];"
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_dec)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.dec.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6];"
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_and_op)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.and.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6];"
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_or_op)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.or.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6];"
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_xor_op)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.xor.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6];"
        :
        : "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride,
tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor_override(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const void* gAddrToOverride,
  const int32_t (&tensorCoords)[1],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor_override(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_t<_Op> __op,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec
                  || __op == op_and_op || __op == op_or_op || __op == op_xor_op,
                "");
  if constexpr (__op == op_add)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.add.tile.override::global_address.bulk_group [%0, %1, "
        "{%2}], [%3];"
        :
        : "l"(__tensorMap), "l"(__gAddrToOverride), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_min)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.min.tile.override::global_address.bulk_group [%0, %1, "
        "{%2}], [%3];"
        :
        : "l"(__tensorMap), "l"(__gAddrToOverride), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.max.tile.override::global_address.bulk_group [%0, %1, "
        "{%2}], [%3];"
        :
        : "l"(__tensorMap), "l"(__gAddrToOverride), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_inc)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.inc.tile.override::global_address.bulk_group [%0, %1, "
        "{%2}], [%3];"
        :
        : "l"(__tensorMap), "l"(__gAddrToOverride), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_dec)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.dec.tile.override::global_address.bulk_group [%0, %1, "
        "{%2}], [%3];"
        :
        : "l"(__tensorMap), "l"(__gAddrToOverride), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_and_op)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.and.tile.override::global_address.bulk_group [%0, %1, "
        "{%2}], [%3];"
        :
        : "l"(__tensorMap), "l"(__gAddrToOverride), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_or_op)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.or.tile.override::global_address.bulk_group [%0, %1, {%2}], "
        "[%3];"
        :
        : "l"(__tensorMap), "l"(__gAddrToOverride), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_xor_op)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.xor.tile.override::global_address.bulk_group [%0, %1, "
        "{%2}], [%3];"
        :
        : "l"(__tensorMap), "l"(__gAddrToOverride), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.reduce.async.bulk.tensor.1d.dst.src.op.tile.override::global_address.override::global_dim.bulk_group [tensorMap,
gAddrToOverride, tensorSizeToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor_override(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const void* gAddrToOverride,
  const B16 (&tensorSizeToOverride)[1],
  const int32_t (&tensorCoords)[1],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor_override(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_t<_Op> __op,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const _B16 (&__tensorSizeToOverride)[1],
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec
                  || __op == op_and_op || __op == op_or_op || __op == op_xor_op,
                "");
  static_assert(sizeof(_B16) == 2, "");
  if constexpr (__op == op_add)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.add.tile.override::global_address.override::global_dim.bulk_"
        "group [%0, %1, {%2}, {%3}], [%4];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_min)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.min.tile.override::global_address.override::global_dim.bulk_"
        "group [%0, %1, {%2}, {%3}], [%4];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.max.tile.override::global_address.override::global_dim.bulk_"
        "group [%0, %1, {%2}, {%3}], [%4];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_inc)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.inc.tile.override::global_address.override::global_dim.bulk_"
        "group [%0, %1, {%2}, {%3}], [%4];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_dec)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.dec.tile.override::global_address.override::global_dim.bulk_"
        "group [%0, %1, {%2}, {%3}], [%4];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_and_op)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.and.tile.override::global_address.override::global_dim.bulk_"
        "group [%0, %1, {%2}, {%3}], [%4];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_or_op)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.or.tile.override::global_address.override::global_dim.bulk_"
        "group [%0, %1, {%2}, {%3}], [%4];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_xor_op)
  {
    asm("cp.reduce.async.bulk.tensor.1d.global.shared::cta.xor.tile.override::global_address.override::global_dim.bulk_"
        "group [%0, %1, {%2}, {%3}], [%4];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "r"(__tensorCoords[0]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride,
tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor_override(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const void* gAddrToOverride,
  const int32_t (&tensorCoords)[2],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor_override(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_t<_Op> __op,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec
                  || __op == op_and_op || __op == op_or_op || __op == op_xor_op,
                "");
  if constexpr (__op == op_add)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3}], [%4];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_min)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.min.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3}], [%4];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.max.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3}], [%4];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_inc)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.inc.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3}], [%4];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_dec)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.dec.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3}], [%4];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_and_op)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.and.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3}], [%4];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_or_op)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.or.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3}], [%4];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_xor_op)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.xor.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3}], [%4];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.reduce.async.bulk.tensor.2d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group
[tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true, cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor_override(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const void* gAddrToOverride,
  const B16 (&tensorSizeToOverride)[2],
  const B32 (&tensorLowerStrideToOverride)[1],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[2],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true,
          ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor_override(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_t<_Op> __op,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const _B16 (&__tensorSizeToOverride)[2],
  const _B32 (&__tensorLowerStrideToOverride)[1],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec
                  || __op == op_and_op || __op == op_or_op || __op == op_xor_op,
                "");
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  if constexpr (__op == op_add)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3}, {%4}, %5, {%6, %7}], [%8];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_min)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.min.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3}, {%4}, %5, {%6, %7}], [%8];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.max.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3}, {%4}, %5, {%6, %7}], [%8];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_inc)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.inc.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3}, {%4}, %5, {%6, %7}], [%8];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_dec)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.dec.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3}, {%4}, %5, {%6, %7}], [%8];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_and_op)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.and.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3}, {%4}, %5, {%6, %7}], [%8];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_or_op)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.or.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3}, {%4}, %5, {%6, %7}], [%8];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_xor_op)
  {
    asm("cp.reduce.async.bulk.tensor.2d.global.shared::cta.xor.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3}, {%4}, %5, {%6, %7}], [%8];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride,
tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor_override(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const void* gAddrToOverride,
  const int32_t (&tensorCoords)[3],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor_override(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_t<_Op> __op,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec
                  || __op == op_and_op || __op == op_or_op || __op == op_xor_op,
                "");
  if constexpr (__op == op_add)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3, %4}], [%5];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_min)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.min.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3, %4}], [%5];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.max.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3, %4}], [%5];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_inc)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.inc.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3, %4}], [%5];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_dec)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.dec.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3, %4}], [%5];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_and_op)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.and.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3, %4}], [%5];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_or_op)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.or.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3, %4}], [%5];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_xor_op)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.xor.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3, %4}], [%5];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.reduce.async.bulk.tensor.3d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group
[tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true, cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor_override(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const void* gAddrToOverride,
  const B16 (&tensorSizeToOverride)[3],
  const B32 (&tensorLowerStrideToOverride)[2],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[3],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true,
          ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor_override(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_t<_Op> __op,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const _B16 (&__tensorSizeToOverride)[3],
  const _B32 (&__tensorLowerStrideToOverride)[2],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec
                  || __op == op_and_op || __op == op_or_op || __op == op_xor_op,
                "");
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  if constexpr (__op == op_add)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3, %4}, {%5, %6}, %7, {%8, %9, %10}], [%11];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_min)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.min.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3, %4}, {%5, %6}, %7, {%8, %9, %10}], [%11];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.max.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3, %4}, {%5, %6}, %7, {%8, %9, %10}], [%11];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_inc)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.inc.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3, %4}, {%5, %6}, %7, {%8, %9, %10}], [%11];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_dec)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.dec.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3, %4}, {%5, %6}, %7, {%8, %9, %10}], [%11];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_and_op)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.and.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3, %4}, {%5, %6}, %7, {%8, %9, %10}], [%11];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_or_op)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.or.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3, %4}, {%5, %6}, %7, {%8, %9, %10}], [%11];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_xor_op)
  {
    asm("cp.reduce.async.bulk.tensor.3d.global.shared::cta.xor.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3, %4}, {%5, %6}, %7, {%8, %9, %10}], [%11];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride,
tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor_override(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const void* gAddrToOverride,
  const int32_t (&tensorCoords)[4],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor_override(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_t<_Op> __op,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec
                  || __op == op_and_op || __op == op_or_op || __op == op_xor_op,
                "");
  if constexpr (__op == op_add)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3, %4, %5}], [%6];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_min)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.min.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3, %4, %5}], [%6];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.max.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3, %4, %5}], [%6];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_inc)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.inc.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3, %4, %5}], [%6];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_dec)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.dec.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3, %4, %5}], [%6];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_and_op)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.and.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3, %4, %5}], [%6];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_or_op)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.or.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3, %4, %5}], [%6];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_xor_op)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.xor.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3, %4, %5}], [%6];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.reduce.async.bulk.tensor.4d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group
[tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true, cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor_override(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const void* gAddrToOverride,
  const B16 (&tensorSizeToOverride)[4],
  const B32 (&tensorLowerStrideToOverride)[3],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[4],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true,
          ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor_override(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_t<_Op> __op,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const _B16 (&__tensorSizeToOverride)[4],
  const _B32 (&__tensorLowerStrideToOverride)[3],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec
                  || __op == op_and_op || __op == op_or_op || __op == op_xor_op,
                "");
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  if constexpr (__op == op_add)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3, %4, %5}, {%6, %7, %8}, %9, {%10, %11, %12, %13}], [%14];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_min)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.min.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3, %4, %5}, {%6, %7, %8}, %9, {%10, %11, %12, %13}], [%14];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.max.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3, %4, %5}, {%6, %7, %8}, %9, {%10, %11, %12, %13}], [%14];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_inc)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.inc.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3, %4, %5}, {%6, %7, %8}, %9, {%10, %11, %12, %13}], [%14];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_dec)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.dec.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3, %4, %5}, {%6, %7, %8}, %9, {%10, %11, %12, %13}], [%14];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_and_op)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.and.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3, %4, %5}, {%6, %7, %8}, %9, {%10, %11, %12, %13}], [%14];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_or_op)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.or.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3, %4, %5}, {%6, %7, %8}, %9, {%10, %11, %12, %13}], [%14];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_xor_op)
  {
    asm("cp.reduce.async.bulk.tensor.4d.global.shared::cta.xor.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3, %4, %5}, {%6, %7, %8}, %9, {%10, %11, %12, %13}], [%14];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.override::global_address.bulk_group [tensorMap, gAddrToOverride,
tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor_override(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const void* gAddrToOverride,
  const int32_t (&tensorCoords)[5],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor_override(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_t<_Op> __op,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec
                  || __op == op_and_op || __op == op_or_op || __op == op_xor_op,
                "");
  if constexpr (__op == op_add)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3, %4, %5, %6}], [%7];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_min)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.min.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3, %4, %5, %6}], [%7];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.max.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3, %4, %5, %6}], [%7];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_inc)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.inc.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3, %4, %5, %6}], [%7];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_dec)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.dec.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3, %4, %5, %6}], [%7];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_and_op)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.and.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3, %4, %5, %6}], [%7];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_or_op)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.or.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3, %4, %5, %6}], [%7];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_xor_op)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.xor.tile.override::global_address.bulk_group [%0, %1, {%2, "
        "%3, %4, %5, %6}], [%7];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.reduce.async.bulk.tensor.5d.dst.src.op.tile.override::global_address.override::global_dim_stride.bulk_group
[tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
// .dst       = { .global }
// .src       = { .shared::cta }
// .op        = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true, cuda::ptx::dot_op Op>
__device__ static inline void cp_reduce_async_bulk_tensor_override(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  cuda::ptx::op_t<Op> op,
  const void* tensorMap,
  const void* gAddrToOverride,
  const B16 (&tensorSizeToOverride)[5],
  const B32 (&tensorLowerStrideToOverride)[4],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[5],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true,
          ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline void cp_reduce_async_bulk_tensor_override(
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::op_t<_Op> __op,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const _B16 (&__tensorSizeToOverride)[5],
  const _B32 (&__tensorLowerStrideToOverride)[4],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  static_assert(__op == op_add || __op == op_min || __op == op_max || __op == op_inc || __op == op_dec
                  || __op == op_and_op || __op == op_or_op || __op == op_xor_op,
                "");
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  if constexpr (__op == op_add)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3, %4, %5, %6}, {%7, %8, %9, %10}, %11, {%12, %13, %14, %15, %16}], [%17];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_min)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.min.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3, %4, %5, %6}, {%7, %8, %9, %10}, %11, {%12, %13, %14, %15, %16}], [%17];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.max.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3, %4, %5, %6}, {%7, %8, %9, %10}, %11, {%12, %13, %14, %15, %16}], [%17];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_inc)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.inc.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3, %4, %5, %6}, {%7, %8, %9, %10}, %11, {%12, %13, %14, %15, %16}], [%17];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_dec)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.dec.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3, %4, %5, %6}, {%7, %8, %9, %10}, %11, {%12, %13, %14, %15, %16}], [%17];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_and_op)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.and.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3, %4, %5, %6}, {%7, %8, %9, %10}, %11, {%12, %13, %14, %15, %16}], [%17];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_or_op)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.or.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3, %4, %5, %6}, {%7, %8, %9, %10}, %11, {%12, %13, %14, %15, %16}], [%17];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
  else if constexpr (__op == op_xor_op)
  {
    asm("cp.reduce.async.bulk.tensor.5d.global.shared::cta.xor.tile.override::global_address.override::global_dim_"
        "stride.bulk_group [%0, %1, {%2, %3, %4, %5, %6}, {%7, %8, %9, %10}, %11, {%12, %13, %14, %15, %16}], [%17];"
        :
        : "l"(__tensorMap),
          "l"(__gAddrToOverride),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
          "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
          "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__srcMem))
        : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

#endif // _CUDA_PTX_GENERATED_CP_REDUCE_ASYNC_BULK_TENSOR_H_
