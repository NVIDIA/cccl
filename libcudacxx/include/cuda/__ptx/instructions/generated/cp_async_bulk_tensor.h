// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_CP_ASYNC_BULK_TENSOR_H_
#define _CUDA_PTX_GENERATED_CP_ASYNC_BULK_TENSOR_H_

/*
// cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar];//
1a. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[1],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_cluster_t,
  space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[1],
  _CUDA_VSTD::uint64_t* __smem_bar)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2}], [%3];// "
         "1a." : : "r"(__as_ptr_smem(__dstMem)),
         "l"(__tensorMap),
         "r"(__tensorCoords[0]),
         "r"(__as_ptr_smem(__smem_bar)) : "memory");),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.1d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3a. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
template <typename = void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[1],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_global_t,
  space_shared_t,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[1],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (asm("cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group [%0, {%1}], [%2]; // 3a." : : "l"(__tensorMap),
         "r"(__tensorCoords[0]),
         "r"(__as_ptr_smem(__srcMem)) : "memory");),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar];//
1b. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[2],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_cluster_t,
  space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[2],
  _CUDA_VSTD::uint64_t* __smem_bar)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], "
         "[%4];// 1b." : : "r"(__as_ptr_smem(__dstMem)),
         "l"(__tensorMap),
         "r"(__tensorCoords[0]),
         "r"(__tensorCoords[1]),
         "r"(__as_ptr_smem(__smem_bar)) : "memory");),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.2d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3b. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
template <typename = void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[2],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_global_t,
  space_shared_t,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[2],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (asm("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%1, %2}], [%3]; // 3b." : : "l"(__tensorMap),
         "r"(__tensorCoords[0]),
         "r"(__tensorCoords[1]),
         "r"(__as_ptr_smem(__srcMem)) : "memory");),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar];//
1c. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[3],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_cluster_t,
  space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[3],
  _CUDA_VSTD::uint64_t* __smem_bar)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], "
         "[%5];// 1c." : : "r"(__as_ptr_smem(__dstMem)),
         "l"(__tensorMap),
         "r"(__tensorCoords[0]),
         "r"(__tensorCoords[1]),
         "r"(__tensorCoords[2]),
         "r"(__as_ptr_smem(__smem_bar)) : "memory");),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.3d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3c. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
template <typename = void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[3],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_global_t,
  space_shared_t,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[3],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (asm("cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3}], [%4]; // 3c." : : "l"(
           __tensorMap),
         "r"(__tensorCoords[0]),
         "r"(__tensorCoords[1]),
         "r"(__tensorCoords[2]),
         "r"(__as_ptr_smem(__srcMem)) : "memory");),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar];//
1d. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[4],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_cluster_t,
  space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[4],
  _CUDA_VSTD::uint64_t* __smem_bar)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, "
         "%5}], [%6];// 1d." : : "r"(__as_ptr_smem(__dstMem)),
         "l"(__tensorMap),
         "r"(__tensorCoords[0]),
         "r"(__tensorCoords[1]),
         "r"(__tensorCoords[2]),
         "r"(__tensorCoords[3]),
         "r"(__as_ptr_smem(__smem_bar)) : "memory");),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.4d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3d. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
template <typename = void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[4],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_global_t,
  space_shared_t,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[4],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (asm("cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5]; // 3d." : : "l"(
           __tensorMap),
         "r"(__tensorCoords[0]),
         "r"(__tensorCoords[1]),
         "r"(__tensorCoords[2]),
         "r"(__tensorCoords[3]),
         "r"(__as_ptr_smem(__srcMem)) : "memory");),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar];//
1e. PTX ISA 80, SM_90
// .dst       = { .shared::cluster }
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_cluster_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[5],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_cluster_t,
  space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[5],
  _CUDA_VSTD::uint64_t* __smem_bar)
{
  // __space == space_cluster (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, "
         "%6}], [%7];// 1e." : : "r"(__as_ptr_smem(__dstMem)),
         "l"(__tensorMap),
         "r"(__tensorCoords[0]),
         "r"(__tensorCoords[1]),
         "r"(__tensorCoords[2]),
         "r"(__tensorCoords[3]),
         "r"(__tensorCoords[4]),
         "r"(__as_ptr_smem(__smem_bar)) : "memory");),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.5d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3e. PTX ISA 80, SM_90
// .dst       = { .global }
// .src       = { .shared::cta }
template <typename = void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_global_t,
  cuda::ptx::space_shared_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[5],
  const void* srcMem);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  space_global_t,
  space_shared_t,
  const void* __tensorMap,
  const _CUDA_VSTD::int32_t (&__tensorCoords)[5],
  const void* __srcMem)
{
  // __space == space_global (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (asm("cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6]; // 3e." : : "l"(
           __tensorMap),
         "r"(__tensorCoords[0]),
         "r"(__tensorCoords[1]),
         "r"(__tensorCoords[2]),
         "r"(__tensorCoords[3]),
         "r"(__tensorCoords[4]),
         "r"(__as_ptr_smem(__srcMem)) : "memory");),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();));
}
#endif // __cccl_ptx_isa >= 800

#endif // _CUDA_PTX_GENERATED_CP_ASYNC_BULK_TENSOR_H_
