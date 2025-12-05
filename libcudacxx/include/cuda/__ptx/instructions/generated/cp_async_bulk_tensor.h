// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_CP_ASYNC_BULK_TENSOR_H_
#define _CUDA_PTX_GENERATED_CP_ASYNC_BULK_TENSOR_H_

/*
// cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar]; //
PTX ISA 80, SM_90
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
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  ::cuda::std::uint64_t* __smem_bar)
{
// __space == space_cluster (due to parameter type constraint)
// __space == space_global (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2}], [%3];"
      :
      : "r"(__as_ptr_smem(__dstMem)), "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar]; //
PTX ISA 86, SM_90
// .dst       = { .shared::cta }
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_shared_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[1],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  ::cuda::std::uint64_t* __smem_bar)
{
// __space == space_shared (due to parameter type constraint)
// __space == space_global (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2}], [%3];"
      :
      : "r"(__as_ptr_smem(__dstMem)), "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group [dstMem], [tensorMap, tensorCoords],
[smem_bar]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
// .dst       = { .shared::cta }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_shared_t,
  cuda::ptx::space_global_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[1],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_cp_async_bulk_tensor_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  if constexpr (__cta_group == cta_group_1)
  {
    asm("cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 [%0], [%1, {%2}], "
        "[%3];"
        :
        : "r"(__as_ptr_smem(__dstMem)), "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2 [%0], [%1, {%2}], "
        "[%3];"
        :
        : "r"(__as_ptr_smem(__dstMem)), "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_async_bulk_tensor_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// cp.async.bulk.tensor.1d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
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
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  const void* __srcMem)
{
// __space == space_global (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group [%0, {%1}], [%2];"
      :
      : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__as_ptr_smem(__srcMem))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar]; //
PTX ISA 80, SM_90
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
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  ::cuda::std::uint64_t* __smem_bar)
{
// __space == space_cluster (due to parameter type constraint)
// __space == space_global (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar]; //
PTX ISA 86, SM_90
// .dst       = { .shared::cta }
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_shared_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[2],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  ::cuda::std::uint64_t* __smem_bar)
{
// __space == space_shared (due to parameter type constraint)
// __space == space_global (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group [dstMem], [tensorMap, tensorCoords],
[smem_bar]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
// .dst       = { .shared::cta }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_shared_t,
  cuda::ptx::space_global_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[2],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_cp_async_bulk_tensor_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  if constexpr (__cta_group == cta_group_1)
  {
    asm("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 [%0], [%1, {%2, "
        "%3}], [%4];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2 [%0], [%1, {%2, "
        "%3}], [%4];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_async_bulk_tensor_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// cp.async.bulk.tensor.2d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
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
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  const void* __srcMem)
{
// __space == space_global (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%1, %2}], [%3];"
      :
      : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__tensorCoords[1]), "r"(__as_ptr_smem(__srcMem))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar]; //
PTX ISA 80, SM_90
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
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  ::cuda::std::uint64_t* __smem_bar)
{
// __space == space_cluster (due to parameter type constraint)
// __space == space_global (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar]; //
PTX ISA 86, SM_90
// .dst       = { .shared::cta }
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_shared_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[3],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  ::cuda::std::uint64_t* __smem_bar)
{
// __space == space_shared (due to parameter type constraint)
// __space == space_global (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group [dstMem], [tensorMap, tensorCoords],
[smem_bar]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
// .dst       = { .shared::cta }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_shared_t,
  cuda::ptx::space_global_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[3],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_cp_async_bulk_tensor_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  if constexpr (__cta_group == cta_group_1)
  {
    asm("cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 [%0], [%1, {%2, %3, "
        "%4}], [%5];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2 [%0], [%1, {%2, %3, "
        "%4}], [%5];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_async_bulk_tensor_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// cp.async.bulk.tensor.3d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
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
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  const void* __srcMem)
{
// __space == space_global (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3}], [%4];"
      :
      : "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__as_ptr_smem(__srcMem))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar]; //
PTX ISA 80, SM_90
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
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  ::cuda::std::uint64_t* __smem_bar)
{
// __space == space_cluster (due to parameter type constraint)
// __space == space_global (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], "
      "[%6];"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar]; //
PTX ISA 86, SM_90
// .dst       = { .shared::cta }
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_shared_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[4],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  ::cuda::std::uint64_t* __smem_bar)
{
// __space == space_shared (due to parameter type constraint)
// __space == space_global (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], [%6];"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group [dstMem], [tensorMap, tensorCoords],
[smem_bar]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
// .dst       = { .shared::cta }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_shared_t,
  cuda::ptx::space_global_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[4],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_cp_async_bulk_tensor_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  if constexpr (__cta_group == cta_group_1)
  {
    asm("cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 [%0], [%1, {%2, %3, "
        "%4, %5}], [%6];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2 [%0], [%1, {%2, %3, "
        "%4, %5}], [%6];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_async_bulk_tensor_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// cp.async.bulk.tensor.4d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
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
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  const void* __srcMem)
{
// __space == space_global (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5];"
      :
      : "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__as_ptr_smem(__srcMem))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar]; //
PTX ISA 80, SM_90
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
  ::cuda::ptx::space_cluster_t,
  ::cuda::ptx::space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  ::cuda::std::uint64_t* __smem_bar)
{
// __space == space_cluster (due to parameter type constraint)
// __space == space_global (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, "
      "%6}], [%7];"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar]; //
PTX ISA 86, SM_90
// .dst       = { .shared::cta }
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_shared_t,
  cuda::ptx::space_global_t,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[5],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::space_global_t,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  ::cuda::std::uint64_t* __smem_bar)
{
// __space == space_shared (due to parameter type constraint)
// __space == space_global (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], "
      "[%7];"
      :
      : "r"(__as_ptr_smem(__dstMem)),
        "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__smem_bar))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group [dstMem], [tensorMap, tensorCoords],
[smem_bar]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
// .dst       = { .shared::cta }
// .src       = { .global }
// .cta_group = { .cta_group::1, .cta_group::2 }
template <cuda::ptx::dot_cta_group Cta_Group>
__device__ static inline void cp_async_bulk_tensor(
  cuda::ptx::space_shared_t,
  cuda::ptx::space_global_t,
  cuda::ptx::cta_group_t<Cta_Group> cta_group,
  void* dstMem,
  const void* tensorMap,
  const int32_t (&tensorCoords)[5],
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_cp_async_bulk_tensor_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_cta_group _Cta_Group>
_CCCL_DEVICE static inline void cp_async_bulk_tensor(
  ::cuda::ptx::space_shared_t,
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::cta_group_t<_Cta_Group> __cta_group,
  void* __dstMem,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  ::cuda::std::uint64_t* __smem_bar)
{
  // __space == space_shared (due to parameter type constraint)
  // __space == space_global (due to parameter type constraint)
  static_assert(__cta_group == cta_group_1 || __cta_group == cta_group_2, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  if constexpr (__cta_group == cta_group_1)
  {
    asm("cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 [%0], [%1, {%2, %3, "
        "%4, %5, %6}], [%7];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }
  else if constexpr (__cta_group == cta_group_2)
  {
    asm("cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2 [%0], [%1, {%2, %3, "
        "%4, %5, %6}], [%7];"
        :
        : "r"(__as_ptr_smem(__dstMem)),
          "l"(__tensorMap),
          "r"(__tensorCoords[0]),
          "r"(__tensorCoords[1]),
          "r"(__tensorCoords[2]),
          "r"(__tensorCoords[3]),
          "r"(__tensorCoords[4]),
          "r"(__as_ptr_smem(__smem_bar))
        : "memory");
  }

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_async_bulk_tensor_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f_depending_on_the_variant__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// cp.async.bulk.tensor.5d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
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
  ::cuda::ptx::space_global_t,
  ::cuda::ptx::space_shared_t,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  const void* __srcMem)
{
// __space == space_global (due to parameter type constraint)
// __space == space_shared (due to parameter type constraint)
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 900
  asm("cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6];"
      :
      : "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "r"(__as_ptr_smem(__srcMem))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_cp_async_bulk_tensor_is_not_supported_before_SM_90__();
#  endif
}
#endif // __cccl_ptx_isa >= 800

#endif // _CUDA_PTX_GENERATED_CP_ASYNC_BULK_TENSOR_H_
