// This file was automatically generated. Do not edit.

namespace cuda_ptx
{

/*
// st.global.b8 [addr], src; // PTX ISA 10, SM_50
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline void st_global(
  B8* addr,
  B8 src);
*/
#if __libcuda_ptx_isa >= 100
extern "C" __device__ void __cuda_ptx_st_global_is_not_supported_before_SM_50__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline void st_global(B8* addr, B8 src)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 500
  asm volatile("st.global.b8 [%0], %1;" : : "l"(__as_ptr_gmem(addr)), "r"(__b8_as_u32(src)) : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_is_not_supported_before_SM_50__();
#  endif
}
#endif // __libcuda_ptx_isa >= 100

/*
// st.global.b16 [addr], src; // PTX ISA 10, SM_50
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void st_global(
  B16* addr,
  B16 src);
*/
#if __libcuda_ptx_isa >= 100
extern "C" __device__ void __cuda_ptx_st_global_is_not_supported_before_SM_50__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void st_global(B16* addr, B16 src)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 500
  asm volatile("st.global.b16 [%0], %1;"
               :
               : "l"(__as_ptr_gmem(addr)), "h"(/*as_b16*/ *reinterpret_cast<const std::int16_t*>(&src))
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_is_not_supported_before_SM_50__();
#  endif
}
#endif // __libcuda_ptx_isa >= 100

/*
// st.global.b32 [addr], src; // PTX ISA 10, SM_50
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void st_global(
  B32* addr,
  B32 src);
*/
#if __libcuda_ptx_isa >= 100
extern "C" __device__ void __cuda_ptx_st_global_is_not_supported_before_SM_50__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void st_global(B32* addr, B32 src)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 500
  asm volatile("st.global.b32 [%0], %1;"
               :
               : "l"(__as_ptr_gmem(addr)), "r"(/*as_b32*/ *reinterpret_cast<const std::int32_t*>(&src))
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_is_not_supported_before_SM_50__();
#  endif
}
#endif // __libcuda_ptx_isa >= 100

/*
// st.global.b64 [addr], src; // PTX ISA 10, SM_50
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void st_global(
  B64* addr,
  B64 src);
*/
#if __libcuda_ptx_isa >= 100
extern "C" __device__ void __cuda_ptx_st_global_is_not_supported_before_SM_50__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void st_global(B64* addr, B64 src)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 500
  asm volatile("st.global.b64 [%0], %1;"
               :
               : "l"(__as_ptr_gmem(addr)), "l"(/*as_b64*/ *reinterpret_cast<const std::int64_t*>(&src))
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_is_not_supported_before_SM_50__();
#  endif
}
#endif // __libcuda_ptx_isa >= 100

/*
// st.global.b128 [addr], src; // PTX ISA 83, SM_70
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline void st_global(
  B128* addr,
  B128 src);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_st_global_is_not_supported_before_SM_70__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline void st_global(B128* addr, B128 src)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 700
  asm volatile(
    "{\n\t .reg .b128 B128_src; \n\t"
    "mov.b128 B128_src, {%1, %2}; \n"
    "st.global.b128 [%0], B128_src;\n\t"
    "}"
    :
    : "l"(__as_ptr_gmem(addr)), "l"((*reinterpret_cast<long2*>(&src)).x), "l"((*reinterpret_cast<long2*>(&src)).y)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_is_not_supported_before_SM_70__();
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// st.global.L1::evict_normal.b8 [addr], src; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline void st_global_L1_evict_normal(
  B8* addr,
  B8 src);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_st_global_L1_evict_normal_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline void st_global_L1_evict_normal(B8* addr, B8 src)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  asm volatile("st.global.L1::evict_normal.b8 [%0], %1;"
               :
               : "l"(__as_ptr_gmem(addr)), "r"(__b8_as_u32(src))
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_L1_evict_normal_is_not_supported_before_SM_75__();
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// st.global.L1::evict_normal.b16 [addr], src; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void st_global_L1_evict_normal(
  B16* addr,
  B16 src);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_st_global_L1_evict_normal_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void st_global_L1_evict_normal(B16* addr, B16 src)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  asm volatile("st.global.L1::evict_normal.b16 [%0], %1;"
               :
               : "l"(__as_ptr_gmem(addr)), "h"(/*as_b16*/ *reinterpret_cast<const std::int16_t*>(&src))
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_L1_evict_normal_is_not_supported_before_SM_75__();
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// st.global.L1::evict_normal.b32 [addr], src; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void st_global_L1_evict_normal(
  B32* addr,
  B32 src);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_st_global_L1_evict_normal_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void st_global_L1_evict_normal(B32* addr, B32 src)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  asm volatile("st.global.L1::evict_normal.b32 [%0], %1;"
               :
               : "l"(__as_ptr_gmem(addr)), "r"(/*as_b32*/ *reinterpret_cast<const std::int32_t*>(&src))
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_L1_evict_normal_is_not_supported_before_SM_75__();
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// st.global.L1::evict_normal.b64 [addr], src; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void st_global_L1_evict_normal(
  B64* addr,
  B64 src);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_st_global_L1_evict_normal_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void st_global_L1_evict_normal(B64* addr, B64 src)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  asm volatile("st.global.L1::evict_normal.b64 [%0], %1;"
               :
               : "l"(__as_ptr_gmem(addr)), "l"(/*as_b64*/ *reinterpret_cast<const std::int64_t*>(&src))
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_L1_evict_normal_is_not_supported_before_SM_75__();
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// st.global.L1::evict_normal.b128 [addr], src; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline void st_global_L1_evict_normal(
  B128* addr,
  B128 src);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_st_global_L1_evict_normal_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline void st_global_L1_evict_normal(B128* addr, B128 src)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  asm volatile(
    "{\n\t .reg .b128 B128_src; \n\t"
    "mov.b128 B128_src, {%1, %2}; \n"
    "st.global.L1::evict_normal.b128 [%0], B128_src;\n\t"
    "}"
    :
    : "l"(__as_ptr_gmem(addr)), "l"((*reinterpret_cast<long2*>(&src)).x), "l"((*reinterpret_cast<long2*>(&src)).y)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_L1_evict_normal_is_not_supported_before_SM_75__();
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// st.global.L1::evict_unchanged.b8 [addr], src; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline void st_global_L1_evict_unchanged(
  B8* addr,
  B8 src);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_st_global_L1_evict_unchanged_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline void st_global_L1_evict_unchanged(B8* addr, B8 src)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  asm volatile("st.global.L1::evict_unchanged.b8 [%0], %1;"
               :
               : "l"(__as_ptr_gmem(addr)), "r"(__b8_as_u32(src))
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_L1_evict_unchanged_is_not_supported_before_SM_75__();
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// st.global.L1::evict_unchanged.b16 [addr], src; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void st_global_L1_evict_unchanged(
  B16* addr,
  B16 src);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_st_global_L1_evict_unchanged_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void st_global_L1_evict_unchanged(B16* addr, B16 src)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  asm volatile("st.global.L1::evict_unchanged.b16 [%0], %1;"
               :
               : "l"(__as_ptr_gmem(addr)), "h"(/*as_b16*/ *reinterpret_cast<const std::int16_t*>(&src))
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_L1_evict_unchanged_is_not_supported_before_SM_75__();
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// st.global.L1::evict_unchanged.b32 [addr], src; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void st_global_L1_evict_unchanged(
  B32* addr,
  B32 src);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_st_global_L1_evict_unchanged_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void st_global_L1_evict_unchanged(B32* addr, B32 src)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  asm volatile("st.global.L1::evict_unchanged.b32 [%0], %1;"
               :
               : "l"(__as_ptr_gmem(addr)), "r"(/*as_b32*/ *reinterpret_cast<const std::int32_t*>(&src))
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_L1_evict_unchanged_is_not_supported_before_SM_75__();
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// st.global.L1::evict_unchanged.b64 [addr], src; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void st_global_L1_evict_unchanged(
  B64* addr,
  B64 src);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_st_global_L1_evict_unchanged_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void st_global_L1_evict_unchanged(B64* addr, B64 src)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  asm volatile("st.global.L1::evict_unchanged.b64 [%0], %1;"
               :
               : "l"(__as_ptr_gmem(addr)), "l"(/*as_b64*/ *reinterpret_cast<const std::int64_t*>(&src))
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_L1_evict_unchanged_is_not_supported_before_SM_75__();
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// st.global.L1::evict_unchanged.b128 [addr], src; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline void st_global_L1_evict_unchanged(
  B128* addr,
  B128 src);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_st_global_L1_evict_unchanged_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline void st_global_L1_evict_unchanged(B128* addr, B128 src)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  asm volatile(
    "{\n\t .reg .b128 B128_src; \n\t"
    "mov.b128 B128_src, {%1, %2}; \n"
    "st.global.L1::evict_unchanged.b128 [%0], B128_src;\n\t"
    "}"
    :
    : "l"(__as_ptr_gmem(addr)), "l"((*reinterpret_cast<long2*>(&src)).x), "l"((*reinterpret_cast<long2*>(&src)).y)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_L1_evict_unchanged_is_not_supported_before_SM_75__();
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// st.global.L1::evict_first.b8 [addr], src; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline void st_global_L1_evict_first(
  B8* addr,
  B8 src);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_st_global_L1_evict_first_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline void st_global_L1_evict_first(B8* addr, B8 src)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  asm volatile("st.global.L1::evict_first.b8 [%0], %1;" : : "l"(__as_ptr_gmem(addr)), "r"(__b8_as_u32(src)) : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_L1_evict_first_is_not_supported_before_SM_75__();
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// st.global.L1::evict_first.b16 [addr], src; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void st_global_L1_evict_first(
  B16* addr,
  B16 src);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_st_global_L1_evict_first_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void st_global_L1_evict_first(B16* addr, B16 src)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  asm volatile("st.global.L1::evict_first.b16 [%0], %1;"
               :
               : "l"(__as_ptr_gmem(addr)), "h"(/*as_b16*/ *reinterpret_cast<const std::int16_t*>(&src))
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_L1_evict_first_is_not_supported_before_SM_75__();
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// st.global.L1::evict_first.b32 [addr], src; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void st_global_L1_evict_first(
  B32* addr,
  B32 src);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_st_global_L1_evict_first_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void st_global_L1_evict_first(B32* addr, B32 src)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  asm volatile("st.global.L1::evict_first.b32 [%0], %1;"
               :
               : "l"(__as_ptr_gmem(addr)), "r"(/*as_b32*/ *reinterpret_cast<const std::int32_t*>(&src))
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_L1_evict_first_is_not_supported_before_SM_75__();
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// st.global.L1::evict_first.b64 [addr], src; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void st_global_L1_evict_first(
  B64* addr,
  B64 src);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_st_global_L1_evict_first_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void st_global_L1_evict_first(B64* addr, B64 src)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  asm volatile("st.global.L1::evict_first.b64 [%0], %1;"
               :
               : "l"(__as_ptr_gmem(addr)), "l"(/*as_b64*/ *reinterpret_cast<const std::int64_t*>(&src))
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_L1_evict_first_is_not_supported_before_SM_75__();
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// st.global.L1::evict_first.b128 [addr], src; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline void st_global_L1_evict_first(
  B128* addr,
  B128 src);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_st_global_L1_evict_first_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline void st_global_L1_evict_first(B128* addr, B128 src)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  asm volatile(
    "{\n\t .reg .b128 B128_src; \n\t"
    "mov.b128 B128_src, {%1, %2}; \n"
    "st.global.L1::evict_first.b128 [%0], B128_src;\n\t"
    "}"
    :
    : "l"(__as_ptr_gmem(addr)), "l"((*reinterpret_cast<long2*>(&src)).x), "l"((*reinterpret_cast<long2*>(&src)).y)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_L1_evict_first_is_not_supported_before_SM_75__();
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// st.global.L1::evict_last.b8 [addr], src; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline void st_global_L1_evict_last(
  B8* addr,
  B8 src);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_st_global_L1_evict_last_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline void st_global_L1_evict_last(B8* addr, B8 src)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  asm volatile("st.global.L1::evict_last.b8 [%0], %1;" : : "l"(__as_ptr_gmem(addr)), "r"(__b8_as_u32(src)) : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_L1_evict_last_is_not_supported_before_SM_75__();
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// st.global.L1::evict_last.b16 [addr], src; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void st_global_L1_evict_last(
  B16* addr,
  B16 src);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_st_global_L1_evict_last_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void st_global_L1_evict_last(B16* addr, B16 src)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  asm volatile("st.global.L1::evict_last.b16 [%0], %1;"
               :
               : "l"(__as_ptr_gmem(addr)), "h"(/*as_b16*/ *reinterpret_cast<const std::int16_t*>(&src))
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_L1_evict_last_is_not_supported_before_SM_75__();
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// st.global.L1::evict_last.b32 [addr], src; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void st_global_L1_evict_last(
  B32* addr,
  B32 src);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_st_global_L1_evict_last_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void st_global_L1_evict_last(B32* addr, B32 src)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  asm volatile("st.global.L1::evict_last.b32 [%0], %1;"
               :
               : "l"(__as_ptr_gmem(addr)), "r"(/*as_b32*/ *reinterpret_cast<const std::int32_t*>(&src))
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_L1_evict_last_is_not_supported_before_SM_75__();
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// st.global.L1::evict_last.b64 [addr], src; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void st_global_L1_evict_last(
  B64* addr,
  B64 src);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_st_global_L1_evict_last_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void st_global_L1_evict_last(B64* addr, B64 src)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  asm volatile("st.global.L1::evict_last.b64 [%0], %1;"
               :
               : "l"(__as_ptr_gmem(addr)), "l"(/*as_b64*/ *reinterpret_cast<const std::int64_t*>(&src))
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_L1_evict_last_is_not_supported_before_SM_75__();
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// st.global.L1::evict_last.b128 [addr], src; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline void st_global_L1_evict_last(
  B128* addr,
  B128 src);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_st_global_L1_evict_last_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline void st_global_L1_evict_last(B128* addr, B128 src)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  asm volatile(
    "{\n\t .reg .b128 B128_src; \n\t"
    "mov.b128 B128_src, {%1, %2}; \n"
    "st.global.L1::evict_last.b128 [%0], B128_src;\n\t"
    "}"
    :
    : "l"(__as_ptr_gmem(addr)), "l"((*reinterpret_cast<long2*>(&src)).x), "l"((*reinterpret_cast<long2*>(&src)).y)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_L1_evict_last_is_not_supported_before_SM_75__();
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// st.global.L1::no_allocate.b8 [addr], src; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline void st_global_L1_no_allocate(
  B8* addr,
  B8 src);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_st_global_L1_no_allocate_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline void st_global_L1_no_allocate(B8* addr, B8 src)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  asm volatile("st.global.L1::no_allocate.b8 [%0], %1;" : : "l"(__as_ptr_gmem(addr)), "r"(__b8_as_u32(src)) : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_L1_no_allocate_is_not_supported_before_SM_75__();
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// st.global.L1::no_allocate.b16 [addr], src; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void st_global_L1_no_allocate(
  B16* addr,
  B16 src);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_st_global_L1_no_allocate_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void st_global_L1_no_allocate(B16* addr, B16 src)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  asm volatile("st.global.L1::no_allocate.b16 [%0], %1;"
               :
               : "l"(__as_ptr_gmem(addr)), "h"(/*as_b16*/ *reinterpret_cast<const std::int16_t*>(&src))
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_L1_no_allocate_is_not_supported_before_SM_75__();
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// st.global.L1::no_allocate.b32 [addr], src; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void st_global_L1_no_allocate(
  B32* addr,
  B32 src);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_st_global_L1_no_allocate_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void st_global_L1_no_allocate(B32* addr, B32 src)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  asm volatile("st.global.L1::no_allocate.b32 [%0], %1;"
               :
               : "l"(__as_ptr_gmem(addr)), "r"(/*as_b32*/ *reinterpret_cast<const std::int32_t*>(&src))
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_L1_no_allocate_is_not_supported_before_SM_75__();
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// st.global.L1::no_allocate.b64 [addr], src; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void st_global_L1_no_allocate(
  B64* addr,
  B64 src);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_st_global_L1_no_allocate_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline void st_global_L1_no_allocate(B64* addr, B64 src)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  asm volatile("st.global.L1::no_allocate.b64 [%0], %1;"
               :
               : "l"(__as_ptr_gmem(addr)), "l"(/*as_b64*/ *reinterpret_cast<const std::int64_t*>(&src))
               : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_L1_no_allocate_is_not_supported_before_SM_75__();
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// st.global.L1::no_allocate.b128 [addr], src; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline void st_global_L1_no_allocate(
  B128* addr,
  B128 src);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_st_global_L1_no_allocate_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline void st_global_L1_no_allocate(B128* addr, B128 src)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  asm volatile(
    "{\n\t .reg .b128 B128_src; \n\t"
    "mov.b128 B128_src, {%1, %2}; \n"
    "st.global.L1::no_allocate.b128 [%0], B128_src;\n\t"
    "}"
    :
    : "l"(__as_ptr_gmem(addr)), "l"((*reinterpret_cast<long2*>(&src)).x), "l"((*reinterpret_cast<long2*>(&src)).y)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_global_L1_no_allocate_is_not_supported_before_SM_75__();
#  endif
}
#endif // __libcuda_ptx_isa >= 830

} // namespace cuda_ptx
