// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_TCGEN05_LD_H_
#define _CUDA_PTX_GENERATED_TCGEN05_LD_H_

/*
// tcgen05.ld.sync.aligned.16x64b.x1.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b(
  B32 (&out)[1],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b(_B32 (&__out)[1], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x64b.x1.b32 {%0}, [%1];" : "=r"(__out[0]) : "r"(__taddr) : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x1.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b_pack_16b(
  B32 (&out)[1],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b_pack_16b(_B32 (&__out)[1], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x64b.x1.pack::16b.b32 {%0}, [%1];" : "=r"(__out[0]) : "r"(__taddr) : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x2.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b(
  B32 (&out)[2],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b(_B32 (&__out)[2], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x64b.x2.b32 {%0, %1}, [%2];"
      : "=r"(__out[0]), "=r"(__out[1])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x2.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b_pack_16b(
  B32 (&out)[2],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b_pack_16b(_B32 (&__out)[2], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x64b.x2.pack::16b.b32 {%0, %1}, [%2];"
      : "=r"(__out[0]), "=r"(__out[1])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x4.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b(
  B32 (&out)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b(_B32 (&__out)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x64b.x4.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x4.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b_pack_16b(
  B32 (&out)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b_pack_16b(_B32 (&__out)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x64b.x4.pack::16b.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x8.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b(
  B32 (&out)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b(_B32 (&__out)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x64b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x8.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b_pack_16b(
  B32 (&out)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b_pack_16b(_B32 (&__out)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x64b.x8.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x16.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b(
  B32 (&out)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b(_B32 (&__out)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x64b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
      "[%16];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x16.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b_pack_16b(
  B32 (&out)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b_pack_16b(_B32 (&__out)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x64b.x16.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15}, [%16];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x32.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b(
  B32 (&out)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b(_B32 (&__out)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.16x64b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x32.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b_pack_16b(
  B32 (&out)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b_pack_16b(_B32 (&__out)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.16x64b.x32.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x64.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b(
  B32 (&out)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b(_B32 (&__out)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.16x64b.x64.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63}, [%64];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x64.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b_pack_16b(
  B32 (&out)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b_pack_16b(_B32 (&__out)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.16x64b.x64.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63}, [%64];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x128.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b(
  B32 (&out)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b(_B32 (&__out)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.16x64b.x128.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, "
    "%82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, "
    "%104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
    "%123, %124, %125, %126, %127}, [%128];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63]),
      "=r"(__out[64]),
      "=r"(__out[65]),
      "=r"(__out[66]),
      "=r"(__out[67]),
      "=r"(__out[68]),
      "=r"(__out[69]),
      "=r"(__out[70]),
      "=r"(__out[71]),
      "=r"(__out[72]),
      "=r"(__out[73]),
      "=r"(__out[74]),
      "=r"(__out[75]),
      "=r"(__out[76]),
      "=r"(__out[77]),
      "=r"(__out[78]),
      "=r"(__out[79]),
      "=r"(__out[80]),
      "=r"(__out[81]),
      "=r"(__out[82]),
      "=r"(__out[83]),
      "=r"(__out[84]),
      "=r"(__out[85]),
      "=r"(__out[86]),
      "=r"(__out[87]),
      "=r"(__out[88]),
      "=r"(__out[89]),
      "=r"(__out[90]),
      "=r"(__out[91]),
      "=r"(__out[92]),
      "=r"(__out[93]),
      "=r"(__out[94]),
      "=r"(__out[95]),
      "=r"(__out[96]),
      "=r"(__out[97]),
      "=r"(__out[98]),
      "=r"(__out[99]),
      "=r"(__out[100]),
      "=r"(__out[101]),
      "=r"(__out[102]),
      "=r"(__out[103]),
      "=r"(__out[104]),
      "=r"(__out[105]),
      "=r"(__out[106]),
      "=r"(__out[107]),
      "=r"(__out[108]),
      "=r"(__out[109]),
      "=r"(__out[110]),
      "=r"(__out[111]),
      "=r"(__out[112]),
      "=r"(__out[113]),
      "=r"(__out[114]),
      "=r"(__out[115]),
      "=r"(__out[116]),
      "=r"(__out[117]),
      "=r"(__out[118]),
      "=r"(__out[119]),
      "=r"(__out[120]),
      "=r"(__out[121]),
      "=r"(__out[122]),
      "=r"(__out[123]),
      "=r"(__out[124]),
      "=r"(__out[125]),
      "=r"(__out[126]),
      "=r"(__out[127])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x128.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b_pack_16b(
  B32 (&out)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b_pack_16b(_B32 (&__out)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.16x64b.x128.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
    "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, "
    "%102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, "
    "%121, %122, %123, %124, %125, %126, %127}, [%128];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63]),
      "=r"(__out[64]),
      "=r"(__out[65]),
      "=r"(__out[66]),
      "=r"(__out[67]),
      "=r"(__out[68]),
      "=r"(__out[69]),
      "=r"(__out[70]),
      "=r"(__out[71]),
      "=r"(__out[72]),
      "=r"(__out[73]),
      "=r"(__out[74]),
      "=r"(__out[75]),
      "=r"(__out[76]),
      "=r"(__out[77]),
      "=r"(__out[78]),
      "=r"(__out[79]),
      "=r"(__out[80]),
      "=r"(__out[81]),
      "=r"(__out[82]),
      "=r"(__out[83]),
      "=r"(__out[84]),
      "=r"(__out[85]),
      "=r"(__out[86]),
      "=r"(__out[87]),
      "=r"(__out[88]),
      "=r"(__out[89]),
      "=r"(__out[90]),
      "=r"(__out[91]),
      "=r"(__out[92]),
      "=r"(__out[93]),
      "=r"(__out[94]),
      "=r"(__out[95]),
      "=r"(__out[96]),
      "=r"(__out[97]),
      "=r"(__out[98]),
      "=r"(__out[99]),
      "=r"(__out[100]),
      "=r"(__out[101]),
      "=r"(__out[102]),
      "=r"(__out[103]),
      "=r"(__out[104]),
      "=r"(__out[105]),
      "=r"(__out[106]),
      "=r"(__out[107]),
      "=r"(__out[108]),
      "=r"(__out[109]),
      "=r"(__out[110]),
      "=r"(__out[111]),
      "=r"(__out[112]),
      "=r"(__out[113]),
      "=r"(__out[114]),
      "=r"(__out[115]),
      "=r"(__out[116]),
      "=r"(__out[117]),
      "=r"(__out[118]),
      "=r"(__out[119]),
      "=r"(__out[120]),
      "=r"(__out[121]),
      "=r"(__out[122]),
      "=r"(__out[123]),
      "=r"(__out[124]),
      "=r"(__out[125]),
      "=r"(__out[126]),
      "=r"(__out[127])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x64b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x1.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b(
  B32 (&out)[2],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b(_B32 (&__out)[2], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x128b.x1.b32 {%0, %1}, [%2];"
      : "=r"(__out[0]), "=r"(__out[1])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x1.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b_pack_16b(
  B32 (&out)[2],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x128b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b_pack_16b(_B32 (&__out)[2], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x128b.x1.pack::16b.b32 {%0, %1}, [%2];"
      : "=r"(__out[0]), "=r"(__out[1])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x128b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x2.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b(
  B32 (&out)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b(_B32 (&__out)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x128b.x2.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x2.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b_pack_16b(
  B32 (&out)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x128b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b_pack_16b(_B32 (&__out)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x128b.x2.pack::16b.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x128b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x4.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b(
  B32 (&out)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b(_B32 (&__out)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x128b.x4.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b_pack_16b(
  B32 (&out)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x128b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b_pack_16b(_B32 (&__out)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x128b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x8.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b(
  B32 (&out)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b(_B32 (&__out)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x128b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
      "[%16];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x8.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b_pack_16b(
  B32 (&out)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x128b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b_pack_16b(_B32 (&__out)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x128b.x8.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15}, [%16];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x128b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x16.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b(
  B32 (&out)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b(_B32 (&__out)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.16x128b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x16.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b_pack_16b(
  B32 (&out)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x128b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b_pack_16b(_B32 (&__out)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.16x128b.x16.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x128b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x32.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b(
  B32 (&out)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b(_B32 (&__out)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.16x128b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63}, [%64];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x32.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b_pack_16b(
  B32 (&out)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x128b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b_pack_16b(_B32 (&__out)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.16x128b.x32.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63}, [%64];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x128b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x64.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b(
  B32 (&out)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b(_B32 (&__out)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.16x128b.x64.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, "
    "%82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, "
    "%104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
    "%123, %124, %125, %126, %127}, [%128];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63]),
      "=r"(__out[64]),
      "=r"(__out[65]),
      "=r"(__out[66]),
      "=r"(__out[67]),
      "=r"(__out[68]),
      "=r"(__out[69]),
      "=r"(__out[70]),
      "=r"(__out[71]),
      "=r"(__out[72]),
      "=r"(__out[73]),
      "=r"(__out[74]),
      "=r"(__out[75]),
      "=r"(__out[76]),
      "=r"(__out[77]),
      "=r"(__out[78]),
      "=r"(__out[79]),
      "=r"(__out[80]),
      "=r"(__out[81]),
      "=r"(__out[82]),
      "=r"(__out[83]),
      "=r"(__out[84]),
      "=r"(__out[85]),
      "=r"(__out[86]),
      "=r"(__out[87]),
      "=r"(__out[88]),
      "=r"(__out[89]),
      "=r"(__out[90]),
      "=r"(__out[91]),
      "=r"(__out[92]),
      "=r"(__out[93]),
      "=r"(__out[94]),
      "=r"(__out[95]),
      "=r"(__out[96]),
      "=r"(__out[97]),
      "=r"(__out[98]),
      "=r"(__out[99]),
      "=r"(__out[100]),
      "=r"(__out[101]),
      "=r"(__out[102]),
      "=r"(__out[103]),
      "=r"(__out[104]),
      "=r"(__out[105]),
      "=r"(__out[106]),
      "=r"(__out[107]),
      "=r"(__out[108]),
      "=r"(__out[109]),
      "=r"(__out[110]),
      "=r"(__out[111]),
      "=r"(__out[112]),
      "=r"(__out[113]),
      "=r"(__out[114]),
      "=r"(__out[115]),
      "=r"(__out[116]),
      "=r"(__out[117]),
      "=r"(__out[118]),
      "=r"(__out[119]),
      "=r"(__out[120]),
      "=r"(__out[121]),
      "=r"(__out[122]),
      "=r"(__out[123]),
      "=r"(__out[124]),
      "=r"(__out[125]),
      "=r"(__out[126]),
      "=r"(__out[127])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x128b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x64.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b_pack_16b(
  B32 (&out)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x128b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b_pack_16b(_B32 (&__out)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.16x128b.x64.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
    "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, "
    "%102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, "
    "%121, %122, %123, %124, %125, %126, %127}, [%128];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63]),
      "=r"(__out[64]),
      "=r"(__out[65]),
      "=r"(__out[66]),
      "=r"(__out[67]),
      "=r"(__out[68]),
      "=r"(__out[69]),
      "=r"(__out[70]),
      "=r"(__out[71]),
      "=r"(__out[72]),
      "=r"(__out[73]),
      "=r"(__out[74]),
      "=r"(__out[75]),
      "=r"(__out[76]),
      "=r"(__out[77]),
      "=r"(__out[78]),
      "=r"(__out[79]),
      "=r"(__out[80]),
      "=r"(__out[81]),
      "=r"(__out[82]),
      "=r"(__out[83]),
      "=r"(__out[84]),
      "=r"(__out[85]),
      "=r"(__out[86]),
      "=r"(__out[87]),
      "=r"(__out[88]),
      "=r"(__out[89]),
      "=r"(__out[90]),
      "=r"(__out[91]),
      "=r"(__out[92]),
      "=r"(__out[93]),
      "=r"(__out[94]),
      "=r"(__out[95]),
      "=r"(__out[96]),
      "=r"(__out[97]),
      "=r"(__out[98]),
      "=r"(__out[99]),
      "=r"(__out[100]),
      "=r"(__out[101]),
      "=r"(__out[102]),
      "=r"(__out[103]),
      "=r"(__out[104]),
      "=r"(__out[105]),
      "=r"(__out[106]),
      "=r"(__out[107]),
      "=r"(__out[108]),
      "=r"(__out[109]),
      "=r"(__out[110]),
      "=r"(__out[111]),
      "=r"(__out[112]),
      "=r"(__out[113]),
      "=r"(__out[114]),
      "=r"(__out[115]),
      "=r"(__out[116]),
      "=r"(__out[117]),
      "=r"(__out[118]),
      "=r"(__out[119]),
      "=r"(__out[120]),
      "=r"(__out[121]),
      "=r"(__out[122]),
      "=r"(__out[123]),
      "=r"(__out[124]),
      "=r"(__out[125]),
      "=r"(__out[126]),
      "=r"(__out[127])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x128b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x1.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b(
  B32 (&out)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b(_B32 (&__out)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x1.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b_pack_16b(
  B32 (&out)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x256b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b_pack_16b(_B32 (&__out)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x256b.x1.pack::16b.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x256b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x2.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b(
  B32 (&out)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b(_B32 (&__out)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x2.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b_pack_16b(
  B32 (&out)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x256b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b_pack_16b(_B32 (&__out)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x256b.x2.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x256b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x4.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b(
  B32 (&out)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b(_B32 (&__out)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
      "[%16];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x4.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b_pack_16b(
  B32 (&out)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x256b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b_pack_16b(_B32 (&__out)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x256b.x4.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15}, [%16];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x256b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x8.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b(
  B32 (&out)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b(_B32 (&__out)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.16x256b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x8.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b_pack_16b(
  B32 (&out)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x256b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b_pack_16b(_B32 (&__out)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.16x256b.x8.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x256b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x16.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b(
  B32 (&out)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b(_B32 (&__out)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.16x256b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63}, [%64];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x16.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b_pack_16b(
  B32 (&out)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x256b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b_pack_16b(_B32 (&__out)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.16x256b.x16.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63}, [%64];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x256b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x32.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b(
  B32 (&out)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b(_B32 (&__out)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.16x256b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, "
    "%82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, "
    "%104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
    "%123, %124, %125, %126, %127}, [%128];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63]),
      "=r"(__out[64]),
      "=r"(__out[65]),
      "=r"(__out[66]),
      "=r"(__out[67]),
      "=r"(__out[68]),
      "=r"(__out[69]),
      "=r"(__out[70]),
      "=r"(__out[71]),
      "=r"(__out[72]),
      "=r"(__out[73]),
      "=r"(__out[74]),
      "=r"(__out[75]),
      "=r"(__out[76]),
      "=r"(__out[77]),
      "=r"(__out[78]),
      "=r"(__out[79]),
      "=r"(__out[80]),
      "=r"(__out[81]),
      "=r"(__out[82]),
      "=r"(__out[83]),
      "=r"(__out[84]),
      "=r"(__out[85]),
      "=r"(__out[86]),
      "=r"(__out[87]),
      "=r"(__out[88]),
      "=r"(__out[89]),
      "=r"(__out[90]),
      "=r"(__out[91]),
      "=r"(__out[92]),
      "=r"(__out[93]),
      "=r"(__out[94]),
      "=r"(__out[95]),
      "=r"(__out[96]),
      "=r"(__out[97]),
      "=r"(__out[98]),
      "=r"(__out[99]),
      "=r"(__out[100]),
      "=r"(__out[101]),
      "=r"(__out[102]),
      "=r"(__out[103]),
      "=r"(__out[104]),
      "=r"(__out[105]),
      "=r"(__out[106]),
      "=r"(__out[107]),
      "=r"(__out[108]),
      "=r"(__out[109]),
      "=r"(__out[110]),
      "=r"(__out[111]),
      "=r"(__out[112]),
      "=r"(__out[113]),
      "=r"(__out[114]),
      "=r"(__out[115]),
      "=r"(__out[116]),
      "=r"(__out[117]),
      "=r"(__out[118]),
      "=r"(__out[119]),
      "=r"(__out[120]),
      "=r"(__out[121]),
      "=r"(__out[122]),
      "=r"(__out[123]),
      "=r"(__out[124]),
      "=r"(__out[125]),
      "=r"(__out[126]),
      "=r"(__out[127])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x256b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x32.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b_pack_16b(
  B32 (&out)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x256b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b_pack_16b(_B32 (&__out)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.16x256b.x32.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
    "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, "
    "%102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, "
    "%121, %122, %123, %124, %125, %126, %127}, [%128];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63]),
      "=r"(__out[64]),
      "=r"(__out[65]),
      "=r"(__out[66]),
      "=r"(__out[67]),
      "=r"(__out[68]),
      "=r"(__out[69]),
      "=r"(__out[70]),
      "=r"(__out[71]),
      "=r"(__out[72]),
      "=r"(__out[73]),
      "=r"(__out[74]),
      "=r"(__out[75]),
      "=r"(__out[76]),
      "=r"(__out[77]),
      "=r"(__out[78]),
      "=r"(__out[79]),
      "=r"(__out[80]),
      "=r"(__out[81]),
      "=r"(__out[82]),
      "=r"(__out[83]),
      "=r"(__out[84]),
      "=r"(__out[85]),
      "=r"(__out[86]),
      "=r"(__out[87]),
      "=r"(__out[88]),
      "=r"(__out[89]),
      "=r"(__out[90]),
      "=r"(__out[91]),
      "=r"(__out[92]),
      "=r"(__out[93]),
      "=r"(__out[94]),
      "=r"(__out[95]),
      "=r"(__out[96]),
      "=r"(__out[97]),
      "=r"(__out[98]),
      "=r"(__out[99]),
      "=r"(__out[100]),
      "=r"(__out[101]),
      "=r"(__out[102]),
      "=r"(__out[103]),
      "=r"(__out[104]),
      "=r"(__out[105]),
      "=r"(__out[106]),
      "=r"(__out[107]),
      "=r"(__out[108]),
      "=r"(__out[109]),
      "=r"(__out[110]),
      "=r"(__out[111]),
      "=r"(__out[112]),
      "=r"(__out[113]),
      "=r"(__out[114]),
      "=r"(__out[115]),
      "=r"(__out[116]),
      "=r"(__out[117]),
      "=r"(__out[118]),
      "=r"(__out[119]),
      "=r"(__out[120]),
      "=r"(__out[121]),
      "=r"(__out[122]),
      "=r"(__out[123]),
      "=r"(__out[124]),
      "=r"(__out[125]),
      "=r"(__out[126]),
      "=r"(__out[127])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x256b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x1.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b(
  B32 (&out)[1],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b(_B32 (&__out)[1], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(__out[0]) : "r"(__taddr) : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x1.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b_pack_16b(
  B32 (&out)[1],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b_pack_16b(_B32 (&__out)[1], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.32x32b.x1.pack::16b.b32 {%0}, [%1];" : "=r"(__out[0]) : "r"(__taddr) : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x2.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b(
  B32 (&out)[2],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b(_B32 (&__out)[2], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0, %1}, [%2];"
      : "=r"(__out[0]), "=r"(__out[1])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x2.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b_pack_16b(
  B32 (&out)[2],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b_pack_16b(_B32 (&__out)[2], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.32x32b.x2.pack::16b.b32 {%0, %1}, [%2];"
      : "=r"(__out[0]), "=r"(__out[1])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x4.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b(
  B32 (&out)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b(_B32 (&__out)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x4.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b_pack_16b(
  B32 (&out)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b_pack_16b(_B32 (&__out)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.32x32b.x4.pack::16b.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x8.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b(
  B32 (&out)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b(_B32 (&__out)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x8.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b_pack_16b(
  B32 (&out)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b_pack_16b(_B32 (&__out)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.32x32b.x8.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x16.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b(
  B32 (&out)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b(_B32 (&__out)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
      "[%16];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x16.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b_pack_16b(
  B32 (&out)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b_pack_16b(_B32 (&__out)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.32x32b.x16.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15}, [%16];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15])
      : "r"(__taddr)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x32.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b(
  B32 (&out)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b(_B32 (&__out)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x32.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b_pack_16b(
  B32 (&out)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b_pack_16b(_B32 (&__out)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.32x32b.x32.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x64.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b(
  B32 (&out)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b(_B32 (&__out)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.32x32b.x64.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63}, [%64];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x64.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b_pack_16b(
  B32 (&out)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b_pack_16b(_B32 (&__out)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.32x32b.x64.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63}, [%64];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x128.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a,
SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b(
  B32 (&out)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b(_B32 (&__out)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.32x32b.x128.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, "
    "%82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, "
    "%104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
    "%123, %124, %125, %126, %127}, [%128];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63]),
      "=r"(__out[64]),
      "=r"(__out[65]),
      "=r"(__out[66]),
      "=r"(__out[67]),
      "=r"(__out[68]),
      "=r"(__out[69]),
      "=r"(__out[70]),
      "=r"(__out[71]),
      "=r"(__out[72]),
      "=r"(__out[73]),
      "=r"(__out[74]),
      "=r"(__out[75]),
      "=r"(__out[76]),
      "=r"(__out[77]),
      "=r"(__out[78]),
      "=r"(__out[79]),
      "=r"(__out[80]),
      "=r"(__out[81]),
      "=r"(__out[82]),
      "=r"(__out[83]),
      "=r"(__out[84]),
      "=r"(__out[85]),
      "=r"(__out[86]),
      "=r"(__out[87]),
      "=r"(__out[88]),
      "=r"(__out[89]),
      "=r"(__out[90]),
      "=r"(__out[91]),
      "=r"(__out[92]),
      "=r"(__out[93]),
      "=r"(__out[94]),
      "=r"(__out[95]),
      "=r"(__out[96]),
      "=r"(__out[97]),
      "=r"(__out[98]),
      "=r"(__out[99]),
      "=r"(__out[100]),
      "=r"(__out[101]),
      "=r"(__out[102]),
      "=r"(__out[103]),
      "=r"(__out[104]),
      "=r"(__out[105]),
      "=r"(__out[106]),
      "=r"(__out[107]),
      "=r"(__out[108]),
      "=r"(__out[109]),
      "=r"(__out[110]),
      "=r"(__out[111]),
      "=r"(__out[112]),
      "=r"(__out[113]),
      "=r"(__out[114]),
      "=r"(__out[115]),
      "=r"(__out[116]),
      "=r"(__out[117]),
      "=r"(__out[118]),
      "=r"(__out[119]),
      "=r"(__out[120]),
      "=r"(__out[121]),
      "=r"(__out[122]),
      "=r"(__out[123]),
      "=r"(__out[124]),
      "=r"(__out[125]),
      "=r"(__out[126]),
      "=r"(__out[127])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x128.pack::16b.b32 out, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b_pack_16b(
  B32 (&out)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b_pack_16b(_B32 (&__out)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.32x32b.x128.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
    "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, "
    "%102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, "
    "%121, %122, %123, %124, %125, %126, %127}, [%128];"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63]),
      "=r"(__out[64]),
      "=r"(__out[65]),
      "=r"(__out[66]),
      "=r"(__out[67]),
      "=r"(__out[68]),
      "=r"(__out[69]),
      "=r"(__out[70]),
      "=r"(__out[71]),
      "=r"(__out[72]),
      "=r"(__out[73]),
      "=r"(__out[74]),
      "=r"(__out[75]),
      "=r"(__out[76]),
      "=r"(__out[77]),
      "=r"(__out[78]),
      "=r"(__out[79]),
      "=r"(__out[80]),
      "=r"(__out[81]),
      "=r"(__out[82]),
      "=r"(__out[83]),
      "=r"(__out[84]),
      "=r"(__out[85]),
      "=r"(__out[86]),
      "=r"(__out[87]),
      "=r"(__out[88]),
      "=r"(__out[89]),
      "=r"(__out[90]),
      "=r"(__out[91]),
      "=r"(__out[92]),
      "=r"(__out[93]),
      "=r"(__out[94]),
      "=r"(__out[95]),
      "=r"(__out[96]),
      "=r"(__out[97]),
      "=r"(__out[98]),
      "=r"(__out[99]),
      "=r"(__out[100]),
      "=r"(__out[101]),
      "=r"(__out[102]),
      "=r"(__out[103]),
      "=r"(__out[104]),
      "=r"(__out[105]),
      "=r"(__out[106]),
      "=r"(__out[107]),
      "=r"(__out[108]),
      "=r"(__out[109]),
      "=r"(__out[110]),
      "=r"(__out[111]),
      "=r"(__out[112]),
      "=r"(__out[113]),
      "=r"(__out[114]),
      "=r"(__out[115]),
      "=r"(__out[116]),
      "=r"(__out[117]),
      "=r"(__out[118]),
      "=r"(__out[119]),
      "=r"(__out[120]),
      "=r"(__out[121]),
      "=r"(__out[122]),
      "=r"(__out[123]),
      "=r"(__out[124]),
      "=r"(__out[125]),
      "=r"(__out[126]),
      "=r"(__out[127])
    : "r"(__taddr)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_32x32b_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x1.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2(
  B32 (&out)[1],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void
tcgen05_ld_16x32bx2(_B32 (&__out)[1], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x32bx2.x1.b32 {%0}, [%1], %2;"
      : "=r"(__out[0])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x1.pack::16b.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2_pack_16b(
  B32 (&out)[1],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void tcgen05_ld_16x32bx2_pack_16b(
  _B32 (&__out)[1], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x32bx2.x1.pack::16b.b32 {%0}, [%1], %2;"
      : "=r"(__out[0])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x2.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2(
  B32 (&out)[2],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void
tcgen05_ld_16x32bx2(_B32 (&__out)[2], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x32bx2.x2.b32 {%0, %1}, [%2], %3;"
      : "=r"(__out[0]), "=r"(__out[1])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x2.pack::16b.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2_pack_16b(
  B32 (&out)[2],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void tcgen05_ld_16x32bx2_pack_16b(
  _B32 (&__out)[2], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x32bx2.x2.pack::16b.b32 {%0, %1}, [%2], %3;"
      : "=r"(__out[0]), "=r"(__out[1])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x4.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2(
  B32 (&out)[4],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void
tcgen05_ld_16x32bx2(_B32 (&__out)[4], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x32bx2.x4.b32 {%0, %1, %2, %3}, [%4], %5;"
      : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x4.pack::16b.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2_pack_16b(
  B32 (&out)[4],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void tcgen05_ld_16x32bx2_pack_16b(
  _B32 (&__out)[4], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x32bx2.x4.pack::16b.b32 {%0, %1, %2, %3}, [%4], %5;"
      : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x8.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2(
  B32 (&out)[8],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void
tcgen05_ld_16x32bx2(_B32 (&__out)[8], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x32bx2.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x8.pack::16b.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2_pack_16b(
  B32 (&out)[8],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void tcgen05_ld_16x32bx2_pack_16b(
  _B32 (&__out)[8], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x32bx2.x8.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x16.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2(
  B32 (&out)[16],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void
tcgen05_ld_16x32bx2(_B32 (&__out)[16], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x32bx2.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
      "%15}, [%16], %17;"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x16.pack::16b.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2_pack_16b(
  B32 (&out)[16],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void tcgen05_ld_16x32bx2_pack_16b(
  _B32 (&__out)[16], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm("tcgen05.ld.sync.aligned.16x32bx2.x16.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15}, [%16], %17;"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x32.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2(
  B32 (&out)[32],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void
tcgen05_ld_16x32bx2(_B32 (&__out)[32], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], %33;"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31])
    : "r"(__taddr), "n"(__immHalfSplitoff.value)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x32.pack::16b.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2_pack_16b(
  B32 (&out)[32],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void tcgen05_ld_16x32bx2_pack_16b(
  _B32 (&__out)[32], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.16x32bx2.x32.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], %33;"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31])
    : "r"(__taddr), "n"(__immHalfSplitoff.value)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x64.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2(
  B32 (&out)[64],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void
tcgen05_ld_16x32bx2(_B32 (&__out)[64], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.16x32bx2.x64.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63}, [%64], %65;"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63])
    : "r"(__taddr), "n"(__immHalfSplitoff.value)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x64.pack::16b.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2_pack_16b(
  B32 (&out)[64],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void tcgen05_ld_16x32bx2_pack_16b(
  _B32 (&__out)[64], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.16x32bx2.x64.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63}, [%64], %65;"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63])
    : "r"(__taddr), "n"(__immHalfSplitoff.value)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x128.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2(
  B32 (&out)[128],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void
tcgen05_ld_16x32bx2(_B32 (&__out)[128], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.16x32bx2.x128.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, "
    "%82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, "
    "%104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
    "%123, %124, %125, %126, %127}, [%128], %129;"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63]),
      "=r"(__out[64]),
      "=r"(__out[65]),
      "=r"(__out[66]),
      "=r"(__out[67]),
      "=r"(__out[68]),
      "=r"(__out[69]),
      "=r"(__out[70]),
      "=r"(__out[71]),
      "=r"(__out[72]),
      "=r"(__out[73]),
      "=r"(__out[74]),
      "=r"(__out[75]),
      "=r"(__out[76]),
      "=r"(__out[77]),
      "=r"(__out[78]),
      "=r"(__out[79]),
      "=r"(__out[80]),
      "=r"(__out[81]),
      "=r"(__out[82]),
      "=r"(__out[83]),
      "=r"(__out[84]),
      "=r"(__out[85]),
      "=r"(__out[86]),
      "=r"(__out[87]),
      "=r"(__out[88]),
      "=r"(__out[89]),
      "=r"(__out[90]),
      "=r"(__out[91]),
      "=r"(__out[92]),
      "=r"(__out[93]),
      "=r"(__out[94]),
      "=r"(__out[95]),
      "=r"(__out[96]),
      "=r"(__out[97]),
      "=r"(__out[98]),
      "=r"(__out[99]),
      "=r"(__out[100]),
      "=r"(__out[101]),
      "=r"(__out[102]),
      "=r"(__out[103]),
      "=r"(__out[104]),
      "=r"(__out[105]),
      "=r"(__out[106]),
      "=r"(__out[107]),
      "=r"(__out[108]),
      "=r"(__out[109]),
      "=r"(__out[110]),
      "=r"(__out[111]),
      "=r"(__out[112]),
      "=r"(__out[113]),
      "=r"(__out[114]),
      "=r"(__out[115]),
      "=r"(__out[116]),
      "=r"(__out[117]),
      "=r"(__out[118]),
      "=r"(__out[119]),
      "=r"(__out[120]),
      "=r"(__out[121]),
      "=r"(__out[122]),
      "=r"(__out[123]),
      "=r"(__out[124]),
      "=r"(__out[125]),
      "=r"(__out[126]),
      "=r"(__out[127])
    : "r"(__taddr), "n"(__immHalfSplitoff.value)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x128.pack::16b.b32 out, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f,
SM_103a, SM_103f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2_pack_16b(
  B32 (&out)[128],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void tcgen05_ld_16x32bx2_pack_16b(
  _B32 (&__out)[128], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1000) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) \
    || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)                                \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  asm(
    "tcgen05.ld.sync.aligned.16x32bx2.x128.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
    "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, "
    "%102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, "
    "%121, %122, %123, %124, %125, %126, %127}, [%128], %129;"
    : "=r"(__out[0]),
      "=r"(__out[1]),
      "=r"(__out[2]),
      "=r"(__out[3]),
      "=r"(__out[4]),
      "=r"(__out[5]),
      "=r"(__out[6]),
      "=r"(__out[7]),
      "=r"(__out[8]),
      "=r"(__out[9]),
      "=r"(__out[10]),
      "=r"(__out[11]),
      "=r"(__out[12]),
      "=r"(__out[13]),
      "=r"(__out[14]),
      "=r"(__out[15]),
      "=r"(__out[16]),
      "=r"(__out[17]),
      "=r"(__out[18]),
      "=r"(__out[19]),
      "=r"(__out[20]),
      "=r"(__out[21]),
      "=r"(__out[22]),
      "=r"(__out[23]),
      "=r"(__out[24]),
      "=r"(__out[25]),
      "=r"(__out[26]),
      "=r"(__out[27]),
      "=r"(__out[28]),
      "=r"(__out[29]),
      "=r"(__out[30]),
      "=r"(__out[31]),
      "=r"(__out[32]),
      "=r"(__out[33]),
      "=r"(__out[34]),
      "=r"(__out[35]),
      "=r"(__out[36]),
      "=r"(__out[37]),
      "=r"(__out[38]),
      "=r"(__out[39]),
      "=r"(__out[40]),
      "=r"(__out[41]),
      "=r"(__out[42]),
      "=r"(__out[43]),
      "=r"(__out[44]),
      "=r"(__out[45]),
      "=r"(__out[46]),
      "=r"(__out[47]),
      "=r"(__out[48]),
      "=r"(__out[49]),
      "=r"(__out[50]),
      "=r"(__out[51]),
      "=r"(__out[52]),
      "=r"(__out[53]),
      "=r"(__out[54]),
      "=r"(__out[55]),
      "=r"(__out[56]),
      "=r"(__out[57]),
      "=r"(__out[58]),
      "=r"(__out[59]),
      "=r"(__out[60]),
      "=r"(__out[61]),
      "=r"(__out[62]),
      "=r"(__out[63]),
      "=r"(__out[64]),
      "=r"(__out[65]),
      "=r"(__out[66]),
      "=r"(__out[67]),
      "=r"(__out[68]),
      "=r"(__out[69]),
      "=r"(__out[70]),
      "=r"(__out[71]),
      "=r"(__out[72]),
      "=r"(__out[73]),
      "=r"(__out[74]),
      "=r"(__out[75]),
      "=r"(__out[76]),
      "=r"(__out[77]),
      "=r"(__out[78]),
      "=r"(__out[79]),
      "=r"(__out[80]),
      "=r"(__out[81]),
      "=r"(__out[82]),
      "=r"(__out[83]),
      "=r"(__out[84]),
      "=r"(__out[85]),
      "=r"(__out[86]),
      "=r"(__out[87]),
      "=r"(__out[88]),
      "=r"(__out[89]),
      "=r"(__out[90]),
      "=r"(__out[91]),
      "=r"(__out[92]),
      "=r"(__out[93]),
      "=r"(__out[94]),
      "=r"(__out[95]),
      "=r"(__out[96]),
      "=r"(__out[97]),
      "=r"(__out[98]),
      "=r"(__out[99]),
      "=r"(__out[100]),
      "=r"(__out[101]),
      "=r"(__out[102]),
      "=r"(__out[103]),
      "=r"(__out[104]),
      "=r"(__out[105]),
      "=r"(__out[106]),
      "=r"(__out[107]),
      "=r"(__out[108]),
      "=r"(__out[109]),
      "=r"(__out[110]),
      "=r"(__out[111]),
      "=r"(__out[112]),
      "=r"(__out[113]),
      "=r"(__out[114]),
      "=r"(__out[115]),
      "=r"(__out[116]),
      "=r"(__out[117]),
      "=r"(__out[118]),
      "=r"(__out[119]),
      "=r"(__out[120]),
      "=r"(__out[121]),
      "=r"(__out[122]),
      "=r"(__out[123]),
      "=r"(__out[124]),
      "=r"(__out[125]),
      "=r"(__out[126]),
      "=r"(__out[127])
    : "r"(__taddr), "n"(__immHalfSplitoff.value)
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_16x32bx2_pack_16b_is_only_supported_on_SM_100a_100f_103a_103f_110a_110f__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.red.sync.aligned.32x32b.x2.u32.op out, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline uint32_t tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  uint32_t (&out)[2],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::uint32_t
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, ::cuda::std::uint32_t (&__out)[2], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  ::cuda::std::uint32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x2.u32.min {%0, %1}, %2, [%3];"
        : "=r"(__out[0]), "=r"(__out[1]), "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x2.u32.max {%0, %1}, %2, [%3];"
        : "=r"(__out[0]), "=r"(__out[1]), "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x2.s32.op out, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline int32_t tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  int32_t (&out)[2],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::int32_t
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, ::cuda::std::int32_t (&__out)[2], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  ::cuda::std::int32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x2.s32.min {%0, %1}, %2, [%3];"
        : "=r"(__out[0]), "=r"(__out[1]), "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x2.s32.max {%0, %1}, %2, [%3];"
        : "=r"(__out[0]), "=r"(__out[1]), "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x2.f32.op.abs out, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_110a,
SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_32x32b_abs(
  cuda::ptx::op_t<Op> op,
  float (&out)[2],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_32x32b_abs_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float
tcgen05_ld_red_32x32b_abs(::cuda::ptx::op_t<_Op> __op, float (&__out)[2], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x2.f32.min.abs {%0, %1}, %2, [%3];"
        : "=f"(__out[0]), "=f"(__out[1]), "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x2.f32.max.abs {%0, %1}, %2, [%3];"
        : "=f"(__out[0]), "=f"(__out[1]), "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_32x32b_abs_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x2.f32.op out, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  float (&out)[2],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, float (&__out)[2], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x2.f32.min {%0, %1}, %2, [%3];"
        : "=f"(__out[0]), "=f"(__out[1]), "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x2.f32.max {%0, %1}, %2, [%3];"
        : "=f"(__out[0]), "=f"(__out[1]), "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x4.u32.op out, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline uint32_t tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  uint32_t (&out)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::uint32_t
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, ::cuda::std::uint32_t (&__out)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  ::cuda::std::uint32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x4.u32.min {%0, %1, %2, %3}, %4, [%5];"
        : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3]), "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x4.u32.max {%0, %1, %2, %3}, %4, [%5];"
        : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3]), "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x4.s32.op out, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline int32_t tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  int32_t (&out)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::int32_t
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, ::cuda::std::int32_t (&__out)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  ::cuda::std::int32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x4.s32.min {%0, %1, %2, %3}, %4, [%5];"
        : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3]), "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x4.s32.max {%0, %1, %2, %3}, %4, [%5];"
        : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3]), "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x4.f32.op.abs out, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_110a,
SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_32x32b_abs(
  cuda::ptx::op_t<Op> op,
  float (&out)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_32x32b_abs_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float
tcgen05_ld_red_32x32b_abs(::cuda::ptx::op_t<_Op> __op, float (&__out)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x4.f32.min.abs {%0, %1, %2, %3}, %4, [%5];"
        : "=f"(__out[0]), "=f"(__out[1]), "=f"(__out[2]), "=f"(__out[3]), "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x4.f32.max.abs {%0, %1, %2, %3}, %4, [%5];"
        : "=f"(__out[0]), "=f"(__out[1]), "=f"(__out[2]), "=f"(__out[3]), "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_32x32b_abs_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x4.f32.op out, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  float (&out)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, float (&__out)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x4.f32.min {%0, %1, %2, %3}, %4, [%5];"
        : "=f"(__out[0]), "=f"(__out[1]), "=f"(__out[2]), "=f"(__out[3]), "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x4.f32.max {%0, %1, %2, %3}, %4, [%5];"
        : "=f"(__out[0]), "=f"(__out[1]), "=f"(__out[2]), "=f"(__out[3]), "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x8.u32.op out, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline uint32_t tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  uint32_t (&out)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::uint32_t
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, ::cuda::std::uint32_t (&__out)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  ::cuda::std::uint32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x8.u32.min {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9];"
        : "=r"(__out[0]),
          "=r"(__out[1]),
          "=r"(__out[2]),
          "=r"(__out[3]),
          "=r"(__out[4]),
          "=r"(__out[5]),
          "=r"(__out[6]),
          "=r"(__out[7]),
          "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x8.u32.max {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9];"
        : "=r"(__out[0]),
          "=r"(__out[1]),
          "=r"(__out[2]),
          "=r"(__out[3]),
          "=r"(__out[4]),
          "=r"(__out[5]),
          "=r"(__out[6]),
          "=r"(__out[7]),
          "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x8.s32.op out, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline int32_t tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  int32_t (&out)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::int32_t
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, ::cuda::std::int32_t (&__out)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  ::cuda::std::int32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x8.s32.min {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9];"
        : "=r"(__out[0]),
          "=r"(__out[1]),
          "=r"(__out[2]),
          "=r"(__out[3]),
          "=r"(__out[4]),
          "=r"(__out[5]),
          "=r"(__out[6]),
          "=r"(__out[7]),
          "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x8.s32.max {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9];"
        : "=r"(__out[0]),
          "=r"(__out[1]),
          "=r"(__out[2]),
          "=r"(__out[3]),
          "=r"(__out[4]),
          "=r"(__out[5]),
          "=r"(__out[6]),
          "=r"(__out[7]),
          "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x8.f32.op.abs out, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_110a,
SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_32x32b_abs(
  cuda::ptx::op_t<Op> op,
  float (&out)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_32x32b_abs_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float
tcgen05_ld_red_32x32b_abs(::cuda::ptx::op_t<_Op> __op, float (&__out)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x8.f32.min.abs {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9];"
        : "=f"(__out[0]),
          "=f"(__out[1]),
          "=f"(__out[2]),
          "=f"(__out[3]),
          "=f"(__out[4]),
          "=f"(__out[5]),
          "=f"(__out[6]),
          "=f"(__out[7]),
          "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x8.f32.max.abs {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9];"
        : "=f"(__out[0]),
          "=f"(__out[1]),
          "=f"(__out[2]),
          "=f"(__out[3]),
          "=f"(__out[4]),
          "=f"(__out[5]),
          "=f"(__out[6]),
          "=f"(__out[7]),
          "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_32x32b_abs_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x8.f32.op out, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  float (&out)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, float (&__out)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x8.f32.min {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9];"
        : "=f"(__out[0]),
          "=f"(__out[1]),
          "=f"(__out[2]),
          "=f"(__out[3]),
          "=f"(__out[4]),
          "=f"(__out[5]),
          "=f"(__out[6]),
          "=f"(__out[7]),
          "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x8.f32.max {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9];"
        : "=f"(__out[0]),
          "=f"(__out[1]),
          "=f"(__out[2]),
          "=f"(__out[3]),
          "=f"(__out[4]),
          "=f"(__out[5]),
          "=f"(__out[6]),
          "=f"(__out[7]),
          "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x16.u32.op out, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline uint32_t tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  uint32_t (&out)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::uint32_t
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, ::cuda::std::uint32_t (&__out)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  ::cuda::std::uint32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x16.u32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
        "%14, %15}, %16, [%17];"
        : "=r"(__out[0]),
          "=r"(__out[1]),
          "=r"(__out[2]),
          "=r"(__out[3]),
          "=r"(__out[4]),
          "=r"(__out[5]),
          "=r"(__out[6]),
          "=r"(__out[7]),
          "=r"(__out[8]),
          "=r"(__out[9]),
          "=r"(__out[10]),
          "=r"(__out[11]),
          "=r"(__out[12]),
          "=r"(__out[13]),
          "=r"(__out[14]),
          "=r"(__out[15]),
          "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x16.u32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
        "%14, %15}, %16, [%17];"
        : "=r"(__out[0]),
          "=r"(__out[1]),
          "=r"(__out[2]),
          "=r"(__out[3]),
          "=r"(__out[4]),
          "=r"(__out[5]),
          "=r"(__out[6]),
          "=r"(__out[7]),
          "=r"(__out[8]),
          "=r"(__out[9]),
          "=r"(__out[10]),
          "=r"(__out[11]),
          "=r"(__out[12]),
          "=r"(__out[13]),
          "=r"(__out[14]),
          "=r"(__out[15]),
          "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x16.s32.op out, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline int32_t tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  int32_t (&out)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::int32_t
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, ::cuda::std::int32_t (&__out)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  ::cuda::std::int32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x16.s32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
        "%14, %15}, %16, [%17];"
        : "=r"(__out[0]),
          "=r"(__out[1]),
          "=r"(__out[2]),
          "=r"(__out[3]),
          "=r"(__out[4]),
          "=r"(__out[5]),
          "=r"(__out[6]),
          "=r"(__out[7]),
          "=r"(__out[8]),
          "=r"(__out[9]),
          "=r"(__out[10]),
          "=r"(__out[11]),
          "=r"(__out[12]),
          "=r"(__out[13]),
          "=r"(__out[14]),
          "=r"(__out[15]),
          "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x16.s32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
        "%14, %15}, %16, [%17];"
        : "=r"(__out[0]),
          "=r"(__out[1]),
          "=r"(__out[2]),
          "=r"(__out[3]),
          "=r"(__out[4]),
          "=r"(__out[5]),
          "=r"(__out[6]),
          "=r"(__out[7]),
          "=r"(__out[8]),
          "=r"(__out[9]),
          "=r"(__out[10]),
          "=r"(__out[11]),
          "=r"(__out[12]),
          "=r"(__out[13]),
          "=r"(__out[14]),
          "=r"(__out[15]),
          "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x16.f32.op.abs out, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_110a,
SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_32x32b_abs(
  cuda::ptx::op_t<Op> op,
  float (&out)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_32x32b_abs_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float
tcgen05_ld_red_32x32b_abs(::cuda::ptx::op_t<_Op> __op, float (&__out)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x16.f32.min.abs {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
        "%13, %14, %15}, %16, [%17];"
        : "=f"(__out[0]),
          "=f"(__out[1]),
          "=f"(__out[2]),
          "=f"(__out[3]),
          "=f"(__out[4]),
          "=f"(__out[5]),
          "=f"(__out[6]),
          "=f"(__out[7]),
          "=f"(__out[8]),
          "=f"(__out[9]),
          "=f"(__out[10]),
          "=f"(__out[11]),
          "=f"(__out[12]),
          "=f"(__out[13]),
          "=f"(__out[14]),
          "=f"(__out[15]),
          "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x16.f32.max.abs {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
        "%13, %14, %15}, %16, [%17];"
        : "=f"(__out[0]),
          "=f"(__out[1]),
          "=f"(__out[2]),
          "=f"(__out[3]),
          "=f"(__out[4]),
          "=f"(__out[5]),
          "=f"(__out[6]),
          "=f"(__out[7]),
          "=f"(__out[8]),
          "=f"(__out[9]),
          "=f"(__out[10]),
          "=f"(__out[11]),
          "=f"(__out[12]),
          "=f"(__out[13]),
          "=f"(__out[14]),
          "=f"(__out[15]),
          "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_32x32b_abs_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x16.f32.op out, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  float (&out)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, float (&__out)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x16.f32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
        "%14, %15}, %16, [%17];"
        : "=f"(__out[0]),
          "=f"(__out[1]),
          "=f"(__out[2]),
          "=f"(__out[3]),
          "=f"(__out[4]),
          "=f"(__out[5]),
          "=f"(__out[6]),
          "=f"(__out[7]),
          "=f"(__out[8]),
          "=f"(__out[9]),
          "=f"(__out[10]),
          "=f"(__out[11]),
          "=f"(__out[12]),
          "=f"(__out[13]),
          "=f"(__out[14]),
          "=f"(__out[15]),
          "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x16.f32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
        "%14, %15}, %16, [%17];"
        : "=f"(__out[0]),
          "=f"(__out[1]),
          "=f"(__out[2]),
          "=f"(__out[3]),
          "=f"(__out[4]),
          "=f"(__out[5]),
          "=f"(__out[6]),
          "=f"(__out[7]),
          "=f"(__out[8]),
          "=f"(__out[9]),
          "=f"(__out[10]),
          "=f"(__out[11]),
          "=f"(__out[12]),
          "=f"(__out[13]),
          "=f"(__out[14]),
          "=f"(__out[15]),
          "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x32.u32.op out, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline uint32_t tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  uint32_t (&out)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::uint32_t
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, ::cuda::std::uint32_t (&__out)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  ::cuda::std::uint32_t __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x32.u32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15]),
        "=r"(__out[16]),
        "=r"(__out[17]),
        "=r"(__out[18]),
        "=r"(__out[19]),
        "=r"(__out[20]),
        "=r"(__out[21]),
        "=r"(__out[22]),
        "=r"(__out[23]),
        "=r"(__out[24]),
        "=r"(__out[25]),
        "=r"(__out[26]),
        "=r"(__out[27]),
        "=r"(__out[28]),
        "=r"(__out[29]),
        "=r"(__out[30]),
        "=r"(__out[31]),
        "=r"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x32.u32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15]),
        "=r"(__out[16]),
        "=r"(__out[17]),
        "=r"(__out[18]),
        "=r"(__out[19]),
        "=r"(__out[20]),
        "=r"(__out[21]),
        "=r"(__out[22]),
        "=r"(__out[23]),
        "=r"(__out[24]),
        "=r"(__out[25]),
        "=r"(__out[26]),
        "=r"(__out[27]),
        "=r"(__out[28]),
        "=r"(__out[29]),
        "=r"(__out[30]),
        "=r"(__out[31]),
        "=r"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x32.s32.op out, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline int32_t tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  int32_t (&out)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::int32_t
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, ::cuda::std::int32_t (&__out)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  ::cuda::std::int32_t __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x32.s32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15]),
        "=r"(__out[16]),
        "=r"(__out[17]),
        "=r"(__out[18]),
        "=r"(__out[19]),
        "=r"(__out[20]),
        "=r"(__out[21]),
        "=r"(__out[22]),
        "=r"(__out[23]),
        "=r"(__out[24]),
        "=r"(__out[25]),
        "=r"(__out[26]),
        "=r"(__out[27]),
        "=r"(__out[28]),
        "=r"(__out[29]),
        "=r"(__out[30]),
        "=r"(__out[31]),
        "=r"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x32.s32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15]),
        "=r"(__out[16]),
        "=r"(__out[17]),
        "=r"(__out[18]),
        "=r"(__out[19]),
        "=r"(__out[20]),
        "=r"(__out[21]),
        "=r"(__out[22]),
        "=r"(__out[23]),
        "=r"(__out[24]),
        "=r"(__out[25]),
        "=r"(__out[26]),
        "=r"(__out[27]),
        "=r"(__out[28]),
        "=r"(__out[29]),
        "=r"(__out[30]),
        "=r"(__out[31]),
        "=r"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x32.f32.op.abs out, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_110a,
SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_32x32b_abs(
  cuda::ptx::op_t<Op> op,
  float (&out)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_32x32b_abs_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float
tcgen05_ld_red_32x32b_abs(::cuda::ptx::op_t<_Op> __op, float (&__out)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  float __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x32.f32.min.abs {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
      : "=f"(__out[0]),
        "=f"(__out[1]),
        "=f"(__out[2]),
        "=f"(__out[3]),
        "=f"(__out[4]),
        "=f"(__out[5]),
        "=f"(__out[6]),
        "=f"(__out[7]),
        "=f"(__out[8]),
        "=f"(__out[9]),
        "=f"(__out[10]),
        "=f"(__out[11]),
        "=f"(__out[12]),
        "=f"(__out[13]),
        "=f"(__out[14]),
        "=f"(__out[15]),
        "=f"(__out[16]),
        "=f"(__out[17]),
        "=f"(__out[18]),
        "=f"(__out[19]),
        "=f"(__out[20]),
        "=f"(__out[21]),
        "=f"(__out[22]),
        "=f"(__out[23]),
        "=f"(__out[24]),
        "=f"(__out[25]),
        "=f"(__out[26]),
        "=f"(__out[27]),
        "=f"(__out[28]),
        "=f"(__out[29]),
        "=f"(__out[30]),
        "=f"(__out[31]),
        "=f"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x32.f32.max.abs {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
      : "=f"(__out[0]),
        "=f"(__out[1]),
        "=f"(__out[2]),
        "=f"(__out[3]),
        "=f"(__out[4]),
        "=f"(__out[5]),
        "=f"(__out[6]),
        "=f"(__out[7]),
        "=f"(__out[8]),
        "=f"(__out[9]),
        "=f"(__out[10]),
        "=f"(__out[11]),
        "=f"(__out[12]),
        "=f"(__out[13]),
        "=f"(__out[14]),
        "=f"(__out[15]),
        "=f"(__out[16]),
        "=f"(__out[17]),
        "=f"(__out[18]),
        "=f"(__out[19]),
        "=f"(__out[20]),
        "=f"(__out[21]),
        "=f"(__out[22]),
        "=f"(__out[23]),
        "=f"(__out[24]),
        "=f"(__out[25]),
        "=f"(__out[26]),
        "=f"(__out[27]),
        "=f"(__out[28]),
        "=f"(__out[29]),
        "=f"(__out[30]),
        "=f"(__out[31]),
        "=f"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_32x32b_abs_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x32.f32.op out, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  float (&out)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, float (&__out)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  float __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x32.f32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
      : "=f"(__out[0]),
        "=f"(__out[1]),
        "=f"(__out[2]),
        "=f"(__out[3]),
        "=f"(__out[4]),
        "=f"(__out[5]),
        "=f"(__out[6]),
        "=f"(__out[7]),
        "=f"(__out[8]),
        "=f"(__out[9]),
        "=f"(__out[10]),
        "=f"(__out[11]),
        "=f"(__out[12]),
        "=f"(__out[13]),
        "=f"(__out[14]),
        "=f"(__out[15]),
        "=f"(__out[16]),
        "=f"(__out[17]),
        "=f"(__out[18]),
        "=f"(__out[19]),
        "=f"(__out[20]),
        "=f"(__out[21]),
        "=f"(__out[22]),
        "=f"(__out[23]),
        "=f"(__out[24]),
        "=f"(__out[25]),
        "=f"(__out[26]),
        "=f"(__out[27]),
        "=f"(__out[28]),
        "=f"(__out[29]),
        "=f"(__out[30]),
        "=f"(__out[31]),
        "=f"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x32.f32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
      : "=f"(__out[0]),
        "=f"(__out[1]),
        "=f"(__out[2]),
        "=f"(__out[3]),
        "=f"(__out[4]),
        "=f"(__out[5]),
        "=f"(__out[6]),
        "=f"(__out[7]),
        "=f"(__out[8]),
        "=f"(__out[9]),
        "=f"(__out[10]),
        "=f"(__out[11]),
        "=f"(__out[12]),
        "=f"(__out[13]),
        "=f"(__out[14]),
        "=f"(__out[15]),
        "=f"(__out[16]),
        "=f"(__out[17]),
        "=f"(__out[18]),
        "=f"(__out[19]),
        "=f"(__out[20]),
        "=f"(__out[21]),
        "=f"(__out[22]),
        "=f"(__out[23]),
        "=f"(__out[24]),
        "=f"(__out[25]),
        "=f"(__out[26]),
        "=f"(__out[27]),
        "=f"(__out[28]),
        "=f"(__out[29]),
        "=f"(__out[30]),
        "=f"(__out[31]),
        "=f"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x64.u32.op out, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline uint32_t tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  uint32_t (&out)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::uint32_t
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, ::cuda::std::uint32_t (&__out)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  ::cuda::std::uint32_t __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x64.u32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63}, %64, [%65];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15]),
        "=r"(__out[16]),
        "=r"(__out[17]),
        "=r"(__out[18]),
        "=r"(__out[19]),
        "=r"(__out[20]),
        "=r"(__out[21]),
        "=r"(__out[22]),
        "=r"(__out[23]),
        "=r"(__out[24]),
        "=r"(__out[25]),
        "=r"(__out[26]),
        "=r"(__out[27]),
        "=r"(__out[28]),
        "=r"(__out[29]),
        "=r"(__out[30]),
        "=r"(__out[31]),
        "=r"(__out[32]),
        "=r"(__out[33]),
        "=r"(__out[34]),
        "=r"(__out[35]),
        "=r"(__out[36]),
        "=r"(__out[37]),
        "=r"(__out[38]),
        "=r"(__out[39]),
        "=r"(__out[40]),
        "=r"(__out[41]),
        "=r"(__out[42]),
        "=r"(__out[43]),
        "=r"(__out[44]),
        "=r"(__out[45]),
        "=r"(__out[46]),
        "=r"(__out[47]),
        "=r"(__out[48]),
        "=r"(__out[49]),
        "=r"(__out[50]),
        "=r"(__out[51]),
        "=r"(__out[52]),
        "=r"(__out[53]),
        "=r"(__out[54]),
        "=r"(__out[55]),
        "=r"(__out[56]),
        "=r"(__out[57]),
        "=r"(__out[58]),
        "=r"(__out[59]),
        "=r"(__out[60]),
        "=r"(__out[61]),
        "=r"(__out[62]),
        "=r"(__out[63]),
        "=r"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x64.u32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63}, %64, [%65];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15]),
        "=r"(__out[16]),
        "=r"(__out[17]),
        "=r"(__out[18]),
        "=r"(__out[19]),
        "=r"(__out[20]),
        "=r"(__out[21]),
        "=r"(__out[22]),
        "=r"(__out[23]),
        "=r"(__out[24]),
        "=r"(__out[25]),
        "=r"(__out[26]),
        "=r"(__out[27]),
        "=r"(__out[28]),
        "=r"(__out[29]),
        "=r"(__out[30]),
        "=r"(__out[31]),
        "=r"(__out[32]),
        "=r"(__out[33]),
        "=r"(__out[34]),
        "=r"(__out[35]),
        "=r"(__out[36]),
        "=r"(__out[37]),
        "=r"(__out[38]),
        "=r"(__out[39]),
        "=r"(__out[40]),
        "=r"(__out[41]),
        "=r"(__out[42]),
        "=r"(__out[43]),
        "=r"(__out[44]),
        "=r"(__out[45]),
        "=r"(__out[46]),
        "=r"(__out[47]),
        "=r"(__out[48]),
        "=r"(__out[49]),
        "=r"(__out[50]),
        "=r"(__out[51]),
        "=r"(__out[52]),
        "=r"(__out[53]),
        "=r"(__out[54]),
        "=r"(__out[55]),
        "=r"(__out[56]),
        "=r"(__out[57]),
        "=r"(__out[58]),
        "=r"(__out[59]),
        "=r"(__out[60]),
        "=r"(__out[61]),
        "=r"(__out[62]),
        "=r"(__out[63]),
        "=r"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x64.s32.op out, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline int32_t tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  int32_t (&out)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::int32_t
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, ::cuda::std::int32_t (&__out)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  ::cuda::std::int32_t __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x64.s32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63}, %64, [%65];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15]),
        "=r"(__out[16]),
        "=r"(__out[17]),
        "=r"(__out[18]),
        "=r"(__out[19]),
        "=r"(__out[20]),
        "=r"(__out[21]),
        "=r"(__out[22]),
        "=r"(__out[23]),
        "=r"(__out[24]),
        "=r"(__out[25]),
        "=r"(__out[26]),
        "=r"(__out[27]),
        "=r"(__out[28]),
        "=r"(__out[29]),
        "=r"(__out[30]),
        "=r"(__out[31]),
        "=r"(__out[32]),
        "=r"(__out[33]),
        "=r"(__out[34]),
        "=r"(__out[35]),
        "=r"(__out[36]),
        "=r"(__out[37]),
        "=r"(__out[38]),
        "=r"(__out[39]),
        "=r"(__out[40]),
        "=r"(__out[41]),
        "=r"(__out[42]),
        "=r"(__out[43]),
        "=r"(__out[44]),
        "=r"(__out[45]),
        "=r"(__out[46]),
        "=r"(__out[47]),
        "=r"(__out[48]),
        "=r"(__out[49]),
        "=r"(__out[50]),
        "=r"(__out[51]),
        "=r"(__out[52]),
        "=r"(__out[53]),
        "=r"(__out[54]),
        "=r"(__out[55]),
        "=r"(__out[56]),
        "=r"(__out[57]),
        "=r"(__out[58]),
        "=r"(__out[59]),
        "=r"(__out[60]),
        "=r"(__out[61]),
        "=r"(__out[62]),
        "=r"(__out[63]),
        "=r"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x64.s32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63}, %64, [%65];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15]),
        "=r"(__out[16]),
        "=r"(__out[17]),
        "=r"(__out[18]),
        "=r"(__out[19]),
        "=r"(__out[20]),
        "=r"(__out[21]),
        "=r"(__out[22]),
        "=r"(__out[23]),
        "=r"(__out[24]),
        "=r"(__out[25]),
        "=r"(__out[26]),
        "=r"(__out[27]),
        "=r"(__out[28]),
        "=r"(__out[29]),
        "=r"(__out[30]),
        "=r"(__out[31]),
        "=r"(__out[32]),
        "=r"(__out[33]),
        "=r"(__out[34]),
        "=r"(__out[35]),
        "=r"(__out[36]),
        "=r"(__out[37]),
        "=r"(__out[38]),
        "=r"(__out[39]),
        "=r"(__out[40]),
        "=r"(__out[41]),
        "=r"(__out[42]),
        "=r"(__out[43]),
        "=r"(__out[44]),
        "=r"(__out[45]),
        "=r"(__out[46]),
        "=r"(__out[47]),
        "=r"(__out[48]),
        "=r"(__out[49]),
        "=r"(__out[50]),
        "=r"(__out[51]),
        "=r"(__out[52]),
        "=r"(__out[53]),
        "=r"(__out[54]),
        "=r"(__out[55]),
        "=r"(__out[56]),
        "=r"(__out[57]),
        "=r"(__out[58]),
        "=r"(__out[59]),
        "=r"(__out[60]),
        "=r"(__out[61]),
        "=r"(__out[62]),
        "=r"(__out[63]),
        "=r"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x64.f32.op.abs out, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_110a,
SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_32x32b_abs(
  cuda::ptx::op_t<Op> op,
  float (&out)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_32x32b_abs_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float
tcgen05_ld_red_32x32b_abs(::cuda::ptx::op_t<_Op> __op, float (&__out)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  float __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x64.f32.min.abs {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63}, %64, [%65];"
      : "=f"(__out[0]),
        "=f"(__out[1]),
        "=f"(__out[2]),
        "=f"(__out[3]),
        "=f"(__out[4]),
        "=f"(__out[5]),
        "=f"(__out[6]),
        "=f"(__out[7]),
        "=f"(__out[8]),
        "=f"(__out[9]),
        "=f"(__out[10]),
        "=f"(__out[11]),
        "=f"(__out[12]),
        "=f"(__out[13]),
        "=f"(__out[14]),
        "=f"(__out[15]),
        "=f"(__out[16]),
        "=f"(__out[17]),
        "=f"(__out[18]),
        "=f"(__out[19]),
        "=f"(__out[20]),
        "=f"(__out[21]),
        "=f"(__out[22]),
        "=f"(__out[23]),
        "=f"(__out[24]),
        "=f"(__out[25]),
        "=f"(__out[26]),
        "=f"(__out[27]),
        "=f"(__out[28]),
        "=f"(__out[29]),
        "=f"(__out[30]),
        "=f"(__out[31]),
        "=f"(__out[32]),
        "=f"(__out[33]),
        "=f"(__out[34]),
        "=f"(__out[35]),
        "=f"(__out[36]),
        "=f"(__out[37]),
        "=f"(__out[38]),
        "=f"(__out[39]),
        "=f"(__out[40]),
        "=f"(__out[41]),
        "=f"(__out[42]),
        "=f"(__out[43]),
        "=f"(__out[44]),
        "=f"(__out[45]),
        "=f"(__out[46]),
        "=f"(__out[47]),
        "=f"(__out[48]),
        "=f"(__out[49]),
        "=f"(__out[50]),
        "=f"(__out[51]),
        "=f"(__out[52]),
        "=f"(__out[53]),
        "=f"(__out[54]),
        "=f"(__out[55]),
        "=f"(__out[56]),
        "=f"(__out[57]),
        "=f"(__out[58]),
        "=f"(__out[59]),
        "=f"(__out[60]),
        "=f"(__out[61]),
        "=f"(__out[62]),
        "=f"(__out[63]),
        "=f"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x64.f32.max.abs {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63}, %64, [%65];"
      : "=f"(__out[0]),
        "=f"(__out[1]),
        "=f"(__out[2]),
        "=f"(__out[3]),
        "=f"(__out[4]),
        "=f"(__out[5]),
        "=f"(__out[6]),
        "=f"(__out[7]),
        "=f"(__out[8]),
        "=f"(__out[9]),
        "=f"(__out[10]),
        "=f"(__out[11]),
        "=f"(__out[12]),
        "=f"(__out[13]),
        "=f"(__out[14]),
        "=f"(__out[15]),
        "=f"(__out[16]),
        "=f"(__out[17]),
        "=f"(__out[18]),
        "=f"(__out[19]),
        "=f"(__out[20]),
        "=f"(__out[21]),
        "=f"(__out[22]),
        "=f"(__out[23]),
        "=f"(__out[24]),
        "=f"(__out[25]),
        "=f"(__out[26]),
        "=f"(__out[27]),
        "=f"(__out[28]),
        "=f"(__out[29]),
        "=f"(__out[30]),
        "=f"(__out[31]),
        "=f"(__out[32]),
        "=f"(__out[33]),
        "=f"(__out[34]),
        "=f"(__out[35]),
        "=f"(__out[36]),
        "=f"(__out[37]),
        "=f"(__out[38]),
        "=f"(__out[39]),
        "=f"(__out[40]),
        "=f"(__out[41]),
        "=f"(__out[42]),
        "=f"(__out[43]),
        "=f"(__out[44]),
        "=f"(__out[45]),
        "=f"(__out[46]),
        "=f"(__out[47]),
        "=f"(__out[48]),
        "=f"(__out[49]),
        "=f"(__out[50]),
        "=f"(__out[51]),
        "=f"(__out[52]),
        "=f"(__out[53]),
        "=f"(__out[54]),
        "=f"(__out[55]),
        "=f"(__out[56]),
        "=f"(__out[57]),
        "=f"(__out[58]),
        "=f"(__out[59]),
        "=f"(__out[60]),
        "=f"(__out[61]),
        "=f"(__out[62]),
        "=f"(__out[63]),
        "=f"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_32x32b_abs_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x64.f32.op out, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  float (&out)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, float (&__out)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  float __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x64.f32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63}, %64, [%65];"
      : "=f"(__out[0]),
        "=f"(__out[1]),
        "=f"(__out[2]),
        "=f"(__out[3]),
        "=f"(__out[4]),
        "=f"(__out[5]),
        "=f"(__out[6]),
        "=f"(__out[7]),
        "=f"(__out[8]),
        "=f"(__out[9]),
        "=f"(__out[10]),
        "=f"(__out[11]),
        "=f"(__out[12]),
        "=f"(__out[13]),
        "=f"(__out[14]),
        "=f"(__out[15]),
        "=f"(__out[16]),
        "=f"(__out[17]),
        "=f"(__out[18]),
        "=f"(__out[19]),
        "=f"(__out[20]),
        "=f"(__out[21]),
        "=f"(__out[22]),
        "=f"(__out[23]),
        "=f"(__out[24]),
        "=f"(__out[25]),
        "=f"(__out[26]),
        "=f"(__out[27]),
        "=f"(__out[28]),
        "=f"(__out[29]),
        "=f"(__out[30]),
        "=f"(__out[31]),
        "=f"(__out[32]),
        "=f"(__out[33]),
        "=f"(__out[34]),
        "=f"(__out[35]),
        "=f"(__out[36]),
        "=f"(__out[37]),
        "=f"(__out[38]),
        "=f"(__out[39]),
        "=f"(__out[40]),
        "=f"(__out[41]),
        "=f"(__out[42]),
        "=f"(__out[43]),
        "=f"(__out[44]),
        "=f"(__out[45]),
        "=f"(__out[46]),
        "=f"(__out[47]),
        "=f"(__out[48]),
        "=f"(__out[49]),
        "=f"(__out[50]),
        "=f"(__out[51]),
        "=f"(__out[52]),
        "=f"(__out[53]),
        "=f"(__out[54]),
        "=f"(__out[55]),
        "=f"(__out[56]),
        "=f"(__out[57]),
        "=f"(__out[58]),
        "=f"(__out[59]),
        "=f"(__out[60]),
        "=f"(__out[61]),
        "=f"(__out[62]),
        "=f"(__out[63]),
        "=f"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x64.f32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63}, %64, [%65];"
      : "=f"(__out[0]),
        "=f"(__out[1]),
        "=f"(__out[2]),
        "=f"(__out[3]),
        "=f"(__out[4]),
        "=f"(__out[5]),
        "=f"(__out[6]),
        "=f"(__out[7]),
        "=f"(__out[8]),
        "=f"(__out[9]),
        "=f"(__out[10]),
        "=f"(__out[11]),
        "=f"(__out[12]),
        "=f"(__out[13]),
        "=f"(__out[14]),
        "=f"(__out[15]),
        "=f"(__out[16]),
        "=f"(__out[17]),
        "=f"(__out[18]),
        "=f"(__out[19]),
        "=f"(__out[20]),
        "=f"(__out[21]),
        "=f"(__out[22]),
        "=f"(__out[23]),
        "=f"(__out[24]),
        "=f"(__out[25]),
        "=f"(__out[26]),
        "=f"(__out[27]),
        "=f"(__out[28]),
        "=f"(__out[29]),
        "=f"(__out[30]),
        "=f"(__out[31]),
        "=f"(__out[32]),
        "=f"(__out[33]),
        "=f"(__out[34]),
        "=f"(__out[35]),
        "=f"(__out[36]),
        "=f"(__out[37]),
        "=f"(__out[38]),
        "=f"(__out[39]),
        "=f"(__out[40]),
        "=f"(__out[41]),
        "=f"(__out[42]),
        "=f"(__out[43]),
        "=f"(__out[44]),
        "=f"(__out[45]),
        "=f"(__out[46]),
        "=f"(__out[47]),
        "=f"(__out[48]),
        "=f"(__out[49]),
        "=f"(__out[50]),
        "=f"(__out[51]),
        "=f"(__out[52]),
        "=f"(__out[53]),
        "=f"(__out[54]),
        "=f"(__out[55]),
        "=f"(__out[56]),
        "=f"(__out[57]),
        "=f"(__out[58]),
        "=f"(__out[59]),
        "=f"(__out[60]),
        "=f"(__out[61]),
        "=f"(__out[62]),
        "=f"(__out[63]),
        "=f"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x128.u32.op out, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_110a,
SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline uint32_t tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  uint32_t (&out)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::uint32_t
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, ::cuda::std::uint32_t (&__out)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  ::cuda::std::uint32_t __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x128.u32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
      "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, "
      "%102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, "
      "%120, %121, %122, %123, %124, %125, %126, %127}, %128, [%129];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15]),
        "=r"(__out[16]),
        "=r"(__out[17]),
        "=r"(__out[18]),
        "=r"(__out[19]),
        "=r"(__out[20]),
        "=r"(__out[21]),
        "=r"(__out[22]),
        "=r"(__out[23]),
        "=r"(__out[24]),
        "=r"(__out[25]),
        "=r"(__out[26]),
        "=r"(__out[27]),
        "=r"(__out[28]),
        "=r"(__out[29]),
        "=r"(__out[30]),
        "=r"(__out[31]),
        "=r"(__out[32]),
        "=r"(__out[33]),
        "=r"(__out[34]),
        "=r"(__out[35]),
        "=r"(__out[36]),
        "=r"(__out[37]),
        "=r"(__out[38]),
        "=r"(__out[39]),
        "=r"(__out[40]),
        "=r"(__out[41]),
        "=r"(__out[42]),
        "=r"(__out[43]),
        "=r"(__out[44]),
        "=r"(__out[45]),
        "=r"(__out[46]),
        "=r"(__out[47]),
        "=r"(__out[48]),
        "=r"(__out[49]),
        "=r"(__out[50]),
        "=r"(__out[51]),
        "=r"(__out[52]),
        "=r"(__out[53]),
        "=r"(__out[54]),
        "=r"(__out[55]),
        "=r"(__out[56]),
        "=r"(__out[57]),
        "=r"(__out[58]),
        "=r"(__out[59]),
        "=r"(__out[60]),
        "=r"(__out[61]),
        "=r"(__out[62]),
        "=r"(__out[63]),
        "=r"(__out[64]),
        "=r"(__out[65]),
        "=r"(__out[66]),
        "=r"(__out[67]),
        "=r"(__out[68]),
        "=r"(__out[69]),
        "=r"(__out[70]),
        "=r"(__out[71]),
        "=r"(__out[72]),
        "=r"(__out[73]),
        "=r"(__out[74]),
        "=r"(__out[75]),
        "=r"(__out[76]),
        "=r"(__out[77]),
        "=r"(__out[78]),
        "=r"(__out[79]),
        "=r"(__out[80]),
        "=r"(__out[81]),
        "=r"(__out[82]),
        "=r"(__out[83]),
        "=r"(__out[84]),
        "=r"(__out[85]),
        "=r"(__out[86]),
        "=r"(__out[87]),
        "=r"(__out[88]),
        "=r"(__out[89]),
        "=r"(__out[90]),
        "=r"(__out[91]),
        "=r"(__out[92]),
        "=r"(__out[93]),
        "=r"(__out[94]),
        "=r"(__out[95]),
        "=r"(__out[96]),
        "=r"(__out[97]),
        "=r"(__out[98]),
        "=r"(__out[99]),
        "=r"(__out[100]),
        "=r"(__out[101]),
        "=r"(__out[102]),
        "=r"(__out[103]),
        "=r"(__out[104]),
        "=r"(__out[105]),
        "=r"(__out[106]),
        "=r"(__out[107]),
        "=r"(__out[108]),
        "=r"(__out[109]),
        "=r"(__out[110]),
        "=r"(__out[111]),
        "=r"(__out[112]),
        "=r"(__out[113]),
        "=r"(__out[114]),
        "=r"(__out[115]),
        "=r"(__out[116]),
        "=r"(__out[117]),
        "=r"(__out[118]),
        "=r"(__out[119]),
        "=r"(__out[120]),
        "=r"(__out[121]),
        "=r"(__out[122]),
        "=r"(__out[123]),
        "=r"(__out[124]),
        "=r"(__out[125]),
        "=r"(__out[126]),
        "=r"(__out[127]),
        "=r"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x128.u32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
      "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, "
      "%102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, "
      "%120, %121, %122, %123, %124, %125, %126, %127}, %128, [%129];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15]),
        "=r"(__out[16]),
        "=r"(__out[17]),
        "=r"(__out[18]),
        "=r"(__out[19]),
        "=r"(__out[20]),
        "=r"(__out[21]),
        "=r"(__out[22]),
        "=r"(__out[23]),
        "=r"(__out[24]),
        "=r"(__out[25]),
        "=r"(__out[26]),
        "=r"(__out[27]),
        "=r"(__out[28]),
        "=r"(__out[29]),
        "=r"(__out[30]),
        "=r"(__out[31]),
        "=r"(__out[32]),
        "=r"(__out[33]),
        "=r"(__out[34]),
        "=r"(__out[35]),
        "=r"(__out[36]),
        "=r"(__out[37]),
        "=r"(__out[38]),
        "=r"(__out[39]),
        "=r"(__out[40]),
        "=r"(__out[41]),
        "=r"(__out[42]),
        "=r"(__out[43]),
        "=r"(__out[44]),
        "=r"(__out[45]),
        "=r"(__out[46]),
        "=r"(__out[47]),
        "=r"(__out[48]),
        "=r"(__out[49]),
        "=r"(__out[50]),
        "=r"(__out[51]),
        "=r"(__out[52]),
        "=r"(__out[53]),
        "=r"(__out[54]),
        "=r"(__out[55]),
        "=r"(__out[56]),
        "=r"(__out[57]),
        "=r"(__out[58]),
        "=r"(__out[59]),
        "=r"(__out[60]),
        "=r"(__out[61]),
        "=r"(__out[62]),
        "=r"(__out[63]),
        "=r"(__out[64]),
        "=r"(__out[65]),
        "=r"(__out[66]),
        "=r"(__out[67]),
        "=r"(__out[68]),
        "=r"(__out[69]),
        "=r"(__out[70]),
        "=r"(__out[71]),
        "=r"(__out[72]),
        "=r"(__out[73]),
        "=r"(__out[74]),
        "=r"(__out[75]),
        "=r"(__out[76]),
        "=r"(__out[77]),
        "=r"(__out[78]),
        "=r"(__out[79]),
        "=r"(__out[80]),
        "=r"(__out[81]),
        "=r"(__out[82]),
        "=r"(__out[83]),
        "=r"(__out[84]),
        "=r"(__out[85]),
        "=r"(__out[86]),
        "=r"(__out[87]),
        "=r"(__out[88]),
        "=r"(__out[89]),
        "=r"(__out[90]),
        "=r"(__out[91]),
        "=r"(__out[92]),
        "=r"(__out[93]),
        "=r"(__out[94]),
        "=r"(__out[95]),
        "=r"(__out[96]),
        "=r"(__out[97]),
        "=r"(__out[98]),
        "=r"(__out[99]),
        "=r"(__out[100]),
        "=r"(__out[101]),
        "=r"(__out[102]),
        "=r"(__out[103]),
        "=r"(__out[104]),
        "=r"(__out[105]),
        "=r"(__out[106]),
        "=r"(__out[107]),
        "=r"(__out[108]),
        "=r"(__out[109]),
        "=r"(__out[110]),
        "=r"(__out[111]),
        "=r"(__out[112]),
        "=r"(__out[113]),
        "=r"(__out[114]),
        "=r"(__out[115]),
        "=r"(__out[116]),
        "=r"(__out[117]),
        "=r"(__out[118]),
        "=r"(__out[119]),
        "=r"(__out[120]),
        "=r"(__out[121]),
        "=r"(__out[122]),
        "=r"(__out[123]),
        "=r"(__out[124]),
        "=r"(__out[125]),
        "=r"(__out[126]),
        "=r"(__out[127]),
        "=r"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x128.s32.op out, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_110a,
SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline int32_t tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  int32_t (&out)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::int32_t
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, ::cuda::std::int32_t (&__out)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  ::cuda::std::int32_t __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x128.s32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
      "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, "
      "%102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, "
      "%120, %121, %122, %123, %124, %125, %126, %127}, %128, [%129];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15]),
        "=r"(__out[16]),
        "=r"(__out[17]),
        "=r"(__out[18]),
        "=r"(__out[19]),
        "=r"(__out[20]),
        "=r"(__out[21]),
        "=r"(__out[22]),
        "=r"(__out[23]),
        "=r"(__out[24]),
        "=r"(__out[25]),
        "=r"(__out[26]),
        "=r"(__out[27]),
        "=r"(__out[28]),
        "=r"(__out[29]),
        "=r"(__out[30]),
        "=r"(__out[31]),
        "=r"(__out[32]),
        "=r"(__out[33]),
        "=r"(__out[34]),
        "=r"(__out[35]),
        "=r"(__out[36]),
        "=r"(__out[37]),
        "=r"(__out[38]),
        "=r"(__out[39]),
        "=r"(__out[40]),
        "=r"(__out[41]),
        "=r"(__out[42]),
        "=r"(__out[43]),
        "=r"(__out[44]),
        "=r"(__out[45]),
        "=r"(__out[46]),
        "=r"(__out[47]),
        "=r"(__out[48]),
        "=r"(__out[49]),
        "=r"(__out[50]),
        "=r"(__out[51]),
        "=r"(__out[52]),
        "=r"(__out[53]),
        "=r"(__out[54]),
        "=r"(__out[55]),
        "=r"(__out[56]),
        "=r"(__out[57]),
        "=r"(__out[58]),
        "=r"(__out[59]),
        "=r"(__out[60]),
        "=r"(__out[61]),
        "=r"(__out[62]),
        "=r"(__out[63]),
        "=r"(__out[64]),
        "=r"(__out[65]),
        "=r"(__out[66]),
        "=r"(__out[67]),
        "=r"(__out[68]),
        "=r"(__out[69]),
        "=r"(__out[70]),
        "=r"(__out[71]),
        "=r"(__out[72]),
        "=r"(__out[73]),
        "=r"(__out[74]),
        "=r"(__out[75]),
        "=r"(__out[76]),
        "=r"(__out[77]),
        "=r"(__out[78]),
        "=r"(__out[79]),
        "=r"(__out[80]),
        "=r"(__out[81]),
        "=r"(__out[82]),
        "=r"(__out[83]),
        "=r"(__out[84]),
        "=r"(__out[85]),
        "=r"(__out[86]),
        "=r"(__out[87]),
        "=r"(__out[88]),
        "=r"(__out[89]),
        "=r"(__out[90]),
        "=r"(__out[91]),
        "=r"(__out[92]),
        "=r"(__out[93]),
        "=r"(__out[94]),
        "=r"(__out[95]),
        "=r"(__out[96]),
        "=r"(__out[97]),
        "=r"(__out[98]),
        "=r"(__out[99]),
        "=r"(__out[100]),
        "=r"(__out[101]),
        "=r"(__out[102]),
        "=r"(__out[103]),
        "=r"(__out[104]),
        "=r"(__out[105]),
        "=r"(__out[106]),
        "=r"(__out[107]),
        "=r"(__out[108]),
        "=r"(__out[109]),
        "=r"(__out[110]),
        "=r"(__out[111]),
        "=r"(__out[112]),
        "=r"(__out[113]),
        "=r"(__out[114]),
        "=r"(__out[115]),
        "=r"(__out[116]),
        "=r"(__out[117]),
        "=r"(__out[118]),
        "=r"(__out[119]),
        "=r"(__out[120]),
        "=r"(__out[121]),
        "=r"(__out[122]),
        "=r"(__out[123]),
        "=r"(__out[124]),
        "=r"(__out[125]),
        "=r"(__out[126]),
        "=r"(__out[127]),
        "=r"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x128.s32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
      "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, "
      "%102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, "
      "%120, %121, %122, %123, %124, %125, %126, %127}, %128, [%129];"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15]),
        "=r"(__out[16]),
        "=r"(__out[17]),
        "=r"(__out[18]),
        "=r"(__out[19]),
        "=r"(__out[20]),
        "=r"(__out[21]),
        "=r"(__out[22]),
        "=r"(__out[23]),
        "=r"(__out[24]),
        "=r"(__out[25]),
        "=r"(__out[26]),
        "=r"(__out[27]),
        "=r"(__out[28]),
        "=r"(__out[29]),
        "=r"(__out[30]),
        "=r"(__out[31]),
        "=r"(__out[32]),
        "=r"(__out[33]),
        "=r"(__out[34]),
        "=r"(__out[35]),
        "=r"(__out[36]),
        "=r"(__out[37]),
        "=r"(__out[38]),
        "=r"(__out[39]),
        "=r"(__out[40]),
        "=r"(__out[41]),
        "=r"(__out[42]),
        "=r"(__out[43]),
        "=r"(__out[44]),
        "=r"(__out[45]),
        "=r"(__out[46]),
        "=r"(__out[47]),
        "=r"(__out[48]),
        "=r"(__out[49]),
        "=r"(__out[50]),
        "=r"(__out[51]),
        "=r"(__out[52]),
        "=r"(__out[53]),
        "=r"(__out[54]),
        "=r"(__out[55]),
        "=r"(__out[56]),
        "=r"(__out[57]),
        "=r"(__out[58]),
        "=r"(__out[59]),
        "=r"(__out[60]),
        "=r"(__out[61]),
        "=r"(__out[62]),
        "=r"(__out[63]),
        "=r"(__out[64]),
        "=r"(__out[65]),
        "=r"(__out[66]),
        "=r"(__out[67]),
        "=r"(__out[68]),
        "=r"(__out[69]),
        "=r"(__out[70]),
        "=r"(__out[71]),
        "=r"(__out[72]),
        "=r"(__out[73]),
        "=r"(__out[74]),
        "=r"(__out[75]),
        "=r"(__out[76]),
        "=r"(__out[77]),
        "=r"(__out[78]),
        "=r"(__out[79]),
        "=r"(__out[80]),
        "=r"(__out[81]),
        "=r"(__out[82]),
        "=r"(__out[83]),
        "=r"(__out[84]),
        "=r"(__out[85]),
        "=r"(__out[86]),
        "=r"(__out[87]),
        "=r"(__out[88]),
        "=r"(__out[89]),
        "=r"(__out[90]),
        "=r"(__out[91]),
        "=r"(__out[92]),
        "=r"(__out[93]),
        "=r"(__out[94]),
        "=r"(__out[95]),
        "=r"(__out[96]),
        "=r"(__out[97]),
        "=r"(__out[98]),
        "=r"(__out[99]),
        "=r"(__out[100]),
        "=r"(__out[101]),
        "=r"(__out[102]),
        "=r"(__out[103]),
        "=r"(__out[104]),
        "=r"(__out[105]),
        "=r"(__out[106]),
        "=r"(__out[107]),
        "=r"(__out[108]),
        "=r"(__out[109]),
        "=r"(__out[110]),
        "=r"(__out[111]),
        "=r"(__out[112]),
        "=r"(__out[113]),
        "=r"(__out[114]),
        "=r"(__out[115]),
        "=r"(__out[116]),
        "=r"(__out[117]),
        "=r"(__out[118]),
        "=r"(__out[119]),
        "=r"(__out[120]),
        "=r"(__out[121]),
        "=r"(__out[122]),
        "=r"(__out[123]),
        "=r"(__out[124]),
        "=r"(__out[125]),
        "=r"(__out[126]),
        "=r"(__out[127]),
        "=r"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x128.f32.op.abs out, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_110a,
SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_32x32b_abs(
  cuda::ptx::op_t<Op> op,
  float (&out)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_32x32b_abs_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float
tcgen05_ld_red_32x32b_abs(::cuda::ptx::op_t<_Op> __op, float (&__out)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  float __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x128.f32.min.abs {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
      "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, "
      "%35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
      "%57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, "
      "%79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, "
      "%101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, "
      "%119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, [%129];"
      : "=f"(__out[0]),
        "=f"(__out[1]),
        "=f"(__out[2]),
        "=f"(__out[3]),
        "=f"(__out[4]),
        "=f"(__out[5]),
        "=f"(__out[6]),
        "=f"(__out[7]),
        "=f"(__out[8]),
        "=f"(__out[9]),
        "=f"(__out[10]),
        "=f"(__out[11]),
        "=f"(__out[12]),
        "=f"(__out[13]),
        "=f"(__out[14]),
        "=f"(__out[15]),
        "=f"(__out[16]),
        "=f"(__out[17]),
        "=f"(__out[18]),
        "=f"(__out[19]),
        "=f"(__out[20]),
        "=f"(__out[21]),
        "=f"(__out[22]),
        "=f"(__out[23]),
        "=f"(__out[24]),
        "=f"(__out[25]),
        "=f"(__out[26]),
        "=f"(__out[27]),
        "=f"(__out[28]),
        "=f"(__out[29]),
        "=f"(__out[30]),
        "=f"(__out[31]),
        "=f"(__out[32]),
        "=f"(__out[33]),
        "=f"(__out[34]),
        "=f"(__out[35]),
        "=f"(__out[36]),
        "=f"(__out[37]),
        "=f"(__out[38]),
        "=f"(__out[39]),
        "=f"(__out[40]),
        "=f"(__out[41]),
        "=f"(__out[42]),
        "=f"(__out[43]),
        "=f"(__out[44]),
        "=f"(__out[45]),
        "=f"(__out[46]),
        "=f"(__out[47]),
        "=f"(__out[48]),
        "=f"(__out[49]),
        "=f"(__out[50]),
        "=f"(__out[51]),
        "=f"(__out[52]),
        "=f"(__out[53]),
        "=f"(__out[54]),
        "=f"(__out[55]),
        "=f"(__out[56]),
        "=f"(__out[57]),
        "=f"(__out[58]),
        "=f"(__out[59]),
        "=f"(__out[60]),
        "=f"(__out[61]),
        "=f"(__out[62]),
        "=f"(__out[63]),
        "=f"(__out[64]),
        "=f"(__out[65]),
        "=f"(__out[66]),
        "=f"(__out[67]),
        "=f"(__out[68]),
        "=f"(__out[69]),
        "=f"(__out[70]),
        "=f"(__out[71]),
        "=f"(__out[72]),
        "=f"(__out[73]),
        "=f"(__out[74]),
        "=f"(__out[75]),
        "=f"(__out[76]),
        "=f"(__out[77]),
        "=f"(__out[78]),
        "=f"(__out[79]),
        "=f"(__out[80]),
        "=f"(__out[81]),
        "=f"(__out[82]),
        "=f"(__out[83]),
        "=f"(__out[84]),
        "=f"(__out[85]),
        "=f"(__out[86]),
        "=f"(__out[87]),
        "=f"(__out[88]),
        "=f"(__out[89]),
        "=f"(__out[90]),
        "=f"(__out[91]),
        "=f"(__out[92]),
        "=f"(__out[93]),
        "=f"(__out[94]),
        "=f"(__out[95]),
        "=f"(__out[96]),
        "=f"(__out[97]),
        "=f"(__out[98]),
        "=f"(__out[99]),
        "=f"(__out[100]),
        "=f"(__out[101]),
        "=f"(__out[102]),
        "=f"(__out[103]),
        "=f"(__out[104]),
        "=f"(__out[105]),
        "=f"(__out[106]),
        "=f"(__out[107]),
        "=f"(__out[108]),
        "=f"(__out[109]),
        "=f"(__out[110]),
        "=f"(__out[111]),
        "=f"(__out[112]),
        "=f"(__out[113]),
        "=f"(__out[114]),
        "=f"(__out[115]),
        "=f"(__out[116]),
        "=f"(__out[117]),
        "=f"(__out[118]),
        "=f"(__out[119]),
        "=f"(__out[120]),
        "=f"(__out[121]),
        "=f"(__out[122]),
        "=f"(__out[123]),
        "=f"(__out[124]),
        "=f"(__out[125]),
        "=f"(__out[126]),
        "=f"(__out[127]),
        "=f"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x128.f32.max.abs {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
      "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, "
      "%35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
      "%57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, "
      "%79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, "
      "%101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, "
      "%119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, [%129];"
      : "=f"(__out[0]),
        "=f"(__out[1]),
        "=f"(__out[2]),
        "=f"(__out[3]),
        "=f"(__out[4]),
        "=f"(__out[5]),
        "=f"(__out[6]),
        "=f"(__out[7]),
        "=f"(__out[8]),
        "=f"(__out[9]),
        "=f"(__out[10]),
        "=f"(__out[11]),
        "=f"(__out[12]),
        "=f"(__out[13]),
        "=f"(__out[14]),
        "=f"(__out[15]),
        "=f"(__out[16]),
        "=f"(__out[17]),
        "=f"(__out[18]),
        "=f"(__out[19]),
        "=f"(__out[20]),
        "=f"(__out[21]),
        "=f"(__out[22]),
        "=f"(__out[23]),
        "=f"(__out[24]),
        "=f"(__out[25]),
        "=f"(__out[26]),
        "=f"(__out[27]),
        "=f"(__out[28]),
        "=f"(__out[29]),
        "=f"(__out[30]),
        "=f"(__out[31]),
        "=f"(__out[32]),
        "=f"(__out[33]),
        "=f"(__out[34]),
        "=f"(__out[35]),
        "=f"(__out[36]),
        "=f"(__out[37]),
        "=f"(__out[38]),
        "=f"(__out[39]),
        "=f"(__out[40]),
        "=f"(__out[41]),
        "=f"(__out[42]),
        "=f"(__out[43]),
        "=f"(__out[44]),
        "=f"(__out[45]),
        "=f"(__out[46]),
        "=f"(__out[47]),
        "=f"(__out[48]),
        "=f"(__out[49]),
        "=f"(__out[50]),
        "=f"(__out[51]),
        "=f"(__out[52]),
        "=f"(__out[53]),
        "=f"(__out[54]),
        "=f"(__out[55]),
        "=f"(__out[56]),
        "=f"(__out[57]),
        "=f"(__out[58]),
        "=f"(__out[59]),
        "=f"(__out[60]),
        "=f"(__out[61]),
        "=f"(__out[62]),
        "=f"(__out[63]),
        "=f"(__out[64]),
        "=f"(__out[65]),
        "=f"(__out[66]),
        "=f"(__out[67]),
        "=f"(__out[68]),
        "=f"(__out[69]),
        "=f"(__out[70]),
        "=f"(__out[71]),
        "=f"(__out[72]),
        "=f"(__out[73]),
        "=f"(__out[74]),
        "=f"(__out[75]),
        "=f"(__out[76]),
        "=f"(__out[77]),
        "=f"(__out[78]),
        "=f"(__out[79]),
        "=f"(__out[80]),
        "=f"(__out[81]),
        "=f"(__out[82]),
        "=f"(__out[83]),
        "=f"(__out[84]),
        "=f"(__out[85]),
        "=f"(__out[86]),
        "=f"(__out[87]),
        "=f"(__out[88]),
        "=f"(__out[89]),
        "=f"(__out[90]),
        "=f"(__out[91]),
        "=f"(__out[92]),
        "=f"(__out[93]),
        "=f"(__out[94]),
        "=f"(__out[95]),
        "=f"(__out[96]),
        "=f"(__out[97]),
        "=f"(__out[98]),
        "=f"(__out[99]),
        "=f"(__out[100]),
        "=f"(__out[101]),
        "=f"(__out[102]),
        "=f"(__out[103]),
        "=f"(__out[104]),
        "=f"(__out[105]),
        "=f"(__out[106]),
        "=f"(__out[107]),
        "=f"(__out[108]),
        "=f"(__out[109]),
        "=f"(__out[110]),
        "=f"(__out[111]),
        "=f"(__out[112]),
        "=f"(__out[113]),
        "=f"(__out[114]),
        "=f"(__out[115]),
        "=f"(__out[116]),
        "=f"(__out[117]),
        "=f"(__out[118]),
        "=f"(__out[119]),
        "=f"(__out[120]),
        "=f"(__out[121]),
        "=f"(__out[122]),
        "=f"(__out[123]),
        "=f"(__out[124]),
        "=f"(__out[125]),
        "=f"(__out[126]),
        "=f"(__out[127]),
        "=f"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_32x32b_abs_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x128.f32.op out, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_110a,
SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  float (&out)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, float (&__out)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  float __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x128.f32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
      "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, "
      "%102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, "
      "%120, %121, %122, %123, %124, %125, %126, %127}, %128, [%129];"
      : "=f"(__out[0]),
        "=f"(__out[1]),
        "=f"(__out[2]),
        "=f"(__out[3]),
        "=f"(__out[4]),
        "=f"(__out[5]),
        "=f"(__out[6]),
        "=f"(__out[7]),
        "=f"(__out[8]),
        "=f"(__out[9]),
        "=f"(__out[10]),
        "=f"(__out[11]),
        "=f"(__out[12]),
        "=f"(__out[13]),
        "=f"(__out[14]),
        "=f"(__out[15]),
        "=f"(__out[16]),
        "=f"(__out[17]),
        "=f"(__out[18]),
        "=f"(__out[19]),
        "=f"(__out[20]),
        "=f"(__out[21]),
        "=f"(__out[22]),
        "=f"(__out[23]),
        "=f"(__out[24]),
        "=f"(__out[25]),
        "=f"(__out[26]),
        "=f"(__out[27]),
        "=f"(__out[28]),
        "=f"(__out[29]),
        "=f"(__out[30]),
        "=f"(__out[31]),
        "=f"(__out[32]),
        "=f"(__out[33]),
        "=f"(__out[34]),
        "=f"(__out[35]),
        "=f"(__out[36]),
        "=f"(__out[37]),
        "=f"(__out[38]),
        "=f"(__out[39]),
        "=f"(__out[40]),
        "=f"(__out[41]),
        "=f"(__out[42]),
        "=f"(__out[43]),
        "=f"(__out[44]),
        "=f"(__out[45]),
        "=f"(__out[46]),
        "=f"(__out[47]),
        "=f"(__out[48]),
        "=f"(__out[49]),
        "=f"(__out[50]),
        "=f"(__out[51]),
        "=f"(__out[52]),
        "=f"(__out[53]),
        "=f"(__out[54]),
        "=f"(__out[55]),
        "=f"(__out[56]),
        "=f"(__out[57]),
        "=f"(__out[58]),
        "=f"(__out[59]),
        "=f"(__out[60]),
        "=f"(__out[61]),
        "=f"(__out[62]),
        "=f"(__out[63]),
        "=f"(__out[64]),
        "=f"(__out[65]),
        "=f"(__out[66]),
        "=f"(__out[67]),
        "=f"(__out[68]),
        "=f"(__out[69]),
        "=f"(__out[70]),
        "=f"(__out[71]),
        "=f"(__out[72]),
        "=f"(__out[73]),
        "=f"(__out[74]),
        "=f"(__out[75]),
        "=f"(__out[76]),
        "=f"(__out[77]),
        "=f"(__out[78]),
        "=f"(__out[79]),
        "=f"(__out[80]),
        "=f"(__out[81]),
        "=f"(__out[82]),
        "=f"(__out[83]),
        "=f"(__out[84]),
        "=f"(__out[85]),
        "=f"(__out[86]),
        "=f"(__out[87]),
        "=f"(__out[88]),
        "=f"(__out[89]),
        "=f"(__out[90]),
        "=f"(__out[91]),
        "=f"(__out[92]),
        "=f"(__out[93]),
        "=f"(__out[94]),
        "=f"(__out[95]),
        "=f"(__out[96]),
        "=f"(__out[97]),
        "=f"(__out[98]),
        "=f"(__out[99]),
        "=f"(__out[100]),
        "=f"(__out[101]),
        "=f"(__out[102]),
        "=f"(__out[103]),
        "=f"(__out[104]),
        "=f"(__out[105]),
        "=f"(__out[106]),
        "=f"(__out[107]),
        "=f"(__out[108]),
        "=f"(__out[109]),
        "=f"(__out[110]),
        "=f"(__out[111]),
        "=f"(__out[112]),
        "=f"(__out[113]),
        "=f"(__out[114]),
        "=f"(__out[115]),
        "=f"(__out[116]),
        "=f"(__out[117]),
        "=f"(__out[118]),
        "=f"(__out[119]),
        "=f"(__out[120]),
        "=f"(__out[121]),
        "=f"(__out[122]),
        "=f"(__out[123]),
        "=f"(__out[124]),
        "=f"(__out[125]),
        "=f"(__out[126]),
        "=f"(__out[127]),
        "=f"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x128.f32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
      "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, "
      "%102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, "
      "%120, %121, %122, %123, %124, %125, %126, %127}, %128, [%129];"
      : "=f"(__out[0]),
        "=f"(__out[1]),
        "=f"(__out[2]),
        "=f"(__out[3]),
        "=f"(__out[4]),
        "=f"(__out[5]),
        "=f"(__out[6]),
        "=f"(__out[7]),
        "=f"(__out[8]),
        "=f"(__out[9]),
        "=f"(__out[10]),
        "=f"(__out[11]),
        "=f"(__out[12]),
        "=f"(__out[13]),
        "=f"(__out[14]),
        "=f"(__out[15]),
        "=f"(__out[16]),
        "=f"(__out[17]),
        "=f"(__out[18]),
        "=f"(__out[19]),
        "=f"(__out[20]),
        "=f"(__out[21]),
        "=f"(__out[22]),
        "=f"(__out[23]),
        "=f"(__out[24]),
        "=f"(__out[25]),
        "=f"(__out[26]),
        "=f"(__out[27]),
        "=f"(__out[28]),
        "=f"(__out[29]),
        "=f"(__out[30]),
        "=f"(__out[31]),
        "=f"(__out[32]),
        "=f"(__out[33]),
        "=f"(__out[34]),
        "=f"(__out[35]),
        "=f"(__out[36]),
        "=f"(__out[37]),
        "=f"(__out[38]),
        "=f"(__out[39]),
        "=f"(__out[40]),
        "=f"(__out[41]),
        "=f"(__out[42]),
        "=f"(__out[43]),
        "=f"(__out[44]),
        "=f"(__out[45]),
        "=f"(__out[46]),
        "=f"(__out[47]),
        "=f"(__out[48]),
        "=f"(__out[49]),
        "=f"(__out[50]),
        "=f"(__out[51]),
        "=f"(__out[52]),
        "=f"(__out[53]),
        "=f"(__out[54]),
        "=f"(__out[55]),
        "=f"(__out[56]),
        "=f"(__out[57]),
        "=f"(__out[58]),
        "=f"(__out[59]),
        "=f"(__out[60]),
        "=f"(__out[61]),
        "=f"(__out[62]),
        "=f"(__out[63]),
        "=f"(__out[64]),
        "=f"(__out[65]),
        "=f"(__out[66]),
        "=f"(__out[67]),
        "=f"(__out[68]),
        "=f"(__out[69]),
        "=f"(__out[70]),
        "=f"(__out[71]),
        "=f"(__out[72]),
        "=f"(__out[73]),
        "=f"(__out[74]),
        "=f"(__out[75]),
        "=f"(__out[76]),
        "=f"(__out[77]),
        "=f"(__out[78]),
        "=f"(__out[79]),
        "=f"(__out[80]),
        "=f"(__out[81]),
        "=f"(__out[82]),
        "=f"(__out[83]),
        "=f"(__out[84]),
        "=f"(__out[85]),
        "=f"(__out[86]),
        "=f"(__out[87]),
        "=f"(__out[88]),
        "=f"(__out[89]),
        "=f"(__out[90]),
        "=f"(__out[91]),
        "=f"(__out[92]),
        "=f"(__out[93]),
        "=f"(__out[94]),
        "=f"(__out[95]),
        "=f"(__out[96]),
        "=f"(__out[97]),
        "=f"(__out[98]),
        "=f"(__out[99]),
        "=f"(__out[100]),
        "=f"(__out[101]),
        "=f"(__out[102]),
        "=f"(__out[103]),
        "=f"(__out[104]),
        "=f"(__out[105]),
        "=f"(__out[106]),
        "=f"(__out[107]),
        "=f"(__out[108]),
        "=f"(__out[109]),
        "=f"(__out[110]),
        "=f"(__out[111]),
        "=f"(__out[112]),
        "=f"(__out[113]),
        "=f"(__out[114]),
        "=f"(__out[115]),
        "=f"(__out[116]),
        "=f"(__out[117]),
        "=f"(__out[118]),
        "=f"(__out[119]),
        "=f"(__out[120]),
        "=f"(__out[121]),
        "=f"(__out[122]),
        "=f"(__out[123]),
        "=f"(__out[124]),
        "=f"(__out[125]),
        "=f"(__out[126]),
        "=f"(__out[127]),
        "=f"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_32x32b_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x2.u32.op out, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline uint32_t tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  uint32_t (&out)[2],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::uint32_t tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  ::cuda::std::uint32_t (&__out)[2],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  ::cuda::std::uint32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x2.u32.min {%0, %1}, %2, [%3], %4;"
        : "=r"(__out[0]), "=r"(__out[1]), "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x2.u32.max {%0, %1}, %2, [%3], %4;"
        : "=r"(__out[0]), "=r"(__out[1]), "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x2.s32.op out, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline int32_t tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  int32_t (&out)[2],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::int32_t tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  ::cuda::std::int32_t (&__out)[2],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  ::cuda::std::int32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x2.s32.min {%0, %1}, %2, [%3], %4;"
        : "=r"(__out[0]), "=r"(__out[1]), "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x2.s32.max {%0, %1}, %2, [%3], %4;"
        : "=r"(__out[0]), "=r"(__out[1]), "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x2.f32.op.abs out, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_16x32bx2_abs(
  cuda::ptx::op_t<Op> op,
  float (&out)[2],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_16x32bx2_abs_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_16x32bx2_abs(
  ::cuda::ptx::op_t<_Op> __op,
  float (&__out)[2],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x2.f32.min.abs {%0, %1}, %2, [%3], %4;"
        : "=f"(__out[0]), "=f"(__out[1]), "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x2.f32.max.abs {%0, %1}, %2, [%3], %4;"
        : "=f"(__out[0]), "=f"(__out[1]), "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_16x32bx2_abs_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x2.f32.op out, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  float (&out)[2],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  float (&__out)[2],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x2.f32.min {%0, %1}, %2, [%3], %4;"
        : "=f"(__out[0]), "=f"(__out[1]), "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x2.f32.max {%0, %1}, %2, [%3], %4;"
        : "=f"(__out[0]), "=f"(__out[1]), "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x4.u32.op out, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline uint32_t tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  uint32_t (&out)[4],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::uint32_t tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  ::cuda::std::uint32_t (&__out)[4],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  ::cuda::std::uint32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x4.u32.min {%0, %1, %2, %3}, %4, [%5], %6;"
        : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3]), "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x4.u32.max {%0, %1, %2, %3}, %4, [%5], %6;"
        : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3]), "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x4.s32.op out, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline int32_t tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  int32_t (&out)[4],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::int32_t tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  ::cuda::std::int32_t (&__out)[4],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  ::cuda::std::int32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x4.s32.min {%0, %1, %2, %3}, %4, [%5], %6;"
        : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3]), "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x4.s32.max {%0, %1, %2, %3}, %4, [%5], %6;"
        : "=r"(__out[0]), "=r"(__out[1]), "=r"(__out[2]), "=r"(__out[3]), "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.op.abs out, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_16x32bx2_abs(
  cuda::ptx::op_t<Op> op,
  float (&out)[4],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_16x32bx2_abs_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_16x32bx2_abs(
  ::cuda::ptx::op_t<_Op> __op,
  float (&__out)[4],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.min.abs {%0, %1, %2, %3}, %4, [%5], %6;"
        : "=f"(__out[0]), "=f"(__out[1]), "=f"(__out[2]), "=f"(__out[3]), "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.max.abs {%0, %1, %2, %3}, %4, [%5], %6;"
        : "=f"(__out[0]), "=f"(__out[1]), "=f"(__out[2]), "=f"(__out[3]), "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_16x32bx2_abs_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.op out, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  float (&out)[4],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  float (&__out)[4],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.min {%0, %1, %2, %3}, %4, [%5], %6;"
        : "=f"(__out[0]), "=f"(__out[1]), "=f"(__out[2]), "=f"(__out[3]), "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.max {%0, %1, %2, %3}, %4, [%5], %6;"
        : "=f"(__out[0]), "=f"(__out[1]), "=f"(__out[2]), "=f"(__out[3]), "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x8.u32.op out, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline uint32_t tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  uint32_t (&out)[8],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::uint32_t tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  ::cuda::std::uint32_t (&__out)[8],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  ::cuda::std::uint32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x8.u32.min {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9], %10;"
        : "=r"(__out[0]),
          "=r"(__out[1]),
          "=r"(__out[2]),
          "=r"(__out[3]),
          "=r"(__out[4]),
          "=r"(__out[5]),
          "=r"(__out[6]),
          "=r"(__out[7]),
          "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x8.u32.max {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9], %10;"
        : "=r"(__out[0]),
          "=r"(__out[1]),
          "=r"(__out[2]),
          "=r"(__out[3]),
          "=r"(__out[4]),
          "=r"(__out[5]),
          "=r"(__out[6]),
          "=r"(__out[7]),
          "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x8.s32.op out, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline int32_t tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  int32_t (&out)[8],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::int32_t tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  ::cuda::std::int32_t (&__out)[8],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  ::cuda::std::int32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x8.s32.min {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9], %10;"
        : "=r"(__out[0]),
          "=r"(__out[1]),
          "=r"(__out[2]),
          "=r"(__out[3]),
          "=r"(__out[4]),
          "=r"(__out[5]),
          "=r"(__out[6]),
          "=r"(__out[7]),
          "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x8.s32.max {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9], %10;"
        : "=r"(__out[0]),
          "=r"(__out[1]),
          "=r"(__out[2]),
          "=r"(__out[3]),
          "=r"(__out[4]),
          "=r"(__out[5]),
          "=r"(__out[6]),
          "=r"(__out[7]),
          "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x8.f32.op.abs out, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_16x32bx2_abs(
  cuda::ptx::op_t<Op> op,
  float (&out)[8],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_16x32bx2_abs_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_16x32bx2_abs(
  ::cuda::ptx::op_t<_Op> __op,
  float (&__out)[8],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x8.f32.min.abs {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9], %10;"
        : "=f"(__out[0]),
          "=f"(__out[1]),
          "=f"(__out[2]),
          "=f"(__out[3]),
          "=f"(__out[4]),
          "=f"(__out[5]),
          "=f"(__out[6]),
          "=f"(__out[7]),
          "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x8.f32.max.abs {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9], %10;"
        : "=f"(__out[0]),
          "=f"(__out[1]),
          "=f"(__out[2]),
          "=f"(__out[3]),
          "=f"(__out[4]),
          "=f"(__out[5]),
          "=f"(__out[6]),
          "=f"(__out[7]),
          "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_16x32bx2_abs_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x8.f32.op out, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  float (&out)[8],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  float (&__out)[8],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x8.f32.min {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9], %10;"
        : "=f"(__out[0]),
          "=f"(__out[1]),
          "=f"(__out[2]),
          "=f"(__out[3]),
          "=f"(__out[4]),
          "=f"(__out[5]),
          "=f"(__out[6]),
          "=f"(__out[7]),
          "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x8.f32.max {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9], %10;"
        : "=f"(__out[0]),
          "=f"(__out[1]),
          "=f"(__out[2]),
          "=f"(__out[3]),
          "=f"(__out[4]),
          "=f"(__out[5]),
          "=f"(__out[6]),
          "=f"(__out[7]),
          "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x16.u32.op out, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline uint32_t tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  uint32_t (&out)[16],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::uint32_t tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  ::cuda::std::uint32_t (&__out)[16],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  ::cuda::std::uint32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x16.u32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
        "%14, %15}, %16, [%17], %18;"
        : "=r"(__out[0]),
          "=r"(__out[1]),
          "=r"(__out[2]),
          "=r"(__out[3]),
          "=r"(__out[4]),
          "=r"(__out[5]),
          "=r"(__out[6]),
          "=r"(__out[7]),
          "=r"(__out[8]),
          "=r"(__out[9]),
          "=r"(__out[10]),
          "=r"(__out[11]),
          "=r"(__out[12]),
          "=r"(__out[13]),
          "=r"(__out[14]),
          "=r"(__out[15]),
          "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x16.u32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
        "%14, %15}, %16, [%17], %18;"
        : "=r"(__out[0]),
          "=r"(__out[1]),
          "=r"(__out[2]),
          "=r"(__out[3]),
          "=r"(__out[4]),
          "=r"(__out[5]),
          "=r"(__out[6]),
          "=r"(__out[7]),
          "=r"(__out[8]),
          "=r"(__out[9]),
          "=r"(__out[10]),
          "=r"(__out[11]),
          "=r"(__out[12]),
          "=r"(__out[13]),
          "=r"(__out[14]),
          "=r"(__out[15]),
          "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x16.s32.op out, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline int32_t tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  int32_t (&out)[16],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::int32_t tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  ::cuda::std::int32_t (&__out)[16],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  ::cuda::std::int32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x16.s32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
        "%14, %15}, %16, [%17], %18;"
        : "=r"(__out[0]),
          "=r"(__out[1]),
          "=r"(__out[2]),
          "=r"(__out[3]),
          "=r"(__out[4]),
          "=r"(__out[5]),
          "=r"(__out[6]),
          "=r"(__out[7]),
          "=r"(__out[8]),
          "=r"(__out[9]),
          "=r"(__out[10]),
          "=r"(__out[11]),
          "=r"(__out[12]),
          "=r"(__out[13]),
          "=r"(__out[14]),
          "=r"(__out[15]),
          "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x16.s32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
        "%14, %15}, %16, [%17], %18;"
        : "=r"(__out[0]),
          "=r"(__out[1]),
          "=r"(__out[2]),
          "=r"(__out[3]),
          "=r"(__out[4]),
          "=r"(__out[5]),
          "=r"(__out[6]),
          "=r"(__out[7]),
          "=r"(__out[8]),
          "=r"(__out[9]),
          "=r"(__out[10]),
          "=r"(__out[11]),
          "=r"(__out[12]),
          "=r"(__out[13]),
          "=r"(__out[14]),
          "=r"(__out[15]),
          "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x16.f32.op.abs out, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_16x32bx2_abs(
  cuda::ptx::op_t<Op> op,
  float (&out)[16],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_16x32bx2_abs_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_16x32bx2_abs(
  ::cuda::ptx::op_t<_Op> __op,
  float (&__out)[16],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x16.f32.min.abs {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
        "%13, %14, %15}, %16, [%17], %18;"
        : "=f"(__out[0]),
          "=f"(__out[1]),
          "=f"(__out[2]),
          "=f"(__out[3]),
          "=f"(__out[4]),
          "=f"(__out[5]),
          "=f"(__out[6]),
          "=f"(__out[7]),
          "=f"(__out[8]),
          "=f"(__out[9]),
          "=f"(__out[10]),
          "=f"(__out[11]),
          "=f"(__out[12]),
          "=f"(__out[13]),
          "=f"(__out[14]),
          "=f"(__out[15]),
          "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x16.f32.max.abs {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
        "%13, %14, %15}, %16, [%17], %18;"
        : "=f"(__out[0]),
          "=f"(__out[1]),
          "=f"(__out[2]),
          "=f"(__out[3]),
          "=f"(__out[4]),
          "=f"(__out[5]),
          "=f"(__out[6]),
          "=f"(__out[7]),
          "=f"(__out[8]),
          "=f"(__out[9]),
          "=f"(__out[10]),
          "=f"(__out[11]),
          "=f"(__out[12]),
          "=f"(__out[13]),
          "=f"(__out[14]),
          "=f"(__out[15]),
          "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_16x32bx2_abs_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x16.f32.op out, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  float (&out)[16],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  float (&__out)[16],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x16.f32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
        "%14, %15}, %16, [%17], %18;"
        : "=f"(__out[0]),
          "=f"(__out[1]),
          "=f"(__out[2]),
          "=f"(__out[3]),
          "=f"(__out[4]),
          "=f"(__out[5]),
          "=f"(__out[6]),
          "=f"(__out[7]),
          "=f"(__out[8]),
          "=f"(__out[9]),
          "=f"(__out[10]),
          "=f"(__out[11]),
          "=f"(__out[12]),
          "=f"(__out[13]),
          "=f"(__out[14]),
          "=f"(__out[15]),
          "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x16.f32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
        "%14, %15}, %16, [%17], %18;"
        : "=f"(__out[0]),
          "=f"(__out[1]),
          "=f"(__out[2]),
          "=f"(__out[3]),
          "=f"(__out[4]),
          "=f"(__out[5]),
          "=f"(__out[6]),
          "=f"(__out[7]),
          "=f"(__out[8]),
          "=f"(__out[9]),
          "=f"(__out[10]),
          "=f"(__out[11]),
          "=f"(__out[12]),
          "=f"(__out[13]),
          "=f"(__out[14]),
          "=f"(__out[15]),
          "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x32.u32.op out, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline uint32_t tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  uint32_t (&out)[32],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::uint32_t tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  ::cuda::std::uint32_t (&__out)[32],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  ::cuda::std::uint32_t __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x32.u32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33], %34;"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15]),
        "=r"(__out[16]),
        "=r"(__out[17]),
        "=r"(__out[18]),
        "=r"(__out[19]),
        "=r"(__out[20]),
        "=r"(__out[21]),
        "=r"(__out[22]),
        "=r"(__out[23]),
        "=r"(__out[24]),
        "=r"(__out[25]),
        "=r"(__out[26]),
        "=r"(__out[27]),
        "=r"(__out[28]),
        "=r"(__out[29]),
        "=r"(__out[30]),
        "=r"(__out[31]),
        "=r"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x32.u32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33], %34;"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15]),
        "=r"(__out[16]),
        "=r"(__out[17]),
        "=r"(__out[18]),
        "=r"(__out[19]),
        "=r"(__out[20]),
        "=r"(__out[21]),
        "=r"(__out[22]),
        "=r"(__out[23]),
        "=r"(__out[24]),
        "=r"(__out[25]),
        "=r"(__out[26]),
        "=r"(__out[27]),
        "=r"(__out[28]),
        "=r"(__out[29]),
        "=r"(__out[30]),
        "=r"(__out[31]),
        "=r"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x32.s32.op out, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline int32_t tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  int32_t (&out)[32],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::int32_t tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  ::cuda::std::int32_t (&__out)[32],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  ::cuda::std::int32_t __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x32.s32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33], %34;"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15]),
        "=r"(__out[16]),
        "=r"(__out[17]),
        "=r"(__out[18]),
        "=r"(__out[19]),
        "=r"(__out[20]),
        "=r"(__out[21]),
        "=r"(__out[22]),
        "=r"(__out[23]),
        "=r"(__out[24]),
        "=r"(__out[25]),
        "=r"(__out[26]),
        "=r"(__out[27]),
        "=r"(__out[28]),
        "=r"(__out[29]),
        "=r"(__out[30]),
        "=r"(__out[31]),
        "=r"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x32.s32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33], %34;"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15]),
        "=r"(__out[16]),
        "=r"(__out[17]),
        "=r"(__out[18]),
        "=r"(__out[19]),
        "=r"(__out[20]),
        "=r"(__out[21]),
        "=r"(__out[22]),
        "=r"(__out[23]),
        "=r"(__out[24]),
        "=r"(__out[25]),
        "=r"(__out[26]),
        "=r"(__out[27]),
        "=r"(__out[28]),
        "=r"(__out[29]),
        "=r"(__out[30]),
        "=r"(__out[31]),
        "=r"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.op.abs out, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_16x32bx2_abs(
  cuda::ptx::op_t<Op> op,
  float (&out)[32],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_16x32bx2_abs_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_16x32bx2_abs(
  ::cuda::ptx::op_t<_Op> __op,
  float (&__out)[32],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  float __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.min.abs {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
      "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33], %34;"
      : "=f"(__out[0]),
        "=f"(__out[1]),
        "=f"(__out[2]),
        "=f"(__out[3]),
        "=f"(__out[4]),
        "=f"(__out[5]),
        "=f"(__out[6]),
        "=f"(__out[7]),
        "=f"(__out[8]),
        "=f"(__out[9]),
        "=f"(__out[10]),
        "=f"(__out[11]),
        "=f"(__out[12]),
        "=f"(__out[13]),
        "=f"(__out[14]),
        "=f"(__out[15]),
        "=f"(__out[16]),
        "=f"(__out[17]),
        "=f"(__out[18]),
        "=f"(__out[19]),
        "=f"(__out[20]),
        "=f"(__out[21]),
        "=f"(__out[22]),
        "=f"(__out[23]),
        "=f"(__out[24]),
        "=f"(__out[25]),
        "=f"(__out[26]),
        "=f"(__out[27]),
        "=f"(__out[28]),
        "=f"(__out[29]),
        "=f"(__out[30]),
        "=f"(__out[31]),
        "=f"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.max.abs {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
      "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33], %34;"
      : "=f"(__out[0]),
        "=f"(__out[1]),
        "=f"(__out[2]),
        "=f"(__out[3]),
        "=f"(__out[4]),
        "=f"(__out[5]),
        "=f"(__out[6]),
        "=f"(__out[7]),
        "=f"(__out[8]),
        "=f"(__out[9]),
        "=f"(__out[10]),
        "=f"(__out[11]),
        "=f"(__out[12]),
        "=f"(__out[13]),
        "=f"(__out[14]),
        "=f"(__out[15]),
        "=f"(__out[16]),
        "=f"(__out[17]),
        "=f"(__out[18]),
        "=f"(__out[19]),
        "=f"(__out[20]),
        "=f"(__out[21]),
        "=f"(__out[22]),
        "=f"(__out[23]),
        "=f"(__out[24]),
        "=f"(__out[25]),
        "=f"(__out[26]),
        "=f"(__out[27]),
        "=f"(__out[28]),
        "=f"(__out[29]),
        "=f"(__out[30]),
        "=f"(__out[31]),
        "=f"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_16x32bx2_abs_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.op out, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  float (&out)[32],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  float (&__out)[32],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  float __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33], %34;"
      : "=f"(__out[0]),
        "=f"(__out[1]),
        "=f"(__out[2]),
        "=f"(__out[3]),
        "=f"(__out[4]),
        "=f"(__out[5]),
        "=f"(__out[6]),
        "=f"(__out[7]),
        "=f"(__out[8]),
        "=f"(__out[9]),
        "=f"(__out[10]),
        "=f"(__out[11]),
        "=f"(__out[12]),
        "=f"(__out[13]),
        "=f"(__out[14]),
        "=f"(__out[15]),
        "=f"(__out[16]),
        "=f"(__out[17]),
        "=f"(__out[18]),
        "=f"(__out[19]),
        "=f"(__out[20]),
        "=f"(__out[21]),
        "=f"(__out[22]),
        "=f"(__out[23]),
        "=f"(__out[24]),
        "=f"(__out[25]),
        "=f"(__out[26]),
        "=f"(__out[27]),
        "=f"(__out[28]),
        "=f"(__out[29]),
        "=f"(__out[30]),
        "=f"(__out[31]),
        "=f"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33], %34;"
      : "=f"(__out[0]),
        "=f"(__out[1]),
        "=f"(__out[2]),
        "=f"(__out[3]),
        "=f"(__out[4]),
        "=f"(__out[5]),
        "=f"(__out[6]),
        "=f"(__out[7]),
        "=f"(__out[8]),
        "=f"(__out[9]),
        "=f"(__out[10]),
        "=f"(__out[11]),
        "=f"(__out[12]),
        "=f"(__out[13]),
        "=f"(__out[14]),
        "=f"(__out[15]),
        "=f"(__out[16]),
        "=f"(__out[17]),
        "=f"(__out[18]),
        "=f"(__out[19]),
        "=f"(__out[20]),
        "=f"(__out[21]),
        "=f"(__out[22]),
        "=f"(__out[23]),
        "=f"(__out[24]),
        "=f"(__out[25]),
        "=f"(__out[26]),
        "=f"(__out[27]),
        "=f"(__out[28]),
        "=f"(__out[29]),
        "=f"(__out[30]),
        "=f"(__out[31]),
        "=f"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x64.u32.op out, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline uint32_t tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  uint32_t (&out)[64],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::uint32_t tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  ::cuda::std::uint32_t (&__out)[64],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  ::cuda::std::uint32_t __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x64.u32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63}, %64, [%65], %66;"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15]),
        "=r"(__out[16]),
        "=r"(__out[17]),
        "=r"(__out[18]),
        "=r"(__out[19]),
        "=r"(__out[20]),
        "=r"(__out[21]),
        "=r"(__out[22]),
        "=r"(__out[23]),
        "=r"(__out[24]),
        "=r"(__out[25]),
        "=r"(__out[26]),
        "=r"(__out[27]),
        "=r"(__out[28]),
        "=r"(__out[29]),
        "=r"(__out[30]),
        "=r"(__out[31]),
        "=r"(__out[32]),
        "=r"(__out[33]),
        "=r"(__out[34]),
        "=r"(__out[35]),
        "=r"(__out[36]),
        "=r"(__out[37]),
        "=r"(__out[38]),
        "=r"(__out[39]),
        "=r"(__out[40]),
        "=r"(__out[41]),
        "=r"(__out[42]),
        "=r"(__out[43]),
        "=r"(__out[44]),
        "=r"(__out[45]),
        "=r"(__out[46]),
        "=r"(__out[47]),
        "=r"(__out[48]),
        "=r"(__out[49]),
        "=r"(__out[50]),
        "=r"(__out[51]),
        "=r"(__out[52]),
        "=r"(__out[53]),
        "=r"(__out[54]),
        "=r"(__out[55]),
        "=r"(__out[56]),
        "=r"(__out[57]),
        "=r"(__out[58]),
        "=r"(__out[59]),
        "=r"(__out[60]),
        "=r"(__out[61]),
        "=r"(__out[62]),
        "=r"(__out[63]),
        "=r"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x64.u32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63}, %64, [%65], %66;"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15]),
        "=r"(__out[16]),
        "=r"(__out[17]),
        "=r"(__out[18]),
        "=r"(__out[19]),
        "=r"(__out[20]),
        "=r"(__out[21]),
        "=r"(__out[22]),
        "=r"(__out[23]),
        "=r"(__out[24]),
        "=r"(__out[25]),
        "=r"(__out[26]),
        "=r"(__out[27]),
        "=r"(__out[28]),
        "=r"(__out[29]),
        "=r"(__out[30]),
        "=r"(__out[31]),
        "=r"(__out[32]),
        "=r"(__out[33]),
        "=r"(__out[34]),
        "=r"(__out[35]),
        "=r"(__out[36]),
        "=r"(__out[37]),
        "=r"(__out[38]),
        "=r"(__out[39]),
        "=r"(__out[40]),
        "=r"(__out[41]),
        "=r"(__out[42]),
        "=r"(__out[43]),
        "=r"(__out[44]),
        "=r"(__out[45]),
        "=r"(__out[46]),
        "=r"(__out[47]),
        "=r"(__out[48]),
        "=r"(__out[49]),
        "=r"(__out[50]),
        "=r"(__out[51]),
        "=r"(__out[52]),
        "=r"(__out[53]),
        "=r"(__out[54]),
        "=r"(__out[55]),
        "=r"(__out[56]),
        "=r"(__out[57]),
        "=r"(__out[58]),
        "=r"(__out[59]),
        "=r"(__out[60]),
        "=r"(__out[61]),
        "=r"(__out[62]),
        "=r"(__out[63]),
        "=r"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x64.s32.op out, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline int32_t tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  int32_t (&out)[64],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::int32_t tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  ::cuda::std::int32_t (&__out)[64],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  ::cuda::std::int32_t __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x64.s32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63}, %64, [%65], %66;"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15]),
        "=r"(__out[16]),
        "=r"(__out[17]),
        "=r"(__out[18]),
        "=r"(__out[19]),
        "=r"(__out[20]),
        "=r"(__out[21]),
        "=r"(__out[22]),
        "=r"(__out[23]),
        "=r"(__out[24]),
        "=r"(__out[25]),
        "=r"(__out[26]),
        "=r"(__out[27]),
        "=r"(__out[28]),
        "=r"(__out[29]),
        "=r"(__out[30]),
        "=r"(__out[31]),
        "=r"(__out[32]),
        "=r"(__out[33]),
        "=r"(__out[34]),
        "=r"(__out[35]),
        "=r"(__out[36]),
        "=r"(__out[37]),
        "=r"(__out[38]),
        "=r"(__out[39]),
        "=r"(__out[40]),
        "=r"(__out[41]),
        "=r"(__out[42]),
        "=r"(__out[43]),
        "=r"(__out[44]),
        "=r"(__out[45]),
        "=r"(__out[46]),
        "=r"(__out[47]),
        "=r"(__out[48]),
        "=r"(__out[49]),
        "=r"(__out[50]),
        "=r"(__out[51]),
        "=r"(__out[52]),
        "=r"(__out[53]),
        "=r"(__out[54]),
        "=r"(__out[55]),
        "=r"(__out[56]),
        "=r"(__out[57]),
        "=r"(__out[58]),
        "=r"(__out[59]),
        "=r"(__out[60]),
        "=r"(__out[61]),
        "=r"(__out[62]),
        "=r"(__out[63]),
        "=r"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x64.s32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63}, %64, [%65], %66;"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15]),
        "=r"(__out[16]),
        "=r"(__out[17]),
        "=r"(__out[18]),
        "=r"(__out[19]),
        "=r"(__out[20]),
        "=r"(__out[21]),
        "=r"(__out[22]),
        "=r"(__out[23]),
        "=r"(__out[24]),
        "=r"(__out[25]),
        "=r"(__out[26]),
        "=r"(__out[27]),
        "=r"(__out[28]),
        "=r"(__out[29]),
        "=r"(__out[30]),
        "=r"(__out[31]),
        "=r"(__out[32]),
        "=r"(__out[33]),
        "=r"(__out[34]),
        "=r"(__out[35]),
        "=r"(__out[36]),
        "=r"(__out[37]),
        "=r"(__out[38]),
        "=r"(__out[39]),
        "=r"(__out[40]),
        "=r"(__out[41]),
        "=r"(__out[42]),
        "=r"(__out[43]),
        "=r"(__out[44]),
        "=r"(__out[45]),
        "=r"(__out[46]),
        "=r"(__out[47]),
        "=r"(__out[48]),
        "=r"(__out[49]),
        "=r"(__out[50]),
        "=r"(__out[51]),
        "=r"(__out[52]),
        "=r"(__out[53]),
        "=r"(__out[54]),
        "=r"(__out[55]),
        "=r"(__out[56]),
        "=r"(__out[57]),
        "=r"(__out[58]),
        "=r"(__out[59]),
        "=r"(__out[60]),
        "=r"(__out[61]),
        "=r"(__out[62]),
        "=r"(__out[63]),
        "=r"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x64.f32.op.abs out, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_16x32bx2_abs(
  cuda::ptx::op_t<Op> op,
  float (&out)[64],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_16x32bx2_abs_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_16x32bx2_abs(
  ::cuda::ptx::op_t<_Op> __op,
  float (&__out)[64],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  float __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x64.f32.min.abs {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
      "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, "
      "%35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
      "%57, %58, %59, %60, %61, %62, %63}, %64, [%65], %66;"
      : "=f"(__out[0]),
        "=f"(__out[1]),
        "=f"(__out[2]),
        "=f"(__out[3]),
        "=f"(__out[4]),
        "=f"(__out[5]),
        "=f"(__out[6]),
        "=f"(__out[7]),
        "=f"(__out[8]),
        "=f"(__out[9]),
        "=f"(__out[10]),
        "=f"(__out[11]),
        "=f"(__out[12]),
        "=f"(__out[13]),
        "=f"(__out[14]),
        "=f"(__out[15]),
        "=f"(__out[16]),
        "=f"(__out[17]),
        "=f"(__out[18]),
        "=f"(__out[19]),
        "=f"(__out[20]),
        "=f"(__out[21]),
        "=f"(__out[22]),
        "=f"(__out[23]),
        "=f"(__out[24]),
        "=f"(__out[25]),
        "=f"(__out[26]),
        "=f"(__out[27]),
        "=f"(__out[28]),
        "=f"(__out[29]),
        "=f"(__out[30]),
        "=f"(__out[31]),
        "=f"(__out[32]),
        "=f"(__out[33]),
        "=f"(__out[34]),
        "=f"(__out[35]),
        "=f"(__out[36]),
        "=f"(__out[37]),
        "=f"(__out[38]),
        "=f"(__out[39]),
        "=f"(__out[40]),
        "=f"(__out[41]),
        "=f"(__out[42]),
        "=f"(__out[43]),
        "=f"(__out[44]),
        "=f"(__out[45]),
        "=f"(__out[46]),
        "=f"(__out[47]),
        "=f"(__out[48]),
        "=f"(__out[49]),
        "=f"(__out[50]),
        "=f"(__out[51]),
        "=f"(__out[52]),
        "=f"(__out[53]),
        "=f"(__out[54]),
        "=f"(__out[55]),
        "=f"(__out[56]),
        "=f"(__out[57]),
        "=f"(__out[58]),
        "=f"(__out[59]),
        "=f"(__out[60]),
        "=f"(__out[61]),
        "=f"(__out[62]),
        "=f"(__out[63]),
        "=f"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x64.f32.max.abs {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
      "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, "
      "%35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
      "%57, %58, %59, %60, %61, %62, %63}, %64, [%65], %66;"
      : "=f"(__out[0]),
        "=f"(__out[1]),
        "=f"(__out[2]),
        "=f"(__out[3]),
        "=f"(__out[4]),
        "=f"(__out[5]),
        "=f"(__out[6]),
        "=f"(__out[7]),
        "=f"(__out[8]),
        "=f"(__out[9]),
        "=f"(__out[10]),
        "=f"(__out[11]),
        "=f"(__out[12]),
        "=f"(__out[13]),
        "=f"(__out[14]),
        "=f"(__out[15]),
        "=f"(__out[16]),
        "=f"(__out[17]),
        "=f"(__out[18]),
        "=f"(__out[19]),
        "=f"(__out[20]),
        "=f"(__out[21]),
        "=f"(__out[22]),
        "=f"(__out[23]),
        "=f"(__out[24]),
        "=f"(__out[25]),
        "=f"(__out[26]),
        "=f"(__out[27]),
        "=f"(__out[28]),
        "=f"(__out[29]),
        "=f"(__out[30]),
        "=f"(__out[31]),
        "=f"(__out[32]),
        "=f"(__out[33]),
        "=f"(__out[34]),
        "=f"(__out[35]),
        "=f"(__out[36]),
        "=f"(__out[37]),
        "=f"(__out[38]),
        "=f"(__out[39]),
        "=f"(__out[40]),
        "=f"(__out[41]),
        "=f"(__out[42]),
        "=f"(__out[43]),
        "=f"(__out[44]),
        "=f"(__out[45]),
        "=f"(__out[46]),
        "=f"(__out[47]),
        "=f"(__out[48]),
        "=f"(__out[49]),
        "=f"(__out[50]),
        "=f"(__out[51]),
        "=f"(__out[52]),
        "=f"(__out[53]),
        "=f"(__out[54]),
        "=f"(__out[55]),
        "=f"(__out[56]),
        "=f"(__out[57]),
        "=f"(__out[58]),
        "=f"(__out[59]),
        "=f"(__out[60]),
        "=f"(__out[61]),
        "=f"(__out[62]),
        "=f"(__out[63]),
        "=f"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_16x32bx2_abs_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x64.f32.op out, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  float (&out)[64],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  float (&__out)[64],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  float __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x64.f32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63}, %64, [%65], %66;"
      : "=f"(__out[0]),
        "=f"(__out[1]),
        "=f"(__out[2]),
        "=f"(__out[3]),
        "=f"(__out[4]),
        "=f"(__out[5]),
        "=f"(__out[6]),
        "=f"(__out[7]),
        "=f"(__out[8]),
        "=f"(__out[9]),
        "=f"(__out[10]),
        "=f"(__out[11]),
        "=f"(__out[12]),
        "=f"(__out[13]),
        "=f"(__out[14]),
        "=f"(__out[15]),
        "=f"(__out[16]),
        "=f"(__out[17]),
        "=f"(__out[18]),
        "=f"(__out[19]),
        "=f"(__out[20]),
        "=f"(__out[21]),
        "=f"(__out[22]),
        "=f"(__out[23]),
        "=f"(__out[24]),
        "=f"(__out[25]),
        "=f"(__out[26]),
        "=f"(__out[27]),
        "=f"(__out[28]),
        "=f"(__out[29]),
        "=f"(__out[30]),
        "=f"(__out[31]),
        "=f"(__out[32]),
        "=f"(__out[33]),
        "=f"(__out[34]),
        "=f"(__out[35]),
        "=f"(__out[36]),
        "=f"(__out[37]),
        "=f"(__out[38]),
        "=f"(__out[39]),
        "=f"(__out[40]),
        "=f"(__out[41]),
        "=f"(__out[42]),
        "=f"(__out[43]),
        "=f"(__out[44]),
        "=f"(__out[45]),
        "=f"(__out[46]),
        "=f"(__out[47]),
        "=f"(__out[48]),
        "=f"(__out[49]),
        "=f"(__out[50]),
        "=f"(__out[51]),
        "=f"(__out[52]),
        "=f"(__out[53]),
        "=f"(__out[54]),
        "=f"(__out[55]),
        "=f"(__out[56]),
        "=f"(__out[57]),
        "=f"(__out[58]),
        "=f"(__out[59]),
        "=f"(__out[60]),
        "=f"(__out[61]),
        "=f"(__out[62]),
        "=f"(__out[63]),
        "=f"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x64.f32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63}, %64, [%65], %66;"
      : "=f"(__out[0]),
        "=f"(__out[1]),
        "=f"(__out[2]),
        "=f"(__out[3]),
        "=f"(__out[4]),
        "=f"(__out[5]),
        "=f"(__out[6]),
        "=f"(__out[7]),
        "=f"(__out[8]),
        "=f"(__out[9]),
        "=f"(__out[10]),
        "=f"(__out[11]),
        "=f"(__out[12]),
        "=f"(__out[13]),
        "=f"(__out[14]),
        "=f"(__out[15]),
        "=f"(__out[16]),
        "=f"(__out[17]),
        "=f"(__out[18]),
        "=f"(__out[19]),
        "=f"(__out[20]),
        "=f"(__out[21]),
        "=f"(__out[22]),
        "=f"(__out[23]),
        "=f"(__out[24]),
        "=f"(__out[25]),
        "=f"(__out[26]),
        "=f"(__out[27]),
        "=f"(__out[28]),
        "=f"(__out[29]),
        "=f"(__out[30]),
        "=f"(__out[31]),
        "=f"(__out[32]),
        "=f"(__out[33]),
        "=f"(__out[34]),
        "=f"(__out[35]),
        "=f"(__out[36]),
        "=f"(__out[37]),
        "=f"(__out[38]),
        "=f"(__out[39]),
        "=f"(__out[40]),
        "=f"(__out[41]),
        "=f"(__out[42]),
        "=f"(__out[43]),
        "=f"(__out[44]),
        "=f"(__out[45]),
        "=f"(__out[46]),
        "=f"(__out[47]),
        "=f"(__out[48]),
        "=f"(__out[49]),
        "=f"(__out[50]),
        "=f"(__out[51]),
        "=f"(__out[52]),
        "=f"(__out[53]),
        "=f"(__out[54]),
        "=f"(__out[55]),
        "=f"(__out[56]),
        "=f"(__out[57]),
        "=f"(__out[58]),
        "=f"(__out[59]),
        "=f"(__out[60]),
        "=f"(__out[61]),
        "=f"(__out[62]),
        "=f"(__out[63]),
        "=f"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x128.u32.op out, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline uint32_t tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  uint32_t (&out)[128],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::uint32_t tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  ::cuda::std::uint32_t (&__out)[128],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  ::cuda::std::uint32_t __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x128.u32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
      "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, "
      "%102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, "
      "%120, %121, %122, %123, %124, %125, %126, %127}, %128, [%129], %130;"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15]),
        "=r"(__out[16]),
        "=r"(__out[17]),
        "=r"(__out[18]),
        "=r"(__out[19]),
        "=r"(__out[20]),
        "=r"(__out[21]),
        "=r"(__out[22]),
        "=r"(__out[23]),
        "=r"(__out[24]),
        "=r"(__out[25]),
        "=r"(__out[26]),
        "=r"(__out[27]),
        "=r"(__out[28]),
        "=r"(__out[29]),
        "=r"(__out[30]),
        "=r"(__out[31]),
        "=r"(__out[32]),
        "=r"(__out[33]),
        "=r"(__out[34]),
        "=r"(__out[35]),
        "=r"(__out[36]),
        "=r"(__out[37]),
        "=r"(__out[38]),
        "=r"(__out[39]),
        "=r"(__out[40]),
        "=r"(__out[41]),
        "=r"(__out[42]),
        "=r"(__out[43]),
        "=r"(__out[44]),
        "=r"(__out[45]),
        "=r"(__out[46]),
        "=r"(__out[47]),
        "=r"(__out[48]),
        "=r"(__out[49]),
        "=r"(__out[50]),
        "=r"(__out[51]),
        "=r"(__out[52]),
        "=r"(__out[53]),
        "=r"(__out[54]),
        "=r"(__out[55]),
        "=r"(__out[56]),
        "=r"(__out[57]),
        "=r"(__out[58]),
        "=r"(__out[59]),
        "=r"(__out[60]),
        "=r"(__out[61]),
        "=r"(__out[62]),
        "=r"(__out[63]),
        "=r"(__out[64]),
        "=r"(__out[65]),
        "=r"(__out[66]),
        "=r"(__out[67]),
        "=r"(__out[68]),
        "=r"(__out[69]),
        "=r"(__out[70]),
        "=r"(__out[71]),
        "=r"(__out[72]),
        "=r"(__out[73]),
        "=r"(__out[74]),
        "=r"(__out[75]),
        "=r"(__out[76]),
        "=r"(__out[77]),
        "=r"(__out[78]),
        "=r"(__out[79]),
        "=r"(__out[80]),
        "=r"(__out[81]),
        "=r"(__out[82]),
        "=r"(__out[83]),
        "=r"(__out[84]),
        "=r"(__out[85]),
        "=r"(__out[86]),
        "=r"(__out[87]),
        "=r"(__out[88]),
        "=r"(__out[89]),
        "=r"(__out[90]),
        "=r"(__out[91]),
        "=r"(__out[92]),
        "=r"(__out[93]),
        "=r"(__out[94]),
        "=r"(__out[95]),
        "=r"(__out[96]),
        "=r"(__out[97]),
        "=r"(__out[98]),
        "=r"(__out[99]),
        "=r"(__out[100]),
        "=r"(__out[101]),
        "=r"(__out[102]),
        "=r"(__out[103]),
        "=r"(__out[104]),
        "=r"(__out[105]),
        "=r"(__out[106]),
        "=r"(__out[107]),
        "=r"(__out[108]),
        "=r"(__out[109]),
        "=r"(__out[110]),
        "=r"(__out[111]),
        "=r"(__out[112]),
        "=r"(__out[113]),
        "=r"(__out[114]),
        "=r"(__out[115]),
        "=r"(__out[116]),
        "=r"(__out[117]),
        "=r"(__out[118]),
        "=r"(__out[119]),
        "=r"(__out[120]),
        "=r"(__out[121]),
        "=r"(__out[122]),
        "=r"(__out[123]),
        "=r"(__out[124]),
        "=r"(__out[125]),
        "=r"(__out[126]),
        "=r"(__out[127]),
        "=r"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x128.u32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
      "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, "
      "%102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, "
      "%120, %121, %122, %123, %124, %125, %126, %127}, %128, [%129], %130;"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15]),
        "=r"(__out[16]),
        "=r"(__out[17]),
        "=r"(__out[18]),
        "=r"(__out[19]),
        "=r"(__out[20]),
        "=r"(__out[21]),
        "=r"(__out[22]),
        "=r"(__out[23]),
        "=r"(__out[24]),
        "=r"(__out[25]),
        "=r"(__out[26]),
        "=r"(__out[27]),
        "=r"(__out[28]),
        "=r"(__out[29]),
        "=r"(__out[30]),
        "=r"(__out[31]),
        "=r"(__out[32]),
        "=r"(__out[33]),
        "=r"(__out[34]),
        "=r"(__out[35]),
        "=r"(__out[36]),
        "=r"(__out[37]),
        "=r"(__out[38]),
        "=r"(__out[39]),
        "=r"(__out[40]),
        "=r"(__out[41]),
        "=r"(__out[42]),
        "=r"(__out[43]),
        "=r"(__out[44]),
        "=r"(__out[45]),
        "=r"(__out[46]),
        "=r"(__out[47]),
        "=r"(__out[48]),
        "=r"(__out[49]),
        "=r"(__out[50]),
        "=r"(__out[51]),
        "=r"(__out[52]),
        "=r"(__out[53]),
        "=r"(__out[54]),
        "=r"(__out[55]),
        "=r"(__out[56]),
        "=r"(__out[57]),
        "=r"(__out[58]),
        "=r"(__out[59]),
        "=r"(__out[60]),
        "=r"(__out[61]),
        "=r"(__out[62]),
        "=r"(__out[63]),
        "=r"(__out[64]),
        "=r"(__out[65]),
        "=r"(__out[66]),
        "=r"(__out[67]),
        "=r"(__out[68]),
        "=r"(__out[69]),
        "=r"(__out[70]),
        "=r"(__out[71]),
        "=r"(__out[72]),
        "=r"(__out[73]),
        "=r"(__out[74]),
        "=r"(__out[75]),
        "=r"(__out[76]),
        "=r"(__out[77]),
        "=r"(__out[78]),
        "=r"(__out[79]),
        "=r"(__out[80]),
        "=r"(__out[81]),
        "=r"(__out[82]),
        "=r"(__out[83]),
        "=r"(__out[84]),
        "=r"(__out[85]),
        "=r"(__out[86]),
        "=r"(__out[87]),
        "=r"(__out[88]),
        "=r"(__out[89]),
        "=r"(__out[90]),
        "=r"(__out[91]),
        "=r"(__out[92]),
        "=r"(__out[93]),
        "=r"(__out[94]),
        "=r"(__out[95]),
        "=r"(__out[96]),
        "=r"(__out[97]),
        "=r"(__out[98]),
        "=r"(__out[99]),
        "=r"(__out[100]),
        "=r"(__out[101]),
        "=r"(__out[102]),
        "=r"(__out[103]),
        "=r"(__out[104]),
        "=r"(__out[105]),
        "=r"(__out[106]),
        "=r"(__out[107]),
        "=r"(__out[108]),
        "=r"(__out[109]),
        "=r"(__out[110]),
        "=r"(__out[111]),
        "=r"(__out[112]),
        "=r"(__out[113]),
        "=r"(__out[114]),
        "=r"(__out[115]),
        "=r"(__out[116]),
        "=r"(__out[117]),
        "=r"(__out[118]),
        "=r"(__out[119]),
        "=r"(__out[120]),
        "=r"(__out[121]),
        "=r"(__out[122]),
        "=r"(__out[123]),
        "=r"(__out[124]),
        "=r"(__out[125]),
        "=r"(__out[126]),
        "=r"(__out[127]),
        "=r"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x128.s32.op out, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline int32_t tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  int32_t (&out)[128],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::int32_t tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  ::cuda::std::int32_t (&__out)[128],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  ::cuda::std::int32_t __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x128.s32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
      "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, "
      "%102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, "
      "%120, %121, %122, %123, %124, %125, %126, %127}, %128, [%129], %130;"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15]),
        "=r"(__out[16]),
        "=r"(__out[17]),
        "=r"(__out[18]),
        "=r"(__out[19]),
        "=r"(__out[20]),
        "=r"(__out[21]),
        "=r"(__out[22]),
        "=r"(__out[23]),
        "=r"(__out[24]),
        "=r"(__out[25]),
        "=r"(__out[26]),
        "=r"(__out[27]),
        "=r"(__out[28]),
        "=r"(__out[29]),
        "=r"(__out[30]),
        "=r"(__out[31]),
        "=r"(__out[32]),
        "=r"(__out[33]),
        "=r"(__out[34]),
        "=r"(__out[35]),
        "=r"(__out[36]),
        "=r"(__out[37]),
        "=r"(__out[38]),
        "=r"(__out[39]),
        "=r"(__out[40]),
        "=r"(__out[41]),
        "=r"(__out[42]),
        "=r"(__out[43]),
        "=r"(__out[44]),
        "=r"(__out[45]),
        "=r"(__out[46]),
        "=r"(__out[47]),
        "=r"(__out[48]),
        "=r"(__out[49]),
        "=r"(__out[50]),
        "=r"(__out[51]),
        "=r"(__out[52]),
        "=r"(__out[53]),
        "=r"(__out[54]),
        "=r"(__out[55]),
        "=r"(__out[56]),
        "=r"(__out[57]),
        "=r"(__out[58]),
        "=r"(__out[59]),
        "=r"(__out[60]),
        "=r"(__out[61]),
        "=r"(__out[62]),
        "=r"(__out[63]),
        "=r"(__out[64]),
        "=r"(__out[65]),
        "=r"(__out[66]),
        "=r"(__out[67]),
        "=r"(__out[68]),
        "=r"(__out[69]),
        "=r"(__out[70]),
        "=r"(__out[71]),
        "=r"(__out[72]),
        "=r"(__out[73]),
        "=r"(__out[74]),
        "=r"(__out[75]),
        "=r"(__out[76]),
        "=r"(__out[77]),
        "=r"(__out[78]),
        "=r"(__out[79]),
        "=r"(__out[80]),
        "=r"(__out[81]),
        "=r"(__out[82]),
        "=r"(__out[83]),
        "=r"(__out[84]),
        "=r"(__out[85]),
        "=r"(__out[86]),
        "=r"(__out[87]),
        "=r"(__out[88]),
        "=r"(__out[89]),
        "=r"(__out[90]),
        "=r"(__out[91]),
        "=r"(__out[92]),
        "=r"(__out[93]),
        "=r"(__out[94]),
        "=r"(__out[95]),
        "=r"(__out[96]),
        "=r"(__out[97]),
        "=r"(__out[98]),
        "=r"(__out[99]),
        "=r"(__out[100]),
        "=r"(__out[101]),
        "=r"(__out[102]),
        "=r"(__out[103]),
        "=r"(__out[104]),
        "=r"(__out[105]),
        "=r"(__out[106]),
        "=r"(__out[107]),
        "=r"(__out[108]),
        "=r"(__out[109]),
        "=r"(__out[110]),
        "=r"(__out[111]),
        "=r"(__out[112]),
        "=r"(__out[113]),
        "=r"(__out[114]),
        "=r"(__out[115]),
        "=r"(__out[116]),
        "=r"(__out[117]),
        "=r"(__out[118]),
        "=r"(__out[119]),
        "=r"(__out[120]),
        "=r"(__out[121]),
        "=r"(__out[122]),
        "=r"(__out[123]),
        "=r"(__out[124]),
        "=r"(__out[125]),
        "=r"(__out[126]),
        "=r"(__out[127]),
        "=r"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x128.s32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
      "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, "
      "%102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, "
      "%120, %121, %122, %123, %124, %125, %126, %127}, %128, [%129], %130;"
      : "=r"(__out[0]),
        "=r"(__out[1]),
        "=r"(__out[2]),
        "=r"(__out[3]),
        "=r"(__out[4]),
        "=r"(__out[5]),
        "=r"(__out[6]),
        "=r"(__out[7]),
        "=r"(__out[8]),
        "=r"(__out[9]),
        "=r"(__out[10]),
        "=r"(__out[11]),
        "=r"(__out[12]),
        "=r"(__out[13]),
        "=r"(__out[14]),
        "=r"(__out[15]),
        "=r"(__out[16]),
        "=r"(__out[17]),
        "=r"(__out[18]),
        "=r"(__out[19]),
        "=r"(__out[20]),
        "=r"(__out[21]),
        "=r"(__out[22]),
        "=r"(__out[23]),
        "=r"(__out[24]),
        "=r"(__out[25]),
        "=r"(__out[26]),
        "=r"(__out[27]),
        "=r"(__out[28]),
        "=r"(__out[29]),
        "=r"(__out[30]),
        "=r"(__out[31]),
        "=r"(__out[32]),
        "=r"(__out[33]),
        "=r"(__out[34]),
        "=r"(__out[35]),
        "=r"(__out[36]),
        "=r"(__out[37]),
        "=r"(__out[38]),
        "=r"(__out[39]),
        "=r"(__out[40]),
        "=r"(__out[41]),
        "=r"(__out[42]),
        "=r"(__out[43]),
        "=r"(__out[44]),
        "=r"(__out[45]),
        "=r"(__out[46]),
        "=r"(__out[47]),
        "=r"(__out[48]),
        "=r"(__out[49]),
        "=r"(__out[50]),
        "=r"(__out[51]),
        "=r"(__out[52]),
        "=r"(__out[53]),
        "=r"(__out[54]),
        "=r"(__out[55]),
        "=r"(__out[56]),
        "=r"(__out[57]),
        "=r"(__out[58]),
        "=r"(__out[59]),
        "=r"(__out[60]),
        "=r"(__out[61]),
        "=r"(__out[62]),
        "=r"(__out[63]),
        "=r"(__out[64]),
        "=r"(__out[65]),
        "=r"(__out[66]),
        "=r"(__out[67]),
        "=r"(__out[68]),
        "=r"(__out[69]),
        "=r"(__out[70]),
        "=r"(__out[71]),
        "=r"(__out[72]),
        "=r"(__out[73]),
        "=r"(__out[74]),
        "=r"(__out[75]),
        "=r"(__out[76]),
        "=r"(__out[77]),
        "=r"(__out[78]),
        "=r"(__out[79]),
        "=r"(__out[80]),
        "=r"(__out[81]),
        "=r"(__out[82]),
        "=r"(__out[83]),
        "=r"(__out[84]),
        "=r"(__out[85]),
        "=r"(__out[86]),
        "=r"(__out[87]),
        "=r"(__out[88]),
        "=r"(__out[89]),
        "=r"(__out[90]),
        "=r"(__out[91]),
        "=r"(__out[92]),
        "=r"(__out[93]),
        "=r"(__out[94]),
        "=r"(__out[95]),
        "=r"(__out[96]),
        "=r"(__out[97]),
        "=r"(__out[98]),
        "=r"(__out[99]),
        "=r"(__out[100]),
        "=r"(__out[101]),
        "=r"(__out[102]),
        "=r"(__out[103]),
        "=r"(__out[104]),
        "=r"(__out[105]),
        "=r"(__out[106]),
        "=r"(__out[107]),
        "=r"(__out[108]),
        "=r"(__out[109]),
        "=r"(__out[110]),
        "=r"(__out[111]),
        "=r"(__out[112]),
        "=r"(__out[113]),
        "=r"(__out[114]),
        "=r"(__out[115]),
        "=r"(__out[116]),
        "=r"(__out[117]),
        "=r"(__out[118]),
        "=r"(__out[119]),
        "=r"(__out[120]),
        "=r"(__out[121]),
        "=r"(__out[122]),
        "=r"(__out[123]),
        "=r"(__out[124]),
        "=r"(__out[125]),
        "=r"(__out[126]),
        "=r"(__out[127]),
        "=r"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x128.f32.op.abs out, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_16x32bx2_abs(
  cuda::ptx::op_t<Op> op,
  float (&out)[128],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_16x32bx2_abs_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_16x32bx2_abs(
  ::cuda::ptx::op_t<_Op> __op,
  float (&__out)[128],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  float __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x128.f32.min.abs {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
      "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, "
      "%35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
      "%57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, "
      "%79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, "
      "%101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, "
      "%119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, [%129], %130;"
      : "=f"(__out[0]),
        "=f"(__out[1]),
        "=f"(__out[2]),
        "=f"(__out[3]),
        "=f"(__out[4]),
        "=f"(__out[5]),
        "=f"(__out[6]),
        "=f"(__out[7]),
        "=f"(__out[8]),
        "=f"(__out[9]),
        "=f"(__out[10]),
        "=f"(__out[11]),
        "=f"(__out[12]),
        "=f"(__out[13]),
        "=f"(__out[14]),
        "=f"(__out[15]),
        "=f"(__out[16]),
        "=f"(__out[17]),
        "=f"(__out[18]),
        "=f"(__out[19]),
        "=f"(__out[20]),
        "=f"(__out[21]),
        "=f"(__out[22]),
        "=f"(__out[23]),
        "=f"(__out[24]),
        "=f"(__out[25]),
        "=f"(__out[26]),
        "=f"(__out[27]),
        "=f"(__out[28]),
        "=f"(__out[29]),
        "=f"(__out[30]),
        "=f"(__out[31]),
        "=f"(__out[32]),
        "=f"(__out[33]),
        "=f"(__out[34]),
        "=f"(__out[35]),
        "=f"(__out[36]),
        "=f"(__out[37]),
        "=f"(__out[38]),
        "=f"(__out[39]),
        "=f"(__out[40]),
        "=f"(__out[41]),
        "=f"(__out[42]),
        "=f"(__out[43]),
        "=f"(__out[44]),
        "=f"(__out[45]),
        "=f"(__out[46]),
        "=f"(__out[47]),
        "=f"(__out[48]),
        "=f"(__out[49]),
        "=f"(__out[50]),
        "=f"(__out[51]),
        "=f"(__out[52]),
        "=f"(__out[53]),
        "=f"(__out[54]),
        "=f"(__out[55]),
        "=f"(__out[56]),
        "=f"(__out[57]),
        "=f"(__out[58]),
        "=f"(__out[59]),
        "=f"(__out[60]),
        "=f"(__out[61]),
        "=f"(__out[62]),
        "=f"(__out[63]),
        "=f"(__out[64]),
        "=f"(__out[65]),
        "=f"(__out[66]),
        "=f"(__out[67]),
        "=f"(__out[68]),
        "=f"(__out[69]),
        "=f"(__out[70]),
        "=f"(__out[71]),
        "=f"(__out[72]),
        "=f"(__out[73]),
        "=f"(__out[74]),
        "=f"(__out[75]),
        "=f"(__out[76]),
        "=f"(__out[77]),
        "=f"(__out[78]),
        "=f"(__out[79]),
        "=f"(__out[80]),
        "=f"(__out[81]),
        "=f"(__out[82]),
        "=f"(__out[83]),
        "=f"(__out[84]),
        "=f"(__out[85]),
        "=f"(__out[86]),
        "=f"(__out[87]),
        "=f"(__out[88]),
        "=f"(__out[89]),
        "=f"(__out[90]),
        "=f"(__out[91]),
        "=f"(__out[92]),
        "=f"(__out[93]),
        "=f"(__out[94]),
        "=f"(__out[95]),
        "=f"(__out[96]),
        "=f"(__out[97]),
        "=f"(__out[98]),
        "=f"(__out[99]),
        "=f"(__out[100]),
        "=f"(__out[101]),
        "=f"(__out[102]),
        "=f"(__out[103]),
        "=f"(__out[104]),
        "=f"(__out[105]),
        "=f"(__out[106]),
        "=f"(__out[107]),
        "=f"(__out[108]),
        "=f"(__out[109]),
        "=f"(__out[110]),
        "=f"(__out[111]),
        "=f"(__out[112]),
        "=f"(__out[113]),
        "=f"(__out[114]),
        "=f"(__out[115]),
        "=f"(__out[116]),
        "=f"(__out[117]),
        "=f"(__out[118]),
        "=f"(__out[119]),
        "=f"(__out[120]),
        "=f"(__out[121]),
        "=f"(__out[122]),
        "=f"(__out[123]),
        "=f"(__out[124]),
        "=f"(__out[125]),
        "=f"(__out[126]),
        "=f"(__out[127]),
        "=f"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x128.f32.max.abs {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
      "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, "
      "%35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
      "%57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, "
      "%79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, "
      "%101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, "
      "%119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, [%129], %130;"
      : "=f"(__out[0]),
        "=f"(__out[1]),
        "=f"(__out[2]),
        "=f"(__out[3]),
        "=f"(__out[4]),
        "=f"(__out[5]),
        "=f"(__out[6]),
        "=f"(__out[7]),
        "=f"(__out[8]),
        "=f"(__out[9]),
        "=f"(__out[10]),
        "=f"(__out[11]),
        "=f"(__out[12]),
        "=f"(__out[13]),
        "=f"(__out[14]),
        "=f"(__out[15]),
        "=f"(__out[16]),
        "=f"(__out[17]),
        "=f"(__out[18]),
        "=f"(__out[19]),
        "=f"(__out[20]),
        "=f"(__out[21]),
        "=f"(__out[22]),
        "=f"(__out[23]),
        "=f"(__out[24]),
        "=f"(__out[25]),
        "=f"(__out[26]),
        "=f"(__out[27]),
        "=f"(__out[28]),
        "=f"(__out[29]),
        "=f"(__out[30]),
        "=f"(__out[31]),
        "=f"(__out[32]),
        "=f"(__out[33]),
        "=f"(__out[34]),
        "=f"(__out[35]),
        "=f"(__out[36]),
        "=f"(__out[37]),
        "=f"(__out[38]),
        "=f"(__out[39]),
        "=f"(__out[40]),
        "=f"(__out[41]),
        "=f"(__out[42]),
        "=f"(__out[43]),
        "=f"(__out[44]),
        "=f"(__out[45]),
        "=f"(__out[46]),
        "=f"(__out[47]),
        "=f"(__out[48]),
        "=f"(__out[49]),
        "=f"(__out[50]),
        "=f"(__out[51]),
        "=f"(__out[52]),
        "=f"(__out[53]),
        "=f"(__out[54]),
        "=f"(__out[55]),
        "=f"(__out[56]),
        "=f"(__out[57]),
        "=f"(__out[58]),
        "=f"(__out[59]),
        "=f"(__out[60]),
        "=f"(__out[61]),
        "=f"(__out[62]),
        "=f"(__out[63]),
        "=f"(__out[64]),
        "=f"(__out[65]),
        "=f"(__out[66]),
        "=f"(__out[67]),
        "=f"(__out[68]),
        "=f"(__out[69]),
        "=f"(__out[70]),
        "=f"(__out[71]),
        "=f"(__out[72]),
        "=f"(__out[73]),
        "=f"(__out[74]),
        "=f"(__out[75]),
        "=f"(__out[76]),
        "=f"(__out[77]),
        "=f"(__out[78]),
        "=f"(__out[79]),
        "=f"(__out[80]),
        "=f"(__out[81]),
        "=f"(__out[82]),
        "=f"(__out[83]),
        "=f"(__out[84]),
        "=f"(__out[85]),
        "=f"(__out[86]),
        "=f"(__out[87]),
        "=f"(__out[88]),
        "=f"(__out[89]),
        "=f"(__out[90]),
        "=f"(__out[91]),
        "=f"(__out[92]),
        "=f"(__out[93]),
        "=f"(__out[94]),
        "=f"(__out[95]),
        "=f"(__out[96]),
        "=f"(__out[97]),
        "=f"(__out[98]),
        "=f"(__out[99]),
        "=f"(__out[100]),
        "=f"(__out[101]),
        "=f"(__out[102]),
        "=f"(__out[103]),
        "=f"(__out[104]),
        "=f"(__out[105]),
        "=f"(__out[106]),
        "=f"(__out[107]),
        "=f"(__out[108]),
        "=f"(__out[109]),
        "=f"(__out[110]),
        "=f"(__out[111]),
        "=f"(__out[112]),
        "=f"(__out[113]),
        "=f"(__out[114]),
        "=f"(__out[115]),
        "=f"(__out[116]),
        "=f"(__out[117]),
        "=f"(__out[118]),
        "=f"(__out[119]),
        "=f"(__out[120]),
        "=f"(__out[121]),
        "=f"(__out[122]),
        "=f"(__out[123]),
        "=f"(__out[124]),
        "=f"(__out[125]),
        "=f"(__out[126]),
        "=f"(__out[127]),
        "=f"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_16x32bx2_abs_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x128.f32.op out, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  float (&out)[128],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
extern "C" _CCCL_DEVICE void
__cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  float (&__out)[128],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1030) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1100) \
    || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(103) || __CUDA_HAS_ARCH_FAMILY_SPECIFIC(110)
  float __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x128.f32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
      "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, "
      "%102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, "
      "%120, %121, %122, %123, %124, %125, %126, %127}, %128, [%129], %130;"
      : "=f"(__out[0]),
        "=f"(__out[1]),
        "=f"(__out[2]),
        "=f"(__out[3]),
        "=f"(__out[4]),
        "=f"(__out[5]),
        "=f"(__out[6]),
        "=f"(__out[7]),
        "=f"(__out[8]),
        "=f"(__out[9]),
        "=f"(__out[10]),
        "=f"(__out[11]),
        "=f"(__out[12]),
        "=f"(__out[13]),
        "=f"(__out[14]),
        "=f"(__out[15]),
        "=f"(__out[16]),
        "=f"(__out[17]),
        "=f"(__out[18]),
        "=f"(__out[19]),
        "=f"(__out[20]),
        "=f"(__out[21]),
        "=f"(__out[22]),
        "=f"(__out[23]),
        "=f"(__out[24]),
        "=f"(__out[25]),
        "=f"(__out[26]),
        "=f"(__out[27]),
        "=f"(__out[28]),
        "=f"(__out[29]),
        "=f"(__out[30]),
        "=f"(__out[31]),
        "=f"(__out[32]),
        "=f"(__out[33]),
        "=f"(__out[34]),
        "=f"(__out[35]),
        "=f"(__out[36]),
        "=f"(__out[37]),
        "=f"(__out[38]),
        "=f"(__out[39]),
        "=f"(__out[40]),
        "=f"(__out[41]),
        "=f"(__out[42]),
        "=f"(__out[43]),
        "=f"(__out[44]),
        "=f"(__out[45]),
        "=f"(__out[46]),
        "=f"(__out[47]),
        "=f"(__out[48]),
        "=f"(__out[49]),
        "=f"(__out[50]),
        "=f"(__out[51]),
        "=f"(__out[52]),
        "=f"(__out[53]),
        "=f"(__out[54]),
        "=f"(__out[55]),
        "=f"(__out[56]),
        "=f"(__out[57]),
        "=f"(__out[58]),
        "=f"(__out[59]),
        "=f"(__out[60]),
        "=f"(__out[61]),
        "=f"(__out[62]),
        "=f"(__out[63]),
        "=f"(__out[64]),
        "=f"(__out[65]),
        "=f"(__out[66]),
        "=f"(__out[67]),
        "=f"(__out[68]),
        "=f"(__out[69]),
        "=f"(__out[70]),
        "=f"(__out[71]),
        "=f"(__out[72]),
        "=f"(__out[73]),
        "=f"(__out[74]),
        "=f"(__out[75]),
        "=f"(__out[76]),
        "=f"(__out[77]),
        "=f"(__out[78]),
        "=f"(__out[79]),
        "=f"(__out[80]),
        "=f"(__out[81]),
        "=f"(__out[82]),
        "=f"(__out[83]),
        "=f"(__out[84]),
        "=f"(__out[85]),
        "=f"(__out[86]),
        "=f"(__out[87]),
        "=f"(__out[88]),
        "=f"(__out[89]),
        "=f"(__out[90]),
        "=f"(__out[91]),
        "=f"(__out[92]),
        "=f"(__out[93]),
        "=f"(__out[94]),
        "=f"(__out[95]),
        "=f"(__out[96]),
        "=f"(__out[97]),
        "=f"(__out[98]),
        "=f"(__out[99]),
        "=f"(__out[100]),
        "=f"(__out[101]),
        "=f"(__out[102]),
        "=f"(__out[103]),
        "=f"(__out[104]),
        "=f"(__out[105]),
        "=f"(__out[106]),
        "=f"(__out[107]),
        "=f"(__out[108]),
        "=f"(__out[109]),
        "=f"(__out[110]),
        "=f"(__out[111]),
        "=f"(__out[112]),
        "=f"(__out[113]),
        "=f"(__out[114]),
        "=f"(__out[115]),
        "=f"(__out[116]),
        "=f"(__out[117]),
        "=f"(__out[118]),
        "=f"(__out[119]),
        "=f"(__out[120]),
        "=f"(__out[121]),
        "=f"(__out[122]),
        "=f"(__out[123]),
        "=f"(__out[124]),
        "=f"(__out[125]),
        "=f"(__out[126]),
        "=f"(__out[127]),
        "=f"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x128.f32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
      "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, "
      "%102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, "
      "%120, %121, %122, %123, %124, %125, %126, %127}, %128, [%129], %130;"
      : "=f"(__out[0]),
        "=f"(__out[1]),
        "=f"(__out[2]),
        "=f"(__out[3]),
        "=f"(__out[4]),
        "=f"(__out[5]),
        "=f"(__out[6]),
        "=f"(__out[7]),
        "=f"(__out[8]),
        "=f"(__out[9]),
        "=f"(__out[10]),
        "=f"(__out[11]),
        "=f"(__out[12]),
        "=f"(__out[13]),
        "=f"(__out[14]),
        "=f"(__out[15]),
        "=f"(__out[16]),
        "=f"(__out[17]),
        "=f"(__out[18]),
        "=f"(__out[19]),
        "=f"(__out[20]),
        "=f"(__out[21]),
        "=f"(__out[22]),
        "=f"(__out[23]),
        "=f"(__out[24]),
        "=f"(__out[25]),
        "=f"(__out[26]),
        "=f"(__out[27]),
        "=f"(__out[28]),
        "=f"(__out[29]),
        "=f"(__out[30]),
        "=f"(__out[31]),
        "=f"(__out[32]),
        "=f"(__out[33]),
        "=f"(__out[34]),
        "=f"(__out[35]),
        "=f"(__out[36]),
        "=f"(__out[37]),
        "=f"(__out[38]),
        "=f"(__out[39]),
        "=f"(__out[40]),
        "=f"(__out[41]),
        "=f"(__out[42]),
        "=f"(__out[43]),
        "=f"(__out[44]),
        "=f"(__out[45]),
        "=f"(__out[46]),
        "=f"(__out[47]),
        "=f"(__out[48]),
        "=f"(__out[49]),
        "=f"(__out[50]),
        "=f"(__out[51]),
        "=f"(__out[52]),
        "=f"(__out[53]),
        "=f"(__out[54]),
        "=f"(__out[55]),
        "=f"(__out[56]),
        "=f"(__out[57]),
        "=f"(__out[58]),
        "=f"(__out[59]),
        "=f"(__out[60]),
        "=f"(__out[61]),
        "=f"(__out[62]),
        "=f"(__out[63]),
        "=f"(__out[64]),
        "=f"(__out[65]),
        "=f"(__out[66]),
        "=f"(__out[67]),
        "=f"(__out[68]),
        "=f"(__out[69]),
        "=f"(__out[70]),
        "=f"(__out[71]),
        "=f"(__out[72]),
        "=f"(__out[73]),
        "=f"(__out[74]),
        "=f"(__out[75]),
        "=f"(__out[76]),
        "=f"(__out[77]),
        "=f"(__out[78]),
        "=f"(__out[79]),
        "=f"(__out[80]),
        "=f"(__out[81]),
        "=f"(__out[82]),
        "=f"(__out[83]),
        "=f"(__out[84]),
        "=f"(__out[85]),
        "=f"(__out[86]),
        "=f"(__out[87]),
        "=f"(__out[88]),
        "=f"(__out[89]),
        "=f"(__out[90]),
        "=f"(__out[91]),
        "=f"(__out[92]),
        "=f"(__out[93]),
        "=f"(__out[94]),
        "=f"(__out[95]),
        "=f"(__out[96]),
        "=f"(__out[97]),
        "=f"(__out[98]),
        "=f"(__out[99]),
        "=f"(__out[100]),
        "=f"(__out[101]),
        "=f"(__out[102]),
        "=f"(__out[103]),
        "=f"(__out[104]),
        "=f"(__out[105]),
        "=f"(__out[106]),
        "=f"(__out[107]),
        "=f"(__out[108]),
        "=f"(__out[109]),
        "=f"(__out[110]),
        "=f"(__out[111]),
        "=f"(__out[112]),
        "=f"(__out[113]),
        "=f"(__out[114]),
        "=f"(__out[115]),
        "=f"(__out[116]),
        "=f"(__out[117]),
        "=f"(__out[118]),
        "=f"(__out[119]),
        "=f"(__out[120]),
        "=f"(__out[121]),
        "=f"(__out[122]),
        "=f"(__out[123]),
        "=f"(__out[124]),
        "=f"(__out[125]),
        "=f"(__out[126]),
        "=f"(__out[127]),
        "=f"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  return __redval;

#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_ld_red_16x32bx2_is_only_supported_on_SM_103a_103f_110a_110f_depending_on_the_variant__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 880

#endif // _CUDA_PTX_GENERATED_TCGEN05_LD_H_
