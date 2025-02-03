// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_TCGEN05_ST_H_
#define _CUDA_PTX_GENERATED_TCGEN05_ST_H_

/*
// tcgen05.st.sync.aligned.16x64b.x1.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b(
  uint32_t taddr,
  const B32 (&values)[1]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x64b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[1])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x64b.x1.b32 [%0], {%1};" : : "r"(__taddr), "r"(__as_b32(__values[0])) : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x1.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[1]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[1])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x64b.x1.unpack::16b.b32 [%0], {%1};"
      :
      : "r"(__taddr), "r"(__as_b32(__values[0]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x2.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b(
  uint32_t taddr,
  const B32 (&values)[2]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x64b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[2])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x64b.x2.b32 [%0], {%1, %2};"
      :
      : "r"(__taddr), "r"(__as_b32(__values[0])), "r"(__as_b32(__values[1]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x2.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[2]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[2])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x64b.x2.unpack::16b.b32 [%0], {%1, %2};"
      :
      : "r"(__taddr), "r"(__as_b32(__values[0])), "r"(__as_b32(__values[1]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x4.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b(
  uint32_t taddr,
  const B32 (&values)[4]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x64b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[4])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x64b.x4.b32 [%0], {%1, %2, %3, %4};"
      :
      : "r"(__taddr),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x4.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[4]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[4])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x64b.x4.unpack::16b.b32 [%0], {%1, %2, %3, %4};"
      :
      : "r"(__taddr),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x8.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b(
  uint32_t taddr,
  const B32 (&values)[8]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x64b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[8])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x64b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
      :
      : "r"(__taddr),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3])),
        "r"(__as_b32(__values[4])),
        "r"(__as_b32(__values[5])),
        "r"(__as_b32(__values[6])),
        "r"(__as_b32(__values[7]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x8.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[8]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[8])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x64b.x8.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
      :
      : "r"(__taddr),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3])),
        "r"(__as_b32(__values[4])),
        "r"(__as_b32(__values[5])),
        "r"(__as_b32(__values[6])),
        "r"(__as_b32(__values[7]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x16.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b(
  uint32_t taddr,
  const B32 (&values)[16]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x64b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[16])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x64b.x16.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16};"
      :
      : "r"(__taddr),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3])),
        "r"(__as_b32(__values[4])),
        "r"(__as_b32(__values[5])),
        "r"(__as_b32(__values[6])),
        "r"(__as_b32(__values[7])),
        "r"(__as_b32(__values[8])),
        "r"(__as_b32(__values[9])),
        "r"(__as_b32(__values[10])),
        "r"(__as_b32(__values[11])),
        "r"(__as_b32(__values[12])),
        "r"(__as_b32(__values[13])),
        "r"(__as_b32(__values[14])),
        "r"(__as_b32(__values[15]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x16.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[16]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[16])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x64b.x16.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
      "%13, %14, %15, %16};"
      :
      : "r"(__taddr),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3])),
        "r"(__as_b32(__values[4])),
        "r"(__as_b32(__values[5])),
        "r"(__as_b32(__values[6])),
        "r"(__as_b32(__values[7])),
        "r"(__as_b32(__values[8])),
        "r"(__as_b32(__values[9])),
        "r"(__as_b32(__values[10])),
        "r"(__as_b32(__values[11])),
        "r"(__as_b32(__values[12])),
        "r"(__as_b32(__values[13])),
        "r"(__as_b32(__values[14])),
        "r"(__as_b32(__values[15]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x32.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b(
  uint32_t taddr,
  const B32 (&values)[32]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x64b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[32])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.16x64b.x32.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
    :
    : "r"(__taddr),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x32.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[32]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[32])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.16x64b.x32.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
    :
    : "r"(__taddr),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x64.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b(
  uint32_t taddr,
  const B32 (&values)[64]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x64b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[64])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.16x64b.x64.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64};"
    :
    : "r"(__taddr),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31])),
      "r"(__as_b32(__values[32])),
      "r"(__as_b32(__values[33])),
      "r"(__as_b32(__values[34])),
      "r"(__as_b32(__values[35])),
      "r"(__as_b32(__values[36])),
      "r"(__as_b32(__values[37])),
      "r"(__as_b32(__values[38])),
      "r"(__as_b32(__values[39])),
      "r"(__as_b32(__values[40])),
      "r"(__as_b32(__values[41])),
      "r"(__as_b32(__values[42])),
      "r"(__as_b32(__values[43])),
      "r"(__as_b32(__values[44])),
      "r"(__as_b32(__values[45])),
      "r"(__as_b32(__values[46])),
      "r"(__as_b32(__values[47])),
      "r"(__as_b32(__values[48])),
      "r"(__as_b32(__values[49])),
      "r"(__as_b32(__values[50])),
      "r"(__as_b32(__values[51])),
      "r"(__as_b32(__values[52])),
      "r"(__as_b32(__values[53])),
      "r"(__as_b32(__values[54])),
      "r"(__as_b32(__values[55])),
      "r"(__as_b32(__values[56])),
      "r"(__as_b32(__values[57])),
      "r"(__as_b32(__values[58])),
      "r"(__as_b32(__values[59])),
      "r"(__as_b32(__values[60])),
      "r"(__as_b32(__values[61])),
      "r"(__as_b32(__values[62])),
      "r"(__as_b32(__values[63]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x64.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[64]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[64])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.16x64b.x64.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63, %64};"
    :
    : "r"(__taddr),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31])),
      "r"(__as_b32(__values[32])),
      "r"(__as_b32(__values[33])),
      "r"(__as_b32(__values[34])),
      "r"(__as_b32(__values[35])),
      "r"(__as_b32(__values[36])),
      "r"(__as_b32(__values[37])),
      "r"(__as_b32(__values[38])),
      "r"(__as_b32(__values[39])),
      "r"(__as_b32(__values[40])),
      "r"(__as_b32(__values[41])),
      "r"(__as_b32(__values[42])),
      "r"(__as_b32(__values[43])),
      "r"(__as_b32(__values[44])),
      "r"(__as_b32(__values[45])),
      "r"(__as_b32(__values[46])),
      "r"(__as_b32(__values[47])),
      "r"(__as_b32(__values[48])),
      "r"(__as_b32(__values[49])),
      "r"(__as_b32(__values[50])),
      "r"(__as_b32(__values[51])),
      "r"(__as_b32(__values[52])),
      "r"(__as_b32(__values[53])),
      "r"(__as_b32(__values[54])),
      "r"(__as_b32(__values[55])),
      "r"(__as_b32(__values[56])),
      "r"(__as_b32(__values[57])),
      "r"(__as_b32(__values[58])),
      "r"(__as_b32(__values[59])),
      "r"(__as_b32(__values[60])),
      "r"(__as_b32(__values[61])),
      "r"(__as_b32(__values[62])),
      "r"(__as_b32(__values[63]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x128.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b(
  uint32_t taddr,
  const B32 (&values)[128]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x64b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[128])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.16x64b.x128.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, "
    "%82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, "
    "%104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
    "%123, %124, %125, %126, %127, %128};"
    :
    : "r"(__taddr),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31])),
      "r"(__as_b32(__values[32])),
      "r"(__as_b32(__values[33])),
      "r"(__as_b32(__values[34])),
      "r"(__as_b32(__values[35])),
      "r"(__as_b32(__values[36])),
      "r"(__as_b32(__values[37])),
      "r"(__as_b32(__values[38])),
      "r"(__as_b32(__values[39])),
      "r"(__as_b32(__values[40])),
      "r"(__as_b32(__values[41])),
      "r"(__as_b32(__values[42])),
      "r"(__as_b32(__values[43])),
      "r"(__as_b32(__values[44])),
      "r"(__as_b32(__values[45])),
      "r"(__as_b32(__values[46])),
      "r"(__as_b32(__values[47])),
      "r"(__as_b32(__values[48])),
      "r"(__as_b32(__values[49])),
      "r"(__as_b32(__values[50])),
      "r"(__as_b32(__values[51])),
      "r"(__as_b32(__values[52])),
      "r"(__as_b32(__values[53])),
      "r"(__as_b32(__values[54])),
      "r"(__as_b32(__values[55])),
      "r"(__as_b32(__values[56])),
      "r"(__as_b32(__values[57])),
      "r"(__as_b32(__values[58])),
      "r"(__as_b32(__values[59])),
      "r"(__as_b32(__values[60])),
      "r"(__as_b32(__values[61])),
      "r"(__as_b32(__values[62])),
      "r"(__as_b32(__values[63])),
      "r"(__as_b32(__values[64])),
      "r"(__as_b32(__values[65])),
      "r"(__as_b32(__values[66])),
      "r"(__as_b32(__values[67])),
      "r"(__as_b32(__values[68])),
      "r"(__as_b32(__values[69])),
      "r"(__as_b32(__values[70])),
      "r"(__as_b32(__values[71])),
      "r"(__as_b32(__values[72])),
      "r"(__as_b32(__values[73])),
      "r"(__as_b32(__values[74])),
      "r"(__as_b32(__values[75])),
      "r"(__as_b32(__values[76])),
      "r"(__as_b32(__values[77])),
      "r"(__as_b32(__values[78])),
      "r"(__as_b32(__values[79])),
      "r"(__as_b32(__values[80])),
      "r"(__as_b32(__values[81])),
      "r"(__as_b32(__values[82])),
      "r"(__as_b32(__values[83])),
      "r"(__as_b32(__values[84])),
      "r"(__as_b32(__values[85])),
      "r"(__as_b32(__values[86])),
      "r"(__as_b32(__values[87])),
      "r"(__as_b32(__values[88])),
      "r"(__as_b32(__values[89])),
      "r"(__as_b32(__values[90])),
      "r"(__as_b32(__values[91])),
      "r"(__as_b32(__values[92])),
      "r"(__as_b32(__values[93])),
      "r"(__as_b32(__values[94])),
      "r"(__as_b32(__values[95])),
      "r"(__as_b32(__values[96])),
      "r"(__as_b32(__values[97])),
      "r"(__as_b32(__values[98])),
      "r"(__as_b32(__values[99])),
      "r"(__as_b32(__values[100])),
      "r"(__as_b32(__values[101])),
      "r"(__as_b32(__values[102])),
      "r"(__as_b32(__values[103])),
      "r"(__as_b32(__values[104])),
      "r"(__as_b32(__values[105])),
      "r"(__as_b32(__values[106])),
      "r"(__as_b32(__values[107])),
      "r"(__as_b32(__values[108])),
      "r"(__as_b32(__values[109])),
      "r"(__as_b32(__values[110])),
      "r"(__as_b32(__values[111])),
      "r"(__as_b32(__values[112])),
      "r"(__as_b32(__values[113])),
      "r"(__as_b32(__values[114])),
      "r"(__as_b32(__values[115])),
      "r"(__as_b32(__values[116])),
      "r"(__as_b32(__values[117])),
      "r"(__as_b32(__values[118])),
      "r"(__as_b32(__values[119])),
      "r"(__as_b32(__values[120])),
      "r"(__as_b32(__values[121])),
      "r"(__as_b32(__values[122])),
      "r"(__as_b32(__values[123])),
      "r"(__as_b32(__values[124])),
      "r"(__as_b32(__values[125])),
      "r"(__as_b32(__values[126])),
      "r"(__as_b32(__values[127]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x64b.x128.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x64b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[128]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x64b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[128])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.16x64b.x128.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
    "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, "
    "%35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
    "%57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, "
    "%79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, "
    "%101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, "
    "%120, %121, %122, %123, %124, %125, %126, %127, %128};"
    :
    : "r"(__taddr),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31])),
      "r"(__as_b32(__values[32])),
      "r"(__as_b32(__values[33])),
      "r"(__as_b32(__values[34])),
      "r"(__as_b32(__values[35])),
      "r"(__as_b32(__values[36])),
      "r"(__as_b32(__values[37])),
      "r"(__as_b32(__values[38])),
      "r"(__as_b32(__values[39])),
      "r"(__as_b32(__values[40])),
      "r"(__as_b32(__values[41])),
      "r"(__as_b32(__values[42])),
      "r"(__as_b32(__values[43])),
      "r"(__as_b32(__values[44])),
      "r"(__as_b32(__values[45])),
      "r"(__as_b32(__values[46])),
      "r"(__as_b32(__values[47])),
      "r"(__as_b32(__values[48])),
      "r"(__as_b32(__values[49])),
      "r"(__as_b32(__values[50])),
      "r"(__as_b32(__values[51])),
      "r"(__as_b32(__values[52])),
      "r"(__as_b32(__values[53])),
      "r"(__as_b32(__values[54])),
      "r"(__as_b32(__values[55])),
      "r"(__as_b32(__values[56])),
      "r"(__as_b32(__values[57])),
      "r"(__as_b32(__values[58])),
      "r"(__as_b32(__values[59])),
      "r"(__as_b32(__values[60])),
      "r"(__as_b32(__values[61])),
      "r"(__as_b32(__values[62])),
      "r"(__as_b32(__values[63])),
      "r"(__as_b32(__values[64])),
      "r"(__as_b32(__values[65])),
      "r"(__as_b32(__values[66])),
      "r"(__as_b32(__values[67])),
      "r"(__as_b32(__values[68])),
      "r"(__as_b32(__values[69])),
      "r"(__as_b32(__values[70])),
      "r"(__as_b32(__values[71])),
      "r"(__as_b32(__values[72])),
      "r"(__as_b32(__values[73])),
      "r"(__as_b32(__values[74])),
      "r"(__as_b32(__values[75])),
      "r"(__as_b32(__values[76])),
      "r"(__as_b32(__values[77])),
      "r"(__as_b32(__values[78])),
      "r"(__as_b32(__values[79])),
      "r"(__as_b32(__values[80])),
      "r"(__as_b32(__values[81])),
      "r"(__as_b32(__values[82])),
      "r"(__as_b32(__values[83])),
      "r"(__as_b32(__values[84])),
      "r"(__as_b32(__values[85])),
      "r"(__as_b32(__values[86])),
      "r"(__as_b32(__values[87])),
      "r"(__as_b32(__values[88])),
      "r"(__as_b32(__values[89])),
      "r"(__as_b32(__values[90])),
      "r"(__as_b32(__values[91])),
      "r"(__as_b32(__values[92])),
      "r"(__as_b32(__values[93])),
      "r"(__as_b32(__values[94])),
      "r"(__as_b32(__values[95])),
      "r"(__as_b32(__values[96])),
      "r"(__as_b32(__values[97])),
      "r"(__as_b32(__values[98])),
      "r"(__as_b32(__values[99])),
      "r"(__as_b32(__values[100])),
      "r"(__as_b32(__values[101])),
      "r"(__as_b32(__values[102])),
      "r"(__as_b32(__values[103])),
      "r"(__as_b32(__values[104])),
      "r"(__as_b32(__values[105])),
      "r"(__as_b32(__values[106])),
      "r"(__as_b32(__values[107])),
      "r"(__as_b32(__values[108])),
      "r"(__as_b32(__values[109])),
      "r"(__as_b32(__values[110])),
      "r"(__as_b32(__values[111])),
      "r"(__as_b32(__values[112])),
      "r"(__as_b32(__values[113])),
      "r"(__as_b32(__values[114])),
      "r"(__as_b32(__values[115])),
      "r"(__as_b32(__values[116])),
      "r"(__as_b32(__values[117])),
      "r"(__as_b32(__values[118])),
      "r"(__as_b32(__values[119])),
      "r"(__as_b32(__values[120])),
      "r"(__as_b32(__values[121])),
      "r"(__as_b32(__values[122])),
      "r"(__as_b32(__values[123])),
      "r"(__as_b32(__values[124])),
      "r"(__as_b32(__values[125])),
      "r"(__as_b32(__values[126])),
      "r"(__as_b32(__values[127]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x64b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x128b.x1.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x128b(
  uint32_t taddr,
  const B32 (&values)[2]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x128b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x128b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[2])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x128b.x1.b32 [%0], {%1, %2};"
      :
      : "r"(__taddr), "r"(__as_b32(__values[0])), "r"(__as_b32(__values[1]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x128b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x128b.x1.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x128b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[2]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x128b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x128b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[2])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x128b.x1.unpack::16b.b32 [%0], {%1, %2};"
      :
      : "r"(__taddr), "r"(__as_b32(__values[0])), "r"(__as_b32(__values[1]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x128b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x128b.x2.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x128b(
  uint32_t taddr,
  const B32 (&values)[4]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x128b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x128b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[4])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x128b.x2.b32 [%0], {%1, %2, %3, %4};"
      :
      : "r"(__taddr),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x128b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x128b.x2.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x128b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[4]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x128b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x128b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[4])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x128b.x2.unpack::16b.b32 [%0], {%1, %2, %3, %4};"
      :
      : "r"(__taddr),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x128b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x128b.x4.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x128b(
  uint32_t taddr,
  const B32 (&values)[8]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x128b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x128b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[8])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x128b.x4.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
      :
      : "r"(__taddr),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3])),
        "r"(__as_b32(__values[4])),
        "r"(__as_b32(__values[5])),
        "r"(__as_b32(__values[6])),
        "r"(__as_b32(__values[7]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x128b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x128b.x4.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x128b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[8]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x128b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x128b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[8])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x128b.x4.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
      :
      : "r"(__taddr),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3])),
        "r"(__as_b32(__values[4])),
        "r"(__as_b32(__values[5])),
        "r"(__as_b32(__values[6])),
        "r"(__as_b32(__values[7]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x128b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x128b.x8.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x128b(
  uint32_t taddr,
  const B32 (&values)[16]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x128b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x128b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[16])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x128b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16};"
      :
      : "r"(__taddr),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3])),
        "r"(__as_b32(__values[4])),
        "r"(__as_b32(__values[5])),
        "r"(__as_b32(__values[6])),
        "r"(__as_b32(__values[7])),
        "r"(__as_b32(__values[8])),
        "r"(__as_b32(__values[9])),
        "r"(__as_b32(__values[10])),
        "r"(__as_b32(__values[11])),
        "r"(__as_b32(__values[12])),
        "r"(__as_b32(__values[13])),
        "r"(__as_b32(__values[14])),
        "r"(__as_b32(__values[15]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x128b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x128b.x8.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x128b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[16]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x128b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x128b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[16])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x128b.x8.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
      "%13, %14, %15, %16};"
      :
      : "r"(__taddr),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3])),
        "r"(__as_b32(__values[4])),
        "r"(__as_b32(__values[5])),
        "r"(__as_b32(__values[6])),
        "r"(__as_b32(__values[7])),
        "r"(__as_b32(__values[8])),
        "r"(__as_b32(__values[9])),
        "r"(__as_b32(__values[10])),
        "r"(__as_b32(__values[11])),
        "r"(__as_b32(__values[12])),
        "r"(__as_b32(__values[13])),
        "r"(__as_b32(__values[14])),
        "r"(__as_b32(__values[15]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x128b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x128b.x16.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x128b(
  uint32_t taddr,
  const B32 (&values)[32]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x128b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x128b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[32])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.16x128b.x16.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
    :
    : "r"(__taddr),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x128b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x128b.x16.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x128b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[32]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x128b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x128b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[32])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.16x128b.x16.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
    "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
    :
    : "r"(__taddr),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x128b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x128b.x32.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x128b(
  uint32_t taddr,
  const B32 (&values)[64]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x128b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x128b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[64])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.16x128b.x32.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64};"
    :
    : "r"(__taddr),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31])),
      "r"(__as_b32(__values[32])),
      "r"(__as_b32(__values[33])),
      "r"(__as_b32(__values[34])),
      "r"(__as_b32(__values[35])),
      "r"(__as_b32(__values[36])),
      "r"(__as_b32(__values[37])),
      "r"(__as_b32(__values[38])),
      "r"(__as_b32(__values[39])),
      "r"(__as_b32(__values[40])),
      "r"(__as_b32(__values[41])),
      "r"(__as_b32(__values[42])),
      "r"(__as_b32(__values[43])),
      "r"(__as_b32(__values[44])),
      "r"(__as_b32(__values[45])),
      "r"(__as_b32(__values[46])),
      "r"(__as_b32(__values[47])),
      "r"(__as_b32(__values[48])),
      "r"(__as_b32(__values[49])),
      "r"(__as_b32(__values[50])),
      "r"(__as_b32(__values[51])),
      "r"(__as_b32(__values[52])),
      "r"(__as_b32(__values[53])),
      "r"(__as_b32(__values[54])),
      "r"(__as_b32(__values[55])),
      "r"(__as_b32(__values[56])),
      "r"(__as_b32(__values[57])),
      "r"(__as_b32(__values[58])),
      "r"(__as_b32(__values[59])),
      "r"(__as_b32(__values[60])),
      "r"(__as_b32(__values[61])),
      "r"(__as_b32(__values[62])),
      "r"(__as_b32(__values[63]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x128b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x128b.x32.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x128b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[64]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x128b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x128b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[64])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.16x128b.x32.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
    "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, "
    "%35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
    "%57, %58, %59, %60, %61, %62, %63, %64};"
    :
    : "r"(__taddr),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31])),
      "r"(__as_b32(__values[32])),
      "r"(__as_b32(__values[33])),
      "r"(__as_b32(__values[34])),
      "r"(__as_b32(__values[35])),
      "r"(__as_b32(__values[36])),
      "r"(__as_b32(__values[37])),
      "r"(__as_b32(__values[38])),
      "r"(__as_b32(__values[39])),
      "r"(__as_b32(__values[40])),
      "r"(__as_b32(__values[41])),
      "r"(__as_b32(__values[42])),
      "r"(__as_b32(__values[43])),
      "r"(__as_b32(__values[44])),
      "r"(__as_b32(__values[45])),
      "r"(__as_b32(__values[46])),
      "r"(__as_b32(__values[47])),
      "r"(__as_b32(__values[48])),
      "r"(__as_b32(__values[49])),
      "r"(__as_b32(__values[50])),
      "r"(__as_b32(__values[51])),
      "r"(__as_b32(__values[52])),
      "r"(__as_b32(__values[53])),
      "r"(__as_b32(__values[54])),
      "r"(__as_b32(__values[55])),
      "r"(__as_b32(__values[56])),
      "r"(__as_b32(__values[57])),
      "r"(__as_b32(__values[58])),
      "r"(__as_b32(__values[59])),
      "r"(__as_b32(__values[60])),
      "r"(__as_b32(__values[61])),
      "r"(__as_b32(__values[62])),
      "r"(__as_b32(__values[63]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x128b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x128b.x64.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x128b(
  uint32_t taddr,
  const B32 (&values)[128]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x128b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x128b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[128])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.16x128b.x64.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, "
    "%82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, "
    "%104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
    "%123, %124, %125, %126, %127, %128};"
    :
    : "r"(__taddr),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31])),
      "r"(__as_b32(__values[32])),
      "r"(__as_b32(__values[33])),
      "r"(__as_b32(__values[34])),
      "r"(__as_b32(__values[35])),
      "r"(__as_b32(__values[36])),
      "r"(__as_b32(__values[37])),
      "r"(__as_b32(__values[38])),
      "r"(__as_b32(__values[39])),
      "r"(__as_b32(__values[40])),
      "r"(__as_b32(__values[41])),
      "r"(__as_b32(__values[42])),
      "r"(__as_b32(__values[43])),
      "r"(__as_b32(__values[44])),
      "r"(__as_b32(__values[45])),
      "r"(__as_b32(__values[46])),
      "r"(__as_b32(__values[47])),
      "r"(__as_b32(__values[48])),
      "r"(__as_b32(__values[49])),
      "r"(__as_b32(__values[50])),
      "r"(__as_b32(__values[51])),
      "r"(__as_b32(__values[52])),
      "r"(__as_b32(__values[53])),
      "r"(__as_b32(__values[54])),
      "r"(__as_b32(__values[55])),
      "r"(__as_b32(__values[56])),
      "r"(__as_b32(__values[57])),
      "r"(__as_b32(__values[58])),
      "r"(__as_b32(__values[59])),
      "r"(__as_b32(__values[60])),
      "r"(__as_b32(__values[61])),
      "r"(__as_b32(__values[62])),
      "r"(__as_b32(__values[63])),
      "r"(__as_b32(__values[64])),
      "r"(__as_b32(__values[65])),
      "r"(__as_b32(__values[66])),
      "r"(__as_b32(__values[67])),
      "r"(__as_b32(__values[68])),
      "r"(__as_b32(__values[69])),
      "r"(__as_b32(__values[70])),
      "r"(__as_b32(__values[71])),
      "r"(__as_b32(__values[72])),
      "r"(__as_b32(__values[73])),
      "r"(__as_b32(__values[74])),
      "r"(__as_b32(__values[75])),
      "r"(__as_b32(__values[76])),
      "r"(__as_b32(__values[77])),
      "r"(__as_b32(__values[78])),
      "r"(__as_b32(__values[79])),
      "r"(__as_b32(__values[80])),
      "r"(__as_b32(__values[81])),
      "r"(__as_b32(__values[82])),
      "r"(__as_b32(__values[83])),
      "r"(__as_b32(__values[84])),
      "r"(__as_b32(__values[85])),
      "r"(__as_b32(__values[86])),
      "r"(__as_b32(__values[87])),
      "r"(__as_b32(__values[88])),
      "r"(__as_b32(__values[89])),
      "r"(__as_b32(__values[90])),
      "r"(__as_b32(__values[91])),
      "r"(__as_b32(__values[92])),
      "r"(__as_b32(__values[93])),
      "r"(__as_b32(__values[94])),
      "r"(__as_b32(__values[95])),
      "r"(__as_b32(__values[96])),
      "r"(__as_b32(__values[97])),
      "r"(__as_b32(__values[98])),
      "r"(__as_b32(__values[99])),
      "r"(__as_b32(__values[100])),
      "r"(__as_b32(__values[101])),
      "r"(__as_b32(__values[102])),
      "r"(__as_b32(__values[103])),
      "r"(__as_b32(__values[104])),
      "r"(__as_b32(__values[105])),
      "r"(__as_b32(__values[106])),
      "r"(__as_b32(__values[107])),
      "r"(__as_b32(__values[108])),
      "r"(__as_b32(__values[109])),
      "r"(__as_b32(__values[110])),
      "r"(__as_b32(__values[111])),
      "r"(__as_b32(__values[112])),
      "r"(__as_b32(__values[113])),
      "r"(__as_b32(__values[114])),
      "r"(__as_b32(__values[115])),
      "r"(__as_b32(__values[116])),
      "r"(__as_b32(__values[117])),
      "r"(__as_b32(__values[118])),
      "r"(__as_b32(__values[119])),
      "r"(__as_b32(__values[120])),
      "r"(__as_b32(__values[121])),
      "r"(__as_b32(__values[122])),
      "r"(__as_b32(__values[123])),
      "r"(__as_b32(__values[124])),
      "r"(__as_b32(__values[125])),
      "r"(__as_b32(__values[126])),
      "r"(__as_b32(__values[127]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x128b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x128b.x64.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x128b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[128]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x128b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x128b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[128])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.16x128b.x64.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
    "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, "
    "%35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
    "%57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, "
    "%79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, "
    "%101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, "
    "%120, %121, %122, %123, %124, %125, %126, %127, %128};"
    :
    : "r"(__taddr),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31])),
      "r"(__as_b32(__values[32])),
      "r"(__as_b32(__values[33])),
      "r"(__as_b32(__values[34])),
      "r"(__as_b32(__values[35])),
      "r"(__as_b32(__values[36])),
      "r"(__as_b32(__values[37])),
      "r"(__as_b32(__values[38])),
      "r"(__as_b32(__values[39])),
      "r"(__as_b32(__values[40])),
      "r"(__as_b32(__values[41])),
      "r"(__as_b32(__values[42])),
      "r"(__as_b32(__values[43])),
      "r"(__as_b32(__values[44])),
      "r"(__as_b32(__values[45])),
      "r"(__as_b32(__values[46])),
      "r"(__as_b32(__values[47])),
      "r"(__as_b32(__values[48])),
      "r"(__as_b32(__values[49])),
      "r"(__as_b32(__values[50])),
      "r"(__as_b32(__values[51])),
      "r"(__as_b32(__values[52])),
      "r"(__as_b32(__values[53])),
      "r"(__as_b32(__values[54])),
      "r"(__as_b32(__values[55])),
      "r"(__as_b32(__values[56])),
      "r"(__as_b32(__values[57])),
      "r"(__as_b32(__values[58])),
      "r"(__as_b32(__values[59])),
      "r"(__as_b32(__values[60])),
      "r"(__as_b32(__values[61])),
      "r"(__as_b32(__values[62])),
      "r"(__as_b32(__values[63])),
      "r"(__as_b32(__values[64])),
      "r"(__as_b32(__values[65])),
      "r"(__as_b32(__values[66])),
      "r"(__as_b32(__values[67])),
      "r"(__as_b32(__values[68])),
      "r"(__as_b32(__values[69])),
      "r"(__as_b32(__values[70])),
      "r"(__as_b32(__values[71])),
      "r"(__as_b32(__values[72])),
      "r"(__as_b32(__values[73])),
      "r"(__as_b32(__values[74])),
      "r"(__as_b32(__values[75])),
      "r"(__as_b32(__values[76])),
      "r"(__as_b32(__values[77])),
      "r"(__as_b32(__values[78])),
      "r"(__as_b32(__values[79])),
      "r"(__as_b32(__values[80])),
      "r"(__as_b32(__values[81])),
      "r"(__as_b32(__values[82])),
      "r"(__as_b32(__values[83])),
      "r"(__as_b32(__values[84])),
      "r"(__as_b32(__values[85])),
      "r"(__as_b32(__values[86])),
      "r"(__as_b32(__values[87])),
      "r"(__as_b32(__values[88])),
      "r"(__as_b32(__values[89])),
      "r"(__as_b32(__values[90])),
      "r"(__as_b32(__values[91])),
      "r"(__as_b32(__values[92])),
      "r"(__as_b32(__values[93])),
      "r"(__as_b32(__values[94])),
      "r"(__as_b32(__values[95])),
      "r"(__as_b32(__values[96])),
      "r"(__as_b32(__values[97])),
      "r"(__as_b32(__values[98])),
      "r"(__as_b32(__values[99])),
      "r"(__as_b32(__values[100])),
      "r"(__as_b32(__values[101])),
      "r"(__as_b32(__values[102])),
      "r"(__as_b32(__values[103])),
      "r"(__as_b32(__values[104])),
      "r"(__as_b32(__values[105])),
      "r"(__as_b32(__values[106])),
      "r"(__as_b32(__values[107])),
      "r"(__as_b32(__values[108])),
      "r"(__as_b32(__values[109])),
      "r"(__as_b32(__values[110])),
      "r"(__as_b32(__values[111])),
      "r"(__as_b32(__values[112])),
      "r"(__as_b32(__values[113])),
      "r"(__as_b32(__values[114])),
      "r"(__as_b32(__values[115])),
      "r"(__as_b32(__values[116])),
      "r"(__as_b32(__values[117])),
      "r"(__as_b32(__values[118])),
      "r"(__as_b32(__values[119])),
      "r"(__as_b32(__values[120])),
      "r"(__as_b32(__values[121])),
      "r"(__as_b32(__values[122])),
      "r"(__as_b32(__values[123])),
      "r"(__as_b32(__values[124])),
      "r"(__as_b32(__values[125])),
      "r"(__as_b32(__values[126])),
      "r"(__as_b32(__values[127]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x128b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x256b.x1.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x256b(
  uint32_t taddr,
  const B32 (&values)[4]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x256b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x256b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[4])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x256b.x1.b32 [%0], {%1, %2, %3, %4};"
      :
      : "r"(__taddr),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x256b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x256b.x1.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x256b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[4]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x256b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x256b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[4])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x256b.x1.unpack::16b.b32 [%0], {%1, %2, %3, %4};"
      :
      : "r"(__taddr),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x256b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x256b.x2.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x256b(
  uint32_t taddr,
  const B32 (&values)[8]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x256b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x256b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[8])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x256b.x2.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
      :
      : "r"(__taddr),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3])),
        "r"(__as_b32(__values[4])),
        "r"(__as_b32(__values[5])),
        "r"(__as_b32(__values[6])),
        "r"(__as_b32(__values[7]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x256b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x256b.x2.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x256b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[8]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x256b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x256b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[8])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x256b.x2.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
      :
      : "r"(__taddr),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3])),
        "r"(__as_b32(__values[4])),
        "r"(__as_b32(__values[5])),
        "r"(__as_b32(__values[6])),
        "r"(__as_b32(__values[7]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x256b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x256b.x4.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x256b(
  uint32_t taddr,
  const B32 (&values)[16]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x256b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x256b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[16])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x256b.x4.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16};"
      :
      : "r"(__taddr),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3])),
        "r"(__as_b32(__values[4])),
        "r"(__as_b32(__values[5])),
        "r"(__as_b32(__values[6])),
        "r"(__as_b32(__values[7])),
        "r"(__as_b32(__values[8])),
        "r"(__as_b32(__values[9])),
        "r"(__as_b32(__values[10])),
        "r"(__as_b32(__values[11])),
        "r"(__as_b32(__values[12])),
        "r"(__as_b32(__values[13])),
        "r"(__as_b32(__values[14])),
        "r"(__as_b32(__values[15]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x256b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x256b.x4.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x256b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[16]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x256b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x256b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[16])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x256b.x4.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
      "%13, %14, %15, %16};"
      :
      : "r"(__taddr),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3])),
        "r"(__as_b32(__values[4])),
        "r"(__as_b32(__values[5])),
        "r"(__as_b32(__values[6])),
        "r"(__as_b32(__values[7])),
        "r"(__as_b32(__values[8])),
        "r"(__as_b32(__values[9])),
        "r"(__as_b32(__values[10])),
        "r"(__as_b32(__values[11])),
        "r"(__as_b32(__values[12])),
        "r"(__as_b32(__values[13])),
        "r"(__as_b32(__values[14])),
        "r"(__as_b32(__values[15]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x256b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x256b.x8.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x256b(
  uint32_t taddr,
  const B32 (&values)[32]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x256b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x256b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[32])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.16x256b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
    :
    : "r"(__taddr),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x256b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x256b.x8.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x256b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[32]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x256b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x256b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[32])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.16x256b.x8.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
    :
    : "r"(__taddr),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x256b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x256b.x16.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x256b(
  uint32_t taddr,
  const B32 (&values)[64]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x256b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x256b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[64])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.16x256b.x16.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64};"
    :
    : "r"(__taddr),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31])),
      "r"(__as_b32(__values[32])),
      "r"(__as_b32(__values[33])),
      "r"(__as_b32(__values[34])),
      "r"(__as_b32(__values[35])),
      "r"(__as_b32(__values[36])),
      "r"(__as_b32(__values[37])),
      "r"(__as_b32(__values[38])),
      "r"(__as_b32(__values[39])),
      "r"(__as_b32(__values[40])),
      "r"(__as_b32(__values[41])),
      "r"(__as_b32(__values[42])),
      "r"(__as_b32(__values[43])),
      "r"(__as_b32(__values[44])),
      "r"(__as_b32(__values[45])),
      "r"(__as_b32(__values[46])),
      "r"(__as_b32(__values[47])),
      "r"(__as_b32(__values[48])),
      "r"(__as_b32(__values[49])),
      "r"(__as_b32(__values[50])),
      "r"(__as_b32(__values[51])),
      "r"(__as_b32(__values[52])),
      "r"(__as_b32(__values[53])),
      "r"(__as_b32(__values[54])),
      "r"(__as_b32(__values[55])),
      "r"(__as_b32(__values[56])),
      "r"(__as_b32(__values[57])),
      "r"(__as_b32(__values[58])),
      "r"(__as_b32(__values[59])),
      "r"(__as_b32(__values[60])),
      "r"(__as_b32(__values[61])),
      "r"(__as_b32(__values[62])),
      "r"(__as_b32(__values[63]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x256b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x256b.x16.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x256b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[64]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x256b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x256b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[64])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.16x256b.x16.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
    "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, "
    "%35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
    "%57, %58, %59, %60, %61, %62, %63, %64};"
    :
    : "r"(__taddr),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31])),
      "r"(__as_b32(__values[32])),
      "r"(__as_b32(__values[33])),
      "r"(__as_b32(__values[34])),
      "r"(__as_b32(__values[35])),
      "r"(__as_b32(__values[36])),
      "r"(__as_b32(__values[37])),
      "r"(__as_b32(__values[38])),
      "r"(__as_b32(__values[39])),
      "r"(__as_b32(__values[40])),
      "r"(__as_b32(__values[41])),
      "r"(__as_b32(__values[42])),
      "r"(__as_b32(__values[43])),
      "r"(__as_b32(__values[44])),
      "r"(__as_b32(__values[45])),
      "r"(__as_b32(__values[46])),
      "r"(__as_b32(__values[47])),
      "r"(__as_b32(__values[48])),
      "r"(__as_b32(__values[49])),
      "r"(__as_b32(__values[50])),
      "r"(__as_b32(__values[51])),
      "r"(__as_b32(__values[52])),
      "r"(__as_b32(__values[53])),
      "r"(__as_b32(__values[54])),
      "r"(__as_b32(__values[55])),
      "r"(__as_b32(__values[56])),
      "r"(__as_b32(__values[57])),
      "r"(__as_b32(__values[58])),
      "r"(__as_b32(__values[59])),
      "r"(__as_b32(__values[60])),
      "r"(__as_b32(__values[61])),
      "r"(__as_b32(__values[62])),
      "r"(__as_b32(__values[63]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x256b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x256b.x32.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x256b(
  uint32_t taddr,
  const B32 (&values)[128]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x256b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x256b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[128])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.16x256b.x32.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, "
    "%82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, "
    "%104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
    "%123, %124, %125, %126, %127, %128};"
    :
    : "r"(__taddr),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31])),
      "r"(__as_b32(__values[32])),
      "r"(__as_b32(__values[33])),
      "r"(__as_b32(__values[34])),
      "r"(__as_b32(__values[35])),
      "r"(__as_b32(__values[36])),
      "r"(__as_b32(__values[37])),
      "r"(__as_b32(__values[38])),
      "r"(__as_b32(__values[39])),
      "r"(__as_b32(__values[40])),
      "r"(__as_b32(__values[41])),
      "r"(__as_b32(__values[42])),
      "r"(__as_b32(__values[43])),
      "r"(__as_b32(__values[44])),
      "r"(__as_b32(__values[45])),
      "r"(__as_b32(__values[46])),
      "r"(__as_b32(__values[47])),
      "r"(__as_b32(__values[48])),
      "r"(__as_b32(__values[49])),
      "r"(__as_b32(__values[50])),
      "r"(__as_b32(__values[51])),
      "r"(__as_b32(__values[52])),
      "r"(__as_b32(__values[53])),
      "r"(__as_b32(__values[54])),
      "r"(__as_b32(__values[55])),
      "r"(__as_b32(__values[56])),
      "r"(__as_b32(__values[57])),
      "r"(__as_b32(__values[58])),
      "r"(__as_b32(__values[59])),
      "r"(__as_b32(__values[60])),
      "r"(__as_b32(__values[61])),
      "r"(__as_b32(__values[62])),
      "r"(__as_b32(__values[63])),
      "r"(__as_b32(__values[64])),
      "r"(__as_b32(__values[65])),
      "r"(__as_b32(__values[66])),
      "r"(__as_b32(__values[67])),
      "r"(__as_b32(__values[68])),
      "r"(__as_b32(__values[69])),
      "r"(__as_b32(__values[70])),
      "r"(__as_b32(__values[71])),
      "r"(__as_b32(__values[72])),
      "r"(__as_b32(__values[73])),
      "r"(__as_b32(__values[74])),
      "r"(__as_b32(__values[75])),
      "r"(__as_b32(__values[76])),
      "r"(__as_b32(__values[77])),
      "r"(__as_b32(__values[78])),
      "r"(__as_b32(__values[79])),
      "r"(__as_b32(__values[80])),
      "r"(__as_b32(__values[81])),
      "r"(__as_b32(__values[82])),
      "r"(__as_b32(__values[83])),
      "r"(__as_b32(__values[84])),
      "r"(__as_b32(__values[85])),
      "r"(__as_b32(__values[86])),
      "r"(__as_b32(__values[87])),
      "r"(__as_b32(__values[88])),
      "r"(__as_b32(__values[89])),
      "r"(__as_b32(__values[90])),
      "r"(__as_b32(__values[91])),
      "r"(__as_b32(__values[92])),
      "r"(__as_b32(__values[93])),
      "r"(__as_b32(__values[94])),
      "r"(__as_b32(__values[95])),
      "r"(__as_b32(__values[96])),
      "r"(__as_b32(__values[97])),
      "r"(__as_b32(__values[98])),
      "r"(__as_b32(__values[99])),
      "r"(__as_b32(__values[100])),
      "r"(__as_b32(__values[101])),
      "r"(__as_b32(__values[102])),
      "r"(__as_b32(__values[103])),
      "r"(__as_b32(__values[104])),
      "r"(__as_b32(__values[105])),
      "r"(__as_b32(__values[106])),
      "r"(__as_b32(__values[107])),
      "r"(__as_b32(__values[108])),
      "r"(__as_b32(__values[109])),
      "r"(__as_b32(__values[110])),
      "r"(__as_b32(__values[111])),
      "r"(__as_b32(__values[112])),
      "r"(__as_b32(__values[113])),
      "r"(__as_b32(__values[114])),
      "r"(__as_b32(__values[115])),
      "r"(__as_b32(__values[116])),
      "r"(__as_b32(__values[117])),
      "r"(__as_b32(__values[118])),
      "r"(__as_b32(__values[119])),
      "r"(__as_b32(__values[120])),
      "r"(__as_b32(__values[121])),
      "r"(__as_b32(__values[122])),
      "r"(__as_b32(__values[123])),
      "r"(__as_b32(__values[124])),
      "r"(__as_b32(__values[125])),
      "r"(__as_b32(__values[126])),
      "r"(__as_b32(__values[127]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x256b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x256b.x32.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x256b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[128]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x256b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_16x256b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[128])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.16x256b.x32.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
    "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, "
    "%35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
    "%57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, "
    "%79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, "
    "%101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, "
    "%120, %121, %122, %123, %124, %125, %126, %127, %128};"
    :
    : "r"(__taddr),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31])),
      "r"(__as_b32(__values[32])),
      "r"(__as_b32(__values[33])),
      "r"(__as_b32(__values[34])),
      "r"(__as_b32(__values[35])),
      "r"(__as_b32(__values[36])),
      "r"(__as_b32(__values[37])),
      "r"(__as_b32(__values[38])),
      "r"(__as_b32(__values[39])),
      "r"(__as_b32(__values[40])),
      "r"(__as_b32(__values[41])),
      "r"(__as_b32(__values[42])),
      "r"(__as_b32(__values[43])),
      "r"(__as_b32(__values[44])),
      "r"(__as_b32(__values[45])),
      "r"(__as_b32(__values[46])),
      "r"(__as_b32(__values[47])),
      "r"(__as_b32(__values[48])),
      "r"(__as_b32(__values[49])),
      "r"(__as_b32(__values[50])),
      "r"(__as_b32(__values[51])),
      "r"(__as_b32(__values[52])),
      "r"(__as_b32(__values[53])),
      "r"(__as_b32(__values[54])),
      "r"(__as_b32(__values[55])),
      "r"(__as_b32(__values[56])),
      "r"(__as_b32(__values[57])),
      "r"(__as_b32(__values[58])),
      "r"(__as_b32(__values[59])),
      "r"(__as_b32(__values[60])),
      "r"(__as_b32(__values[61])),
      "r"(__as_b32(__values[62])),
      "r"(__as_b32(__values[63])),
      "r"(__as_b32(__values[64])),
      "r"(__as_b32(__values[65])),
      "r"(__as_b32(__values[66])),
      "r"(__as_b32(__values[67])),
      "r"(__as_b32(__values[68])),
      "r"(__as_b32(__values[69])),
      "r"(__as_b32(__values[70])),
      "r"(__as_b32(__values[71])),
      "r"(__as_b32(__values[72])),
      "r"(__as_b32(__values[73])),
      "r"(__as_b32(__values[74])),
      "r"(__as_b32(__values[75])),
      "r"(__as_b32(__values[76])),
      "r"(__as_b32(__values[77])),
      "r"(__as_b32(__values[78])),
      "r"(__as_b32(__values[79])),
      "r"(__as_b32(__values[80])),
      "r"(__as_b32(__values[81])),
      "r"(__as_b32(__values[82])),
      "r"(__as_b32(__values[83])),
      "r"(__as_b32(__values[84])),
      "r"(__as_b32(__values[85])),
      "r"(__as_b32(__values[86])),
      "r"(__as_b32(__values[87])),
      "r"(__as_b32(__values[88])),
      "r"(__as_b32(__values[89])),
      "r"(__as_b32(__values[90])),
      "r"(__as_b32(__values[91])),
      "r"(__as_b32(__values[92])),
      "r"(__as_b32(__values[93])),
      "r"(__as_b32(__values[94])),
      "r"(__as_b32(__values[95])),
      "r"(__as_b32(__values[96])),
      "r"(__as_b32(__values[97])),
      "r"(__as_b32(__values[98])),
      "r"(__as_b32(__values[99])),
      "r"(__as_b32(__values[100])),
      "r"(__as_b32(__values[101])),
      "r"(__as_b32(__values[102])),
      "r"(__as_b32(__values[103])),
      "r"(__as_b32(__values[104])),
      "r"(__as_b32(__values[105])),
      "r"(__as_b32(__values[106])),
      "r"(__as_b32(__values[107])),
      "r"(__as_b32(__values[108])),
      "r"(__as_b32(__values[109])),
      "r"(__as_b32(__values[110])),
      "r"(__as_b32(__values[111])),
      "r"(__as_b32(__values[112])),
      "r"(__as_b32(__values[113])),
      "r"(__as_b32(__values[114])),
      "r"(__as_b32(__values[115])),
      "r"(__as_b32(__values[116])),
      "r"(__as_b32(__values[117])),
      "r"(__as_b32(__values[118])),
      "r"(__as_b32(__values[119])),
      "r"(__as_b32(__values[120])),
      "r"(__as_b32(__values[121])),
      "r"(__as_b32(__values[122])),
      "r"(__as_b32(__values[123])),
      "r"(__as_b32(__values[124])),
      "r"(__as_b32(__values[125])),
      "r"(__as_b32(__values[126])),
      "r"(__as_b32(__values[127]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x256b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x1.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b(
  uint32_t taddr,
  const B32 (&values)[1]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_32x32b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[1])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" : : "r"(__taddr), "r"(__as_b32(__values[0])) : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x1.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[1]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[1])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.32x32b.x1.unpack::16b.b32 [%0], {%1};"
      :
      : "r"(__taddr), "r"(__as_b32(__values[0]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x2.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b(
  uint32_t taddr,
  const B32 (&values)[2]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_32x32b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[2])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.32x32b.x2.b32 [%0], {%1, %2};"
      :
      : "r"(__taddr), "r"(__as_b32(__values[0])), "r"(__as_b32(__values[1]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x2.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[2]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[2])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.32x32b.x2.unpack::16b.b32 [%0], {%1, %2};"
      :
      : "r"(__taddr), "r"(__as_b32(__values[0])), "r"(__as_b32(__values[1]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x4.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b(
  uint32_t taddr,
  const B32 (&values)[4]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_32x32b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[4])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};"
      :
      : "r"(__taddr),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x4.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[4]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[4])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.32x32b.x4.unpack::16b.b32 [%0], {%1, %2, %3, %4};"
      :
      : "r"(__taddr),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x8.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b(
  uint32_t taddr,
  const B32 (&values)[8]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_32x32b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[8])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
      :
      : "r"(__taddr),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3])),
        "r"(__as_b32(__values[4])),
        "r"(__as_b32(__values[5])),
        "r"(__as_b32(__values[6])),
        "r"(__as_b32(__values[7]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x8.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[8]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[8])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.32x32b.x8.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
      :
      : "r"(__taddr),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3])),
        "r"(__as_b32(__values[4])),
        "r"(__as_b32(__values[5])),
        "r"(__as_b32(__values[6])),
        "r"(__as_b32(__values[7]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x16.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b(
  uint32_t taddr,
  const B32 (&values)[16]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_32x32b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[16])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16};"
      :
      : "r"(__taddr),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3])),
        "r"(__as_b32(__values[4])),
        "r"(__as_b32(__values[5])),
        "r"(__as_b32(__values[6])),
        "r"(__as_b32(__values[7])),
        "r"(__as_b32(__values[8])),
        "r"(__as_b32(__values[9])),
        "r"(__as_b32(__values[10])),
        "r"(__as_b32(__values[11])),
        "r"(__as_b32(__values[12])),
        "r"(__as_b32(__values[13])),
        "r"(__as_b32(__values[14])),
        "r"(__as_b32(__values[15]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x16.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[16]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[16])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.32x32b.x16.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
      "%13, %14, %15, %16};"
      :
      : "r"(__taddr),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3])),
        "r"(__as_b32(__values[4])),
        "r"(__as_b32(__values[5])),
        "r"(__as_b32(__values[6])),
        "r"(__as_b32(__values[7])),
        "r"(__as_b32(__values[8])),
        "r"(__as_b32(__values[9])),
        "r"(__as_b32(__values[10])),
        "r"(__as_b32(__values[11])),
        "r"(__as_b32(__values[12])),
        "r"(__as_b32(__values[13])),
        "r"(__as_b32(__values[14])),
        "r"(__as_b32(__values[15]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x32.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b(
  uint32_t taddr,
  const B32 (&values)[32]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_32x32b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[32])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
    :
    : "r"(__taddr),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x32.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[32]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[32])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.32x32b.x32.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
    :
    : "r"(__taddr),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x64.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b(
  uint32_t taddr,
  const B32 (&values)[64]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_32x32b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[64])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.32x32b.x64.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64};"
    :
    : "r"(__taddr),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31])),
      "r"(__as_b32(__values[32])),
      "r"(__as_b32(__values[33])),
      "r"(__as_b32(__values[34])),
      "r"(__as_b32(__values[35])),
      "r"(__as_b32(__values[36])),
      "r"(__as_b32(__values[37])),
      "r"(__as_b32(__values[38])),
      "r"(__as_b32(__values[39])),
      "r"(__as_b32(__values[40])),
      "r"(__as_b32(__values[41])),
      "r"(__as_b32(__values[42])),
      "r"(__as_b32(__values[43])),
      "r"(__as_b32(__values[44])),
      "r"(__as_b32(__values[45])),
      "r"(__as_b32(__values[46])),
      "r"(__as_b32(__values[47])),
      "r"(__as_b32(__values[48])),
      "r"(__as_b32(__values[49])),
      "r"(__as_b32(__values[50])),
      "r"(__as_b32(__values[51])),
      "r"(__as_b32(__values[52])),
      "r"(__as_b32(__values[53])),
      "r"(__as_b32(__values[54])),
      "r"(__as_b32(__values[55])),
      "r"(__as_b32(__values[56])),
      "r"(__as_b32(__values[57])),
      "r"(__as_b32(__values[58])),
      "r"(__as_b32(__values[59])),
      "r"(__as_b32(__values[60])),
      "r"(__as_b32(__values[61])),
      "r"(__as_b32(__values[62])),
      "r"(__as_b32(__values[63]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x64.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[64]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[64])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.32x32b.x64.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63, %64};"
    :
    : "r"(__taddr),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31])),
      "r"(__as_b32(__values[32])),
      "r"(__as_b32(__values[33])),
      "r"(__as_b32(__values[34])),
      "r"(__as_b32(__values[35])),
      "r"(__as_b32(__values[36])),
      "r"(__as_b32(__values[37])),
      "r"(__as_b32(__values[38])),
      "r"(__as_b32(__values[39])),
      "r"(__as_b32(__values[40])),
      "r"(__as_b32(__values[41])),
      "r"(__as_b32(__values[42])),
      "r"(__as_b32(__values[43])),
      "r"(__as_b32(__values[44])),
      "r"(__as_b32(__values[45])),
      "r"(__as_b32(__values[46])),
      "r"(__as_b32(__values[47])),
      "r"(__as_b32(__values[48])),
      "r"(__as_b32(__values[49])),
      "r"(__as_b32(__values[50])),
      "r"(__as_b32(__values[51])),
      "r"(__as_b32(__values[52])),
      "r"(__as_b32(__values[53])),
      "r"(__as_b32(__values[54])),
      "r"(__as_b32(__values[55])),
      "r"(__as_b32(__values[56])),
      "r"(__as_b32(__values[57])),
      "r"(__as_b32(__values[58])),
      "r"(__as_b32(__values[59])),
      "r"(__as_b32(__values[60])),
      "r"(__as_b32(__values[61])),
      "r"(__as_b32(__values[62])),
      "r"(__as_b32(__values[63]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x128.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b(
  uint32_t taddr,
  const B32 (&values)[128]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_32x32b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[128])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.32x32b.x128.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, "
    "%82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, "
    "%104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
    "%123, %124, %125, %126, %127, %128};"
    :
    : "r"(__taddr),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31])),
      "r"(__as_b32(__values[32])),
      "r"(__as_b32(__values[33])),
      "r"(__as_b32(__values[34])),
      "r"(__as_b32(__values[35])),
      "r"(__as_b32(__values[36])),
      "r"(__as_b32(__values[37])),
      "r"(__as_b32(__values[38])),
      "r"(__as_b32(__values[39])),
      "r"(__as_b32(__values[40])),
      "r"(__as_b32(__values[41])),
      "r"(__as_b32(__values[42])),
      "r"(__as_b32(__values[43])),
      "r"(__as_b32(__values[44])),
      "r"(__as_b32(__values[45])),
      "r"(__as_b32(__values[46])),
      "r"(__as_b32(__values[47])),
      "r"(__as_b32(__values[48])),
      "r"(__as_b32(__values[49])),
      "r"(__as_b32(__values[50])),
      "r"(__as_b32(__values[51])),
      "r"(__as_b32(__values[52])),
      "r"(__as_b32(__values[53])),
      "r"(__as_b32(__values[54])),
      "r"(__as_b32(__values[55])),
      "r"(__as_b32(__values[56])),
      "r"(__as_b32(__values[57])),
      "r"(__as_b32(__values[58])),
      "r"(__as_b32(__values[59])),
      "r"(__as_b32(__values[60])),
      "r"(__as_b32(__values[61])),
      "r"(__as_b32(__values[62])),
      "r"(__as_b32(__values[63])),
      "r"(__as_b32(__values[64])),
      "r"(__as_b32(__values[65])),
      "r"(__as_b32(__values[66])),
      "r"(__as_b32(__values[67])),
      "r"(__as_b32(__values[68])),
      "r"(__as_b32(__values[69])),
      "r"(__as_b32(__values[70])),
      "r"(__as_b32(__values[71])),
      "r"(__as_b32(__values[72])),
      "r"(__as_b32(__values[73])),
      "r"(__as_b32(__values[74])),
      "r"(__as_b32(__values[75])),
      "r"(__as_b32(__values[76])),
      "r"(__as_b32(__values[77])),
      "r"(__as_b32(__values[78])),
      "r"(__as_b32(__values[79])),
      "r"(__as_b32(__values[80])),
      "r"(__as_b32(__values[81])),
      "r"(__as_b32(__values[82])),
      "r"(__as_b32(__values[83])),
      "r"(__as_b32(__values[84])),
      "r"(__as_b32(__values[85])),
      "r"(__as_b32(__values[86])),
      "r"(__as_b32(__values[87])),
      "r"(__as_b32(__values[88])),
      "r"(__as_b32(__values[89])),
      "r"(__as_b32(__values[90])),
      "r"(__as_b32(__values[91])),
      "r"(__as_b32(__values[92])),
      "r"(__as_b32(__values[93])),
      "r"(__as_b32(__values[94])),
      "r"(__as_b32(__values[95])),
      "r"(__as_b32(__values[96])),
      "r"(__as_b32(__values[97])),
      "r"(__as_b32(__values[98])),
      "r"(__as_b32(__values[99])),
      "r"(__as_b32(__values[100])),
      "r"(__as_b32(__values[101])),
      "r"(__as_b32(__values[102])),
      "r"(__as_b32(__values[103])),
      "r"(__as_b32(__values[104])),
      "r"(__as_b32(__values[105])),
      "r"(__as_b32(__values[106])),
      "r"(__as_b32(__values[107])),
      "r"(__as_b32(__values[108])),
      "r"(__as_b32(__values[109])),
      "r"(__as_b32(__values[110])),
      "r"(__as_b32(__values[111])),
      "r"(__as_b32(__values[112])),
      "r"(__as_b32(__values[113])),
      "r"(__as_b32(__values[114])),
      "r"(__as_b32(__values[115])),
      "r"(__as_b32(__values[116])),
      "r"(__as_b32(__values[117])),
      "r"(__as_b32(__values[118])),
      "r"(__as_b32(__values[119])),
      "r"(__as_b32(__values[120])),
      "r"(__as_b32(__values[121])),
      "r"(__as_b32(__values[122])),
      "r"(__as_b32(__values[123])),
      "r"(__as_b32(__values[124])),
      "r"(__as_b32(__values[125])),
      "r"(__as_b32(__values[126])),
      "r"(__as_b32(__values[127]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.32x32b.x128.unpack::16b.b32 [taddr], values; // PTX ISA 86, SM_100a, SM_101a
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_32x32b_unpack_16b(
  uint32_t taddr,
  const B32 (&values)[128]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_st_32x32b_unpack_16b(_CUDA_VSTD::uint32_t __taddr, const _B32 (&__values)[128])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.32x32b.x128.unpack::16b.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
    "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, "
    "%35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
    "%57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, "
    "%79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, "
    "%101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, "
    "%120, %121, %122, %123, %124, %125, %126, %127, %128};"
    :
    : "r"(__taddr),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31])),
      "r"(__as_b32(__values[32])),
      "r"(__as_b32(__values[33])),
      "r"(__as_b32(__values[34])),
      "r"(__as_b32(__values[35])),
      "r"(__as_b32(__values[36])),
      "r"(__as_b32(__values[37])),
      "r"(__as_b32(__values[38])),
      "r"(__as_b32(__values[39])),
      "r"(__as_b32(__values[40])),
      "r"(__as_b32(__values[41])),
      "r"(__as_b32(__values[42])),
      "r"(__as_b32(__values[43])),
      "r"(__as_b32(__values[44])),
      "r"(__as_b32(__values[45])),
      "r"(__as_b32(__values[46])),
      "r"(__as_b32(__values[47])),
      "r"(__as_b32(__values[48])),
      "r"(__as_b32(__values[49])),
      "r"(__as_b32(__values[50])),
      "r"(__as_b32(__values[51])),
      "r"(__as_b32(__values[52])),
      "r"(__as_b32(__values[53])),
      "r"(__as_b32(__values[54])),
      "r"(__as_b32(__values[55])),
      "r"(__as_b32(__values[56])),
      "r"(__as_b32(__values[57])),
      "r"(__as_b32(__values[58])),
      "r"(__as_b32(__values[59])),
      "r"(__as_b32(__values[60])),
      "r"(__as_b32(__values[61])),
      "r"(__as_b32(__values[62])),
      "r"(__as_b32(__values[63])),
      "r"(__as_b32(__values[64])),
      "r"(__as_b32(__values[65])),
      "r"(__as_b32(__values[66])),
      "r"(__as_b32(__values[67])),
      "r"(__as_b32(__values[68])),
      "r"(__as_b32(__values[69])),
      "r"(__as_b32(__values[70])),
      "r"(__as_b32(__values[71])),
      "r"(__as_b32(__values[72])),
      "r"(__as_b32(__values[73])),
      "r"(__as_b32(__values[74])),
      "r"(__as_b32(__values[75])),
      "r"(__as_b32(__values[76])),
      "r"(__as_b32(__values[77])),
      "r"(__as_b32(__values[78])),
      "r"(__as_b32(__values[79])),
      "r"(__as_b32(__values[80])),
      "r"(__as_b32(__values[81])),
      "r"(__as_b32(__values[82])),
      "r"(__as_b32(__values[83])),
      "r"(__as_b32(__values[84])),
      "r"(__as_b32(__values[85])),
      "r"(__as_b32(__values[86])),
      "r"(__as_b32(__values[87])),
      "r"(__as_b32(__values[88])),
      "r"(__as_b32(__values[89])),
      "r"(__as_b32(__values[90])),
      "r"(__as_b32(__values[91])),
      "r"(__as_b32(__values[92])),
      "r"(__as_b32(__values[93])),
      "r"(__as_b32(__values[94])),
      "r"(__as_b32(__values[95])),
      "r"(__as_b32(__values[96])),
      "r"(__as_b32(__values[97])),
      "r"(__as_b32(__values[98])),
      "r"(__as_b32(__values[99])),
      "r"(__as_b32(__values[100])),
      "r"(__as_b32(__values[101])),
      "r"(__as_b32(__values[102])),
      "r"(__as_b32(__values[103])),
      "r"(__as_b32(__values[104])),
      "r"(__as_b32(__values[105])),
      "r"(__as_b32(__values[106])),
      "r"(__as_b32(__values[107])),
      "r"(__as_b32(__values[108])),
      "r"(__as_b32(__values[109])),
      "r"(__as_b32(__values[110])),
      "r"(__as_b32(__values[111])),
      "r"(__as_b32(__values[112])),
      "r"(__as_b32(__values[113])),
      "r"(__as_b32(__values[114])),
      "r"(__as_b32(__values[115])),
      "r"(__as_b32(__values[116])),
      "r"(__as_b32(__values[117])),
      "r"(__as_b32(__values[118])),
      "r"(__as_b32(__values[119])),
      "r"(__as_b32(__values[120])),
      "r"(__as_b32(__values[121])),
      "r"(__as_b32(__values[122])),
      "r"(__as_b32(__values[123])),
      "r"(__as_b32(__values[124])),
      "r"(__as_b32(__values[125])),
      "r"(__as_b32(__values[126])),
      "r"(__as_b32(__values[127]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_32x32b_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x1.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a, SM_101a
template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[1]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x32bx2_is_not_supported_before_SM_100a_SM_101a__();
template <int _N32, typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void
tcgen05_st_16x32bx2(_CUDA_VSTD::uint32_t __taddr, n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[1])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x32bx2.x1.b32 [%0], %1, {%2};"
      :
      : "r"(__taddr), "n"(__immHalfSplitoff.value), "r"(__as_b32(__values[0]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x1.unpack::16b.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a, SM_101a
template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2_unpack_16b(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[1]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <int _N32, typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void
tcgen05_st_16x32bx2_unpack_16b(_CUDA_VSTD::uint32_t __taddr, n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[1])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x32bx2.x1.unpack::16b.b32 [%0], %1, {%2};"
      :
      : "r"(__taddr), "n"(__immHalfSplitoff.value), "r"(__as_b32(__values[0]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x2.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a, SM_101a
template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[2]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x32bx2_is_not_supported_before_SM_100a_SM_101a__();
template <int _N32, typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void
tcgen05_st_16x32bx2(_CUDA_VSTD::uint32_t __taddr, n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[2])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x32bx2.x2.b32 [%0], %1, {%2, %3};"
      :
      : "r"(__taddr), "n"(__immHalfSplitoff.value), "r"(__as_b32(__values[0])), "r"(__as_b32(__values[1]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x2.unpack::16b.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a, SM_101a
template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2_unpack_16b(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[2]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <int _N32, typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void
tcgen05_st_16x32bx2_unpack_16b(_CUDA_VSTD::uint32_t __taddr, n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[2])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x32bx2.x2.unpack::16b.b32 [%0], %1, {%2, %3};"
      :
      : "r"(__taddr), "n"(__immHalfSplitoff.value), "r"(__as_b32(__values[0])), "r"(__as_b32(__values[1]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x4.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a, SM_101a
template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[4]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x32bx2_is_not_supported_before_SM_100a_SM_101a__();
template <int _N32, typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void
tcgen05_st_16x32bx2(_CUDA_VSTD::uint32_t __taddr, n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[4])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x32bx2.x4.b32 [%0], %1, {%2, %3, %4, %5};"
      :
      : "r"(__taddr),
        "n"(__immHalfSplitoff.value),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x4.unpack::16b.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a, SM_101a
template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2_unpack_16b(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[4]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <int _N32, typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void
tcgen05_st_16x32bx2_unpack_16b(_CUDA_VSTD::uint32_t __taddr, n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[4])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x32bx2.x4.unpack::16b.b32 [%0], %1, {%2, %3, %4, %5};"
      :
      : "r"(__taddr),
        "n"(__immHalfSplitoff.value),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x8.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a, SM_101a
template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[8]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x32bx2_is_not_supported_before_SM_100a_SM_101a__();
template <int _N32, typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void
tcgen05_st_16x32bx2(_CUDA_VSTD::uint32_t __taddr, n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[8])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x32bx2.x8.b32 [%0], %1, {%2, %3, %4, %5, %6, %7, %8, %9};"
      :
      : "r"(__taddr),
        "n"(__immHalfSplitoff.value),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3])),
        "r"(__as_b32(__values[4])),
        "r"(__as_b32(__values[5])),
        "r"(__as_b32(__values[6])),
        "r"(__as_b32(__values[7]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x8.unpack::16b.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a, SM_101a
template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2_unpack_16b(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[8]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <int _N32, typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void
tcgen05_st_16x32bx2_unpack_16b(_CUDA_VSTD::uint32_t __taddr, n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[8])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x32bx2.x8.unpack::16b.b32 [%0], %1, {%2, %3, %4, %5, %6, %7, %8, %9};"
      :
      : "r"(__taddr),
        "n"(__immHalfSplitoff.value),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3])),
        "r"(__as_b32(__values[4])),
        "r"(__as_b32(__values[5])),
        "r"(__as_b32(__values[6])),
        "r"(__as_b32(__values[7]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x16.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a, SM_101a
template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[16]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x32bx2_is_not_supported_before_SM_100a_SM_101a__();
template <int _N32, typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void
tcgen05_st_16x32bx2(_CUDA_VSTD::uint32_t __taddr, n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[16])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x32bx2.x16.b32 [%0], %1, {%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
      "%15, %16, %17};"
      :
      : "r"(__taddr),
        "n"(__immHalfSplitoff.value),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3])),
        "r"(__as_b32(__values[4])),
        "r"(__as_b32(__values[5])),
        "r"(__as_b32(__values[6])),
        "r"(__as_b32(__values[7])),
        "r"(__as_b32(__values[8])),
        "r"(__as_b32(__values[9])),
        "r"(__as_b32(__values[10])),
        "r"(__as_b32(__values[11])),
        "r"(__as_b32(__values[12])),
        "r"(__as_b32(__values[13])),
        "r"(__as_b32(__values[14])),
        "r"(__as_b32(__values[15]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x16.unpack::16b.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a,
SM_101a template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2_unpack_16b(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[16]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <int _N32, typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void
tcgen05_st_16x32bx2_unpack_16b(_CUDA_VSTD::uint32_t __taddr, n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[16])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm("tcgen05.st.sync.aligned.16x32bx2.x16.unpack::16b.b32 [%0], %1, {%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
      "%13, %14, %15, %16, %17};"
      :
      : "r"(__taddr),
        "n"(__immHalfSplitoff.value),
        "r"(__as_b32(__values[0])),
        "r"(__as_b32(__values[1])),
        "r"(__as_b32(__values[2])),
        "r"(__as_b32(__values[3])),
        "r"(__as_b32(__values[4])),
        "r"(__as_b32(__values[5])),
        "r"(__as_b32(__values[6])),
        "r"(__as_b32(__values[7])),
        "r"(__as_b32(__values[8])),
        "r"(__as_b32(__values[9])),
        "r"(__as_b32(__values[10])),
        "r"(__as_b32(__values[11])),
        "r"(__as_b32(__values[12])),
        "r"(__as_b32(__values[13])),
        "r"(__as_b32(__values[14])),
        "r"(__as_b32(__values[15]))
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x32.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a, SM_101a
template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[32]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x32bx2_is_not_supported_before_SM_100a_SM_101a__();
template <int _N32, typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void
tcgen05_st_16x32bx2(_CUDA_VSTD::uint32_t __taddr, n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[32])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.16x32bx2.x32.b32 [%0], %1, {%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33};"
    :
    : "r"(__taddr),
      "n"(__immHalfSplitoff.value),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x32.unpack::16b.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a,
SM_101a template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2_unpack_16b(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[32]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <int _N32, typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void
tcgen05_st_16x32bx2_unpack_16b(_CUDA_VSTD::uint32_t __taddr, n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[32])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.16x32bx2.x32.unpack::16b.b32 [%0], %1, {%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
    "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33};"
    :
    : "r"(__taddr),
      "n"(__immHalfSplitoff.value),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x64.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a, SM_101a
template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[64]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x32bx2_is_not_supported_before_SM_100a_SM_101a__();
template <int _N32, typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void
tcgen05_st_16x32bx2(_CUDA_VSTD::uint32_t __taddr, n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[64])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.16x32bx2.x64.b32 [%0], %1, {%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64, %65};"
    :
    : "r"(__taddr),
      "n"(__immHalfSplitoff.value),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31])),
      "r"(__as_b32(__values[32])),
      "r"(__as_b32(__values[33])),
      "r"(__as_b32(__values[34])),
      "r"(__as_b32(__values[35])),
      "r"(__as_b32(__values[36])),
      "r"(__as_b32(__values[37])),
      "r"(__as_b32(__values[38])),
      "r"(__as_b32(__values[39])),
      "r"(__as_b32(__values[40])),
      "r"(__as_b32(__values[41])),
      "r"(__as_b32(__values[42])),
      "r"(__as_b32(__values[43])),
      "r"(__as_b32(__values[44])),
      "r"(__as_b32(__values[45])),
      "r"(__as_b32(__values[46])),
      "r"(__as_b32(__values[47])),
      "r"(__as_b32(__values[48])),
      "r"(__as_b32(__values[49])),
      "r"(__as_b32(__values[50])),
      "r"(__as_b32(__values[51])),
      "r"(__as_b32(__values[52])),
      "r"(__as_b32(__values[53])),
      "r"(__as_b32(__values[54])),
      "r"(__as_b32(__values[55])),
      "r"(__as_b32(__values[56])),
      "r"(__as_b32(__values[57])),
      "r"(__as_b32(__values[58])),
      "r"(__as_b32(__values[59])),
      "r"(__as_b32(__values[60])),
      "r"(__as_b32(__values[61])),
      "r"(__as_b32(__values[62])),
      "r"(__as_b32(__values[63]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x64.unpack::16b.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a,
SM_101a template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2_unpack_16b(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[64]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <int _N32, typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void
tcgen05_st_16x32bx2_unpack_16b(_CUDA_VSTD::uint32_t __taddr, n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[64])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.16x32bx2.x64.unpack::16b.b32 [%0], %1, {%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
    "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, "
    "%35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
    "%57, %58, %59, %60, %61, %62, %63, %64, %65};"
    :
    : "r"(__taddr),
      "n"(__immHalfSplitoff.value),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31])),
      "r"(__as_b32(__values[32])),
      "r"(__as_b32(__values[33])),
      "r"(__as_b32(__values[34])),
      "r"(__as_b32(__values[35])),
      "r"(__as_b32(__values[36])),
      "r"(__as_b32(__values[37])),
      "r"(__as_b32(__values[38])),
      "r"(__as_b32(__values[39])),
      "r"(__as_b32(__values[40])),
      "r"(__as_b32(__values[41])),
      "r"(__as_b32(__values[42])),
      "r"(__as_b32(__values[43])),
      "r"(__as_b32(__values[44])),
      "r"(__as_b32(__values[45])),
      "r"(__as_b32(__values[46])),
      "r"(__as_b32(__values[47])),
      "r"(__as_b32(__values[48])),
      "r"(__as_b32(__values[49])),
      "r"(__as_b32(__values[50])),
      "r"(__as_b32(__values[51])),
      "r"(__as_b32(__values[52])),
      "r"(__as_b32(__values[53])),
      "r"(__as_b32(__values[54])),
      "r"(__as_b32(__values[55])),
      "r"(__as_b32(__values[56])),
      "r"(__as_b32(__values[57])),
      "r"(__as_b32(__values[58])),
      "r"(__as_b32(__values[59])),
      "r"(__as_b32(__values[60])),
      "r"(__as_b32(__values[61])),
      "r"(__as_b32(__values[62])),
      "r"(__as_b32(__values[63]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x128.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a, SM_101a
template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[128]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x32bx2_is_not_supported_before_SM_100a_SM_101a__();
template <int _N32, typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void
tcgen05_st_16x32bx2(_CUDA_VSTD::uint32_t __taddr, n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[128])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.16x32bx2.x128.b32 [%0], %1, {%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
    "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, "
    "%37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, "
    "%59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, "
    "%81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, "
    "%103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, "
    "%122, %123, %124, %125, %126, %127, %128, %129};"
    :
    : "r"(__taddr),
      "n"(__immHalfSplitoff.value),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31])),
      "r"(__as_b32(__values[32])),
      "r"(__as_b32(__values[33])),
      "r"(__as_b32(__values[34])),
      "r"(__as_b32(__values[35])),
      "r"(__as_b32(__values[36])),
      "r"(__as_b32(__values[37])),
      "r"(__as_b32(__values[38])),
      "r"(__as_b32(__values[39])),
      "r"(__as_b32(__values[40])),
      "r"(__as_b32(__values[41])),
      "r"(__as_b32(__values[42])),
      "r"(__as_b32(__values[43])),
      "r"(__as_b32(__values[44])),
      "r"(__as_b32(__values[45])),
      "r"(__as_b32(__values[46])),
      "r"(__as_b32(__values[47])),
      "r"(__as_b32(__values[48])),
      "r"(__as_b32(__values[49])),
      "r"(__as_b32(__values[50])),
      "r"(__as_b32(__values[51])),
      "r"(__as_b32(__values[52])),
      "r"(__as_b32(__values[53])),
      "r"(__as_b32(__values[54])),
      "r"(__as_b32(__values[55])),
      "r"(__as_b32(__values[56])),
      "r"(__as_b32(__values[57])),
      "r"(__as_b32(__values[58])),
      "r"(__as_b32(__values[59])),
      "r"(__as_b32(__values[60])),
      "r"(__as_b32(__values[61])),
      "r"(__as_b32(__values[62])),
      "r"(__as_b32(__values[63])),
      "r"(__as_b32(__values[64])),
      "r"(__as_b32(__values[65])),
      "r"(__as_b32(__values[66])),
      "r"(__as_b32(__values[67])),
      "r"(__as_b32(__values[68])),
      "r"(__as_b32(__values[69])),
      "r"(__as_b32(__values[70])),
      "r"(__as_b32(__values[71])),
      "r"(__as_b32(__values[72])),
      "r"(__as_b32(__values[73])),
      "r"(__as_b32(__values[74])),
      "r"(__as_b32(__values[75])),
      "r"(__as_b32(__values[76])),
      "r"(__as_b32(__values[77])),
      "r"(__as_b32(__values[78])),
      "r"(__as_b32(__values[79])),
      "r"(__as_b32(__values[80])),
      "r"(__as_b32(__values[81])),
      "r"(__as_b32(__values[82])),
      "r"(__as_b32(__values[83])),
      "r"(__as_b32(__values[84])),
      "r"(__as_b32(__values[85])),
      "r"(__as_b32(__values[86])),
      "r"(__as_b32(__values[87])),
      "r"(__as_b32(__values[88])),
      "r"(__as_b32(__values[89])),
      "r"(__as_b32(__values[90])),
      "r"(__as_b32(__values[91])),
      "r"(__as_b32(__values[92])),
      "r"(__as_b32(__values[93])),
      "r"(__as_b32(__values[94])),
      "r"(__as_b32(__values[95])),
      "r"(__as_b32(__values[96])),
      "r"(__as_b32(__values[97])),
      "r"(__as_b32(__values[98])),
      "r"(__as_b32(__values[99])),
      "r"(__as_b32(__values[100])),
      "r"(__as_b32(__values[101])),
      "r"(__as_b32(__values[102])),
      "r"(__as_b32(__values[103])),
      "r"(__as_b32(__values[104])),
      "r"(__as_b32(__values[105])),
      "r"(__as_b32(__values[106])),
      "r"(__as_b32(__values[107])),
      "r"(__as_b32(__values[108])),
      "r"(__as_b32(__values[109])),
      "r"(__as_b32(__values[110])),
      "r"(__as_b32(__values[111])),
      "r"(__as_b32(__values[112])),
      "r"(__as_b32(__values[113])),
      "r"(__as_b32(__values[114])),
      "r"(__as_b32(__values[115])),
      "r"(__as_b32(__values[116])),
      "r"(__as_b32(__values[117])),
      "r"(__as_b32(__values[118])),
      "r"(__as_b32(__values[119])),
      "r"(__as_b32(__values[120])),
      "r"(__as_b32(__values[121])),
      "r"(__as_b32(__values[122])),
      "r"(__as_b32(__values[123])),
      "r"(__as_b32(__values[124])),
      "r"(__as_b32(__values[125])),
      "r"(__as_b32(__values[126])),
      "r"(__as_b32(__values[127]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.st.sync.aligned.16x32bx2.x128.unpack::16b.b32 [taddr], immHalfSplitoff, values; // PTX ISA 86, SM_100a,
SM_101a template <int N32, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_st_16x32bx2_unpack_16b(
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff,
  const B32 (&values)[128]);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
template <int _N32, typename _B32, _CUDA_VSTD::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void
tcgen05_st_16x32bx2_unpack_16b(_CUDA_VSTD::uint32_t __taddr, n32_t<_N32> __immHalfSplitoff, const _B32 (&__values)[128])
{
  static_assert(sizeof(_B32) == 4, "");
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH_FEAT_SM100_ALL || __CUDA_ARCH_FEAT_SM101_ALL
  asm(
    "tcgen05.st.sync.aligned.16x32bx2.x128.unpack::16b.b32 [%0], %1, {%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
    "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, "
    "%35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
    "%57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, "
    "%79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, "
    "%101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, "
    "%120, %121, %122, %123, %124, %125, %126, %127, %128, %129};"
    :
    : "r"(__taddr),
      "n"(__immHalfSplitoff.value),
      "r"(__as_b32(__values[0])),
      "r"(__as_b32(__values[1])),
      "r"(__as_b32(__values[2])),
      "r"(__as_b32(__values[3])),
      "r"(__as_b32(__values[4])),
      "r"(__as_b32(__values[5])),
      "r"(__as_b32(__values[6])),
      "r"(__as_b32(__values[7])),
      "r"(__as_b32(__values[8])),
      "r"(__as_b32(__values[9])),
      "r"(__as_b32(__values[10])),
      "r"(__as_b32(__values[11])),
      "r"(__as_b32(__values[12])),
      "r"(__as_b32(__values[13])),
      "r"(__as_b32(__values[14])),
      "r"(__as_b32(__values[15])),
      "r"(__as_b32(__values[16])),
      "r"(__as_b32(__values[17])),
      "r"(__as_b32(__values[18])),
      "r"(__as_b32(__values[19])),
      "r"(__as_b32(__values[20])),
      "r"(__as_b32(__values[21])),
      "r"(__as_b32(__values[22])),
      "r"(__as_b32(__values[23])),
      "r"(__as_b32(__values[24])),
      "r"(__as_b32(__values[25])),
      "r"(__as_b32(__values[26])),
      "r"(__as_b32(__values[27])),
      "r"(__as_b32(__values[28])),
      "r"(__as_b32(__values[29])),
      "r"(__as_b32(__values[30])),
      "r"(__as_b32(__values[31])),
      "r"(__as_b32(__values[32])),
      "r"(__as_b32(__values[33])),
      "r"(__as_b32(__values[34])),
      "r"(__as_b32(__values[35])),
      "r"(__as_b32(__values[36])),
      "r"(__as_b32(__values[37])),
      "r"(__as_b32(__values[38])),
      "r"(__as_b32(__values[39])),
      "r"(__as_b32(__values[40])),
      "r"(__as_b32(__values[41])),
      "r"(__as_b32(__values[42])),
      "r"(__as_b32(__values[43])),
      "r"(__as_b32(__values[44])),
      "r"(__as_b32(__values[45])),
      "r"(__as_b32(__values[46])),
      "r"(__as_b32(__values[47])),
      "r"(__as_b32(__values[48])),
      "r"(__as_b32(__values[49])),
      "r"(__as_b32(__values[50])),
      "r"(__as_b32(__values[51])),
      "r"(__as_b32(__values[52])),
      "r"(__as_b32(__values[53])),
      "r"(__as_b32(__values[54])),
      "r"(__as_b32(__values[55])),
      "r"(__as_b32(__values[56])),
      "r"(__as_b32(__values[57])),
      "r"(__as_b32(__values[58])),
      "r"(__as_b32(__values[59])),
      "r"(__as_b32(__values[60])),
      "r"(__as_b32(__values[61])),
      "r"(__as_b32(__values[62])),
      "r"(__as_b32(__values[63])),
      "r"(__as_b32(__values[64])),
      "r"(__as_b32(__values[65])),
      "r"(__as_b32(__values[66])),
      "r"(__as_b32(__values[67])),
      "r"(__as_b32(__values[68])),
      "r"(__as_b32(__values[69])),
      "r"(__as_b32(__values[70])),
      "r"(__as_b32(__values[71])),
      "r"(__as_b32(__values[72])),
      "r"(__as_b32(__values[73])),
      "r"(__as_b32(__values[74])),
      "r"(__as_b32(__values[75])),
      "r"(__as_b32(__values[76])),
      "r"(__as_b32(__values[77])),
      "r"(__as_b32(__values[78])),
      "r"(__as_b32(__values[79])),
      "r"(__as_b32(__values[80])),
      "r"(__as_b32(__values[81])),
      "r"(__as_b32(__values[82])),
      "r"(__as_b32(__values[83])),
      "r"(__as_b32(__values[84])),
      "r"(__as_b32(__values[85])),
      "r"(__as_b32(__values[86])),
      "r"(__as_b32(__values[87])),
      "r"(__as_b32(__values[88])),
      "r"(__as_b32(__values[89])),
      "r"(__as_b32(__values[90])),
      "r"(__as_b32(__values[91])),
      "r"(__as_b32(__values[92])),
      "r"(__as_b32(__values[93])),
      "r"(__as_b32(__values[94])),
      "r"(__as_b32(__values[95])),
      "r"(__as_b32(__values[96])),
      "r"(__as_b32(__values[97])),
      "r"(__as_b32(__values[98])),
      "r"(__as_b32(__values[99])),
      "r"(__as_b32(__values[100])),
      "r"(__as_b32(__values[101])),
      "r"(__as_b32(__values[102])),
      "r"(__as_b32(__values[103])),
      "r"(__as_b32(__values[104])),
      "r"(__as_b32(__values[105])),
      "r"(__as_b32(__values[106])),
      "r"(__as_b32(__values[107])),
      "r"(__as_b32(__values[108])),
      "r"(__as_b32(__values[109])),
      "r"(__as_b32(__values[110])),
      "r"(__as_b32(__values[111])),
      "r"(__as_b32(__values[112])),
      "r"(__as_b32(__values[113])),
      "r"(__as_b32(__values[114])),
      "r"(__as_b32(__values[115])),
      "r"(__as_b32(__values[116])),
      "r"(__as_b32(__values[117])),
      "r"(__as_b32(__values[118])),
      "r"(__as_b32(__values[119])),
      "r"(__as_b32(__values[120])),
      "r"(__as_b32(__values[121])),
      "r"(__as_b32(__values[122])),
      "r"(__as_b32(__values[123])),
      "r"(__as_b32(__values[124])),
      "r"(__as_b32(__values[125])),
      "r"(__as_b32(__values[126])),
      "r"(__as_b32(__values[127]))
    : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_tcgen05_st_16x32bx2_unpack_16b_is_not_supported_before_SM_100a_SM_101a__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

#endif // _CUDA_PTX_GENERATED_TCGEN05_ST_H_
