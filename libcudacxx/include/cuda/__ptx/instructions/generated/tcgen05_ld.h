// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_TCGEN05_LD_H_
#define _CUDA_PTX_GENERATED_TCGEN05_LD_H_

/*
// tcgen05.ld.sync.aligned.16x64b.x1.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b(
  B32 (&out_var)[1],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b(_B32 (&__out_var)[1], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x64b.x1.b32 {%0}, [%1];" : "=r"(__out_var[0]) : "r"(__taddr) : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x1.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b_pack_16b(
  B32 (&out_var)[1],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b_pack_16b(_B32 (&__out_var)[1], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x64b.x1.pack::16b.b32 {%0}, [%1];" : "=r"(__out_var[0]) : "r"(__taddr) : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x2.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b(
  B32 (&out_var)[2],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b(_B32 (&__out_var)[2], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x64b.x2.b32 {%0, %1}, [%2];"
      : "=r"(__out_var[0]), "=r"(__out_var[1])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x2.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b_pack_16b(
  B32 (&out_var)[2],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b_pack_16b(_B32 (&__out_var)[2], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x64b.x2.pack::16b.b32 {%0, %1}, [%2];"
      : "=r"(__out_var[0]), "=r"(__out_var[1])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x4.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b(
  B32 (&out_var)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b(_B32 (&__out_var)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x64b.x4.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out_var[0]), "=r"(__out_var[1]), "=r"(__out_var[2]), "=r"(__out_var[3])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x4.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b_pack_16b(
  B32 (&out_var)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b_pack_16b(_B32 (&__out_var)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x64b.x4.pack::16b.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out_var[0]), "=r"(__out_var[1]), "=r"(__out_var[2]), "=r"(__out_var[3])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x8.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b(
  B32 (&out_var)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b(_B32 (&__out_var)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x64b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x8.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b_pack_16b(
  B32 (&out_var)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b_pack_16b(_B32 (&__out_var)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x64b.x8.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x16.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b(
  B32 (&out_var)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b(_B32 (&__out_var)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x64b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
      "[%16];"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x16.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b_pack_16b(
  B32 (&out_var)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b_pack_16b(_B32 (&__out_var)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x64b.x16.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15}, [%16];"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x32.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b(
  B32 (&out_var)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b(_B32 (&__out_var)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.16x64b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31])
    : "r"(__taddr)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x32.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b_pack_16b(
  B32 (&out_var)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b_pack_16b(_B32 (&__out_var)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.16x64b.x32.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31])
    : "r"(__taddr)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x64.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b(
  B32 (&out_var)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b(_B32 (&__out_var)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.16x64b.x64.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63}, [%64];"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31]),
      "=r"(__out_var[32]),
      "=r"(__out_var[33]),
      "=r"(__out_var[34]),
      "=r"(__out_var[35]),
      "=r"(__out_var[36]),
      "=r"(__out_var[37]),
      "=r"(__out_var[38]),
      "=r"(__out_var[39]),
      "=r"(__out_var[40]),
      "=r"(__out_var[41]),
      "=r"(__out_var[42]),
      "=r"(__out_var[43]),
      "=r"(__out_var[44]),
      "=r"(__out_var[45]),
      "=r"(__out_var[46]),
      "=r"(__out_var[47]),
      "=r"(__out_var[48]),
      "=r"(__out_var[49]),
      "=r"(__out_var[50]),
      "=r"(__out_var[51]),
      "=r"(__out_var[52]),
      "=r"(__out_var[53]),
      "=r"(__out_var[54]),
      "=r"(__out_var[55]),
      "=r"(__out_var[56]),
      "=r"(__out_var[57]),
      "=r"(__out_var[58]),
      "=r"(__out_var[59]),
      "=r"(__out_var[60]),
      "=r"(__out_var[61]),
      "=r"(__out_var[62]),
      "=r"(__out_var[63])
    : "r"(__taddr)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x64.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b_pack_16b(
  B32 (&out_var)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b_pack_16b(_B32 (&__out_var)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.16x64b.x64.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63}, [%64];"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31]),
      "=r"(__out_var[32]),
      "=r"(__out_var[33]),
      "=r"(__out_var[34]),
      "=r"(__out_var[35]),
      "=r"(__out_var[36]),
      "=r"(__out_var[37]),
      "=r"(__out_var[38]),
      "=r"(__out_var[39]),
      "=r"(__out_var[40]),
      "=r"(__out_var[41]),
      "=r"(__out_var[42]),
      "=r"(__out_var[43]),
      "=r"(__out_var[44]),
      "=r"(__out_var[45]),
      "=r"(__out_var[46]),
      "=r"(__out_var[47]),
      "=r"(__out_var[48]),
      "=r"(__out_var[49]),
      "=r"(__out_var[50]),
      "=r"(__out_var[51]),
      "=r"(__out_var[52]),
      "=r"(__out_var[53]),
      "=r"(__out_var[54]),
      "=r"(__out_var[55]),
      "=r"(__out_var[56]),
      "=r"(__out_var[57]),
      "=r"(__out_var[58]),
      "=r"(__out_var[59]),
      "=r"(__out_var[60]),
      "=r"(__out_var[61]),
      "=r"(__out_var[62]),
      "=r"(__out_var[63])
    : "r"(__taddr)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x128.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b(
  B32 (&out_var)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b(_B32 (&__out_var)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.16x64b.x128.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, "
    "%82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, "
    "%104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
    "%123, %124, %125, %126, %127}, [%128];"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31]),
      "=r"(__out_var[32]),
      "=r"(__out_var[33]),
      "=r"(__out_var[34]),
      "=r"(__out_var[35]),
      "=r"(__out_var[36]),
      "=r"(__out_var[37]),
      "=r"(__out_var[38]),
      "=r"(__out_var[39]),
      "=r"(__out_var[40]),
      "=r"(__out_var[41]),
      "=r"(__out_var[42]),
      "=r"(__out_var[43]),
      "=r"(__out_var[44]),
      "=r"(__out_var[45]),
      "=r"(__out_var[46]),
      "=r"(__out_var[47]),
      "=r"(__out_var[48]),
      "=r"(__out_var[49]),
      "=r"(__out_var[50]),
      "=r"(__out_var[51]),
      "=r"(__out_var[52]),
      "=r"(__out_var[53]),
      "=r"(__out_var[54]),
      "=r"(__out_var[55]),
      "=r"(__out_var[56]),
      "=r"(__out_var[57]),
      "=r"(__out_var[58]),
      "=r"(__out_var[59]),
      "=r"(__out_var[60]),
      "=r"(__out_var[61]),
      "=r"(__out_var[62]),
      "=r"(__out_var[63]),
      "=r"(__out_var[64]),
      "=r"(__out_var[65]),
      "=r"(__out_var[66]),
      "=r"(__out_var[67]),
      "=r"(__out_var[68]),
      "=r"(__out_var[69]),
      "=r"(__out_var[70]),
      "=r"(__out_var[71]),
      "=r"(__out_var[72]),
      "=r"(__out_var[73]),
      "=r"(__out_var[74]),
      "=r"(__out_var[75]),
      "=r"(__out_var[76]),
      "=r"(__out_var[77]),
      "=r"(__out_var[78]),
      "=r"(__out_var[79]),
      "=r"(__out_var[80]),
      "=r"(__out_var[81]),
      "=r"(__out_var[82]),
      "=r"(__out_var[83]),
      "=r"(__out_var[84]),
      "=r"(__out_var[85]),
      "=r"(__out_var[86]),
      "=r"(__out_var[87]),
      "=r"(__out_var[88]),
      "=r"(__out_var[89]),
      "=r"(__out_var[90]),
      "=r"(__out_var[91]),
      "=r"(__out_var[92]),
      "=r"(__out_var[93]),
      "=r"(__out_var[94]),
      "=r"(__out_var[95]),
      "=r"(__out_var[96]),
      "=r"(__out_var[97]),
      "=r"(__out_var[98]),
      "=r"(__out_var[99]),
      "=r"(__out_var[100]),
      "=r"(__out_var[101]),
      "=r"(__out_var[102]),
      "=r"(__out_var[103]),
      "=r"(__out_var[104]),
      "=r"(__out_var[105]),
      "=r"(__out_var[106]),
      "=r"(__out_var[107]),
      "=r"(__out_var[108]),
      "=r"(__out_var[109]),
      "=r"(__out_var[110]),
      "=r"(__out_var[111]),
      "=r"(__out_var[112]),
      "=r"(__out_var[113]),
      "=r"(__out_var[114]),
      "=r"(__out_var[115]),
      "=r"(__out_var[116]),
      "=r"(__out_var[117]),
      "=r"(__out_var[118]),
      "=r"(__out_var[119]),
      "=r"(__out_var[120]),
      "=r"(__out_var[121]),
      "=r"(__out_var[122]),
      "=r"(__out_var[123]),
      "=r"(__out_var[124]),
      "=r"(__out_var[125]),
      "=r"(__out_var[126]),
      "=r"(__out_var[127])
    : "r"(__taddr)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x64b.x128.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x64b_pack_16b(
  B32 (&out_var)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x64b_pack_16b(_B32 (&__out_var)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.16x64b.x128.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
    "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, "
    "%102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, "
    "%121, %122, %123, %124, %125, %126, %127}, [%128];"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31]),
      "=r"(__out_var[32]),
      "=r"(__out_var[33]),
      "=r"(__out_var[34]),
      "=r"(__out_var[35]),
      "=r"(__out_var[36]),
      "=r"(__out_var[37]),
      "=r"(__out_var[38]),
      "=r"(__out_var[39]),
      "=r"(__out_var[40]),
      "=r"(__out_var[41]),
      "=r"(__out_var[42]),
      "=r"(__out_var[43]),
      "=r"(__out_var[44]),
      "=r"(__out_var[45]),
      "=r"(__out_var[46]),
      "=r"(__out_var[47]),
      "=r"(__out_var[48]),
      "=r"(__out_var[49]),
      "=r"(__out_var[50]),
      "=r"(__out_var[51]),
      "=r"(__out_var[52]),
      "=r"(__out_var[53]),
      "=r"(__out_var[54]),
      "=r"(__out_var[55]),
      "=r"(__out_var[56]),
      "=r"(__out_var[57]),
      "=r"(__out_var[58]),
      "=r"(__out_var[59]),
      "=r"(__out_var[60]),
      "=r"(__out_var[61]),
      "=r"(__out_var[62]),
      "=r"(__out_var[63]),
      "=r"(__out_var[64]),
      "=r"(__out_var[65]),
      "=r"(__out_var[66]),
      "=r"(__out_var[67]),
      "=r"(__out_var[68]),
      "=r"(__out_var[69]),
      "=r"(__out_var[70]),
      "=r"(__out_var[71]),
      "=r"(__out_var[72]),
      "=r"(__out_var[73]),
      "=r"(__out_var[74]),
      "=r"(__out_var[75]),
      "=r"(__out_var[76]),
      "=r"(__out_var[77]),
      "=r"(__out_var[78]),
      "=r"(__out_var[79]),
      "=r"(__out_var[80]),
      "=r"(__out_var[81]),
      "=r"(__out_var[82]),
      "=r"(__out_var[83]),
      "=r"(__out_var[84]),
      "=r"(__out_var[85]),
      "=r"(__out_var[86]),
      "=r"(__out_var[87]),
      "=r"(__out_var[88]),
      "=r"(__out_var[89]),
      "=r"(__out_var[90]),
      "=r"(__out_var[91]),
      "=r"(__out_var[92]),
      "=r"(__out_var[93]),
      "=r"(__out_var[94]),
      "=r"(__out_var[95]),
      "=r"(__out_var[96]),
      "=r"(__out_var[97]),
      "=r"(__out_var[98]),
      "=r"(__out_var[99]),
      "=r"(__out_var[100]),
      "=r"(__out_var[101]),
      "=r"(__out_var[102]),
      "=r"(__out_var[103]),
      "=r"(__out_var[104]),
      "=r"(__out_var[105]),
      "=r"(__out_var[106]),
      "=r"(__out_var[107]),
      "=r"(__out_var[108]),
      "=r"(__out_var[109]),
      "=r"(__out_var[110]),
      "=r"(__out_var[111]),
      "=r"(__out_var[112]),
      "=r"(__out_var[113]),
      "=r"(__out_var[114]),
      "=r"(__out_var[115]),
      "=r"(__out_var[116]),
      "=r"(__out_var[117]),
      "=r"(__out_var[118]),
      "=r"(__out_var[119]),
      "=r"(__out_var[120]),
      "=r"(__out_var[121]),
      "=r"(__out_var[122]),
      "=r"(__out_var[123]),
      "=r"(__out_var[124]),
      "=r"(__out_var[125]),
      "=r"(__out_var[126]),
      "=r"(__out_var[127])
    : "r"(__taddr)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x1.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b(
  B32 (&out_var)[2],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b(_B32 (&__out_var)[2], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x128b.x1.b32 {%0, %1}, [%2];"
      : "=r"(__out_var[0]), "=r"(__out_var[1])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x1.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b_pack_16b(
  B32 (&out_var)[2],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b_pack_16b(_B32 (&__out_var)[2], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x128b.x1.pack::16b.b32 {%0, %1}, [%2];"
      : "=r"(__out_var[0]), "=r"(__out_var[1])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x2.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b(
  B32 (&out_var)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b(_B32 (&__out_var)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x128b.x2.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out_var[0]), "=r"(__out_var[1]), "=r"(__out_var[2]), "=r"(__out_var[3])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x2.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b_pack_16b(
  B32 (&out_var)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b_pack_16b(_B32 (&__out_var)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x128b.x2.pack::16b.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out_var[0]), "=r"(__out_var[1]), "=r"(__out_var[2]), "=r"(__out_var[3])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x4.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b(
  B32 (&out_var)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b(_B32 (&__out_var)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x128b.x4.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b_pack_16b(
  B32 (&out_var)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b_pack_16b(_B32 (&__out_var)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x8.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b(
  B32 (&out_var)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b(_B32 (&__out_var)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x128b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
      "[%16];"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x8.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b_pack_16b(
  B32 (&out_var)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b_pack_16b(_B32 (&__out_var)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x128b.x8.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15}, [%16];"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x16.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b(
  B32 (&out_var)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b(_B32 (&__out_var)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.16x128b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31])
    : "r"(__taddr)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x16.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b_pack_16b(
  B32 (&out_var)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b_pack_16b(_B32 (&__out_var)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.16x128b.x16.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31])
    : "r"(__taddr)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x32.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b(
  B32 (&out_var)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b(_B32 (&__out_var)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.16x128b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63}, [%64];"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31]),
      "=r"(__out_var[32]),
      "=r"(__out_var[33]),
      "=r"(__out_var[34]),
      "=r"(__out_var[35]),
      "=r"(__out_var[36]),
      "=r"(__out_var[37]),
      "=r"(__out_var[38]),
      "=r"(__out_var[39]),
      "=r"(__out_var[40]),
      "=r"(__out_var[41]),
      "=r"(__out_var[42]),
      "=r"(__out_var[43]),
      "=r"(__out_var[44]),
      "=r"(__out_var[45]),
      "=r"(__out_var[46]),
      "=r"(__out_var[47]),
      "=r"(__out_var[48]),
      "=r"(__out_var[49]),
      "=r"(__out_var[50]),
      "=r"(__out_var[51]),
      "=r"(__out_var[52]),
      "=r"(__out_var[53]),
      "=r"(__out_var[54]),
      "=r"(__out_var[55]),
      "=r"(__out_var[56]),
      "=r"(__out_var[57]),
      "=r"(__out_var[58]),
      "=r"(__out_var[59]),
      "=r"(__out_var[60]),
      "=r"(__out_var[61]),
      "=r"(__out_var[62]),
      "=r"(__out_var[63])
    : "r"(__taddr)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x32.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b_pack_16b(
  B32 (&out_var)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b_pack_16b(_B32 (&__out_var)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.16x128b.x32.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63}, [%64];"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31]),
      "=r"(__out_var[32]),
      "=r"(__out_var[33]),
      "=r"(__out_var[34]),
      "=r"(__out_var[35]),
      "=r"(__out_var[36]),
      "=r"(__out_var[37]),
      "=r"(__out_var[38]),
      "=r"(__out_var[39]),
      "=r"(__out_var[40]),
      "=r"(__out_var[41]),
      "=r"(__out_var[42]),
      "=r"(__out_var[43]),
      "=r"(__out_var[44]),
      "=r"(__out_var[45]),
      "=r"(__out_var[46]),
      "=r"(__out_var[47]),
      "=r"(__out_var[48]),
      "=r"(__out_var[49]),
      "=r"(__out_var[50]),
      "=r"(__out_var[51]),
      "=r"(__out_var[52]),
      "=r"(__out_var[53]),
      "=r"(__out_var[54]),
      "=r"(__out_var[55]),
      "=r"(__out_var[56]),
      "=r"(__out_var[57]),
      "=r"(__out_var[58]),
      "=r"(__out_var[59]),
      "=r"(__out_var[60]),
      "=r"(__out_var[61]),
      "=r"(__out_var[62]),
      "=r"(__out_var[63])
    : "r"(__taddr)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x64.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b(
  B32 (&out_var)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b(_B32 (&__out_var)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.16x128b.x64.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, "
    "%82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, "
    "%104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
    "%123, %124, %125, %126, %127}, [%128];"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31]),
      "=r"(__out_var[32]),
      "=r"(__out_var[33]),
      "=r"(__out_var[34]),
      "=r"(__out_var[35]),
      "=r"(__out_var[36]),
      "=r"(__out_var[37]),
      "=r"(__out_var[38]),
      "=r"(__out_var[39]),
      "=r"(__out_var[40]),
      "=r"(__out_var[41]),
      "=r"(__out_var[42]),
      "=r"(__out_var[43]),
      "=r"(__out_var[44]),
      "=r"(__out_var[45]),
      "=r"(__out_var[46]),
      "=r"(__out_var[47]),
      "=r"(__out_var[48]),
      "=r"(__out_var[49]),
      "=r"(__out_var[50]),
      "=r"(__out_var[51]),
      "=r"(__out_var[52]),
      "=r"(__out_var[53]),
      "=r"(__out_var[54]),
      "=r"(__out_var[55]),
      "=r"(__out_var[56]),
      "=r"(__out_var[57]),
      "=r"(__out_var[58]),
      "=r"(__out_var[59]),
      "=r"(__out_var[60]),
      "=r"(__out_var[61]),
      "=r"(__out_var[62]),
      "=r"(__out_var[63]),
      "=r"(__out_var[64]),
      "=r"(__out_var[65]),
      "=r"(__out_var[66]),
      "=r"(__out_var[67]),
      "=r"(__out_var[68]),
      "=r"(__out_var[69]),
      "=r"(__out_var[70]),
      "=r"(__out_var[71]),
      "=r"(__out_var[72]),
      "=r"(__out_var[73]),
      "=r"(__out_var[74]),
      "=r"(__out_var[75]),
      "=r"(__out_var[76]),
      "=r"(__out_var[77]),
      "=r"(__out_var[78]),
      "=r"(__out_var[79]),
      "=r"(__out_var[80]),
      "=r"(__out_var[81]),
      "=r"(__out_var[82]),
      "=r"(__out_var[83]),
      "=r"(__out_var[84]),
      "=r"(__out_var[85]),
      "=r"(__out_var[86]),
      "=r"(__out_var[87]),
      "=r"(__out_var[88]),
      "=r"(__out_var[89]),
      "=r"(__out_var[90]),
      "=r"(__out_var[91]),
      "=r"(__out_var[92]),
      "=r"(__out_var[93]),
      "=r"(__out_var[94]),
      "=r"(__out_var[95]),
      "=r"(__out_var[96]),
      "=r"(__out_var[97]),
      "=r"(__out_var[98]),
      "=r"(__out_var[99]),
      "=r"(__out_var[100]),
      "=r"(__out_var[101]),
      "=r"(__out_var[102]),
      "=r"(__out_var[103]),
      "=r"(__out_var[104]),
      "=r"(__out_var[105]),
      "=r"(__out_var[106]),
      "=r"(__out_var[107]),
      "=r"(__out_var[108]),
      "=r"(__out_var[109]),
      "=r"(__out_var[110]),
      "=r"(__out_var[111]),
      "=r"(__out_var[112]),
      "=r"(__out_var[113]),
      "=r"(__out_var[114]),
      "=r"(__out_var[115]),
      "=r"(__out_var[116]),
      "=r"(__out_var[117]),
      "=r"(__out_var[118]),
      "=r"(__out_var[119]),
      "=r"(__out_var[120]),
      "=r"(__out_var[121]),
      "=r"(__out_var[122]),
      "=r"(__out_var[123]),
      "=r"(__out_var[124]),
      "=r"(__out_var[125]),
      "=r"(__out_var[126]),
      "=r"(__out_var[127])
    : "r"(__taddr)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x128b.x64.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x128b_pack_16b(
  B32 (&out_var)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x128b_pack_16b(_B32 (&__out_var)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.16x128b.x64.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
    "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, "
    "%102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, "
    "%121, %122, %123, %124, %125, %126, %127}, [%128];"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31]),
      "=r"(__out_var[32]),
      "=r"(__out_var[33]),
      "=r"(__out_var[34]),
      "=r"(__out_var[35]),
      "=r"(__out_var[36]),
      "=r"(__out_var[37]),
      "=r"(__out_var[38]),
      "=r"(__out_var[39]),
      "=r"(__out_var[40]),
      "=r"(__out_var[41]),
      "=r"(__out_var[42]),
      "=r"(__out_var[43]),
      "=r"(__out_var[44]),
      "=r"(__out_var[45]),
      "=r"(__out_var[46]),
      "=r"(__out_var[47]),
      "=r"(__out_var[48]),
      "=r"(__out_var[49]),
      "=r"(__out_var[50]),
      "=r"(__out_var[51]),
      "=r"(__out_var[52]),
      "=r"(__out_var[53]),
      "=r"(__out_var[54]),
      "=r"(__out_var[55]),
      "=r"(__out_var[56]),
      "=r"(__out_var[57]),
      "=r"(__out_var[58]),
      "=r"(__out_var[59]),
      "=r"(__out_var[60]),
      "=r"(__out_var[61]),
      "=r"(__out_var[62]),
      "=r"(__out_var[63]),
      "=r"(__out_var[64]),
      "=r"(__out_var[65]),
      "=r"(__out_var[66]),
      "=r"(__out_var[67]),
      "=r"(__out_var[68]),
      "=r"(__out_var[69]),
      "=r"(__out_var[70]),
      "=r"(__out_var[71]),
      "=r"(__out_var[72]),
      "=r"(__out_var[73]),
      "=r"(__out_var[74]),
      "=r"(__out_var[75]),
      "=r"(__out_var[76]),
      "=r"(__out_var[77]),
      "=r"(__out_var[78]),
      "=r"(__out_var[79]),
      "=r"(__out_var[80]),
      "=r"(__out_var[81]),
      "=r"(__out_var[82]),
      "=r"(__out_var[83]),
      "=r"(__out_var[84]),
      "=r"(__out_var[85]),
      "=r"(__out_var[86]),
      "=r"(__out_var[87]),
      "=r"(__out_var[88]),
      "=r"(__out_var[89]),
      "=r"(__out_var[90]),
      "=r"(__out_var[91]),
      "=r"(__out_var[92]),
      "=r"(__out_var[93]),
      "=r"(__out_var[94]),
      "=r"(__out_var[95]),
      "=r"(__out_var[96]),
      "=r"(__out_var[97]),
      "=r"(__out_var[98]),
      "=r"(__out_var[99]),
      "=r"(__out_var[100]),
      "=r"(__out_var[101]),
      "=r"(__out_var[102]),
      "=r"(__out_var[103]),
      "=r"(__out_var[104]),
      "=r"(__out_var[105]),
      "=r"(__out_var[106]),
      "=r"(__out_var[107]),
      "=r"(__out_var[108]),
      "=r"(__out_var[109]),
      "=r"(__out_var[110]),
      "=r"(__out_var[111]),
      "=r"(__out_var[112]),
      "=r"(__out_var[113]),
      "=r"(__out_var[114]),
      "=r"(__out_var[115]),
      "=r"(__out_var[116]),
      "=r"(__out_var[117]),
      "=r"(__out_var[118]),
      "=r"(__out_var[119]),
      "=r"(__out_var[120]),
      "=r"(__out_var[121]),
      "=r"(__out_var[122]),
      "=r"(__out_var[123]),
      "=r"(__out_var[124]),
      "=r"(__out_var[125]),
      "=r"(__out_var[126]),
      "=r"(__out_var[127])
    : "r"(__taddr)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x1.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b(
  B32 (&out_var)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b(_B32 (&__out_var)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out_var[0]), "=r"(__out_var[1]), "=r"(__out_var[2]), "=r"(__out_var[3])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x1.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b_pack_16b(
  B32 (&out_var)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b_pack_16b(_B32 (&__out_var)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x256b.x1.pack::16b.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out_var[0]), "=r"(__out_var[1]), "=r"(__out_var[2]), "=r"(__out_var[3])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x2.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b(
  B32 (&out_var)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b(_B32 (&__out_var)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x2.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b_pack_16b(
  B32 (&out_var)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b_pack_16b(_B32 (&__out_var)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x256b.x2.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x4.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b(
  B32 (&out_var)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b(_B32 (&__out_var)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
      "[%16];"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x4.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b_pack_16b(
  B32 (&out_var)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b_pack_16b(_B32 (&__out_var)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x256b.x4.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15}, [%16];"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x8.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b(
  B32 (&out_var)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b(_B32 (&__out_var)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.16x256b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31])
    : "r"(__taddr)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x8.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b_pack_16b(
  B32 (&out_var)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b_pack_16b(_B32 (&__out_var)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.16x256b.x8.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31])
    : "r"(__taddr)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x16.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b(
  B32 (&out_var)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b(_B32 (&__out_var)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.16x256b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63}, [%64];"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31]),
      "=r"(__out_var[32]),
      "=r"(__out_var[33]),
      "=r"(__out_var[34]),
      "=r"(__out_var[35]),
      "=r"(__out_var[36]),
      "=r"(__out_var[37]),
      "=r"(__out_var[38]),
      "=r"(__out_var[39]),
      "=r"(__out_var[40]),
      "=r"(__out_var[41]),
      "=r"(__out_var[42]),
      "=r"(__out_var[43]),
      "=r"(__out_var[44]),
      "=r"(__out_var[45]),
      "=r"(__out_var[46]),
      "=r"(__out_var[47]),
      "=r"(__out_var[48]),
      "=r"(__out_var[49]),
      "=r"(__out_var[50]),
      "=r"(__out_var[51]),
      "=r"(__out_var[52]),
      "=r"(__out_var[53]),
      "=r"(__out_var[54]),
      "=r"(__out_var[55]),
      "=r"(__out_var[56]),
      "=r"(__out_var[57]),
      "=r"(__out_var[58]),
      "=r"(__out_var[59]),
      "=r"(__out_var[60]),
      "=r"(__out_var[61]),
      "=r"(__out_var[62]),
      "=r"(__out_var[63])
    : "r"(__taddr)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x16.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b_pack_16b(
  B32 (&out_var)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b_pack_16b(_B32 (&__out_var)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.16x256b.x16.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63}, [%64];"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31]),
      "=r"(__out_var[32]),
      "=r"(__out_var[33]),
      "=r"(__out_var[34]),
      "=r"(__out_var[35]),
      "=r"(__out_var[36]),
      "=r"(__out_var[37]),
      "=r"(__out_var[38]),
      "=r"(__out_var[39]),
      "=r"(__out_var[40]),
      "=r"(__out_var[41]),
      "=r"(__out_var[42]),
      "=r"(__out_var[43]),
      "=r"(__out_var[44]),
      "=r"(__out_var[45]),
      "=r"(__out_var[46]),
      "=r"(__out_var[47]),
      "=r"(__out_var[48]),
      "=r"(__out_var[49]),
      "=r"(__out_var[50]),
      "=r"(__out_var[51]),
      "=r"(__out_var[52]),
      "=r"(__out_var[53]),
      "=r"(__out_var[54]),
      "=r"(__out_var[55]),
      "=r"(__out_var[56]),
      "=r"(__out_var[57]),
      "=r"(__out_var[58]),
      "=r"(__out_var[59]),
      "=r"(__out_var[60]),
      "=r"(__out_var[61]),
      "=r"(__out_var[62]),
      "=r"(__out_var[63])
    : "r"(__taddr)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x32.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b(
  B32 (&out_var)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b(_B32 (&__out_var)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.16x256b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, "
    "%82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, "
    "%104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
    "%123, %124, %125, %126, %127}, [%128];"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31]),
      "=r"(__out_var[32]),
      "=r"(__out_var[33]),
      "=r"(__out_var[34]),
      "=r"(__out_var[35]),
      "=r"(__out_var[36]),
      "=r"(__out_var[37]),
      "=r"(__out_var[38]),
      "=r"(__out_var[39]),
      "=r"(__out_var[40]),
      "=r"(__out_var[41]),
      "=r"(__out_var[42]),
      "=r"(__out_var[43]),
      "=r"(__out_var[44]),
      "=r"(__out_var[45]),
      "=r"(__out_var[46]),
      "=r"(__out_var[47]),
      "=r"(__out_var[48]),
      "=r"(__out_var[49]),
      "=r"(__out_var[50]),
      "=r"(__out_var[51]),
      "=r"(__out_var[52]),
      "=r"(__out_var[53]),
      "=r"(__out_var[54]),
      "=r"(__out_var[55]),
      "=r"(__out_var[56]),
      "=r"(__out_var[57]),
      "=r"(__out_var[58]),
      "=r"(__out_var[59]),
      "=r"(__out_var[60]),
      "=r"(__out_var[61]),
      "=r"(__out_var[62]),
      "=r"(__out_var[63]),
      "=r"(__out_var[64]),
      "=r"(__out_var[65]),
      "=r"(__out_var[66]),
      "=r"(__out_var[67]),
      "=r"(__out_var[68]),
      "=r"(__out_var[69]),
      "=r"(__out_var[70]),
      "=r"(__out_var[71]),
      "=r"(__out_var[72]),
      "=r"(__out_var[73]),
      "=r"(__out_var[74]),
      "=r"(__out_var[75]),
      "=r"(__out_var[76]),
      "=r"(__out_var[77]),
      "=r"(__out_var[78]),
      "=r"(__out_var[79]),
      "=r"(__out_var[80]),
      "=r"(__out_var[81]),
      "=r"(__out_var[82]),
      "=r"(__out_var[83]),
      "=r"(__out_var[84]),
      "=r"(__out_var[85]),
      "=r"(__out_var[86]),
      "=r"(__out_var[87]),
      "=r"(__out_var[88]),
      "=r"(__out_var[89]),
      "=r"(__out_var[90]),
      "=r"(__out_var[91]),
      "=r"(__out_var[92]),
      "=r"(__out_var[93]),
      "=r"(__out_var[94]),
      "=r"(__out_var[95]),
      "=r"(__out_var[96]),
      "=r"(__out_var[97]),
      "=r"(__out_var[98]),
      "=r"(__out_var[99]),
      "=r"(__out_var[100]),
      "=r"(__out_var[101]),
      "=r"(__out_var[102]),
      "=r"(__out_var[103]),
      "=r"(__out_var[104]),
      "=r"(__out_var[105]),
      "=r"(__out_var[106]),
      "=r"(__out_var[107]),
      "=r"(__out_var[108]),
      "=r"(__out_var[109]),
      "=r"(__out_var[110]),
      "=r"(__out_var[111]),
      "=r"(__out_var[112]),
      "=r"(__out_var[113]),
      "=r"(__out_var[114]),
      "=r"(__out_var[115]),
      "=r"(__out_var[116]),
      "=r"(__out_var[117]),
      "=r"(__out_var[118]),
      "=r"(__out_var[119]),
      "=r"(__out_var[120]),
      "=r"(__out_var[121]),
      "=r"(__out_var[122]),
      "=r"(__out_var[123]),
      "=r"(__out_var[124]),
      "=r"(__out_var[125]),
      "=r"(__out_var[126]),
      "=r"(__out_var[127])
    : "r"(__taddr)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x256b.x32.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_16x256b_pack_16b(
  B32 (&out_var)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_16x256b_pack_16b(_B32 (&__out_var)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.16x256b.x32.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
    "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, "
    "%102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, "
    "%121, %122, %123, %124, %125, %126, %127}, [%128];"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31]),
      "=r"(__out_var[32]),
      "=r"(__out_var[33]),
      "=r"(__out_var[34]),
      "=r"(__out_var[35]),
      "=r"(__out_var[36]),
      "=r"(__out_var[37]),
      "=r"(__out_var[38]),
      "=r"(__out_var[39]),
      "=r"(__out_var[40]),
      "=r"(__out_var[41]),
      "=r"(__out_var[42]),
      "=r"(__out_var[43]),
      "=r"(__out_var[44]),
      "=r"(__out_var[45]),
      "=r"(__out_var[46]),
      "=r"(__out_var[47]),
      "=r"(__out_var[48]),
      "=r"(__out_var[49]),
      "=r"(__out_var[50]),
      "=r"(__out_var[51]),
      "=r"(__out_var[52]),
      "=r"(__out_var[53]),
      "=r"(__out_var[54]),
      "=r"(__out_var[55]),
      "=r"(__out_var[56]),
      "=r"(__out_var[57]),
      "=r"(__out_var[58]),
      "=r"(__out_var[59]),
      "=r"(__out_var[60]),
      "=r"(__out_var[61]),
      "=r"(__out_var[62]),
      "=r"(__out_var[63]),
      "=r"(__out_var[64]),
      "=r"(__out_var[65]),
      "=r"(__out_var[66]),
      "=r"(__out_var[67]),
      "=r"(__out_var[68]),
      "=r"(__out_var[69]),
      "=r"(__out_var[70]),
      "=r"(__out_var[71]),
      "=r"(__out_var[72]),
      "=r"(__out_var[73]),
      "=r"(__out_var[74]),
      "=r"(__out_var[75]),
      "=r"(__out_var[76]),
      "=r"(__out_var[77]),
      "=r"(__out_var[78]),
      "=r"(__out_var[79]),
      "=r"(__out_var[80]),
      "=r"(__out_var[81]),
      "=r"(__out_var[82]),
      "=r"(__out_var[83]),
      "=r"(__out_var[84]),
      "=r"(__out_var[85]),
      "=r"(__out_var[86]),
      "=r"(__out_var[87]),
      "=r"(__out_var[88]),
      "=r"(__out_var[89]),
      "=r"(__out_var[90]),
      "=r"(__out_var[91]),
      "=r"(__out_var[92]),
      "=r"(__out_var[93]),
      "=r"(__out_var[94]),
      "=r"(__out_var[95]),
      "=r"(__out_var[96]),
      "=r"(__out_var[97]),
      "=r"(__out_var[98]),
      "=r"(__out_var[99]),
      "=r"(__out_var[100]),
      "=r"(__out_var[101]),
      "=r"(__out_var[102]),
      "=r"(__out_var[103]),
      "=r"(__out_var[104]),
      "=r"(__out_var[105]),
      "=r"(__out_var[106]),
      "=r"(__out_var[107]),
      "=r"(__out_var[108]),
      "=r"(__out_var[109]),
      "=r"(__out_var[110]),
      "=r"(__out_var[111]),
      "=r"(__out_var[112]),
      "=r"(__out_var[113]),
      "=r"(__out_var[114]),
      "=r"(__out_var[115]),
      "=r"(__out_var[116]),
      "=r"(__out_var[117]),
      "=r"(__out_var[118]),
      "=r"(__out_var[119]),
      "=r"(__out_var[120]),
      "=r"(__out_var[121]),
      "=r"(__out_var[122]),
      "=r"(__out_var[123]),
      "=r"(__out_var[124]),
      "=r"(__out_var[125]),
      "=r"(__out_var[126]),
      "=r"(__out_var[127])
    : "r"(__taddr)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x1.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b(
  B32 (&out_var)[1],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b(_B32 (&__out_var)[1], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(__out_var[0]) : "r"(__taddr) : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x1.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b_pack_16b(
  B32 (&out_var)[1],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b_pack_16b(_B32 (&__out_var)[1], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.32x32b.x1.pack::16b.b32 {%0}, [%1];" : "=r"(__out_var[0]) : "r"(__taddr) : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x2.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b(
  B32 (&out_var)[2],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b(_B32 (&__out_var)[2], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0, %1}, [%2];"
      : "=r"(__out_var[0]), "=r"(__out_var[1])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x2.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b_pack_16b(
  B32 (&out_var)[2],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b_pack_16b(_B32 (&__out_var)[2], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.32x32b.x2.pack::16b.b32 {%0, %1}, [%2];"
      : "=r"(__out_var[0]), "=r"(__out_var[1])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x4.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b(
  B32 (&out_var)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b(_B32 (&__out_var)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out_var[0]), "=r"(__out_var[1]), "=r"(__out_var[2]), "=r"(__out_var[3])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x4.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b_pack_16b(
  B32 (&out_var)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b_pack_16b(_B32 (&__out_var)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.32x32b.x4.pack::16b.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(__out_var[0]), "=r"(__out_var[1]), "=r"(__out_var[2]), "=r"(__out_var[3])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x8.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b(
  B32 (&out_var)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b(_B32 (&__out_var)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x8.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b_pack_16b(
  B32 (&out_var)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b_pack_16b(_B32 (&__out_var)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.32x32b.x8.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x16.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b(
  B32 (&out_var)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b(_B32 (&__out_var)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
      "[%16];"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x16.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b_pack_16b(
  B32 (&out_var)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b_pack_16b(_B32 (&__out_var)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.32x32b.x16.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15}, [%16];"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15])
      : "r"(__taddr)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x32.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b(
  B32 (&out_var)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b(_B32 (&__out_var)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31])
    : "r"(__taddr)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x32.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b_pack_16b(
  B32 (&out_var)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b_pack_16b(_B32 (&__out_var)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.32x32b.x32.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31])
    : "r"(__taddr)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x64.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b(
  B32 (&out_var)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b(_B32 (&__out_var)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.32x32b.x64.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63}, [%64];"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31]),
      "=r"(__out_var[32]),
      "=r"(__out_var[33]),
      "=r"(__out_var[34]),
      "=r"(__out_var[35]),
      "=r"(__out_var[36]),
      "=r"(__out_var[37]),
      "=r"(__out_var[38]),
      "=r"(__out_var[39]),
      "=r"(__out_var[40]),
      "=r"(__out_var[41]),
      "=r"(__out_var[42]),
      "=r"(__out_var[43]),
      "=r"(__out_var[44]),
      "=r"(__out_var[45]),
      "=r"(__out_var[46]),
      "=r"(__out_var[47]),
      "=r"(__out_var[48]),
      "=r"(__out_var[49]),
      "=r"(__out_var[50]),
      "=r"(__out_var[51]),
      "=r"(__out_var[52]),
      "=r"(__out_var[53]),
      "=r"(__out_var[54]),
      "=r"(__out_var[55]),
      "=r"(__out_var[56]),
      "=r"(__out_var[57]),
      "=r"(__out_var[58]),
      "=r"(__out_var[59]),
      "=r"(__out_var[60]),
      "=r"(__out_var[61]),
      "=r"(__out_var[62]),
      "=r"(__out_var[63])
    : "r"(__taddr)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x64.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f,
SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b_pack_16b(
  B32 (&out_var)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b_pack_16b(_B32 (&__out_var)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.32x32b.x64.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63}, [%64];"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31]),
      "=r"(__out_var[32]),
      "=r"(__out_var[33]),
      "=r"(__out_var[34]),
      "=r"(__out_var[35]),
      "=r"(__out_var[36]),
      "=r"(__out_var[37]),
      "=r"(__out_var[38]),
      "=r"(__out_var[39]),
      "=r"(__out_var[40]),
      "=r"(__out_var[41]),
      "=r"(__out_var[42]),
      "=r"(__out_var[43]),
      "=r"(__out_var[44]),
      "=r"(__out_var[45]),
      "=r"(__out_var[46]),
      "=r"(__out_var[47]),
      "=r"(__out_var[48]),
      "=r"(__out_var[49]),
      "=r"(__out_var[50]),
      "=r"(__out_var[51]),
      "=r"(__out_var[52]),
      "=r"(__out_var[53]),
      "=r"(__out_var[54]),
      "=r"(__out_var[55]),
      "=r"(__out_var[56]),
      "=r"(__out_var[57]),
      "=r"(__out_var[58]),
      "=r"(__out_var[59]),
      "=r"(__out_var[60]),
      "=r"(__out_var[61]),
      "=r"(__out_var[62]),
      "=r"(__out_var[63])
    : "r"(__taddr)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x128.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b(
  B32 (&out_var)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b(_B32 (&__out_var)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.32x32b.x128.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, "
    "%82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, "
    "%104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
    "%123, %124, %125, %126, %127}, [%128];"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31]),
      "=r"(__out_var[32]),
      "=r"(__out_var[33]),
      "=r"(__out_var[34]),
      "=r"(__out_var[35]),
      "=r"(__out_var[36]),
      "=r"(__out_var[37]),
      "=r"(__out_var[38]),
      "=r"(__out_var[39]),
      "=r"(__out_var[40]),
      "=r"(__out_var[41]),
      "=r"(__out_var[42]),
      "=r"(__out_var[43]),
      "=r"(__out_var[44]),
      "=r"(__out_var[45]),
      "=r"(__out_var[46]),
      "=r"(__out_var[47]),
      "=r"(__out_var[48]),
      "=r"(__out_var[49]),
      "=r"(__out_var[50]),
      "=r"(__out_var[51]),
      "=r"(__out_var[52]),
      "=r"(__out_var[53]),
      "=r"(__out_var[54]),
      "=r"(__out_var[55]),
      "=r"(__out_var[56]),
      "=r"(__out_var[57]),
      "=r"(__out_var[58]),
      "=r"(__out_var[59]),
      "=r"(__out_var[60]),
      "=r"(__out_var[61]),
      "=r"(__out_var[62]),
      "=r"(__out_var[63]),
      "=r"(__out_var[64]),
      "=r"(__out_var[65]),
      "=r"(__out_var[66]),
      "=r"(__out_var[67]),
      "=r"(__out_var[68]),
      "=r"(__out_var[69]),
      "=r"(__out_var[70]),
      "=r"(__out_var[71]),
      "=r"(__out_var[72]),
      "=r"(__out_var[73]),
      "=r"(__out_var[74]),
      "=r"(__out_var[75]),
      "=r"(__out_var[76]),
      "=r"(__out_var[77]),
      "=r"(__out_var[78]),
      "=r"(__out_var[79]),
      "=r"(__out_var[80]),
      "=r"(__out_var[81]),
      "=r"(__out_var[82]),
      "=r"(__out_var[83]),
      "=r"(__out_var[84]),
      "=r"(__out_var[85]),
      "=r"(__out_var[86]),
      "=r"(__out_var[87]),
      "=r"(__out_var[88]),
      "=r"(__out_var[89]),
      "=r"(__out_var[90]),
      "=r"(__out_var[91]),
      "=r"(__out_var[92]),
      "=r"(__out_var[93]),
      "=r"(__out_var[94]),
      "=r"(__out_var[95]),
      "=r"(__out_var[96]),
      "=r"(__out_var[97]),
      "=r"(__out_var[98]),
      "=r"(__out_var[99]),
      "=r"(__out_var[100]),
      "=r"(__out_var[101]),
      "=r"(__out_var[102]),
      "=r"(__out_var[103]),
      "=r"(__out_var[104]),
      "=r"(__out_var[105]),
      "=r"(__out_var[106]),
      "=r"(__out_var[107]),
      "=r"(__out_var[108]),
      "=r"(__out_var[109]),
      "=r"(__out_var[110]),
      "=r"(__out_var[111]),
      "=r"(__out_var[112]),
      "=r"(__out_var[113]),
      "=r"(__out_var[114]),
      "=r"(__out_var[115]),
      "=r"(__out_var[116]),
      "=r"(__out_var[117]),
      "=r"(__out_var[118]),
      "=r"(__out_var[119]),
      "=r"(__out_var[120]),
      "=r"(__out_var[121]),
      "=r"(__out_var[122]),
      "=r"(__out_var[123]),
      "=r"(__out_var[124]),
      "=r"(__out_var[125]),
      "=r"(__out_var[126]),
      "=r"(__out_var[127])
    : "r"(__taddr)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.32x32b.x128.pack::16b.b32 out_var, [taddr]; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline void tcgen05_ld_32x32b_pack_16b(
  B32 (&out_var)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void tcgen05_ld_32x32b_pack_16b(_B32 (&__out_var)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.32x32b.x128.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
    "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, "
    "%102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, "
    "%121, %122, %123, %124, %125, %126, %127}, [%128];"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31]),
      "=r"(__out_var[32]),
      "=r"(__out_var[33]),
      "=r"(__out_var[34]),
      "=r"(__out_var[35]),
      "=r"(__out_var[36]),
      "=r"(__out_var[37]),
      "=r"(__out_var[38]),
      "=r"(__out_var[39]),
      "=r"(__out_var[40]),
      "=r"(__out_var[41]),
      "=r"(__out_var[42]),
      "=r"(__out_var[43]),
      "=r"(__out_var[44]),
      "=r"(__out_var[45]),
      "=r"(__out_var[46]),
      "=r"(__out_var[47]),
      "=r"(__out_var[48]),
      "=r"(__out_var[49]),
      "=r"(__out_var[50]),
      "=r"(__out_var[51]),
      "=r"(__out_var[52]),
      "=r"(__out_var[53]),
      "=r"(__out_var[54]),
      "=r"(__out_var[55]),
      "=r"(__out_var[56]),
      "=r"(__out_var[57]),
      "=r"(__out_var[58]),
      "=r"(__out_var[59]),
      "=r"(__out_var[60]),
      "=r"(__out_var[61]),
      "=r"(__out_var[62]),
      "=r"(__out_var[63]),
      "=r"(__out_var[64]),
      "=r"(__out_var[65]),
      "=r"(__out_var[66]),
      "=r"(__out_var[67]),
      "=r"(__out_var[68]),
      "=r"(__out_var[69]),
      "=r"(__out_var[70]),
      "=r"(__out_var[71]),
      "=r"(__out_var[72]),
      "=r"(__out_var[73]),
      "=r"(__out_var[74]),
      "=r"(__out_var[75]),
      "=r"(__out_var[76]),
      "=r"(__out_var[77]),
      "=r"(__out_var[78]),
      "=r"(__out_var[79]),
      "=r"(__out_var[80]),
      "=r"(__out_var[81]),
      "=r"(__out_var[82]),
      "=r"(__out_var[83]),
      "=r"(__out_var[84]),
      "=r"(__out_var[85]),
      "=r"(__out_var[86]),
      "=r"(__out_var[87]),
      "=r"(__out_var[88]),
      "=r"(__out_var[89]),
      "=r"(__out_var[90]),
      "=r"(__out_var[91]),
      "=r"(__out_var[92]),
      "=r"(__out_var[93]),
      "=r"(__out_var[94]),
      "=r"(__out_var[95]),
      "=r"(__out_var[96]),
      "=r"(__out_var[97]),
      "=r"(__out_var[98]),
      "=r"(__out_var[99]),
      "=r"(__out_var[100]),
      "=r"(__out_var[101]),
      "=r"(__out_var[102]),
      "=r"(__out_var[103]),
      "=r"(__out_var[104]),
      "=r"(__out_var[105]),
      "=r"(__out_var[106]),
      "=r"(__out_var[107]),
      "=r"(__out_var[108]),
      "=r"(__out_var[109]),
      "=r"(__out_var[110]),
      "=r"(__out_var[111]),
      "=r"(__out_var[112]),
      "=r"(__out_var[113]),
      "=r"(__out_var[114]),
      "=r"(__out_var[115]),
      "=r"(__out_var[116]),
      "=r"(__out_var[117]),
      "=r"(__out_var[118]),
      "=r"(__out_var[119]),
      "=r"(__out_var[120]),
      "=r"(__out_var[121]),
      "=r"(__out_var[122]),
      "=r"(__out_var[123]),
      "=r"(__out_var[124]),
      "=r"(__out_var[125]),
      "=r"(__out_var[126]),
      "=r"(__out_var[127])
    : "r"(__taddr)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x1.b32 out_var, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2(
  B32 (&out_var)[1],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void
tcgen05_ld_16x32bx2(_B32 (&__out_var)[1], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x32bx2.x1.b32 {%0}, [%1], %2;"
      : "=r"(__out_var[0])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x1.pack::16b.b32 out_var, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f,
SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true,
int N32>
__device__ static inline void tcgen05_ld_16x32bx2_pack_16b(
  B32 (&out_var)[1],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void tcgen05_ld_16x32bx2_pack_16b(
  _B32 (&__out_var)[1], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x32bx2.x1.pack::16b.b32 {%0}, [%1], %2;"
      : "=r"(__out_var[0])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x2.b32 out_var, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2(
  B32 (&out_var)[2],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void
tcgen05_ld_16x32bx2(_B32 (&__out_var)[2], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x32bx2.x2.b32 {%0, %1}, [%2], %3;"
      : "=r"(__out_var[0]), "=r"(__out_var[1])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x2.pack::16b.b32 out_var, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f,
SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true,
int N32>
__device__ static inline void tcgen05_ld_16x32bx2_pack_16b(
  B32 (&out_var)[2],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void tcgen05_ld_16x32bx2_pack_16b(
  _B32 (&__out_var)[2], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x32bx2.x2.pack::16b.b32 {%0, %1}, [%2], %3;"
      : "=r"(__out_var[0]), "=r"(__out_var[1])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x4.b32 out_var, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2(
  B32 (&out_var)[4],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void
tcgen05_ld_16x32bx2(_B32 (&__out_var)[4], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x32bx2.x4.b32 {%0, %1, %2, %3}, [%4], %5;"
      : "=r"(__out_var[0]), "=r"(__out_var[1]), "=r"(__out_var[2]), "=r"(__out_var[3])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x4.pack::16b.b32 out_var, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f,
SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true,
int N32>
__device__ static inline void tcgen05_ld_16x32bx2_pack_16b(
  B32 (&out_var)[4],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void tcgen05_ld_16x32bx2_pack_16b(
  _B32 (&__out_var)[4], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x32bx2.x4.pack::16b.b32 {%0, %1, %2, %3}, [%4], %5;"
      : "=r"(__out_var[0]), "=r"(__out_var[1]), "=r"(__out_var[2]), "=r"(__out_var[3])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x8.b32 out_var, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2(
  B32 (&out_var)[8],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void
tcgen05_ld_16x32bx2(_B32 (&__out_var)[8], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x32bx2.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x8.pack::16b.b32 out_var, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f,
SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true,
int N32>
__device__ static inline void tcgen05_ld_16x32bx2_pack_16b(
  B32 (&out_var)[8],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void tcgen05_ld_16x32bx2_pack_16b(
  _B32 (&__out_var)[8], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x32bx2.x8.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x16.b32 out_var, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2(
  B32 (&out_var)[16],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void
tcgen05_ld_16x32bx2(_B32 (&__out_var)[16], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x32bx2.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
      "%15}, [%16], %17;"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x16.pack::16b.b32 out_var, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a,
SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4,
bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2_pack_16b(
  B32 (&out_var)[16],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void tcgen05_ld_16x32bx2_pack_16b(
  _B32 (&__out_var)[16], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
  asm("tcgen05.ld.sync.aligned.16x32bx2.x16.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15}, [%16], %17;"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15])
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x32.b32 out_var, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2(
  B32 (&out_var)[32],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void
tcgen05_ld_16x32bx2(_B32 (&__out_var)[32], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], %33;"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31])
    : "r"(__taddr), "n"(__immHalfSplitoff.value)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x32.pack::16b.b32 out_var, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a,
SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4,
bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2_pack_16b(
  B32 (&out_var)[32],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void tcgen05_ld_16x32bx2_pack_16b(
  _B32 (&__out_var)[32], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.16x32bx2.x32.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], %33;"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31])
    : "r"(__taddr), "n"(__immHalfSplitoff.value)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x64.b32 out_var, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2(
  B32 (&out_var)[64],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void
tcgen05_ld_16x32bx2(_B32 (&__out_var)[64], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.16x32bx2.x64.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63}, [%64], %65;"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31]),
      "=r"(__out_var[32]),
      "=r"(__out_var[33]),
      "=r"(__out_var[34]),
      "=r"(__out_var[35]),
      "=r"(__out_var[36]),
      "=r"(__out_var[37]),
      "=r"(__out_var[38]),
      "=r"(__out_var[39]),
      "=r"(__out_var[40]),
      "=r"(__out_var[41]),
      "=r"(__out_var[42]),
      "=r"(__out_var[43]),
      "=r"(__out_var[44]),
      "=r"(__out_var[45]),
      "=r"(__out_var[46]),
      "=r"(__out_var[47]),
      "=r"(__out_var[48]),
      "=r"(__out_var[49]),
      "=r"(__out_var[50]),
      "=r"(__out_var[51]),
      "=r"(__out_var[52]),
      "=r"(__out_var[53]),
      "=r"(__out_var[54]),
      "=r"(__out_var[55]),
      "=r"(__out_var[56]),
      "=r"(__out_var[57]),
      "=r"(__out_var[58]),
      "=r"(__out_var[59]),
      "=r"(__out_var[60]),
      "=r"(__out_var[61]),
      "=r"(__out_var[62]),
      "=r"(__out_var[63])
    : "r"(__taddr), "n"(__immHalfSplitoff.value)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x64.pack::16b.b32 out_var, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a,
SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4,
bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2_pack_16b(
  B32 (&out_var)[64],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void tcgen05_ld_16x32bx2_pack_16b(
  _B32 (&__out_var)[64], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.16x32bx2.x64.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63}, [%64], %65;"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31]),
      "=r"(__out_var[32]),
      "=r"(__out_var[33]),
      "=r"(__out_var[34]),
      "=r"(__out_var[35]),
      "=r"(__out_var[36]),
      "=r"(__out_var[37]),
      "=r"(__out_var[38]),
      "=r"(__out_var[39]),
      "=r"(__out_var[40]),
      "=r"(__out_var[41]),
      "=r"(__out_var[42]),
      "=r"(__out_var[43]),
      "=r"(__out_var[44]),
      "=r"(__out_var[45]),
      "=r"(__out_var[46]),
      "=r"(__out_var[47]),
      "=r"(__out_var[48]),
      "=r"(__out_var[49]),
      "=r"(__out_var[50]),
      "=r"(__out_var[51]),
      "=r"(__out_var[52]),
      "=r"(__out_var[53]),
      "=r"(__out_var[54]),
      "=r"(__out_var[55]),
      "=r"(__out_var[56]),
      "=r"(__out_var[57]),
      "=r"(__out_var[58]),
      "=r"(__out_var[59]),
      "=r"(__out_var[60]),
      "=r"(__out_var[61]),
      "=r"(__out_var[62]),
      "=r"(__out_var[63])
    : "r"(__taddr), "n"(__immHalfSplitoff.value)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x128.b32 out_var, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a, SM_100f,
SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true,
int N32>
__device__ static inline void tcgen05_ld_16x32bx2(
  B32 (&out_var)[128],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void
tcgen05_ld_16x32bx2(_B32 (&__out_var)[128], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.16x32bx2.x128.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, "
    "%38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, "
    "%60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, "
    "%82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, "
    "%104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
    "%123, %124, %125, %126, %127}, [%128], %129;"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31]),
      "=r"(__out_var[32]),
      "=r"(__out_var[33]),
      "=r"(__out_var[34]),
      "=r"(__out_var[35]),
      "=r"(__out_var[36]),
      "=r"(__out_var[37]),
      "=r"(__out_var[38]),
      "=r"(__out_var[39]),
      "=r"(__out_var[40]),
      "=r"(__out_var[41]),
      "=r"(__out_var[42]),
      "=r"(__out_var[43]),
      "=r"(__out_var[44]),
      "=r"(__out_var[45]),
      "=r"(__out_var[46]),
      "=r"(__out_var[47]),
      "=r"(__out_var[48]),
      "=r"(__out_var[49]),
      "=r"(__out_var[50]),
      "=r"(__out_var[51]),
      "=r"(__out_var[52]),
      "=r"(__out_var[53]),
      "=r"(__out_var[54]),
      "=r"(__out_var[55]),
      "=r"(__out_var[56]),
      "=r"(__out_var[57]),
      "=r"(__out_var[58]),
      "=r"(__out_var[59]),
      "=r"(__out_var[60]),
      "=r"(__out_var[61]),
      "=r"(__out_var[62]),
      "=r"(__out_var[63]),
      "=r"(__out_var[64]),
      "=r"(__out_var[65]),
      "=r"(__out_var[66]),
      "=r"(__out_var[67]),
      "=r"(__out_var[68]),
      "=r"(__out_var[69]),
      "=r"(__out_var[70]),
      "=r"(__out_var[71]),
      "=r"(__out_var[72]),
      "=r"(__out_var[73]),
      "=r"(__out_var[74]),
      "=r"(__out_var[75]),
      "=r"(__out_var[76]),
      "=r"(__out_var[77]),
      "=r"(__out_var[78]),
      "=r"(__out_var[79]),
      "=r"(__out_var[80]),
      "=r"(__out_var[81]),
      "=r"(__out_var[82]),
      "=r"(__out_var[83]),
      "=r"(__out_var[84]),
      "=r"(__out_var[85]),
      "=r"(__out_var[86]),
      "=r"(__out_var[87]),
      "=r"(__out_var[88]),
      "=r"(__out_var[89]),
      "=r"(__out_var[90]),
      "=r"(__out_var[91]),
      "=r"(__out_var[92]),
      "=r"(__out_var[93]),
      "=r"(__out_var[94]),
      "=r"(__out_var[95]),
      "=r"(__out_var[96]),
      "=r"(__out_var[97]),
      "=r"(__out_var[98]),
      "=r"(__out_var[99]),
      "=r"(__out_var[100]),
      "=r"(__out_var[101]),
      "=r"(__out_var[102]),
      "=r"(__out_var[103]),
      "=r"(__out_var[104]),
      "=r"(__out_var[105]),
      "=r"(__out_var[106]),
      "=r"(__out_var[107]),
      "=r"(__out_var[108]),
      "=r"(__out_var[109]),
      "=r"(__out_var[110]),
      "=r"(__out_var[111]),
      "=r"(__out_var[112]),
      "=r"(__out_var[113]),
      "=r"(__out_var[114]),
      "=r"(__out_var[115]),
      "=r"(__out_var[116]),
      "=r"(__out_var[117]),
      "=r"(__out_var[118]),
      "=r"(__out_var[119]),
      "=r"(__out_var[120]),
      "=r"(__out_var[121]),
      "=r"(__out_var[122]),
      "=r"(__out_var[123]),
      "=r"(__out_var[124]),
      "=r"(__out_var[125]),
      "=r"(__out_var[126]),
      "=r"(__out_var[127])
    : "r"(__taddr), "n"(__immHalfSplitoff.value)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.sync.aligned.16x32bx2.x128.pack::16b.b32 out_var, [taddr], immHalfSplitoff; // PTX ISA 86, SM_100a,
SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f template <typename B32, enable_if_t<sizeof(B32) == 4,
bool> = true, int N32>
__device__ static inline void tcgen05_ld_16x32bx2_pack_16b(
  B32 (&out_var)[128],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32, ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true, int _N32>
_CCCL_DEVICE static inline void tcgen05_ld_16x32bx2_pack_16b(
  _B32 (&__out_var)[128], ::cuda::std::uint32_t __taddr, ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(sizeof(_B32) == 4, "");
  asm(
    "tcgen05.ld.sync.aligned.16x32bx2.x128.pack::16b.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
    "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
    "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
    "%58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
    "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, "
    "%102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, "
    "%121, %122, %123, %124, %125, %126, %127}, [%128], %129;"
    : "=r"(__out_var[0]),
      "=r"(__out_var[1]),
      "=r"(__out_var[2]),
      "=r"(__out_var[3]),
      "=r"(__out_var[4]),
      "=r"(__out_var[5]),
      "=r"(__out_var[6]),
      "=r"(__out_var[7]),
      "=r"(__out_var[8]),
      "=r"(__out_var[9]),
      "=r"(__out_var[10]),
      "=r"(__out_var[11]),
      "=r"(__out_var[12]),
      "=r"(__out_var[13]),
      "=r"(__out_var[14]),
      "=r"(__out_var[15]),
      "=r"(__out_var[16]),
      "=r"(__out_var[17]),
      "=r"(__out_var[18]),
      "=r"(__out_var[19]),
      "=r"(__out_var[20]),
      "=r"(__out_var[21]),
      "=r"(__out_var[22]),
      "=r"(__out_var[23]),
      "=r"(__out_var[24]),
      "=r"(__out_var[25]),
      "=r"(__out_var[26]),
      "=r"(__out_var[27]),
      "=r"(__out_var[28]),
      "=r"(__out_var[29]),
      "=r"(__out_var[30]),
      "=r"(__out_var[31]),
      "=r"(__out_var[32]),
      "=r"(__out_var[33]),
      "=r"(__out_var[34]),
      "=r"(__out_var[35]),
      "=r"(__out_var[36]),
      "=r"(__out_var[37]),
      "=r"(__out_var[38]),
      "=r"(__out_var[39]),
      "=r"(__out_var[40]),
      "=r"(__out_var[41]),
      "=r"(__out_var[42]),
      "=r"(__out_var[43]),
      "=r"(__out_var[44]),
      "=r"(__out_var[45]),
      "=r"(__out_var[46]),
      "=r"(__out_var[47]),
      "=r"(__out_var[48]),
      "=r"(__out_var[49]),
      "=r"(__out_var[50]),
      "=r"(__out_var[51]),
      "=r"(__out_var[52]),
      "=r"(__out_var[53]),
      "=r"(__out_var[54]),
      "=r"(__out_var[55]),
      "=r"(__out_var[56]),
      "=r"(__out_var[57]),
      "=r"(__out_var[58]),
      "=r"(__out_var[59]),
      "=r"(__out_var[60]),
      "=r"(__out_var[61]),
      "=r"(__out_var[62]),
      "=r"(__out_var[63]),
      "=r"(__out_var[64]),
      "=r"(__out_var[65]),
      "=r"(__out_var[66]),
      "=r"(__out_var[67]),
      "=r"(__out_var[68]),
      "=r"(__out_var[69]),
      "=r"(__out_var[70]),
      "=r"(__out_var[71]),
      "=r"(__out_var[72]),
      "=r"(__out_var[73]),
      "=r"(__out_var[74]),
      "=r"(__out_var[75]),
      "=r"(__out_var[76]),
      "=r"(__out_var[77]),
      "=r"(__out_var[78]),
      "=r"(__out_var[79]),
      "=r"(__out_var[80]),
      "=r"(__out_var[81]),
      "=r"(__out_var[82]),
      "=r"(__out_var[83]),
      "=r"(__out_var[84]),
      "=r"(__out_var[85]),
      "=r"(__out_var[86]),
      "=r"(__out_var[87]),
      "=r"(__out_var[88]),
      "=r"(__out_var[89]),
      "=r"(__out_var[90]),
      "=r"(__out_var[91]),
      "=r"(__out_var[92]),
      "=r"(__out_var[93]),
      "=r"(__out_var[94]),
      "=r"(__out_var[95]),
      "=r"(__out_var[96]),
      "=r"(__out_var[97]),
      "=r"(__out_var[98]),
      "=r"(__out_var[99]),
      "=r"(__out_var[100]),
      "=r"(__out_var[101]),
      "=r"(__out_var[102]),
      "=r"(__out_var[103]),
      "=r"(__out_var[104]),
      "=r"(__out_var[105]),
      "=r"(__out_var[106]),
      "=r"(__out_var[107]),
      "=r"(__out_var[108]),
      "=r"(__out_var[109]),
      "=r"(__out_var[110]),
      "=r"(__out_var[111]),
      "=r"(__out_var[112]),
      "=r"(__out_var[113]),
      "=r"(__out_var[114]),
      "=r"(__out_var[115]),
      "=r"(__out_var[116]),
      "=r"(__out_var[117]),
      "=r"(__out_var[118]),
      "=r"(__out_var[119]),
      "=r"(__out_var[120]),
      "=r"(__out_var[121]),
      "=r"(__out_var[122]),
      "=r"(__out_var[123]),
      "=r"(__out_var[124]),
      "=r"(__out_var[125]),
      "=r"(__out_var[126]),
      "=r"(__out_var[127])
    : "r"(__taddr), "n"(__immHalfSplitoff.value)
    : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.ld.red.sync.aligned.32x32b.x2.u32.op out_var, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline uint32_t tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  uint32_t (&out_var)[2],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::uint32_t
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, ::cuda::std::uint32_t (&__out_var)[2], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  ::cuda::std::uint32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x2.u32.min {%0, %1}, %2, [%3];"
        : "=r"(__out_var[0]), "=r"(__out_var[1]), "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x2.u32.max {%0, %1}, %2, [%3];"
        : "=r"(__out_var[0]), "=r"(__out_var[1]), "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x2.s32.op out_var, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline int32_t tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  int32_t (&out_var)[2],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::int32_t
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, ::cuda::std::int32_t (&__out_var)[2], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  ::cuda::std::int32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x2.s32.min {%0, %1}, %2, [%3];"
        : "=r"(__out_var[0]), "=r"(__out_var[1]), "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x2.s32.max {%0, %1}, %2, [%3];"
        : "=r"(__out_var[0]), "=r"(__out_var[1]), "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x2.f32.op.abs out_var, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_32x32b_abs(
  cuda::ptx::op_t<Op> op,
  float (&out_var)[2],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float
tcgen05_ld_red_32x32b_abs(::cuda::ptx::op_t<_Op> __op, float (&__out_var)[2], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x2.f32.min.abs {%0, %1}, %2, [%3];"
        : "=f"(__out_var[0]), "=f"(__out_var[1]), "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x2.f32.max.abs {%0, %1}, %2, [%3];"
        : "=f"(__out_var[0]), "=f"(__out_var[1]), "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x2.f32.op out_var, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  float (&out_var)[2],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, float (&__out_var)[2], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x2.f32.min {%0, %1}, %2, [%3];"
        : "=f"(__out_var[0]), "=f"(__out_var[1]), "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x2.f32.max {%0, %1}, %2, [%3];"
        : "=f"(__out_var[0]), "=f"(__out_var[1]), "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x4.u32.op out_var, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline uint32_t tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  uint32_t (&out_var)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::uint32_t
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, ::cuda::std::uint32_t (&__out_var)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  ::cuda::std::uint32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x4.u32.min {%0, %1, %2, %3}, %4, [%5];"
        : "=r"(__out_var[0]), "=r"(__out_var[1]), "=r"(__out_var[2]), "=r"(__out_var[3]), "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x4.u32.max {%0, %1, %2, %3}, %4, [%5];"
        : "=r"(__out_var[0]), "=r"(__out_var[1]), "=r"(__out_var[2]), "=r"(__out_var[3]), "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x4.s32.op out_var, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline int32_t tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  int32_t (&out_var)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::int32_t
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, ::cuda::std::int32_t (&__out_var)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  ::cuda::std::int32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x4.s32.min {%0, %1, %2, %3}, %4, [%5];"
        : "=r"(__out_var[0]), "=r"(__out_var[1]), "=r"(__out_var[2]), "=r"(__out_var[3]), "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x4.s32.max {%0, %1, %2, %3}, %4, [%5];"
        : "=r"(__out_var[0]), "=r"(__out_var[1]), "=r"(__out_var[2]), "=r"(__out_var[3]), "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x4.f32.op.abs out_var, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_32x32b_abs(
  cuda::ptx::op_t<Op> op,
  float (&out_var)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float
tcgen05_ld_red_32x32b_abs(::cuda::ptx::op_t<_Op> __op, float (&__out_var)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x4.f32.min.abs {%0, %1, %2, %3}, %4, [%5];"
        : "=f"(__out_var[0]), "=f"(__out_var[1]), "=f"(__out_var[2]), "=f"(__out_var[3]), "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x4.f32.max.abs {%0, %1, %2, %3}, %4, [%5];"
        : "=f"(__out_var[0]), "=f"(__out_var[1]), "=f"(__out_var[2]), "=f"(__out_var[3]), "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x4.f32.op out_var, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  float (&out_var)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, float (&__out_var)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x4.f32.min {%0, %1, %2, %3}, %4, [%5];"
        : "=f"(__out_var[0]), "=f"(__out_var[1]), "=f"(__out_var[2]), "=f"(__out_var[3]), "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x4.f32.max {%0, %1, %2, %3}, %4, [%5];"
        : "=f"(__out_var[0]), "=f"(__out_var[1]), "=f"(__out_var[2]), "=f"(__out_var[3]), "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x8.u32.op out_var, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline uint32_t tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  uint32_t (&out_var)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::uint32_t
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, ::cuda::std::uint32_t (&__out_var)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  ::cuda::std::uint32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x8.u32.min {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9];"
        : "=r"(__out_var[0]),
          "=r"(__out_var[1]),
          "=r"(__out_var[2]),
          "=r"(__out_var[3]),
          "=r"(__out_var[4]),
          "=r"(__out_var[5]),
          "=r"(__out_var[6]),
          "=r"(__out_var[7]),
          "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x8.u32.max {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9];"
        : "=r"(__out_var[0]),
          "=r"(__out_var[1]),
          "=r"(__out_var[2]),
          "=r"(__out_var[3]),
          "=r"(__out_var[4]),
          "=r"(__out_var[5]),
          "=r"(__out_var[6]),
          "=r"(__out_var[7]),
          "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x8.s32.op out_var, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline int32_t tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  int32_t (&out_var)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::int32_t
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, ::cuda::std::int32_t (&__out_var)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  ::cuda::std::int32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x8.s32.min {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9];"
        : "=r"(__out_var[0]),
          "=r"(__out_var[1]),
          "=r"(__out_var[2]),
          "=r"(__out_var[3]),
          "=r"(__out_var[4]),
          "=r"(__out_var[5]),
          "=r"(__out_var[6]),
          "=r"(__out_var[7]),
          "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x8.s32.max {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9];"
        : "=r"(__out_var[0]),
          "=r"(__out_var[1]),
          "=r"(__out_var[2]),
          "=r"(__out_var[3]),
          "=r"(__out_var[4]),
          "=r"(__out_var[5]),
          "=r"(__out_var[6]),
          "=r"(__out_var[7]),
          "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x8.f32.op.abs out_var, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_32x32b_abs(
  cuda::ptx::op_t<Op> op,
  float (&out_var)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float
tcgen05_ld_red_32x32b_abs(::cuda::ptx::op_t<_Op> __op, float (&__out_var)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x8.f32.min.abs {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9];"
        : "=f"(__out_var[0]),
          "=f"(__out_var[1]),
          "=f"(__out_var[2]),
          "=f"(__out_var[3]),
          "=f"(__out_var[4]),
          "=f"(__out_var[5]),
          "=f"(__out_var[6]),
          "=f"(__out_var[7]),
          "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x8.f32.max.abs {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9];"
        : "=f"(__out_var[0]),
          "=f"(__out_var[1]),
          "=f"(__out_var[2]),
          "=f"(__out_var[3]),
          "=f"(__out_var[4]),
          "=f"(__out_var[5]),
          "=f"(__out_var[6]),
          "=f"(__out_var[7]),
          "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x8.f32.op out_var, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  float (&out_var)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, float (&__out_var)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x8.f32.min {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9];"
        : "=f"(__out_var[0]),
          "=f"(__out_var[1]),
          "=f"(__out_var[2]),
          "=f"(__out_var[3]),
          "=f"(__out_var[4]),
          "=f"(__out_var[5]),
          "=f"(__out_var[6]),
          "=f"(__out_var[7]),
          "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x8.f32.max {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9];"
        : "=f"(__out_var[0]),
          "=f"(__out_var[1]),
          "=f"(__out_var[2]),
          "=f"(__out_var[3]),
          "=f"(__out_var[4]),
          "=f"(__out_var[5]),
          "=f"(__out_var[6]),
          "=f"(__out_var[7]),
          "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x16.u32.op out_var, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline uint32_t tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  uint32_t (&out_var)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::uint32_t tcgen05_ld_red_32x32b(
  ::cuda::ptx::op_t<_Op> __op, ::cuda::std::uint32_t (&__out_var)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  ::cuda::std::uint32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x16.u32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
        "%14, %15}, %16, [%17];"
        : "=r"(__out_var[0]),
          "=r"(__out_var[1]),
          "=r"(__out_var[2]),
          "=r"(__out_var[3]),
          "=r"(__out_var[4]),
          "=r"(__out_var[5]),
          "=r"(__out_var[6]),
          "=r"(__out_var[7]),
          "=r"(__out_var[8]),
          "=r"(__out_var[9]),
          "=r"(__out_var[10]),
          "=r"(__out_var[11]),
          "=r"(__out_var[12]),
          "=r"(__out_var[13]),
          "=r"(__out_var[14]),
          "=r"(__out_var[15]),
          "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x16.u32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
        "%14, %15}, %16, [%17];"
        : "=r"(__out_var[0]),
          "=r"(__out_var[1]),
          "=r"(__out_var[2]),
          "=r"(__out_var[3]),
          "=r"(__out_var[4]),
          "=r"(__out_var[5]),
          "=r"(__out_var[6]),
          "=r"(__out_var[7]),
          "=r"(__out_var[8]),
          "=r"(__out_var[9]),
          "=r"(__out_var[10]),
          "=r"(__out_var[11]),
          "=r"(__out_var[12]),
          "=r"(__out_var[13]),
          "=r"(__out_var[14]),
          "=r"(__out_var[15]),
          "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x16.s32.op out_var, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline int32_t tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  int32_t (&out_var)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::int32_t
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, ::cuda::std::int32_t (&__out_var)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  ::cuda::std::int32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x16.s32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
        "%14, %15}, %16, [%17];"
        : "=r"(__out_var[0]),
          "=r"(__out_var[1]),
          "=r"(__out_var[2]),
          "=r"(__out_var[3]),
          "=r"(__out_var[4]),
          "=r"(__out_var[5]),
          "=r"(__out_var[6]),
          "=r"(__out_var[7]),
          "=r"(__out_var[8]),
          "=r"(__out_var[9]),
          "=r"(__out_var[10]),
          "=r"(__out_var[11]),
          "=r"(__out_var[12]),
          "=r"(__out_var[13]),
          "=r"(__out_var[14]),
          "=r"(__out_var[15]),
          "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x16.s32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
        "%14, %15}, %16, [%17];"
        : "=r"(__out_var[0]),
          "=r"(__out_var[1]),
          "=r"(__out_var[2]),
          "=r"(__out_var[3]),
          "=r"(__out_var[4]),
          "=r"(__out_var[5]),
          "=r"(__out_var[6]),
          "=r"(__out_var[7]),
          "=r"(__out_var[8]),
          "=r"(__out_var[9]),
          "=r"(__out_var[10]),
          "=r"(__out_var[11]),
          "=r"(__out_var[12]),
          "=r"(__out_var[13]),
          "=r"(__out_var[14]),
          "=r"(__out_var[15]),
          "=r"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x16.f32.op.abs out_var, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_32x32b_abs(
  cuda::ptx::op_t<Op> op,
  float (&out_var)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float
tcgen05_ld_red_32x32b_abs(::cuda::ptx::op_t<_Op> __op, float (&__out_var)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x16.f32.min.abs {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
        "%13, %14, %15}, %16, [%17];"
        : "=f"(__out_var[0]),
          "=f"(__out_var[1]),
          "=f"(__out_var[2]),
          "=f"(__out_var[3]),
          "=f"(__out_var[4]),
          "=f"(__out_var[5]),
          "=f"(__out_var[6]),
          "=f"(__out_var[7]),
          "=f"(__out_var[8]),
          "=f"(__out_var[9]),
          "=f"(__out_var[10]),
          "=f"(__out_var[11]),
          "=f"(__out_var[12]),
          "=f"(__out_var[13]),
          "=f"(__out_var[14]),
          "=f"(__out_var[15]),
          "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x16.f32.max.abs {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
        "%13, %14, %15}, %16, [%17];"
        : "=f"(__out_var[0]),
          "=f"(__out_var[1]),
          "=f"(__out_var[2]),
          "=f"(__out_var[3]),
          "=f"(__out_var[4]),
          "=f"(__out_var[5]),
          "=f"(__out_var[6]),
          "=f"(__out_var[7]),
          "=f"(__out_var[8]),
          "=f"(__out_var[9]),
          "=f"(__out_var[10]),
          "=f"(__out_var[11]),
          "=f"(__out_var[12]),
          "=f"(__out_var[13]),
          "=f"(__out_var[14]),
          "=f"(__out_var[15]),
          "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x16.f32.op out_var, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  float (&out_var)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, float (&__out_var)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x16.f32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
        "%14, %15}, %16, [%17];"
        : "=f"(__out_var[0]),
          "=f"(__out_var[1]),
          "=f"(__out_var[2]),
          "=f"(__out_var[3]),
          "=f"(__out_var[4]),
          "=f"(__out_var[5]),
          "=f"(__out_var[6]),
          "=f"(__out_var[7]),
          "=f"(__out_var[8]),
          "=f"(__out_var[9]),
          "=f"(__out_var[10]),
          "=f"(__out_var[11]),
          "=f"(__out_var[12]),
          "=f"(__out_var[13]),
          "=f"(__out_var[14]),
          "=f"(__out_var[15]),
          "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.32x32b.x16.f32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
        "%14, %15}, %16, [%17];"
        : "=f"(__out_var[0]),
          "=f"(__out_var[1]),
          "=f"(__out_var[2]),
          "=f"(__out_var[3]),
          "=f"(__out_var[4]),
          "=f"(__out_var[5]),
          "=f"(__out_var[6]),
          "=f"(__out_var[7]),
          "=f"(__out_var[8]),
          "=f"(__out_var[9]),
          "=f"(__out_var[10]),
          "=f"(__out_var[11]),
          "=f"(__out_var[12]),
          "=f"(__out_var[13]),
          "=f"(__out_var[14]),
          "=f"(__out_var[15]),
          "=f"(__redval)
        : "r"(__taddr)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x32.u32.op out_var, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline uint32_t tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  uint32_t (&out_var)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::uint32_t tcgen05_ld_red_32x32b(
  ::cuda::ptx::op_t<_Op> __op, ::cuda::std::uint32_t (&__out_var)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  ::cuda::std::uint32_t __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x32.u32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15]),
        "=r"(__out_var[16]),
        "=r"(__out_var[17]),
        "=r"(__out_var[18]),
        "=r"(__out_var[19]),
        "=r"(__out_var[20]),
        "=r"(__out_var[21]),
        "=r"(__out_var[22]),
        "=r"(__out_var[23]),
        "=r"(__out_var[24]),
        "=r"(__out_var[25]),
        "=r"(__out_var[26]),
        "=r"(__out_var[27]),
        "=r"(__out_var[28]),
        "=r"(__out_var[29]),
        "=r"(__out_var[30]),
        "=r"(__out_var[31]),
        "=r"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x32.u32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15]),
        "=r"(__out_var[16]),
        "=r"(__out_var[17]),
        "=r"(__out_var[18]),
        "=r"(__out_var[19]),
        "=r"(__out_var[20]),
        "=r"(__out_var[21]),
        "=r"(__out_var[22]),
        "=r"(__out_var[23]),
        "=r"(__out_var[24]),
        "=r"(__out_var[25]),
        "=r"(__out_var[26]),
        "=r"(__out_var[27]),
        "=r"(__out_var[28]),
        "=r"(__out_var[29]),
        "=r"(__out_var[30]),
        "=r"(__out_var[31]),
        "=r"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x32.s32.op out_var, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline int32_t tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  int32_t (&out_var)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::int32_t
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, ::cuda::std::int32_t (&__out_var)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  ::cuda::std::int32_t __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x32.s32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15]),
        "=r"(__out_var[16]),
        "=r"(__out_var[17]),
        "=r"(__out_var[18]),
        "=r"(__out_var[19]),
        "=r"(__out_var[20]),
        "=r"(__out_var[21]),
        "=r"(__out_var[22]),
        "=r"(__out_var[23]),
        "=r"(__out_var[24]),
        "=r"(__out_var[25]),
        "=r"(__out_var[26]),
        "=r"(__out_var[27]),
        "=r"(__out_var[28]),
        "=r"(__out_var[29]),
        "=r"(__out_var[30]),
        "=r"(__out_var[31]),
        "=r"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x32.s32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15]),
        "=r"(__out_var[16]),
        "=r"(__out_var[17]),
        "=r"(__out_var[18]),
        "=r"(__out_var[19]),
        "=r"(__out_var[20]),
        "=r"(__out_var[21]),
        "=r"(__out_var[22]),
        "=r"(__out_var[23]),
        "=r"(__out_var[24]),
        "=r"(__out_var[25]),
        "=r"(__out_var[26]),
        "=r"(__out_var[27]),
        "=r"(__out_var[28]),
        "=r"(__out_var[29]),
        "=r"(__out_var[30]),
        "=r"(__out_var[31]),
        "=r"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x32.f32.op.abs out_var, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_32x32b_abs(
  cuda::ptx::op_t<Op> op,
  float (&out_var)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float
tcgen05_ld_red_32x32b_abs(::cuda::ptx::op_t<_Op> __op, float (&__out_var)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x32.f32.min.abs {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
      : "=f"(__out_var[0]),
        "=f"(__out_var[1]),
        "=f"(__out_var[2]),
        "=f"(__out_var[3]),
        "=f"(__out_var[4]),
        "=f"(__out_var[5]),
        "=f"(__out_var[6]),
        "=f"(__out_var[7]),
        "=f"(__out_var[8]),
        "=f"(__out_var[9]),
        "=f"(__out_var[10]),
        "=f"(__out_var[11]),
        "=f"(__out_var[12]),
        "=f"(__out_var[13]),
        "=f"(__out_var[14]),
        "=f"(__out_var[15]),
        "=f"(__out_var[16]),
        "=f"(__out_var[17]),
        "=f"(__out_var[18]),
        "=f"(__out_var[19]),
        "=f"(__out_var[20]),
        "=f"(__out_var[21]),
        "=f"(__out_var[22]),
        "=f"(__out_var[23]),
        "=f"(__out_var[24]),
        "=f"(__out_var[25]),
        "=f"(__out_var[26]),
        "=f"(__out_var[27]),
        "=f"(__out_var[28]),
        "=f"(__out_var[29]),
        "=f"(__out_var[30]),
        "=f"(__out_var[31]),
        "=f"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x32.f32.max.abs {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
      : "=f"(__out_var[0]),
        "=f"(__out_var[1]),
        "=f"(__out_var[2]),
        "=f"(__out_var[3]),
        "=f"(__out_var[4]),
        "=f"(__out_var[5]),
        "=f"(__out_var[6]),
        "=f"(__out_var[7]),
        "=f"(__out_var[8]),
        "=f"(__out_var[9]),
        "=f"(__out_var[10]),
        "=f"(__out_var[11]),
        "=f"(__out_var[12]),
        "=f"(__out_var[13]),
        "=f"(__out_var[14]),
        "=f"(__out_var[15]),
        "=f"(__out_var[16]),
        "=f"(__out_var[17]),
        "=f"(__out_var[18]),
        "=f"(__out_var[19]),
        "=f"(__out_var[20]),
        "=f"(__out_var[21]),
        "=f"(__out_var[22]),
        "=f"(__out_var[23]),
        "=f"(__out_var[24]),
        "=f"(__out_var[25]),
        "=f"(__out_var[26]),
        "=f"(__out_var[27]),
        "=f"(__out_var[28]),
        "=f"(__out_var[29]),
        "=f"(__out_var[30]),
        "=f"(__out_var[31]),
        "=f"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x32.f32.op out_var, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  float (&out_var)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, float (&__out_var)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x32.f32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
      : "=f"(__out_var[0]),
        "=f"(__out_var[1]),
        "=f"(__out_var[2]),
        "=f"(__out_var[3]),
        "=f"(__out_var[4]),
        "=f"(__out_var[5]),
        "=f"(__out_var[6]),
        "=f"(__out_var[7]),
        "=f"(__out_var[8]),
        "=f"(__out_var[9]),
        "=f"(__out_var[10]),
        "=f"(__out_var[11]),
        "=f"(__out_var[12]),
        "=f"(__out_var[13]),
        "=f"(__out_var[14]),
        "=f"(__out_var[15]),
        "=f"(__out_var[16]),
        "=f"(__out_var[17]),
        "=f"(__out_var[18]),
        "=f"(__out_var[19]),
        "=f"(__out_var[20]),
        "=f"(__out_var[21]),
        "=f"(__out_var[22]),
        "=f"(__out_var[23]),
        "=f"(__out_var[24]),
        "=f"(__out_var[25]),
        "=f"(__out_var[26]),
        "=f"(__out_var[27]),
        "=f"(__out_var[28]),
        "=f"(__out_var[29]),
        "=f"(__out_var[30]),
        "=f"(__out_var[31]),
        "=f"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x32.f32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
      : "=f"(__out_var[0]),
        "=f"(__out_var[1]),
        "=f"(__out_var[2]),
        "=f"(__out_var[3]),
        "=f"(__out_var[4]),
        "=f"(__out_var[5]),
        "=f"(__out_var[6]),
        "=f"(__out_var[7]),
        "=f"(__out_var[8]),
        "=f"(__out_var[9]),
        "=f"(__out_var[10]),
        "=f"(__out_var[11]),
        "=f"(__out_var[12]),
        "=f"(__out_var[13]),
        "=f"(__out_var[14]),
        "=f"(__out_var[15]),
        "=f"(__out_var[16]),
        "=f"(__out_var[17]),
        "=f"(__out_var[18]),
        "=f"(__out_var[19]),
        "=f"(__out_var[20]),
        "=f"(__out_var[21]),
        "=f"(__out_var[22]),
        "=f"(__out_var[23]),
        "=f"(__out_var[24]),
        "=f"(__out_var[25]),
        "=f"(__out_var[26]),
        "=f"(__out_var[27]),
        "=f"(__out_var[28]),
        "=f"(__out_var[29]),
        "=f"(__out_var[30]),
        "=f"(__out_var[31]),
        "=f"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x64.u32.op out_var, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline uint32_t tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  uint32_t (&out_var)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::uint32_t tcgen05_ld_red_32x32b(
  ::cuda::ptx::op_t<_Op> __op, ::cuda::std::uint32_t (&__out_var)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  ::cuda::std::uint32_t __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x64.u32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63}, %64, [%65];"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15]),
        "=r"(__out_var[16]),
        "=r"(__out_var[17]),
        "=r"(__out_var[18]),
        "=r"(__out_var[19]),
        "=r"(__out_var[20]),
        "=r"(__out_var[21]),
        "=r"(__out_var[22]),
        "=r"(__out_var[23]),
        "=r"(__out_var[24]),
        "=r"(__out_var[25]),
        "=r"(__out_var[26]),
        "=r"(__out_var[27]),
        "=r"(__out_var[28]),
        "=r"(__out_var[29]),
        "=r"(__out_var[30]),
        "=r"(__out_var[31]),
        "=r"(__out_var[32]),
        "=r"(__out_var[33]),
        "=r"(__out_var[34]),
        "=r"(__out_var[35]),
        "=r"(__out_var[36]),
        "=r"(__out_var[37]),
        "=r"(__out_var[38]),
        "=r"(__out_var[39]),
        "=r"(__out_var[40]),
        "=r"(__out_var[41]),
        "=r"(__out_var[42]),
        "=r"(__out_var[43]),
        "=r"(__out_var[44]),
        "=r"(__out_var[45]),
        "=r"(__out_var[46]),
        "=r"(__out_var[47]),
        "=r"(__out_var[48]),
        "=r"(__out_var[49]),
        "=r"(__out_var[50]),
        "=r"(__out_var[51]),
        "=r"(__out_var[52]),
        "=r"(__out_var[53]),
        "=r"(__out_var[54]),
        "=r"(__out_var[55]),
        "=r"(__out_var[56]),
        "=r"(__out_var[57]),
        "=r"(__out_var[58]),
        "=r"(__out_var[59]),
        "=r"(__out_var[60]),
        "=r"(__out_var[61]),
        "=r"(__out_var[62]),
        "=r"(__out_var[63]),
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
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15]),
        "=r"(__out_var[16]),
        "=r"(__out_var[17]),
        "=r"(__out_var[18]),
        "=r"(__out_var[19]),
        "=r"(__out_var[20]),
        "=r"(__out_var[21]),
        "=r"(__out_var[22]),
        "=r"(__out_var[23]),
        "=r"(__out_var[24]),
        "=r"(__out_var[25]),
        "=r"(__out_var[26]),
        "=r"(__out_var[27]),
        "=r"(__out_var[28]),
        "=r"(__out_var[29]),
        "=r"(__out_var[30]),
        "=r"(__out_var[31]),
        "=r"(__out_var[32]),
        "=r"(__out_var[33]),
        "=r"(__out_var[34]),
        "=r"(__out_var[35]),
        "=r"(__out_var[36]),
        "=r"(__out_var[37]),
        "=r"(__out_var[38]),
        "=r"(__out_var[39]),
        "=r"(__out_var[40]),
        "=r"(__out_var[41]),
        "=r"(__out_var[42]),
        "=r"(__out_var[43]),
        "=r"(__out_var[44]),
        "=r"(__out_var[45]),
        "=r"(__out_var[46]),
        "=r"(__out_var[47]),
        "=r"(__out_var[48]),
        "=r"(__out_var[49]),
        "=r"(__out_var[50]),
        "=r"(__out_var[51]),
        "=r"(__out_var[52]),
        "=r"(__out_var[53]),
        "=r"(__out_var[54]),
        "=r"(__out_var[55]),
        "=r"(__out_var[56]),
        "=r"(__out_var[57]),
        "=r"(__out_var[58]),
        "=r"(__out_var[59]),
        "=r"(__out_var[60]),
        "=r"(__out_var[61]),
        "=r"(__out_var[62]),
        "=r"(__out_var[63]),
        "=r"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x64.s32.op out_var, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline int32_t tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  int32_t (&out_var)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::int32_t
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, ::cuda::std::int32_t (&__out_var)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  ::cuda::std::int32_t __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x64.s32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63}, %64, [%65];"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15]),
        "=r"(__out_var[16]),
        "=r"(__out_var[17]),
        "=r"(__out_var[18]),
        "=r"(__out_var[19]),
        "=r"(__out_var[20]),
        "=r"(__out_var[21]),
        "=r"(__out_var[22]),
        "=r"(__out_var[23]),
        "=r"(__out_var[24]),
        "=r"(__out_var[25]),
        "=r"(__out_var[26]),
        "=r"(__out_var[27]),
        "=r"(__out_var[28]),
        "=r"(__out_var[29]),
        "=r"(__out_var[30]),
        "=r"(__out_var[31]),
        "=r"(__out_var[32]),
        "=r"(__out_var[33]),
        "=r"(__out_var[34]),
        "=r"(__out_var[35]),
        "=r"(__out_var[36]),
        "=r"(__out_var[37]),
        "=r"(__out_var[38]),
        "=r"(__out_var[39]),
        "=r"(__out_var[40]),
        "=r"(__out_var[41]),
        "=r"(__out_var[42]),
        "=r"(__out_var[43]),
        "=r"(__out_var[44]),
        "=r"(__out_var[45]),
        "=r"(__out_var[46]),
        "=r"(__out_var[47]),
        "=r"(__out_var[48]),
        "=r"(__out_var[49]),
        "=r"(__out_var[50]),
        "=r"(__out_var[51]),
        "=r"(__out_var[52]),
        "=r"(__out_var[53]),
        "=r"(__out_var[54]),
        "=r"(__out_var[55]),
        "=r"(__out_var[56]),
        "=r"(__out_var[57]),
        "=r"(__out_var[58]),
        "=r"(__out_var[59]),
        "=r"(__out_var[60]),
        "=r"(__out_var[61]),
        "=r"(__out_var[62]),
        "=r"(__out_var[63]),
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
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15]),
        "=r"(__out_var[16]),
        "=r"(__out_var[17]),
        "=r"(__out_var[18]),
        "=r"(__out_var[19]),
        "=r"(__out_var[20]),
        "=r"(__out_var[21]),
        "=r"(__out_var[22]),
        "=r"(__out_var[23]),
        "=r"(__out_var[24]),
        "=r"(__out_var[25]),
        "=r"(__out_var[26]),
        "=r"(__out_var[27]),
        "=r"(__out_var[28]),
        "=r"(__out_var[29]),
        "=r"(__out_var[30]),
        "=r"(__out_var[31]),
        "=r"(__out_var[32]),
        "=r"(__out_var[33]),
        "=r"(__out_var[34]),
        "=r"(__out_var[35]),
        "=r"(__out_var[36]),
        "=r"(__out_var[37]),
        "=r"(__out_var[38]),
        "=r"(__out_var[39]),
        "=r"(__out_var[40]),
        "=r"(__out_var[41]),
        "=r"(__out_var[42]),
        "=r"(__out_var[43]),
        "=r"(__out_var[44]),
        "=r"(__out_var[45]),
        "=r"(__out_var[46]),
        "=r"(__out_var[47]),
        "=r"(__out_var[48]),
        "=r"(__out_var[49]),
        "=r"(__out_var[50]),
        "=r"(__out_var[51]),
        "=r"(__out_var[52]),
        "=r"(__out_var[53]),
        "=r"(__out_var[54]),
        "=r"(__out_var[55]),
        "=r"(__out_var[56]),
        "=r"(__out_var[57]),
        "=r"(__out_var[58]),
        "=r"(__out_var[59]),
        "=r"(__out_var[60]),
        "=r"(__out_var[61]),
        "=r"(__out_var[62]),
        "=r"(__out_var[63]),
        "=r"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x64.f32.op.abs out_var, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_32x32b_abs(
  cuda::ptx::op_t<Op> op,
  float (&out_var)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float
tcgen05_ld_red_32x32b_abs(::cuda::ptx::op_t<_Op> __op, float (&__out_var)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x64.f32.min.abs {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63}, %64, [%65];"
      : "=f"(__out_var[0]),
        "=f"(__out_var[1]),
        "=f"(__out_var[2]),
        "=f"(__out_var[3]),
        "=f"(__out_var[4]),
        "=f"(__out_var[5]),
        "=f"(__out_var[6]),
        "=f"(__out_var[7]),
        "=f"(__out_var[8]),
        "=f"(__out_var[9]),
        "=f"(__out_var[10]),
        "=f"(__out_var[11]),
        "=f"(__out_var[12]),
        "=f"(__out_var[13]),
        "=f"(__out_var[14]),
        "=f"(__out_var[15]),
        "=f"(__out_var[16]),
        "=f"(__out_var[17]),
        "=f"(__out_var[18]),
        "=f"(__out_var[19]),
        "=f"(__out_var[20]),
        "=f"(__out_var[21]),
        "=f"(__out_var[22]),
        "=f"(__out_var[23]),
        "=f"(__out_var[24]),
        "=f"(__out_var[25]),
        "=f"(__out_var[26]),
        "=f"(__out_var[27]),
        "=f"(__out_var[28]),
        "=f"(__out_var[29]),
        "=f"(__out_var[30]),
        "=f"(__out_var[31]),
        "=f"(__out_var[32]),
        "=f"(__out_var[33]),
        "=f"(__out_var[34]),
        "=f"(__out_var[35]),
        "=f"(__out_var[36]),
        "=f"(__out_var[37]),
        "=f"(__out_var[38]),
        "=f"(__out_var[39]),
        "=f"(__out_var[40]),
        "=f"(__out_var[41]),
        "=f"(__out_var[42]),
        "=f"(__out_var[43]),
        "=f"(__out_var[44]),
        "=f"(__out_var[45]),
        "=f"(__out_var[46]),
        "=f"(__out_var[47]),
        "=f"(__out_var[48]),
        "=f"(__out_var[49]),
        "=f"(__out_var[50]),
        "=f"(__out_var[51]),
        "=f"(__out_var[52]),
        "=f"(__out_var[53]),
        "=f"(__out_var[54]),
        "=f"(__out_var[55]),
        "=f"(__out_var[56]),
        "=f"(__out_var[57]),
        "=f"(__out_var[58]),
        "=f"(__out_var[59]),
        "=f"(__out_var[60]),
        "=f"(__out_var[61]),
        "=f"(__out_var[62]),
        "=f"(__out_var[63]),
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
      : "=f"(__out_var[0]),
        "=f"(__out_var[1]),
        "=f"(__out_var[2]),
        "=f"(__out_var[3]),
        "=f"(__out_var[4]),
        "=f"(__out_var[5]),
        "=f"(__out_var[6]),
        "=f"(__out_var[7]),
        "=f"(__out_var[8]),
        "=f"(__out_var[9]),
        "=f"(__out_var[10]),
        "=f"(__out_var[11]),
        "=f"(__out_var[12]),
        "=f"(__out_var[13]),
        "=f"(__out_var[14]),
        "=f"(__out_var[15]),
        "=f"(__out_var[16]),
        "=f"(__out_var[17]),
        "=f"(__out_var[18]),
        "=f"(__out_var[19]),
        "=f"(__out_var[20]),
        "=f"(__out_var[21]),
        "=f"(__out_var[22]),
        "=f"(__out_var[23]),
        "=f"(__out_var[24]),
        "=f"(__out_var[25]),
        "=f"(__out_var[26]),
        "=f"(__out_var[27]),
        "=f"(__out_var[28]),
        "=f"(__out_var[29]),
        "=f"(__out_var[30]),
        "=f"(__out_var[31]),
        "=f"(__out_var[32]),
        "=f"(__out_var[33]),
        "=f"(__out_var[34]),
        "=f"(__out_var[35]),
        "=f"(__out_var[36]),
        "=f"(__out_var[37]),
        "=f"(__out_var[38]),
        "=f"(__out_var[39]),
        "=f"(__out_var[40]),
        "=f"(__out_var[41]),
        "=f"(__out_var[42]),
        "=f"(__out_var[43]),
        "=f"(__out_var[44]),
        "=f"(__out_var[45]),
        "=f"(__out_var[46]),
        "=f"(__out_var[47]),
        "=f"(__out_var[48]),
        "=f"(__out_var[49]),
        "=f"(__out_var[50]),
        "=f"(__out_var[51]),
        "=f"(__out_var[52]),
        "=f"(__out_var[53]),
        "=f"(__out_var[54]),
        "=f"(__out_var[55]),
        "=f"(__out_var[56]),
        "=f"(__out_var[57]),
        "=f"(__out_var[58]),
        "=f"(__out_var[59]),
        "=f"(__out_var[60]),
        "=f"(__out_var[61]),
        "=f"(__out_var[62]),
        "=f"(__out_var[63]),
        "=f"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x64.f32.op out_var, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  float (&out_var)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, float (&__out_var)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.32x32b.x64.f32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63}, %64, [%65];"
      : "=f"(__out_var[0]),
        "=f"(__out_var[1]),
        "=f"(__out_var[2]),
        "=f"(__out_var[3]),
        "=f"(__out_var[4]),
        "=f"(__out_var[5]),
        "=f"(__out_var[6]),
        "=f"(__out_var[7]),
        "=f"(__out_var[8]),
        "=f"(__out_var[9]),
        "=f"(__out_var[10]),
        "=f"(__out_var[11]),
        "=f"(__out_var[12]),
        "=f"(__out_var[13]),
        "=f"(__out_var[14]),
        "=f"(__out_var[15]),
        "=f"(__out_var[16]),
        "=f"(__out_var[17]),
        "=f"(__out_var[18]),
        "=f"(__out_var[19]),
        "=f"(__out_var[20]),
        "=f"(__out_var[21]),
        "=f"(__out_var[22]),
        "=f"(__out_var[23]),
        "=f"(__out_var[24]),
        "=f"(__out_var[25]),
        "=f"(__out_var[26]),
        "=f"(__out_var[27]),
        "=f"(__out_var[28]),
        "=f"(__out_var[29]),
        "=f"(__out_var[30]),
        "=f"(__out_var[31]),
        "=f"(__out_var[32]),
        "=f"(__out_var[33]),
        "=f"(__out_var[34]),
        "=f"(__out_var[35]),
        "=f"(__out_var[36]),
        "=f"(__out_var[37]),
        "=f"(__out_var[38]),
        "=f"(__out_var[39]),
        "=f"(__out_var[40]),
        "=f"(__out_var[41]),
        "=f"(__out_var[42]),
        "=f"(__out_var[43]),
        "=f"(__out_var[44]),
        "=f"(__out_var[45]),
        "=f"(__out_var[46]),
        "=f"(__out_var[47]),
        "=f"(__out_var[48]),
        "=f"(__out_var[49]),
        "=f"(__out_var[50]),
        "=f"(__out_var[51]),
        "=f"(__out_var[52]),
        "=f"(__out_var[53]),
        "=f"(__out_var[54]),
        "=f"(__out_var[55]),
        "=f"(__out_var[56]),
        "=f"(__out_var[57]),
        "=f"(__out_var[58]),
        "=f"(__out_var[59]),
        "=f"(__out_var[60]),
        "=f"(__out_var[61]),
        "=f"(__out_var[62]),
        "=f"(__out_var[63]),
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
      : "=f"(__out_var[0]),
        "=f"(__out_var[1]),
        "=f"(__out_var[2]),
        "=f"(__out_var[3]),
        "=f"(__out_var[4]),
        "=f"(__out_var[5]),
        "=f"(__out_var[6]),
        "=f"(__out_var[7]),
        "=f"(__out_var[8]),
        "=f"(__out_var[9]),
        "=f"(__out_var[10]),
        "=f"(__out_var[11]),
        "=f"(__out_var[12]),
        "=f"(__out_var[13]),
        "=f"(__out_var[14]),
        "=f"(__out_var[15]),
        "=f"(__out_var[16]),
        "=f"(__out_var[17]),
        "=f"(__out_var[18]),
        "=f"(__out_var[19]),
        "=f"(__out_var[20]),
        "=f"(__out_var[21]),
        "=f"(__out_var[22]),
        "=f"(__out_var[23]),
        "=f"(__out_var[24]),
        "=f"(__out_var[25]),
        "=f"(__out_var[26]),
        "=f"(__out_var[27]),
        "=f"(__out_var[28]),
        "=f"(__out_var[29]),
        "=f"(__out_var[30]),
        "=f"(__out_var[31]),
        "=f"(__out_var[32]),
        "=f"(__out_var[33]),
        "=f"(__out_var[34]),
        "=f"(__out_var[35]),
        "=f"(__out_var[36]),
        "=f"(__out_var[37]),
        "=f"(__out_var[38]),
        "=f"(__out_var[39]),
        "=f"(__out_var[40]),
        "=f"(__out_var[41]),
        "=f"(__out_var[42]),
        "=f"(__out_var[43]),
        "=f"(__out_var[44]),
        "=f"(__out_var[45]),
        "=f"(__out_var[46]),
        "=f"(__out_var[47]),
        "=f"(__out_var[48]),
        "=f"(__out_var[49]),
        "=f"(__out_var[50]),
        "=f"(__out_var[51]),
        "=f"(__out_var[52]),
        "=f"(__out_var[53]),
        "=f"(__out_var[54]),
        "=f"(__out_var[55]),
        "=f"(__out_var[56]),
        "=f"(__out_var[57]),
        "=f"(__out_var[58]),
        "=f"(__out_var[59]),
        "=f"(__out_var[60]),
        "=f"(__out_var[61]),
        "=f"(__out_var[62]),
        "=f"(__out_var[63]),
        "=f"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x128.u32.op out_var, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline uint32_t tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  uint32_t (&out_var)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::uint32_t tcgen05_ld_red_32x32b(
  ::cuda::ptx::op_t<_Op> __op, ::cuda::std::uint32_t (&__out_var)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
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
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15]),
        "=r"(__out_var[16]),
        "=r"(__out_var[17]),
        "=r"(__out_var[18]),
        "=r"(__out_var[19]),
        "=r"(__out_var[20]),
        "=r"(__out_var[21]),
        "=r"(__out_var[22]),
        "=r"(__out_var[23]),
        "=r"(__out_var[24]),
        "=r"(__out_var[25]),
        "=r"(__out_var[26]),
        "=r"(__out_var[27]),
        "=r"(__out_var[28]),
        "=r"(__out_var[29]),
        "=r"(__out_var[30]),
        "=r"(__out_var[31]),
        "=r"(__out_var[32]),
        "=r"(__out_var[33]),
        "=r"(__out_var[34]),
        "=r"(__out_var[35]),
        "=r"(__out_var[36]),
        "=r"(__out_var[37]),
        "=r"(__out_var[38]),
        "=r"(__out_var[39]),
        "=r"(__out_var[40]),
        "=r"(__out_var[41]),
        "=r"(__out_var[42]),
        "=r"(__out_var[43]),
        "=r"(__out_var[44]),
        "=r"(__out_var[45]),
        "=r"(__out_var[46]),
        "=r"(__out_var[47]),
        "=r"(__out_var[48]),
        "=r"(__out_var[49]),
        "=r"(__out_var[50]),
        "=r"(__out_var[51]),
        "=r"(__out_var[52]),
        "=r"(__out_var[53]),
        "=r"(__out_var[54]),
        "=r"(__out_var[55]),
        "=r"(__out_var[56]),
        "=r"(__out_var[57]),
        "=r"(__out_var[58]),
        "=r"(__out_var[59]),
        "=r"(__out_var[60]),
        "=r"(__out_var[61]),
        "=r"(__out_var[62]),
        "=r"(__out_var[63]),
        "=r"(__out_var[64]),
        "=r"(__out_var[65]),
        "=r"(__out_var[66]),
        "=r"(__out_var[67]),
        "=r"(__out_var[68]),
        "=r"(__out_var[69]),
        "=r"(__out_var[70]),
        "=r"(__out_var[71]),
        "=r"(__out_var[72]),
        "=r"(__out_var[73]),
        "=r"(__out_var[74]),
        "=r"(__out_var[75]),
        "=r"(__out_var[76]),
        "=r"(__out_var[77]),
        "=r"(__out_var[78]),
        "=r"(__out_var[79]),
        "=r"(__out_var[80]),
        "=r"(__out_var[81]),
        "=r"(__out_var[82]),
        "=r"(__out_var[83]),
        "=r"(__out_var[84]),
        "=r"(__out_var[85]),
        "=r"(__out_var[86]),
        "=r"(__out_var[87]),
        "=r"(__out_var[88]),
        "=r"(__out_var[89]),
        "=r"(__out_var[90]),
        "=r"(__out_var[91]),
        "=r"(__out_var[92]),
        "=r"(__out_var[93]),
        "=r"(__out_var[94]),
        "=r"(__out_var[95]),
        "=r"(__out_var[96]),
        "=r"(__out_var[97]),
        "=r"(__out_var[98]),
        "=r"(__out_var[99]),
        "=r"(__out_var[100]),
        "=r"(__out_var[101]),
        "=r"(__out_var[102]),
        "=r"(__out_var[103]),
        "=r"(__out_var[104]),
        "=r"(__out_var[105]),
        "=r"(__out_var[106]),
        "=r"(__out_var[107]),
        "=r"(__out_var[108]),
        "=r"(__out_var[109]),
        "=r"(__out_var[110]),
        "=r"(__out_var[111]),
        "=r"(__out_var[112]),
        "=r"(__out_var[113]),
        "=r"(__out_var[114]),
        "=r"(__out_var[115]),
        "=r"(__out_var[116]),
        "=r"(__out_var[117]),
        "=r"(__out_var[118]),
        "=r"(__out_var[119]),
        "=r"(__out_var[120]),
        "=r"(__out_var[121]),
        "=r"(__out_var[122]),
        "=r"(__out_var[123]),
        "=r"(__out_var[124]),
        "=r"(__out_var[125]),
        "=r"(__out_var[126]),
        "=r"(__out_var[127]),
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
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15]),
        "=r"(__out_var[16]),
        "=r"(__out_var[17]),
        "=r"(__out_var[18]),
        "=r"(__out_var[19]),
        "=r"(__out_var[20]),
        "=r"(__out_var[21]),
        "=r"(__out_var[22]),
        "=r"(__out_var[23]),
        "=r"(__out_var[24]),
        "=r"(__out_var[25]),
        "=r"(__out_var[26]),
        "=r"(__out_var[27]),
        "=r"(__out_var[28]),
        "=r"(__out_var[29]),
        "=r"(__out_var[30]),
        "=r"(__out_var[31]),
        "=r"(__out_var[32]),
        "=r"(__out_var[33]),
        "=r"(__out_var[34]),
        "=r"(__out_var[35]),
        "=r"(__out_var[36]),
        "=r"(__out_var[37]),
        "=r"(__out_var[38]),
        "=r"(__out_var[39]),
        "=r"(__out_var[40]),
        "=r"(__out_var[41]),
        "=r"(__out_var[42]),
        "=r"(__out_var[43]),
        "=r"(__out_var[44]),
        "=r"(__out_var[45]),
        "=r"(__out_var[46]),
        "=r"(__out_var[47]),
        "=r"(__out_var[48]),
        "=r"(__out_var[49]),
        "=r"(__out_var[50]),
        "=r"(__out_var[51]),
        "=r"(__out_var[52]),
        "=r"(__out_var[53]),
        "=r"(__out_var[54]),
        "=r"(__out_var[55]),
        "=r"(__out_var[56]),
        "=r"(__out_var[57]),
        "=r"(__out_var[58]),
        "=r"(__out_var[59]),
        "=r"(__out_var[60]),
        "=r"(__out_var[61]),
        "=r"(__out_var[62]),
        "=r"(__out_var[63]),
        "=r"(__out_var[64]),
        "=r"(__out_var[65]),
        "=r"(__out_var[66]),
        "=r"(__out_var[67]),
        "=r"(__out_var[68]),
        "=r"(__out_var[69]),
        "=r"(__out_var[70]),
        "=r"(__out_var[71]),
        "=r"(__out_var[72]),
        "=r"(__out_var[73]),
        "=r"(__out_var[74]),
        "=r"(__out_var[75]),
        "=r"(__out_var[76]),
        "=r"(__out_var[77]),
        "=r"(__out_var[78]),
        "=r"(__out_var[79]),
        "=r"(__out_var[80]),
        "=r"(__out_var[81]),
        "=r"(__out_var[82]),
        "=r"(__out_var[83]),
        "=r"(__out_var[84]),
        "=r"(__out_var[85]),
        "=r"(__out_var[86]),
        "=r"(__out_var[87]),
        "=r"(__out_var[88]),
        "=r"(__out_var[89]),
        "=r"(__out_var[90]),
        "=r"(__out_var[91]),
        "=r"(__out_var[92]),
        "=r"(__out_var[93]),
        "=r"(__out_var[94]),
        "=r"(__out_var[95]),
        "=r"(__out_var[96]),
        "=r"(__out_var[97]),
        "=r"(__out_var[98]),
        "=r"(__out_var[99]),
        "=r"(__out_var[100]),
        "=r"(__out_var[101]),
        "=r"(__out_var[102]),
        "=r"(__out_var[103]),
        "=r"(__out_var[104]),
        "=r"(__out_var[105]),
        "=r"(__out_var[106]),
        "=r"(__out_var[107]),
        "=r"(__out_var[108]),
        "=r"(__out_var[109]),
        "=r"(__out_var[110]),
        "=r"(__out_var[111]),
        "=r"(__out_var[112]),
        "=r"(__out_var[113]),
        "=r"(__out_var[114]),
        "=r"(__out_var[115]),
        "=r"(__out_var[116]),
        "=r"(__out_var[117]),
        "=r"(__out_var[118]),
        "=r"(__out_var[119]),
        "=r"(__out_var[120]),
        "=r"(__out_var[121]),
        "=r"(__out_var[122]),
        "=r"(__out_var[123]),
        "=r"(__out_var[124]),
        "=r"(__out_var[125]),
        "=r"(__out_var[126]),
        "=r"(__out_var[127]),
        "=r"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x128.s32.op out_var, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline int32_t tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  int32_t (&out_var)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::int32_t tcgen05_ld_red_32x32b(
  ::cuda::ptx::op_t<_Op> __op, ::cuda::std::int32_t (&__out_var)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
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
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15]),
        "=r"(__out_var[16]),
        "=r"(__out_var[17]),
        "=r"(__out_var[18]),
        "=r"(__out_var[19]),
        "=r"(__out_var[20]),
        "=r"(__out_var[21]),
        "=r"(__out_var[22]),
        "=r"(__out_var[23]),
        "=r"(__out_var[24]),
        "=r"(__out_var[25]),
        "=r"(__out_var[26]),
        "=r"(__out_var[27]),
        "=r"(__out_var[28]),
        "=r"(__out_var[29]),
        "=r"(__out_var[30]),
        "=r"(__out_var[31]),
        "=r"(__out_var[32]),
        "=r"(__out_var[33]),
        "=r"(__out_var[34]),
        "=r"(__out_var[35]),
        "=r"(__out_var[36]),
        "=r"(__out_var[37]),
        "=r"(__out_var[38]),
        "=r"(__out_var[39]),
        "=r"(__out_var[40]),
        "=r"(__out_var[41]),
        "=r"(__out_var[42]),
        "=r"(__out_var[43]),
        "=r"(__out_var[44]),
        "=r"(__out_var[45]),
        "=r"(__out_var[46]),
        "=r"(__out_var[47]),
        "=r"(__out_var[48]),
        "=r"(__out_var[49]),
        "=r"(__out_var[50]),
        "=r"(__out_var[51]),
        "=r"(__out_var[52]),
        "=r"(__out_var[53]),
        "=r"(__out_var[54]),
        "=r"(__out_var[55]),
        "=r"(__out_var[56]),
        "=r"(__out_var[57]),
        "=r"(__out_var[58]),
        "=r"(__out_var[59]),
        "=r"(__out_var[60]),
        "=r"(__out_var[61]),
        "=r"(__out_var[62]),
        "=r"(__out_var[63]),
        "=r"(__out_var[64]),
        "=r"(__out_var[65]),
        "=r"(__out_var[66]),
        "=r"(__out_var[67]),
        "=r"(__out_var[68]),
        "=r"(__out_var[69]),
        "=r"(__out_var[70]),
        "=r"(__out_var[71]),
        "=r"(__out_var[72]),
        "=r"(__out_var[73]),
        "=r"(__out_var[74]),
        "=r"(__out_var[75]),
        "=r"(__out_var[76]),
        "=r"(__out_var[77]),
        "=r"(__out_var[78]),
        "=r"(__out_var[79]),
        "=r"(__out_var[80]),
        "=r"(__out_var[81]),
        "=r"(__out_var[82]),
        "=r"(__out_var[83]),
        "=r"(__out_var[84]),
        "=r"(__out_var[85]),
        "=r"(__out_var[86]),
        "=r"(__out_var[87]),
        "=r"(__out_var[88]),
        "=r"(__out_var[89]),
        "=r"(__out_var[90]),
        "=r"(__out_var[91]),
        "=r"(__out_var[92]),
        "=r"(__out_var[93]),
        "=r"(__out_var[94]),
        "=r"(__out_var[95]),
        "=r"(__out_var[96]),
        "=r"(__out_var[97]),
        "=r"(__out_var[98]),
        "=r"(__out_var[99]),
        "=r"(__out_var[100]),
        "=r"(__out_var[101]),
        "=r"(__out_var[102]),
        "=r"(__out_var[103]),
        "=r"(__out_var[104]),
        "=r"(__out_var[105]),
        "=r"(__out_var[106]),
        "=r"(__out_var[107]),
        "=r"(__out_var[108]),
        "=r"(__out_var[109]),
        "=r"(__out_var[110]),
        "=r"(__out_var[111]),
        "=r"(__out_var[112]),
        "=r"(__out_var[113]),
        "=r"(__out_var[114]),
        "=r"(__out_var[115]),
        "=r"(__out_var[116]),
        "=r"(__out_var[117]),
        "=r"(__out_var[118]),
        "=r"(__out_var[119]),
        "=r"(__out_var[120]),
        "=r"(__out_var[121]),
        "=r"(__out_var[122]),
        "=r"(__out_var[123]),
        "=r"(__out_var[124]),
        "=r"(__out_var[125]),
        "=r"(__out_var[126]),
        "=r"(__out_var[127]),
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
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15]),
        "=r"(__out_var[16]),
        "=r"(__out_var[17]),
        "=r"(__out_var[18]),
        "=r"(__out_var[19]),
        "=r"(__out_var[20]),
        "=r"(__out_var[21]),
        "=r"(__out_var[22]),
        "=r"(__out_var[23]),
        "=r"(__out_var[24]),
        "=r"(__out_var[25]),
        "=r"(__out_var[26]),
        "=r"(__out_var[27]),
        "=r"(__out_var[28]),
        "=r"(__out_var[29]),
        "=r"(__out_var[30]),
        "=r"(__out_var[31]),
        "=r"(__out_var[32]),
        "=r"(__out_var[33]),
        "=r"(__out_var[34]),
        "=r"(__out_var[35]),
        "=r"(__out_var[36]),
        "=r"(__out_var[37]),
        "=r"(__out_var[38]),
        "=r"(__out_var[39]),
        "=r"(__out_var[40]),
        "=r"(__out_var[41]),
        "=r"(__out_var[42]),
        "=r"(__out_var[43]),
        "=r"(__out_var[44]),
        "=r"(__out_var[45]),
        "=r"(__out_var[46]),
        "=r"(__out_var[47]),
        "=r"(__out_var[48]),
        "=r"(__out_var[49]),
        "=r"(__out_var[50]),
        "=r"(__out_var[51]),
        "=r"(__out_var[52]),
        "=r"(__out_var[53]),
        "=r"(__out_var[54]),
        "=r"(__out_var[55]),
        "=r"(__out_var[56]),
        "=r"(__out_var[57]),
        "=r"(__out_var[58]),
        "=r"(__out_var[59]),
        "=r"(__out_var[60]),
        "=r"(__out_var[61]),
        "=r"(__out_var[62]),
        "=r"(__out_var[63]),
        "=r"(__out_var[64]),
        "=r"(__out_var[65]),
        "=r"(__out_var[66]),
        "=r"(__out_var[67]),
        "=r"(__out_var[68]),
        "=r"(__out_var[69]),
        "=r"(__out_var[70]),
        "=r"(__out_var[71]),
        "=r"(__out_var[72]),
        "=r"(__out_var[73]),
        "=r"(__out_var[74]),
        "=r"(__out_var[75]),
        "=r"(__out_var[76]),
        "=r"(__out_var[77]),
        "=r"(__out_var[78]),
        "=r"(__out_var[79]),
        "=r"(__out_var[80]),
        "=r"(__out_var[81]),
        "=r"(__out_var[82]),
        "=r"(__out_var[83]),
        "=r"(__out_var[84]),
        "=r"(__out_var[85]),
        "=r"(__out_var[86]),
        "=r"(__out_var[87]),
        "=r"(__out_var[88]),
        "=r"(__out_var[89]),
        "=r"(__out_var[90]),
        "=r"(__out_var[91]),
        "=r"(__out_var[92]),
        "=r"(__out_var[93]),
        "=r"(__out_var[94]),
        "=r"(__out_var[95]),
        "=r"(__out_var[96]),
        "=r"(__out_var[97]),
        "=r"(__out_var[98]),
        "=r"(__out_var[99]),
        "=r"(__out_var[100]),
        "=r"(__out_var[101]),
        "=r"(__out_var[102]),
        "=r"(__out_var[103]),
        "=r"(__out_var[104]),
        "=r"(__out_var[105]),
        "=r"(__out_var[106]),
        "=r"(__out_var[107]),
        "=r"(__out_var[108]),
        "=r"(__out_var[109]),
        "=r"(__out_var[110]),
        "=r"(__out_var[111]),
        "=r"(__out_var[112]),
        "=r"(__out_var[113]),
        "=r"(__out_var[114]),
        "=r"(__out_var[115]),
        "=r"(__out_var[116]),
        "=r"(__out_var[117]),
        "=r"(__out_var[118]),
        "=r"(__out_var[119]),
        "=r"(__out_var[120]),
        "=r"(__out_var[121]),
        "=r"(__out_var[122]),
        "=r"(__out_var[123]),
        "=r"(__out_var[124]),
        "=r"(__out_var[125]),
        "=r"(__out_var[126]),
        "=r"(__out_var[127]),
        "=r"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x128.f32.op.abs out_var, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f,
SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_32x32b_abs(
  cuda::ptx::op_t<Op> op,
  float (&out_var)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float
tcgen05_ld_red_32x32b_abs(::cuda::ptx::op_t<_Op> __op, float (&__out_var)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
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
      : "=f"(__out_var[0]),
        "=f"(__out_var[1]),
        "=f"(__out_var[2]),
        "=f"(__out_var[3]),
        "=f"(__out_var[4]),
        "=f"(__out_var[5]),
        "=f"(__out_var[6]),
        "=f"(__out_var[7]),
        "=f"(__out_var[8]),
        "=f"(__out_var[9]),
        "=f"(__out_var[10]),
        "=f"(__out_var[11]),
        "=f"(__out_var[12]),
        "=f"(__out_var[13]),
        "=f"(__out_var[14]),
        "=f"(__out_var[15]),
        "=f"(__out_var[16]),
        "=f"(__out_var[17]),
        "=f"(__out_var[18]),
        "=f"(__out_var[19]),
        "=f"(__out_var[20]),
        "=f"(__out_var[21]),
        "=f"(__out_var[22]),
        "=f"(__out_var[23]),
        "=f"(__out_var[24]),
        "=f"(__out_var[25]),
        "=f"(__out_var[26]),
        "=f"(__out_var[27]),
        "=f"(__out_var[28]),
        "=f"(__out_var[29]),
        "=f"(__out_var[30]),
        "=f"(__out_var[31]),
        "=f"(__out_var[32]),
        "=f"(__out_var[33]),
        "=f"(__out_var[34]),
        "=f"(__out_var[35]),
        "=f"(__out_var[36]),
        "=f"(__out_var[37]),
        "=f"(__out_var[38]),
        "=f"(__out_var[39]),
        "=f"(__out_var[40]),
        "=f"(__out_var[41]),
        "=f"(__out_var[42]),
        "=f"(__out_var[43]),
        "=f"(__out_var[44]),
        "=f"(__out_var[45]),
        "=f"(__out_var[46]),
        "=f"(__out_var[47]),
        "=f"(__out_var[48]),
        "=f"(__out_var[49]),
        "=f"(__out_var[50]),
        "=f"(__out_var[51]),
        "=f"(__out_var[52]),
        "=f"(__out_var[53]),
        "=f"(__out_var[54]),
        "=f"(__out_var[55]),
        "=f"(__out_var[56]),
        "=f"(__out_var[57]),
        "=f"(__out_var[58]),
        "=f"(__out_var[59]),
        "=f"(__out_var[60]),
        "=f"(__out_var[61]),
        "=f"(__out_var[62]),
        "=f"(__out_var[63]),
        "=f"(__out_var[64]),
        "=f"(__out_var[65]),
        "=f"(__out_var[66]),
        "=f"(__out_var[67]),
        "=f"(__out_var[68]),
        "=f"(__out_var[69]),
        "=f"(__out_var[70]),
        "=f"(__out_var[71]),
        "=f"(__out_var[72]),
        "=f"(__out_var[73]),
        "=f"(__out_var[74]),
        "=f"(__out_var[75]),
        "=f"(__out_var[76]),
        "=f"(__out_var[77]),
        "=f"(__out_var[78]),
        "=f"(__out_var[79]),
        "=f"(__out_var[80]),
        "=f"(__out_var[81]),
        "=f"(__out_var[82]),
        "=f"(__out_var[83]),
        "=f"(__out_var[84]),
        "=f"(__out_var[85]),
        "=f"(__out_var[86]),
        "=f"(__out_var[87]),
        "=f"(__out_var[88]),
        "=f"(__out_var[89]),
        "=f"(__out_var[90]),
        "=f"(__out_var[91]),
        "=f"(__out_var[92]),
        "=f"(__out_var[93]),
        "=f"(__out_var[94]),
        "=f"(__out_var[95]),
        "=f"(__out_var[96]),
        "=f"(__out_var[97]),
        "=f"(__out_var[98]),
        "=f"(__out_var[99]),
        "=f"(__out_var[100]),
        "=f"(__out_var[101]),
        "=f"(__out_var[102]),
        "=f"(__out_var[103]),
        "=f"(__out_var[104]),
        "=f"(__out_var[105]),
        "=f"(__out_var[106]),
        "=f"(__out_var[107]),
        "=f"(__out_var[108]),
        "=f"(__out_var[109]),
        "=f"(__out_var[110]),
        "=f"(__out_var[111]),
        "=f"(__out_var[112]),
        "=f"(__out_var[113]),
        "=f"(__out_var[114]),
        "=f"(__out_var[115]),
        "=f"(__out_var[116]),
        "=f"(__out_var[117]),
        "=f"(__out_var[118]),
        "=f"(__out_var[119]),
        "=f"(__out_var[120]),
        "=f"(__out_var[121]),
        "=f"(__out_var[122]),
        "=f"(__out_var[123]),
        "=f"(__out_var[124]),
        "=f"(__out_var[125]),
        "=f"(__out_var[126]),
        "=f"(__out_var[127]),
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
      : "=f"(__out_var[0]),
        "=f"(__out_var[1]),
        "=f"(__out_var[2]),
        "=f"(__out_var[3]),
        "=f"(__out_var[4]),
        "=f"(__out_var[5]),
        "=f"(__out_var[6]),
        "=f"(__out_var[7]),
        "=f"(__out_var[8]),
        "=f"(__out_var[9]),
        "=f"(__out_var[10]),
        "=f"(__out_var[11]),
        "=f"(__out_var[12]),
        "=f"(__out_var[13]),
        "=f"(__out_var[14]),
        "=f"(__out_var[15]),
        "=f"(__out_var[16]),
        "=f"(__out_var[17]),
        "=f"(__out_var[18]),
        "=f"(__out_var[19]),
        "=f"(__out_var[20]),
        "=f"(__out_var[21]),
        "=f"(__out_var[22]),
        "=f"(__out_var[23]),
        "=f"(__out_var[24]),
        "=f"(__out_var[25]),
        "=f"(__out_var[26]),
        "=f"(__out_var[27]),
        "=f"(__out_var[28]),
        "=f"(__out_var[29]),
        "=f"(__out_var[30]),
        "=f"(__out_var[31]),
        "=f"(__out_var[32]),
        "=f"(__out_var[33]),
        "=f"(__out_var[34]),
        "=f"(__out_var[35]),
        "=f"(__out_var[36]),
        "=f"(__out_var[37]),
        "=f"(__out_var[38]),
        "=f"(__out_var[39]),
        "=f"(__out_var[40]),
        "=f"(__out_var[41]),
        "=f"(__out_var[42]),
        "=f"(__out_var[43]),
        "=f"(__out_var[44]),
        "=f"(__out_var[45]),
        "=f"(__out_var[46]),
        "=f"(__out_var[47]),
        "=f"(__out_var[48]),
        "=f"(__out_var[49]),
        "=f"(__out_var[50]),
        "=f"(__out_var[51]),
        "=f"(__out_var[52]),
        "=f"(__out_var[53]),
        "=f"(__out_var[54]),
        "=f"(__out_var[55]),
        "=f"(__out_var[56]),
        "=f"(__out_var[57]),
        "=f"(__out_var[58]),
        "=f"(__out_var[59]),
        "=f"(__out_var[60]),
        "=f"(__out_var[61]),
        "=f"(__out_var[62]),
        "=f"(__out_var[63]),
        "=f"(__out_var[64]),
        "=f"(__out_var[65]),
        "=f"(__out_var[66]),
        "=f"(__out_var[67]),
        "=f"(__out_var[68]),
        "=f"(__out_var[69]),
        "=f"(__out_var[70]),
        "=f"(__out_var[71]),
        "=f"(__out_var[72]),
        "=f"(__out_var[73]),
        "=f"(__out_var[74]),
        "=f"(__out_var[75]),
        "=f"(__out_var[76]),
        "=f"(__out_var[77]),
        "=f"(__out_var[78]),
        "=f"(__out_var[79]),
        "=f"(__out_var[80]),
        "=f"(__out_var[81]),
        "=f"(__out_var[82]),
        "=f"(__out_var[83]),
        "=f"(__out_var[84]),
        "=f"(__out_var[85]),
        "=f"(__out_var[86]),
        "=f"(__out_var[87]),
        "=f"(__out_var[88]),
        "=f"(__out_var[89]),
        "=f"(__out_var[90]),
        "=f"(__out_var[91]),
        "=f"(__out_var[92]),
        "=f"(__out_var[93]),
        "=f"(__out_var[94]),
        "=f"(__out_var[95]),
        "=f"(__out_var[96]),
        "=f"(__out_var[97]),
        "=f"(__out_var[98]),
        "=f"(__out_var[99]),
        "=f"(__out_var[100]),
        "=f"(__out_var[101]),
        "=f"(__out_var[102]),
        "=f"(__out_var[103]),
        "=f"(__out_var[104]),
        "=f"(__out_var[105]),
        "=f"(__out_var[106]),
        "=f"(__out_var[107]),
        "=f"(__out_var[108]),
        "=f"(__out_var[109]),
        "=f"(__out_var[110]),
        "=f"(__out_var[111]),
        "=f"(__out_var[112]),
        "=f"(__out_var[113]),
        "=f"(__out_var[114]),
        "=f"(__out_var[115]),
        "=f"(__out_var[116]),
        "=f"(__out_var[117]),
        "=f"(__out_var[118]),
        "=f"(__out_var[119]),
        "=f"(__out_var[120]),
        "=f"(__out_var[121]),
        "=f"(__out_var[122]),
        "=f"(__out_var[123]),
        "=f"(__out_var[124]),
        "=f"(__out_var[125]),
        "=f"(__out_var[126]),
        "=f"(__out_var[127]),
        "=f"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.32x32b.x128.f32.op out_var, redval, [taddr]; // PTX ISA 88, SM_103a, SM_103f, SM_107a,
SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_32x32b(
  cuda::ptx::op_t<Op> op,
  float (&out_var)[128],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 880
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float
tcgen05_ld_red_32x32b(::cuda::ptx::op_t<_Op> __op, float (&__out_var)[128], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
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
      : "=f"(__out_var[0]),
        "=f"(__out_var[1]),
        "=f"(__out_var[2]),
        "=f"(__out_var[3]),
        "=f"(__out_var[4]),
        "=f"(__out_var[5]),
        "=f"(__out_var[6]),
        "=f"(__out_var[7]),
        "=f"(__out_var[8]),
        "=f"(__out_var[9]),
        "=f"(__out_var[10]),
        "=f"(__out_var[11]),
        "=f"(__out_var[12]),
        "=f"(__out_var[13]),
        "=f"(__out_var[14]),
        "=f"(__out_var[15]),
        "=f"(__out_var[16]),
        "=f"(__out_var[17]),
        "=f"(__out_var[18]),
        "=f"(__out_var[19]),
        "=f"(__out_var[20]),
        "=f"(__out_var[21]),
        "=f"(__out_var[22]),
        "=f"(__out_var[23]),
        "=f"(__out_var[24]),
        "=f"(__out_var[25]),
        "=f"(__out_var[26]),
        "=f"(__out_var[27]),
        "=f"(__out_var[28]),
        "=f"(__out_var[29]),
        "=f"(__out_var[30]),
        "=f"(__out_var[31]),
        "=f"(__out_var[32]),
        "=f"(__out_var[33]),
        "=f"(__out_var[34]),
        "=f"(__out_var[35]),
        "=f"(__out_var[36]),
        "=f"(__out_var[37]),
        "=f"(__out_var[38]),
        "=f"(__out_var[39]),
        "=f"(__out_var[40]),
        "=f"(__out_var[41]),
        "=f"(__out_var[42]),
        "=f"(__out_var[43]),
        "=f"(__out_var[44]),
        "=f"(__out_var[45]),
        "=f"(__out_var[46]),
        "=f"(__out_var[47]),
        "=f"(__out_var[48]),
        "=f"(__out_var[49]),
        "=f"(__out_var[50]),
        "=f"(__out_var[51]),
        "=f"(__out_var[52]),
        "=f"(__out_var[53]),
        "=f"(__out_var[54]),
        "=f"(__out_var[55]),
        "=f"(__out_var[56]),
        "=f"(__out_var[57]),
        "=f"(__out_var[58]),
        "=f"(__out_var[59]),
        "=f"(__out_var[60]),
        "=f"(__out_var[61]),
        "=f"(__out_var[62]),
        "=f"(__out_var[63]),
        "=f"(__out_var[64]),
        "=f"(__out_var[65]),
        "=f"(__out_var[66]),
        "=f"(__out_var[67]),
        "=f"(__out_var[68]),
        "=f"(__out_var[69]),
        "=f"(__out_var[70]),
        "=f"(__out_var[71]),
        "=f"(__out_var[72]),
        "=f"(__out_var[73]),
        "=f"(__out_var[74]),
        "=f"(__out_var[75]),
        "=f"(__out_var[76]),
        "=f"(__out_var[77]),
        "=f"(__out_var[78]),
        "=f"(__out_var[79]),
        "=f"(__out_var[80]),
        "=f"(__out_var[81]),
        "=f"(__out_var[82]),
        "=f"(__out_var[83]),
        "=f"(__out_var[84]),
        "=f"(__out_var[85]),
        "=f"(__out_var[86]),
        "=f"(__out_var[87]),
        "=f"(__out_var[88]),
        "=f"(__out_var[89]),
        "=f"(__out_var[90]),
        "=f"(__out_var[91]),
        "=f"(__out_var[92]),
        "=f"(__out_var[93]),
        "=f"(__out_var[94]),
        "=f"(__out_var[95]),
        "=f"(__out_var[96]),
        "=f"(__out_var[97]),
        "=f"(__out_var[98]),
        "=f"(__out_var[99]),
        "=f"(__out_var[100]),
        "=f"(__out_var[101]),
        "=f"(__out_var[102]),
        "=f"(__out_var[103]),
        "=f"(__out_var[104]),
        "=f"(__out_var[105]),
        "=f"(__out_var[106]),
        "=f"(__out_var[107]),
        "=f"(__out_var[108]),
        "=f"(__out_var[109]),
        "=f"(__out_var[110]),
        "=f"(__out_var[111]),
        "=f"(__out_var[112]),
        "=f"(__out_var[113]),
        "=f"(__out_var[114]),
        "=f"(__out_var[115]),
        "=f"(__out_var[116]),
        "=f"(__out_var[117]),
        "=f"(__out_var[118]),
        "=f"(__out_var[119]),
        "=f"(__out_var[120]),
        "=f"(__out_var[121]),
        "=f"(__out_var[122]),
        "=f"(__out_var[123]),
        "=f"(__out_var[124]),
        "=f"(__out_var[125]),
        "=f"(__out_var[126]),
        "=f"(__out_var[127]),
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
      : "=f"(__out_var[0]),
        "=f"(__out_var[1]),
        "=f"(__out_var[2]),
        "=f"(__out_var[3]),
        "=f"(__out_var[4]),
        "=f"(__out_var[5]),
        "=f"(__out_var[6]),
        "=f"(__out_var[7]),
        "=f"(__out_var[8]),
        "=f"(__out_var[9]),
        "=f"(__out_var[10]),
        "=f"(__out_var[11]),
        "=f"(__out_var[12]),
        "=f"(__out_var[13]),
        "=f"(__out_var[14]),
        "=f"(__out_var[15]),
        "=f"(__out_var[16]),
        "=f"(__out_var[17]),
        "=f"(__out_var[18]),
        "=f"(__out_var[19]),
        "=f"(__out_var[20]),
        "=f"(__out_var[21]),
        "=f"(__out_var[22]),
        "=f"(__out_var[23]),
        "=f"(__out_var[24]),
        "=f"(__out_var[25]),
        "=f"(__out_var[26]),
        "=f"(__out_var[27]),
        "=f"(__out_var[28]),
        "=f"(__out_var[29]),
        "=f"(__out_var[30]),
        "=f"(__out_var[31]),
        "=f"(__out_var[32]),
        "=f"(__out_var[33]),
        "=f"(__out_var[34]),
        "=f"(__out_var[35]),
        "=f"(__out_var[36]),
        "=f"(__out_var[37]),
        "=f"(__out_var[38]),
        "=f"(__out_var[39]),
        "=f"(__out_var[40]),
        "=f"(__out_var[41]),
        "=f"(__out_var[42]),
        "=f"(__out_var[43]),
        "=f"(__out_var[44]),
        "=f"(__out_var[45]),
        "=f"(__out_var[46]),
        "=f"(__out_var[47]),
        "=f"(__out_var[48]),
        "=f"(__out_var[49]),
        "=f"(__out_var[50]),
        "=f"(__out_var[51]),
        "=f"(__out_var[52]),
        "=f"(__out_var[53]),
        "=f"(__out_var[54]),
        "=f"(__out_var[55]),
        "=f"(__out_var[56]),
        "=f"(__out_var[57]),
        "=f"(__out_var[58]),
        "=f"(__out_var[59]),
        "=f"(__out_var[60]),
        "=f"(__out_var[61]),
        "=f"(__out_var[62]),
        "=f"(__out_var[63]),
        "=f"(__out_var[64]),
        "=f"(__out_var[65]),
        "=f"(__out_var[66]),
        "=f"(__out_var[67]),
        "=f"(__out_var[68]),
        "=f"(__out_var[69]),
        "=f"(__out_var[70]),
        "=f"(__out_var[71]),
        "=f"(__out_var[72]),
        "=f"(__out_var[73]),
        "=f"(__out_var[74]),
        "=f"(__out_var[75]),
        "=f"(__out_var[76]),
        "=f"(__out_var[77]),
        "=f"(__out_var[78]),
        "=f"(__out_var[79]),
        "=f"(__out_var[80]),
        "=f"(__out_var[81]),
        "=f"(__out_var[82]),
        "=f"(__out_var[83]),
        "=f"(__out_var[84]),
        "=f"(__out_var[85]),
        "=f"(__out_var[86]),
        "=f"(__out_var[87]),
        "=f"(__out_var[88]),
        "=f"(__out_var[89]),
        "=f"(__out_var[90]),
        "=f"(__out_var[91]),
        "=f"(__out_var[92]),
        "=f"(__out_var[93]),
        "=f"(__out_var[94]),
        "=f"(__out_var[95]),
        "=f"(__out_var[96]),
        "=f"(__out_var[97]),
        "=f"(__out_var[98]),
        "=f"(__out_var[99]),
        "=f"(__out_var[100]),
        "=f"(__out_var[101]),
        "=f"(__out_var[102]),
        "=f"(__out_var[103]),
        "=f"(__out_var[104]),
        "=f"(__out_var[105]),
        "=f"(__out_var[106]),
        "=f"(__out_var[107]),
        "=f"(__out_var[108]),
        "=f"(__out_var[109]),
        "=f"(__out_var[110]),
        "=f"(__out_var[111]),
        "=f"(__out_var[112]),
        "=f"(__out_var[113]),
        "=f"(__out_var[114]),
        "=f"(__out_var[115]),
        "=f"(__out_var[116]),
        "=f"(__out_var[117]),
        "=f"(__out_var[118]),
        "=f"(__out_var[119]),
        "=f"(__out_var[120]),
        "=f"(__out_var[121]),
        "=f"(__out_var[122]),
        "=f"(__out_var[123]),
        "=f"(__out_var[124]),
        "=f"(__out_var[125]),
        "=f"(__out_var[126]),
        "=f"(__out_var[127]),
        "=f"(__redval)
      : "r"(__taddr)
      : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x2.u32.op out_var, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline uint32_t tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  uint32_t (&out_var)[2],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::uint32_t tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  ::cuda::std::uint32_t (&__out_var)[2],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
  ::cuda::std::uint32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x2.u32.min {%0, %1}, %2, [%3], %4;"
        : "=r"(__out_var[0]), "=r"(__out_var[1]), "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x2.u32.max {%0, %1}, %2, [%3], %4;"
        : "=r"(__out_var[0]), "=r"(__out_var[1]), "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x2.s32.op out_var, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline int32_t tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  int32_t (&out_var)[2],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::int32_t tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  ::cuda::std::int32_t (&__out_var)[2],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
  ::cuda::std::int32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x2.s32.min {%0, %1}, %2, [%3], %4;"
        : "=r"(__out_var[0]), "=r"(__out_var[1]), "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x2.s32.max {%0, %1}, %2, [%3], %4;"
        : "=r"(__out_var[0]), "=r"(__out_var[1]), "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x2.f32.op.abs out_var, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_16x32bx2_abs(
  cuda::ptx::op_t<Op> op,
  float (&out_var)[2],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_16x32bx2_abs(
  ::cuda::ptx::op_t<_Op> __op,
  float (&__out_var)[2],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x2.f32.min.abs {%0, %1}, %2, [%3], %4;"
        : "=f"(__out_var[0]), "=f"(__out_var[1]), "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x2.f32.max.abs {%0, %1}, %2, [%3], %4;"
        : "=f"(__out_var[0]), "=f"(__out_var[1]), "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x2.f32.op out_var, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  float (&out_var)[2],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  float (&__out_var)[2],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x2.f32.min {%0, %1}, %2, [%3], %4;"
        : "=f"(__out_var[0]), "=f"(__out_var[1]), "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x2.f32.max {%0, %1}, %2, [%3], %4;"
        : "=f"(__out_var[0]), "=f"(__out_var[1]), "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x4.u32.op out_var, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline uint32_t tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  uint32_t (&out_var)[4],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::uint32_t tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  ::cuda::std::uint32_t (&__out_var)[4],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
  ::cuda::std::uint32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x4.u32.min {%0, %1, %2, %3}, %4, [%5], %6;"
        : "=r"(__out_var[0]), "=r"(__out_var[1]), "=r"(__out_var[2]), "=r"(__out_var[3]), "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x4.u32.max {%0, %1, %2, %3}, %4, [%5], %6;"
        : "=r"(__out_var[0]), "=r"(__out_var[1]), "=r"(__out_var[2]), "=r"(__out_var[3]), "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x4.s32.op out_var, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline int32_t tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  int32_t (&out_var)[4],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::int32_t tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  ::cuda::std::int32_t (&__out_var)[4],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
  ::cuda::std::int32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x4.s32.min {%0, %1, %2, %3}, %4, [%5], %6;"
        : "=r"(__out_var[0]), "=r"(__out_var[1]), "=r"(__out_var[2]), "=r"(__out_var[3]), "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x4.s32.max {%0, %1, %2, %3}, %4, [%5], %6;"
        : "=r"(__out_var[0]), "=r"(__out_var[1]), "=r"(__out_var[2]), "=r"(__out_var[3]), "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.op.abs out_var, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_16x32bx2_abs(
  cuda::ptx::op_t<Op> op,
  float (&out_var)[4],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_16x32bx2_abs(
  ::cuda::ptx::op_t<_Op> __op,
  float (&__out_var)[4],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.min.abs {%0, %1, %2, %3}, %4, [%5], %6;"
        : "=f"(__out_var[0]), "=f"(__out_var[1]), "=f"(__out_var[2]), "=f"(__out_var[3]), "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.max.abs {%0, %1, %2, %3}, %4, [%5], %6;"
        : "=f"(__out_var[0]), "=f"(__out_var[1]), "=f"(__out_var[2]), "=f"(__out_var[3]), "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.op out_var, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  float (&out_var)[4],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  float (&__out_var)[4],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.min {%0, %1, %2, %3}, %4, [%5], %6;"
        : "=f"(__out_var[0]), "=f"(__out_var[1]), "=f"(__out_var[2]), "=f"(__out_var[3]), "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.max {%0, %1, %2, %3}, %4, [%5], %6;"
        : "=f"(__out_var[0]), "=f"(__out_var[1]), "=f"(__out_var[2]), "=f"(__out_var[3]), "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x8.u32.op out_var, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline uint32_t tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  uint32_t (&out_var)[8],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::uint32_t tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  ::cuda::std::uint32_t (&__out_var)[8],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
  ::cuda::std::uint32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x8.u32.min {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9], %10;"
        : "=r"(__out_var[0]),
          "=r"(__out_var[1]),
          "=r"(__out_var[2]),
          "=r"(__out_var[3]),
          "=r"(__out_var[4]),
          "=r"(__out_var[5]),
          "=r"(__out_var[6]),
          "=r"(__out_var[7]),
          "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x8.u32.max {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9], %10;"
        : "=r"(__out_var[0]),
          "=r"(__out_var[1]),
          "=r"(__out_var[2]),
          "=r"(__out_var[3]),
          "=r"(__out_var[4]),
          "=r"(__out_var[5]),
          "=r"(__out_var[6]),
          "=r"(__out_var[7]),
          "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x8.s32.op out_var, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline int32_t tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  int32_t (&out_var)[8],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::int32_t tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  ::cuda::std::int32_t (&__out_var)[8],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
  ::cuda::std::int32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x8.s32.min {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9], %10;"
        : "=r"(__out_var[0]),
          "=r"(__out_var[1]),
          "=r"(__out_var[2]),
          "=r"(__out_var[3]),
          "=r"(__out_var[4]),
          "=r"(__out_var[5]),
          "=r"(__out_var[6]),
          "=r"(__out_var[7]),
          "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x8.s32.max {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9], %10;"
        : "=r"(__out_var[0]),
          "=r"(__out_var[1]),
          "=r"(__out_var[2]),
          "=r"(__out_var[3]),
          "=r"(__out_var[4]),
          "=r"(__out_var[5]),
          "=r"(__out_var[6]),
          "=r"(__out_var[7]),
          "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x8.f32.op.abs out_var, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_16x32bx2_abs(
  cuda::ptx::op_t<Op> op,
  float (&out_var)[8],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_16x32bx2_abs(
  ::cuda::ptx::op_t<_Op> __op,
  float (&__out_var)[8],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x8.f32.min.abs {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9], %10;"
        : "=f"(__out_var[0]),
          "=f"(__out_var[1]),
          "=f"(__out_var[2]),
          "=f"(__out_var[3]),
          "=f"(__out_var[4]),
          "=f"(__out_var[5]),
          "=f"(__out_var[6]),
          "=f"(__out_var[7]),
          "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x8.f32.max.abs {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9], %10;"
        : "=f"(__out_var[0]),
          "=f"(__out_var[1]),
          "=f"(__out_var[2]),
          "=f"(__out_var[3]),
          "=f"(__out_var[4]),
          "=f"(__out_var[5]),
          "=f"(__out_var[6]),
          "=f"(__out_var[7]),
          "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x8.f32.op out_var, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  float (&out_var)[8],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  float (&__out_var)[8],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x8.f32.min {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9], %10;"
        : "=f"(__out_var[0]),
          "=f"(__out_var[1]),
          "=f"(__out_var[2]),
          "=f"(__out_var[3]),
          "=f"(__out_var[4]),
          "=f"(__out_var[5]),
          "=f"(__out_var[6]),
          "=f"(__out_var[7]),
          "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x8.f32.max {%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9], %10;"
        : "=f"(__out_var[0]),
          "=f"(__out_var[1]),
          "=f"(__out_var[2]),
          "=f"(__out_var[3]),
          "=f"(__out_var[4]),
          "=f"(__out_var[5]),
          "=f"(__out_var[6]),
          "=f"(__out_var[7]),
          "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x16.u32.op out_var, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline uint32_t tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  uint32_t (&out_var)[16],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::uint32_t tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  ::cuda::std::uint32_t (&__out_var)[16],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
  ::cuda::std::uint32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x16.u32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
        "%14, %15}, %16, [%17], %18;"
        : "=r"(__out_var[0]),
          "=r"(__out_var[1]),
          "=r"(__out_var[2]),
          "=r"(__out_var[3]),
          "=r"(__out_var[4]),
          "=r"(__out_var[5]),
          "=r"(__out_var[6]),
          "=r"(__out_var[7]),
          "=r"(__out_var[8]),
          "=r"(__out_var[9]),
          "=r"(__out_var[10]),
          "=r"(__out_var[11]),
          "=r"(__out_var[12]),
          "=r"(__out_var[13]),
          "=r"(__out_var[14]),
          "=r"(__out_var[15]),
          "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x16.u32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
        "%14, %15}, %16, [%17], %18;"
        : "=r"(__out_var[0]),
          "=r"(__out_var[1]),
          "=r"(__out_var[2]),
          "=r"(__out_var[3]),
          "=r"(__out_var[4]),
          "=r"(__out_var[5]),
          "=r"(__out_var[6]),
          "=r"(__out_var[7]),
          "=r"(__out_var[8]),
          "=r"(__out_var[9]),
          "=r"(__out_var[10]),
          "=r"(__out_var[11]),
          "=r"(__out_var[12]),
          "=r"(__out_var[13]),
          "=r"(__out_var[14]),
          "=r"(__out_var[15]),
          "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x16.s32.op out_var, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline int32_t tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  int32_t (&out_var)[16],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::int32_t tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  ::cuda::std::int32_t (&__out_var)[16],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
  ::cuda::std::int32_t __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x16.s32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
        "%14, %15}, %16, [%17], %18;"
        : "=r"(__out_var[0]),
          "=r"(__out_var[1]),
          "=r"(__out_var[2]),
          "=r"(__out_var[3]),
          "=r"(__out_var[4]),
          "=r"(__out_var[5]),
          "=r"(__out_var[6]),
          "=r"(__out_var[7]),
          "=r"(__out_var[8]),
          "=r"(__out_var[9]),
          "=r"(__out_var[10]),
          "=r"(__out_var[11]),
          "=r"(__out_var[12]),
          "=r"(__out_var[13]),
          "=r"(__out_var[14]),
          "=r"(__out_var[15]),
          "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x16.s32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
        "%14, %15}, %16, [%17], %18;"
        : "=r"(__out_var[0]),
          "=r"(__out_var[1]),
          "=r"(__out_var[2]),
          "=r"(__out_var[3]),
          "=r"(__out_var[4]),
          "=r"(__out_var[5]),
          "=r"(__out_var[6]),
          "=r"(__out_var[7]),
          "=r"(__out_var[8]),
          "=r"(__out_var[9]),
          "=r"(__out_var[10]),
          "=r"(__out_var[11]),
          "=r"(__out_var[12]),
          "=r"(__out_var[13]),
          "=r"(__out_var[14]),
          "=r"(__out_var[15]),
          "=r"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x16.f32.op.abs out_var, redval, [taddr], immHalfSplitoff; // PTX ISA 88,
SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_16x32bx2_abs(
  cuda::ptx::op_t<Op> op,
  float (&out_var)[16],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_16x32bx2_abs(
  ::cuda::ptx::op_t<_Op> __op,
  float (&__out_var)[16],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x16.f32.min.abs {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
        "%13, %14, %15}, %16, [%17], %18;"
        : "=f"(__out_var[0]),
          "=f"(__out_var[1]),
          "=f"(__out_var[2]),
          "=f"(__out_var[3]),
          "=f"(__out_var[4]),
          "=f"(__out_var[5]),
          "=f"(__out_var[6]),
          "=f"(__out_var[7]),
          "=f"(__out_var[8]),
          "=f"(__out_var[9]),
          "=f"(__out_var[10]),
          "=f"(__out_var[11]),
          "=f"(__out_var[12]),
          "=f"(__out_var[13]),
          "=f"(__out_var[14]),
          "=f"(__out_var[15]),
          "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x16.f32.max.abs {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
        "%13, %14, %15}, %16, [%17], %18;"
        : "=f"(__out_var[0]),
          "=f"(__out_var[1]),
          "=f"(__out_var[2]),
          "=f"(__out_var[3]),
          "=f"(__out_var[4]),
          "=f"(__out_var[5]),
          "=f"(__out_var[6]),
          "=f"(__out_var[7]),
          "=f"(__out_var[8]),
          "=f"(__out_var[9]),
          "=f"(__out_var[10]),
          "=f"(__out_var[11]),
          "=f"(__out_var[12]),
          "=f"(__out_var[13]),
          "=f"(__out_var[14]),
          "=f"(__out_var[15]),
          "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x16.f32.op out_var, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  float (&out_var)[16],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  float (&__out_var)[16],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x16.f32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
        "%14, %15}, %16, [%17], %18;"
        : "=f"(__out_var[0]),
          "=f"(__out_var[1]),
          "=f"(__out_var[2]),
          "=f"(__out_var[3]),
          "=f"(__out_var[4]),
          "=f"(__out_var[5]),
          "=f"(__out_var[6]),
          "=f"(__out_var[7]),
          "=f"(__out_var[8]),
          "=f"(__out_var[9]),
          "=f"(__out_var[10]),
          "=f"(__out_var[11]),
          "=f"(__out_var[12]),
          "=f"(__out_var[13]),
          "=f"(__out_var[14]),
          "=f"(__out_var[15]),
          "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.sync.aligned.16x32bx2.x16.f32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
        "%14, %15}, %16, [%17], %18;"
        : "=f"(__out_var[0]),
          "=f"(__out_var[1]),
          "=f"(__out_var[2]),
          "=f"(__out_var[3]),
          "=f"(__out_var[4]),
          "=f"(__out_var[5]),
          "=f"(__out_var[6]),
          "=f"(__out_var[7]),
          "=f"(__out_var[8]),
          "=f"(__out_var[9]),
          "=f"(__out_var[10]),
          "=f"(__out_var[11]),
          "=f"(__out_var[12]),
          "=f"(__out_var[13]),
          "=f"(__out_var[14]),
          "=f"(__out_var[15]),
          "=f"(__redval)
        : "r"(__taddr), "n"(__immHalfSplitoff.value)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x32.u32.op out_var, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline uint32_t tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  uint32_t (&out_var)[32],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::uint32_t tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  ::cuda::std::uint32_t (&__out_var)[32],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
  ::cuda::std::uint32_t __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x32.u32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33], %34;"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15]),
        "=r"(__out_var[16]),
        "=r"(__out_var[17]),
        "=r"(__out_var[18]),
        "=r"(__out_var[19]),
        "=r"(__out_var[20]),
        "=r"(__out_var[21]),
        "=r"(__out_var[22]),
        "=r"(__out_var[23]),
        "=r"(__out_var[24]),
        "=r"(__out_var[25]),
        "=r"(__out_var[26]),
        "=r"(__out_var[27]),
        "=r"(__out_var[28]),
        "=r"(__out_var[29]),
        "=r"(__out_var[30]),
        "=r"(__out_var[31]),
        "=r"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x32.u32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33], %34;"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15]),
        "=r"(__out_var[16]),
        "=r"(__out_var[17]),
        "=r"(__out_var[18]),
        "=r"(__out_var[19]),
        "=r"(__out_var[20]),
        "=r"(__out_var[21]),
        "=r"(__out_var[22]),
        "=r"(__out_var[23]),
        "=r"(__out_var[24]),
        "=r"(__out_var[25]),
        "=r"(__out_var[26]),
        "=r"(__out_var[27]),
        "=r"(__out_var[28]),
        "=r"(__out_var[29]),
        "=r"(__out_var[30]),
        "=r"(__out_var[31]),
        "=r"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x32.s32.op out_var, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline int32_t tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  int32_t (&out_var)[32],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::int32_t tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  ::cuda::std::int32_t (&__out_var)[32],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
  ::cuda::std::int32_t __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x32.s32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33], %34;"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15]),
        "=r"(__out_var[16]),
        "=r"(__out_var[17]),
        "=r"(__out_var[18]),
        "=r"(__out_var[19]),
        "=r"(__out_var[20]),
        "=r"(__out_var[21]),
        "=r"(__out_var[22]),
        "=r"(__out_var[23]),
        "=r"(__out_var[24]),
        "=r"(__out_var[25]),
        "=r"(__out_var[26]),
        "=r"(__out_var[27]),
        "=r"(__out_var[28]),
        "=r"(__out_var[29]),
        "=r"(__out_var[30]),
        "=r"(__out_var[31]),
        "=r"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x32.s32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33], %34;"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15]),
        "=r"(__out_var[16]),
        "=r"(__out_var[17]),
        "=r"(__out_var[18]),
        "=r"(__out_var[19]),
        "=r"(__out_var[20]),
        "=r"(__out_var[21]),
        "=r"(__out_var[22]),
        "=r"(__out_var[23]),
        "=r"(__out_var[24]),
        "=r"(__out_var[25]),
        "=r"(__out_var[26]),
        "=r"(__out_var[27]),
        "=r"(__out_var[28]),
        "=r"(__out_var[29]),
        "=r"(__out_var[30]),
        "=r"(__out_var[31]),
        "=r"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.op.abs out_var, redval, [taddr], immHalfSplitoff; // PTX ISA 88,
SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_16x32bx2_abs(
  cuda::ptx::op_t<Op> op,
  float (&out_var)[32],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_16x32bx2_abs(
  ::cuda::ptx::op_t<_Op> __op,
  float (&__out_var)[32],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.min.abs {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
      "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33], %34;"
      : "=f"(__out_var[0]),
        "=f"(__out_var[1]),
        "=f"(__out_var[2]),
        "=f"(__out_var[3]),
        "=f"(__out_var[4]),
        "=f"(__out_var[5]),
        "=f"(__out_var[6]),
        "=f"(__out_var[7]),
        "=f"(__out_var[8]),
        "=f"(__out_var[9]),
        "=f"(__out_var[10]),
        "=f"(__out_var[11]),
        "=f"(__out_var[12]),
        "=f"(__out_var[13]),
        "=f"(__out_var[14]),
        "=f"(__out_var[15]),
        "=f"(__out_var[16]),
        "=f"(__out_var[17]),
        "=f"(__out_var[18]),
        "=f"(__out_var[19]),
        "=f"(__out_var[20]),
        "=f"(__out_var[21]),
        "=f"(__out_var[22]),
        "=f"(__out_var[23]),
        "=f"(__out_var[24]),
        "=f"(__out_var[25]),
        "=f"(__out_var[26]),
        "=f"(__out_var[27]),
        "=f"(__out_var[28]),
        "=f"(__out_var[29]),
        "=f"(__out_var[30]),
        "=f"(__out_var[31]),
        "=f"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.max.abs {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
      "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33], %34;"
      : "=f"(__out_var[0]),
        "=f"(__out_var[1]),
        "=f"(__out_var[2]),
        "=f"(__out_var[3]),
        "=f"(__out_var[4]),
        "=f"(__out_var[5]),
        "=f"(__out_var[6]),
        "=f"(__out_var[7]),
        "=f"(__out_var[8]),
        "=f"(__out_var[9]),
        "=f"(__out_var[10]),
        "=f"(__out_var[11]),
        "=f"(__out_var[12]),
        "=f"(__out_var[13]),
        "=f"(__out_var[14]),
        "=f"(__out_var[15]),
        "=f"(__out_var[16]),
        "=f"(__out_var[17]),
        "=f"(__out_var[18]),
        "=f"(__out_var[19]),
        "=f"(__out_var[20]),
        "=f"(__out_var[21]),
        "=f"(__out_var[22]),
        "=f"(__out_var[23]),
        "=f"(__out_var[24]),
        "=f"(__out_var[25]),
        "=f"(__out_var[26]),
        "=f"(__out_var[27]),
        "=f"(__out_var[28]),
        "=f"(__out_var[29]),
        "=f"(__out_var[30]),
        "=f"(__out_var[31]),
        "=f"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.op out_var, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  float (&out_var)[32],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  float (&__out_var)[32],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33], %34;"
      : "=f"(__out_var[0]),
        "=f"(__out_var[1]),
        "=f"(__out_var[2]),
        "=f"(__out_var[3]),
        "=f"(__out_var[4]),
        "=f"(__out_var[5]),
        "=f"(__out_var[6]),
        "=f"(__out_var[7]),
        "=f"(__out_var[8]),
        "=f"(__out_var[9]),
        "=f"(__out_var[10]),
        "=f"(__out_var[11]),
        "=f"(__out_var[12]),
        "=f"(__out_var[13]),
        "=f"(__out_var[14]),
        "=f"(__out_var[15]),
        "=f"(__out_var[16]),
        "=f"(__out_var[17]),
        "=f"(__out_var[18]),
        "=f"(__out_var[19]),
        "=f"(__out_var[20]),
        "=f"(__out_var[21]),
        "=f"(__out_var[22]),
        "=f"(__out_var[23]),
        "=f"(__out_var[24]),
        "=f"(__out_var[25]),
        "=f"(__out_var[26]),
        "=f"(__out_var[27]),
        "=f"(__out_var[28]),
        "=f"(__out_var[29]),
        "=f"(__out_var[30]),
        "=f"(__out_var[31]),
        "=f"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.max {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33], %34;"
      : "=f"(__out_var[0]),
        "=f"(__out_var[1]),
        "=f"(__out_var[2]),
        "=f"(__out_var[3]),
        "=f"(__out_var[4]),
        "=f"(__out_var[5]),
        "=f"(__out_var[6]),
        "=f"(__out_var[7]),
        "=f"(__out_var[8]),
        "=f"(__out_var[9]),
        "=f"(__out_var[10]),
        "=f"(__out_var[11]),
        "=f"(__out_var[12]),
        "=f"(__out_var[13]),
        "=f"(__out_var[14]),
        "=f"(__out_var[15]),
        "=f"(__out_var[16]),
        "=f"(__out_var[17]),
        "=f"(__out_var[18]),
        "=f"(__out_var[19]),
        "=f"(__out_var[20]),
        "=f"(__out_var[21]),
        "=f"(__out_var[22]),
        "=f"(__out_var[23]),
        "=f"(__out_var[24]),
        "=f"(__out_var[25]),
        "=f"(__out_var[26]),
        "=f"(__out_var[27]),
        "=f"(__out_var[28]),
        "=f"(__out_var[29]),
        "=f"(__out_var[30]),
        "=f"(__out_var[31]),
        "=f"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x64.u32.op out_var, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline uint32_t tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  uint32_t (&out_var)[64],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::uint32_t tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  ::cuda::std::uint32_t (&__out_var)[64],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
  ::cuda::std::uint32_t __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x64.u32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63}, %64, [%65], %66;"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15]),
        "=r"(__out_var[16]),
        "=r"(__out_var[17]),
        "=r"(__out_var[18]),
        "=r"(__out_var[19]),
        "=r"(__out_var[20]),
        "=r"(__out_var[21]),
        "=r"(__out_var[22]),
        "=r"(__out_var[23]),
        "=r"(__out_var[24]),
        "=r"(__out_var[25]),
        "=r"(__out_var[26]),
        "=r"(__out_var[27]),
        "=r"(__out_var[28]),
        "=r"(__out_var[29]),
        "=r"(__out_var[30]),
        "=r"(__out_var[31]),
        "=r"(__out_var[32]),
        "=r"(__out_var[33]),
        "=r"(__out_var[34]),
        "=r"(__out_var[35]),
        "=r"(__out_var[36]),
        "=r"(__out_var[37]),
        "=r"(__out_var[38]),
        "=r"(__out_var[39]),
        "=r"(__out_var[40]),
        "=r"(__out_var[41]),
        "=r"(__out_var[42]),
        "=r"(__out_var[43]),
        "=r"(__out_var[44]),
        "=r"(__out_var[45]),
        "=r"(__out_var[46]),
        "=r"(__out_var[47]),
        "=r"(__out_var[48]),
        "=r"(__out_var[49]),
        "=r"(__out_var[50]),
        "=r"(__out_var[51]),
        "=r"(__out_var[52]),
        "=r"(__out_var[53]),
        "=r"(__out_var[54]),
        "=r"(__out_var[55]),
        "=r"(__out_var[56]),
        "=r"(__out_var[57]),
        "=r"(__out_var[58]),
        "=r"(__out_var[59]),
        "=r"(__out_var[60]),
        "=r"(__out_var[61]),
        "=r"(__out_var[62]),
        "=r"(__out_var[63]),
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
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15]),
        "=r"(__out_var[16]),
        "=r"(__out_var[17]),
        "=r"(__out_var[18]),
        "=r"(__out_var[19]),
        "=r"(__out_var[20]),
        "=r"(__out_var[21]),
        "=r"(__out_var[22]),
        "=r"(__out_var[23]),
        "=r"(__out_var[24]),
        "=r"(__out_var[25]),
        "=r"(__out_var[26]),
        "=r"(__out_var[27]),
        "=r"(__out_var[28]),
        "=r"(__out_var[29]),
        "=r"(__out_var[30]),
        "=r"(__out_var[31]),
        "=r"(__out_var[32]),
        "=r"(__out_var[33]),
        "=r"(__out_var[34]),
        "=r"(__out_var[35]),
        "=r"(__out_var[36]),
        "=r"(__out_var[37]),
        "=r"(__out_var[38]),
        "=r"(__out_var[39]),
        "=r"(__out_var[40]),
        "=r"(__out_var[41]),
        "=r"(__out_var[42]),
        "=r"(__out_var[43]),
        "=r"(__out_var[44]),
        "=r"(__out_var[45]),
        "=r"(__out_var[46]),
        "=r"(__out_var[47]),
        "=r"(__out_var[48]),
        "=r"(__out_var[49]),
        "=r"(__out_var[50]),
        "=r"(__out_var[51]),
        "=r"(__out_var[52]),
        "=r"(__out_var[53]),
        "=r"(__out_var[54]),
        "=r"(__out_var[55]),
        "=r"(__out_var[56]),
        "=r"(__out_var[57]),
        "=r"(__out_var[58]),
        "=r"(__out_var[59]),
        "=r"(__out_var[60]),
        "=r"(__out_var[61]),
        "=r"(__out_var[62]),
        "=r"(__out_var[63]),
        "=r"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x64.s32.op out_var, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline int32_t tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  int32_t (&out_var)[64],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::int32_t tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  ::cuda::std::int32_t (&__out_var)[64],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
  ::cuda::std::int32_t __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x64.s32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63}, %64, [%65], %66;"
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15]),
        "=r"(__out_var[16]),
        "=r"(__out_var[17]),
        "=r"(__out_var[18]),
        "=r"(__out_var[19]),
        "=r"(__out_var[20]),
        "=r"(__out_var[21]),
        "=r"(__out_var[22]),
        "=r"(__out_var[23]),
        "=r"(__out_var[24]),
        "=r"(__out_var[25]),
        "=r"(__out_var[26]),
        "=r"(__out_var[27]),
        "=r"(__out_var[28]),
        "=r"(__out_var[29]),
        "=r"(__out_var[30]),
        "=r"(__out_var[31]),
        "=r"(__out_var[32]),
        "=r"(__out_var[33]),
        "=r"(__out_var[34]),
        "=r"(__out_var[35]),
        "=r"(__out_var[36]),
        "=r"(__out_var[37]),
        "=r"(__out_var[38]),
        "=r"(__out_var[39]),
        "=r"(__out_var[40]),
        "=r"(__out_var[41]),
        "=r"(__out_var[42]),
        "=r"(__out_var[43]),
        "=r"(__out_var[44]),
        "=r"(__out_var[45]),
        "=r"(__out_var[46]),
        "=r"(__out_var[47]),
        "=r"(__out_var[48]),
        "=r"(__out_var[49]),
        "=r"(__out_var[50]),
        "=r"(__out_var[51]),
        "=r"(__out_var[52]),
        "=r"(__out_var[53]),
        "=r"(__out_var[54]),
        "=r"(__out_var[55]),
        "=r"(__out_var[56]),
        "=r"(__out_var[57]),
        "=r"(__out_var[58]),
        "=r"(__out_var[59]),
        "=r"(__out_var[60]),
        "=r"(__out_var[61]),
        "=r"(__out_var[62]),
        "=r"(__out_var[63]),
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
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15]),
        "=r"(__out_var[16]),
        "=r"(__out_var[17]),
        "=r"(__out_var[18]),
        "=r"(__out_var[19]),
        "=r"(__out_var[20]),
        "=r"(__out_var[21]),
        "=r"(__out_var[22]),
        "=r"(__out_var[23]),
        "=r"(__out_var[24]),
        "=r"(__out_var[25]),
        "=r"(__out_var[26]),
        "=r"(__out_var[27]),
        "=r"(__out_var[28]),
        "=r"(__out_var[29]),
        "=r"(__out_var[30]),
        "=r"(__out_var[31]),
        "=r"(__out_var[32]),
        "=r"(__out_var[33]),
        "=r"(__out_var[34]),
        "=r"(__out_var[35]),
        "=r"(__out_var[36]),
        "=r"(__out_var[37]),
        "=r"(__out_var[38]),
        "=r"(__out_var[39]),
        "=r"(__out_var[40]),
        "=r"(__out_var[41]),
        "=r"(__out_var[42]),
        "=r"(__out_var[43]),
        "=r"(__out_var[44]),
        "=r"(__out_var[45]),
        "=r"(__out_var[46]),
        "=r"(__out_var[47]),
        "=r"(__out_var[48]),
        "=r"(__out_var[49]),
        "=r"(__out_var[50]),
        "=r"(__out_var[51]),
        "=r"(__out_var[52]),
        "=r"(__out_var[53]),
        "=r"(__out_var[54]),
        "=r"(__out_var[55]),
        "=r"(__out_var[56]),
        "=r"(__out_var[57]),
        "=r"(__out_var[58]),
        "=r"(__out_var[59]),
        "=r"(__out_var[60]),
        "=r"(__out_var[61]),
        "=r"(__out_var[62]),
        "=r"(__out_var[63]),
        "=r"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x64.f32.op.abs out_var, redval, [taddr], immHalfSplitoff; // PTX ISA 88,
SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_16x32bx2_abs(
  cuda::ptx::op_t<Op> op,
  float (&out_var)[64],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_16x32bx2_abs(
  ::cuda::ptx::op_t<_Op> __op,
  float (&__out_var)[64],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x64.f32.min.abs {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, "
      "%13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, "
      "%35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, "
      "%57, %58, %59, %60, %61, %62, %63}, %64, [%65], %66;"
      : "=f"(__out_var[0]),
        "=f"(__out_var[1]),
        "=f"(__out_var[2]),
        "=f"(__out_var[3]),
        "=f"(__out_var[4]),
        "=f"(__out_var[5]),
        "=f"(__out_var[6]),
        "=f"(__out_var[7]),
        "=f"(__out_var[8]),
        "=f"(__out_var[9]),
        "=f"(__out_var[10]),
        "=f"(__out_var[11]),
        "=f"(__out_var[12]),
        "=f"(__out_var[13]),
        "=f"(__out_var[14]),
        "=f"(__out_var[15]),
        "=f"(__out_var[16]),
        "=f"(__out_var[17]),
        "=f"(__out_var[18]),
        "=f"(__out_var[19]),
        "=f"(__out_var[20]),
        "=f"(__out_var[21]),
        "=f"(__out_var[22]),
        "=f"(__out_var[23]),
        "=f"(__out_var[24]),
        "=f"(__out_var[25]),
        "=f"(__out_var[26]),
        "=f"(__out_var[27]),
        "=f"(__out_var[28]),
        "=f"(__out_var[29]),
        "=f"(__out_var[30]),
        "=f"(__out_var[31]),
        "=f"(__out_var[32]),
        "=f"(__out_var[33]),
        "=f"(__out_var[34]),
        "=f"(__out_var[35]),
        "=f"(__out_var[36]),
        "=f"(__out_var[37]),
        "=f"(__out_var[38]),
        "=f"(__out_var[39]),
        "=f"(__out_var[40]),
        "=f"(__out_var[41]),
        "=f"(__out_var[42]),
        "=f"(__out_var[43]),
        "=f"(__out_var[44]),
        "=f"(__out_var[45]),
        "=f"(__out_var[46]),
        "=f"(__out_var[47]),
        "=f"(__out_var[48]),
        "=f"(__out_var[49]),
        "=f"(__out_var[50]),
        "=f"(__out_var[51]),
        "=f"(__out_var[52]),
        "=f"(__out_var[53]),
        "=f"(__out_var[54]),
        "=f"(__out_var[55]),
        "=f"(__out_var[56]),
        "=f"(__out_var[57]),
        "=f"(__out_var[58]),
        "=f"(__out_var[59]),
        "=f"(__out_var[60]),
        "=f"(__out_var[61]),
        "=f"(__out_var[62]),
        "=f"(__out_var[63]),
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
      : "=f"(__out_var[0]),
        "=f"(__out_var[1]),
        "=f"(__out_var[2]),
        "=f"(__out_var[3]),
        "=f"(__out_var[4]),
        "=f"(__out_var[5]),
        "=f"(__out_var[6]),
        "=f"(__out_var[7]),
        "=f"(__out_var[8]),
        "=f"(__out_var[9]),
        "=f"(__out_var[10]),
        "=f"(__out_var[11]),
        "=f"(__out_var[12]),
        "=f"(__out_var[13]),
        "=f"(__out_var[14]),
        "=f"(__out_var[15]),
        "=f"(__out_var[16]),
        "=f"(__out_var[17]),
        "=f"(__out_var[18]),
        "=f"(__out_var[19]),
        "=f"(__out_var[20]),
        "=f"(__out_var[21]),
        "=f"(__out_var[22]),
        "=f"(__out_var[23]),
        "=f"(__out_var[24]),
        "=f"(__out_var[25]),
        "=f"(__out_var[26]),
        "=f"(__out_var[27]),
        "=f"(__out_var[28]),
        "=f"(__out_var[29]),
        "=f"(__out_var[30]),
        "=f"(__out_var[31]),
        "=f"(__out_var[32]),
        "=f"(__out_var[33]),
        "=f"(__out_var[34]),
        "=f"(__out_var[35]),
        "=f"(__out_var[36]),
        "=f"(__out_var[37]),
        "=f"(__out_var[38]),
        "=f"(__out_var[39]),
        "=f"(__out_var[40]),
        "=f"(__out_var[41]),
        "=f"(__out_var[42]),
        "=f"(__out_var[43]),
        "=f"(__out_var[44]),
        "=f"(__out_var[45]),
        "=f"(__out_var[46]),
        "=f"(__out_var[47]),
        "=f"(__out_var[48]),
        "=f"(__out_var[49]),
        "=f"(__out_var[50]),
        "=f"(__out_var[51]),
        "=f"(__out_var[52]),
        "=f"(__out_var[53]),
        "=f"(__out_var[54]),
        "=f"(__out_var[55]),
        "=f"(__out_var[56]),
        "=f"(__out_var[57]),
        "=f"(__out_var[58]),
        "=f"(__out_var[59]),
        "=f"(__out_var[60]),
        "=f"(__out_var[61]),
        "=f"(__out_var[62]),
        "=f"(__out_var[63]),
        "=f"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x64.f32.op out_var, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  float (&out_var)[64],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  float (&__out_var)[64],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.sync.aligned.16x32bx2.x64.f32.min {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, "
      "%36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
      "%58, %59, %60, %61, %62, %63}, %64, [%65], %66;"
      : "=f"(__out_var[0]),
        "=f"(__out_var[1]),
        "=f"(__out_var[2]),
        "=f"(__out_var[3]),
        "=f"(__out_var[4]),
        "=f"(__out_var[5]),
        "=f"(__out_var[6]),
        "=f"(__out_var[7]),
        "=f"(__out_var[8]),
        "=f"(__out_var[9]),
        "=f"(__out_var[10]),
        "=f"(__out_var[11]),
        "=f"(__out_var[12]),
        "=f"(__out_var[13]),
        "=f"(__out_var[14]),
        "=f"(__out_var[15]),
        "=f"(__out_var[16]),
        "=f"(__out_var[17]),
        "=f"(__out_var[18]),
        "=f"(__out_var[19]),
        "=f"(__out_var[20]),
        "=f"(__out_var[21]),
        "=f"(__out_var[22]),
        "=f"(__out_var[23]),
        "=f"(__out_var[24]),
        "=f"(__out_var[25]),
        "=f"(__out_var[26]),
        "=f"(__out_var[27]),
        "=f"(__out_var[28]),
        "=f"(__out_var[29]),
        "=f"(__out_var[30]),
        "=f"(__out_var[31]),
        "=f"(__out_var[32]),
        "=f"(__out_var[33]),
        "=f"(__out_var[34]),
        "=f"(__out_var[35]),
        "=f"(__out_var[36]),
        "=f"(__out_var[37]),
        "=f"(__out_var[38]),
        "=f"(__out_var[39]),
        "=f"(__out_var[40]),
        "=f"(__out_var[41]),
        "=f"(__out_var[42]),
        "=f"(__out_var[43]),
        "=f"(__out_var[44]),
        "=f"(__out_var[45]),
        "=f"(__out_var[46]),
        "=f"(__out_var[47]),
        "=f"(__out_var[48]),
        "=f"(__out_var[49]),
        "=f"(__out_var[50]),
        "=f"(__out_var[51]),
        "=f"(__out_var[52]),
        "=f"(__out_var[53]),
        "=f"(__out_var[54]),
        "=f"(__out_var[55]),
        "=f"(__out_var[56]),
        "=f"(__out_var[57]),
        "=f"(__out_var[58]),
        "=f"(__out_var[59]),
        "=f"(__out_var[60]),
        "=f"(__out_var[61]),
        "=f"(__out_var[62]),
        "=f"(__out_var[63]),
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
      : "=f"(__out_var[0]),
        "=f"(__out_var[1]),
        "=f"(__out_var[2]),
        "=f"(__out_var[3]),
        "=f"(__out_var[4]),
        "=f"(__out_var[5]),
        "=f"(__out_var[6]),
        "=f"(__out_var[7]),
        "=f"(__out_var[8]),
        "=f"(__out_var[9]),
        "=f"(__out_var[10]),
        "=f"(__out_var[11]),
        "=f"(__out_var[12]),
        "=f"(__out_var[13]),
        "=f"(__out_var[14]),
        "=f"(__out_var[15]),
        "=f"(__out_var[16]),
        "=f"(__out_var[17]),
        "=f"(__out_var[18]),
        "=f"(__out_var[19]),
        "=f"(__out_var[20]),
        "=f"(__out_var[21]),
        "=f"(__out_var[22]),
        "=f"(__out_var[23]),
        "=f"(__out_var[24]),
        "=f"(__out_var[25]),
        "=f"(__out_var[26]),
        "=f"(__out_var[27]),
        "=f"(__out_var[28]),
        "=f"(__out_var[29]),
        "=f"(__out_var[30]),
        "=f"(__out_var[31]),
        "=f"(__out_var[32]),
        "=f"(__out_var[33]),
        "=f"(__out_var[34]),
        "=f"(__out_var[35]),
        "=f"(__out_var[36]),
        "=f"(__out_var[37]),
        "=f"(__out_var[38]),
        "=f"(__out_var[39]),
        "=f"(__out_var[40]),
        "=f"(__out_var[41]),
        "=f"(__out_var[42]),
        "=f"(__out_var[43]),
        "=f"(__out_var[44]),
        "=f"(__out_var[45]),
        "=f"(__out_var[46]),
        "=f"(__out_var[47]),
        "=f"(__out_var[48]),
        "=f"(__out_var[49]),
        "=f"(__out_var[50]),
        "=f"(__out_var[51]),
        "=f"(__out_var[52]),
        "=f"(__out_var[53]),
        "=f"(__out_var[54]),
        "=f"(__out_var[55]),
        "=f"(__out_var[56]),
        "=f"(__out_var[57]),
        "=f"(__out_var[58]),
        "=f"(__out_var[59]),
        "=f"(__out_var[60]),
        "=f"(__out_var[61]),
        "=f"(__out_var[62]),
        "=f"(__out_var[63]),
        "=f"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x128.u32.op out_var, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline uint32_t tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  uint32_t (&out_var)[128],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::uint32_t tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  ::cuda::std::uint32_t (&__out_var)[128],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
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
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15]),
        "=r"(__out_var[16]),
        "=r"(__out_var[17]),
        "=r"(__out_var[18]),
        "=r"(__out_var[19]),
        "=r"(__out_var[20]),
        "=r"(__out_var[21]),
        "=r"(__out_var[22]),
        "=r"(__out_var[23]),
        "=r"(__out_var[24]),
        "=r"(__out_var[25]),
        "=r"(__out_var[26]),
        "=r"(__out_var[27]),
        "=r"(__out_var[28]),
        "=r"(__out_var[29]),
        "=r"(__out_var[30]),
        "=r"(__out_var[31]),
        "=r"(__out_var[32]),
        "=r"(__out_var[33]),
        "=r"(__out_var[34]),
        "=r"(__out_var[35]),
        "=r"(__out_var[36]),
        "=r"(__out_var[37]),
        "=r"(__out_var[38]),
        "=r"(__out_var[39]),
        "=r"(__out_var[40]),
        "=r"(__out_var[41]),
        "=r"(__out_var[42]),
        "=r"(__out_var[43]),
        "=r"(__out_var[44]),
        "=r"(__out_var[45]),
        "=r"(__out_var[46]),
        "=r"(__out_var[47]),
        "=r"(__out_var[48]),
        "=r"(__out_var[49]),
        "=r"(__out_var[50]),
        "=r"(__out_var[51]),
        "=r"(__out_var[52]),
        "=r"(__out_var[53]),
        "=r"(__out_var[54]),
        "=r"(__out_var[55]),
        "=r"(__out_var[56]),
        "=r"(__out_var[57]),
        "=r"(__out_var[58]),
        "=r"(__out_var[59]),
        "=r"(__out_var[60]),
        "=r"(__out_var[61]),
        "=r"(__out_var[62]),
        "=r"(__out_var[63]),
        "=r"(__out_var[64]),
        "=r"(__out_var[65]),
        "=r"(__out_var[66]),
        "=r"(__out_var[67]),
        "=r"(__out_var[68]),
        "=r"(__out_var[69]),
        "=r"(__out_var[70]),
        "=r"(__out_var[71]),
        "=r"(__out_var[72]),
        "=r"(__out_var[73]),
        "=r"(__out_var[74]),
        "=r"(__out_var[75]),
        "=r"(__out_var[76]),
        "=r"(__out_var[77]),
        "=r"(__out_var[78]),
        "=r"(__out_var[79]),
        "=r"(__out_var[80]),
        "=r"(__out_var[81]),
        "=r"(__out_var[82]),
        "=r"(__out_var[83]),
        "=r"(__out_var[84]),
        "=r"(__out_var[85]),
        "=r"(__out_var[86]),
        "=r"(__out_var[87]),
        "=r"(__out_var[88]),
        "=r"(__out_var[89]),
        "=r"(__out_var[90]),
        "=r"(__out_var[91]),
        "=r"(__out_var[92]),
        "=r"(__out_var[93]),
        "=r"(__out_var[94]),
        "=r"(__out_var[95]),
        "=r"(__out_var[96]),
        "=r"(__out_var[97]),
        "=r"(__out_var[98]),
        "=r"(__out_var[99]),
        "=r"(__out_var[100]),
        "=r"(__out_var[101]),
        "=r"(__out_var[102]),
        "=r"(__out_var[103]),
        "=r"(__out_var[104]),
        "=r"(__out_var[105]),
        "=r"(__out_var[106]),
        "=r"(__out_var[107]),
        "=r"(__out_var[108]),
        "=r"(__out_var[109]),
        "=r"(__out_var[110]),
        "=r"(__out_var[111]),
        "=r"(__out_var[112]),
        "=r"(__out_var[113]),
        "=r"(__out_var[114]),
        "=r"(__out_var[115]),
        "=r"(__out_var[116]),
        "=r"(__out_var[117]),
        "=r"(__out_var[118]),
        "=r"(__out_var[119]),
        "=r"(__out_var[120]),
        "=r"(__out_var[121]),
        "=r"(__out_var[122]),
        "=r"(__out_var[123]),
        "=r"(__out_var[124]),
        "=r"(__out_var[125]),
        "=r"(__out_var[126]),
        "=r"(__out_var[127]),
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
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15]),
        "=r"(__out_var[16]),
        "=r"(__out_var[17]),
        "=r"(__out_var[18]),
        "=r"(__out_var[19]),
        "=r"(__out_var[20]),
        "=r"(__out_var[21]),
        "=r"(__out_var[22]),
        "=r"(__out_var[23]),
        "=r"(__out_var[24]),
        "=r"(__out_var[25]),
        "=r"(__out_var[26]),
        "=r"(__out_var[27]),
        "=r"(__out_var[28]),
        "=r"(__out_var[29]),
        "=r"(__out_var[30]),
        "=r"(__out_var[31]),
        "=r"(__out_var[32]),
        "=r"(__out_var[33]),
        "=r"(__out_var[34]),
        "=r"(__out_var[35]),
        "=r"(__out_var[36]),
        "=r"(__out_var[37]),
        "=r"(__out_var[38]),
        "=r"(__out_var[39]),
        "=r"(__out_var[40]),
        "=r"(__out_var[41]),
        "=r"(__out_var[42]),
        "=r"(__out_var[43]),
        "=r"(__out_var[44]),
        "=r"(__out_var[45]),
        "=r"(__out_var[46]),
        "=r"(__out_var[47]),
        "=r"(__out_var[48]),
        "=r"(__out_var[49]),
        "=r"(__out_var[50]),
        "=r"(__out_var[51]),
        "=r"(__out_var[52]),
        "=r"(__out_var[53]),
        "=r"(__out_var[54]),
        "=r"(__out_var[55]),
        "=r"(__out_var[56]),
        "=r"(__out_var[57]),
        "=r"(__out_var[58]),
        "=r"(__out_var[59]),
        "=r"(__out_var[60]),
        "=r"(__out_var[61]),
        "=r"(__out_var[62]),
        "=r"(__out_var[63]),
        "=r"(__out_var[64]),
        "=r"(__out_var[65]),
        "=r"(__out_var[66]),
        "=r"(__out_var[67]),
        "=r"(__out_var[68]),
        "=r"(__out_var[69]),
        "=r"(__out_var[70]),
        "=r"(__out_var[71]),
        "=r"(__out_var[72]),
        "=r"(__out_var[73]),
        "=r"(__out_var[74]),
        "=r"(__out_var[75]),
        "=r"(__out_var[76]),
        "=r"(__out_var[77]),
        "=r"(__out_var[78]),
        "=r"(__out_var[79]),
        "=r"(__out_var[80]),
        "=r"(__out_var[81]),
        "=r"(__out_var[82]),
        "=r"(__out_var[83]),
        "=r"(__out_var[84]),
        "=r"(__out_var[85]),
        "=r"(__out_var[86]),
        "=r"(__out_var[87]),
        "=r"(__out_var[88]),
        "=r"(__out_var[89]),
        "=r"(__out_var[90]),
        "=r"(__out_var[91]),
        "=r"(__out_var[92]),
        "=r"(__out_var[93]),
        "=r"(__out_var[94]),
        "=r"(__out_var[95]),
        "=r"(__out_var[96]),
        "=r"(__out_var[97]),
        "=r"(__out_var[98]),
        "=r"(__out_var[99]),
        "=r"(__out_var[100]),
        "=r"(__out_var[101]),
        "=r"(__out_var[102]),
        "=r"(__out_var[103]),
        "=r"(__out_var[104]),
        "=r"(__out_var[105]),
        "=r"(__out_var[106]),
        "=r"(__out_var[107]),
        "=r"(__out_var[108]),
        "=r"(__out_var[109]),
        "=r"(__out_var[110]),
        "=r"(__out_var[111]),
        "=r"(__out_var[112]),
        "=r"(__out_var[113]),
        "=r"(__out_var[114]),
        "=r"(__out_var[115]),
        "=r"(__out_var[116]),
        "=r"(__out_var[117]),
        "=r"(__out_var[118]),
        "=r"(__out_var[119]),
        "=r"(__out_var[120]),
        "=r"(__out_var[121]),
        "=r"(__out_var[122]),
        "=r"(__out_var[123]),
        "=r"(__out_var[124]),
        "=r"(__out_var[125]),
        "=r"(__out_var[126]),
        "=r"(__out_var[127]),
        "=r"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x128.s32.op out_var, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline int32_t tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  int32_t (&out_var)[128],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline ::cuda::std::int32_t tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  ::cuda::std::int32_t (&__out_var)[128],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
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
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15]),
        "=r"(__out_var[16]),
        "=r"(__out_var[17]),
        "=r"(__out_var[18]),
        "=r"(__out_var[19]),
        "=r"(__out_var[20]),
        "=r"(__out_var[21]),
        "=r"(__out_var[22]),
        "=r"(__out_var[23]),
        "=r"(__out_var[24]),
        "=r"(__out_var[25]),
        "=r"(__out_var[26]),
        "=r"(__out_var[27]),
        "=r"(__out_var[28]),
        "=r"(__out_var[29]),
        "=r"(__out_var[30]),
        "=r"(__out_var[31]),
        "=r"(__out_var[32]),
        "=r"(__out_var[33]),
        "=r"(__out_var[34]),
        "=r"(__out_var[35]),
        "=r"(__out_var[36]),
        "=r"(__out_var[37]),
        "=r"(__out_var[38]),
        "=r"(__out_var[39]),
        "=r"(__out_var[40]),
        "=r"(__out_var[41]),
        "=r"(__out_var[42]),
        "=r"(__out_var[43]),
        "=r"(__out_var[44]),
        "=r"(__out_var[45]),
        "=r"(__out_var[46]),
        "=r"(__out_var[47]),
        "=r"(__out_var[48]),
        "=r"(__out_var[49]),
        "=r"(__out_var[50]),
        "=r"(__out_var[51]),
        "=r"(__out_var[52]),
        "=r"(__out_var[53]),
        "=r"(__out_var[54]),
        "=r"(__out_var[55]),
        "=r"(__out_var[56]),
        "=r"(__out_var[57]),
        "=r"(__out_var[58]),
        "=r"(__out_var[59]),
        "=r"(__out_var[60]),
        "=r"(__out_var[61]),
        "=r"(__out_var[62]),
        "=r"(__out_var[63]),
        "=r"(__out_var[64]),
        "=r"(__out_var[65]),
        "=r"(__out_var[66]),
        "=r"(__out_var[67]),
        "=r"(__out_var[68]),
        "=r"(__out_var[69]),
        "=r"(__out_var[70]),
        "=r"(__out_var[71]),
        "=r"(__out_var[72]),
        "=r"(__out_var[73]),
        "=r"(__out_var[74]),
        "=r"(__out_var[75]),
        "=r"(__out_var[76]),
        "=r"(__out_var[77]),
        "=r"(__out_var[78]),
        "=r"(__out_var[79]),
        "=r"(__out_var[80]),
        "=r"(__out_var[81]),
        "=r"(__out_var[82]),
        "=r"(__out_var[83]),
        "=r"(__out_var[84]),
        "=r"(__out_var[85]),
        "=r"(__out_var[86]),
        "=r"(__out_var[87]),
        "=r"(__out_var[88]),
        "=r"(__out_var[89]),
        "=r"(__out_var[90]),
        "=r"(__out_var[91]),
        "=r"(__out_var[92]),
        "=r"(__out_var[93]),
        "=r"(__out_var[94]),
        "=r"(__out_var[95]),
        "=r"(__out_var[96]),
        "=r"(__out_var[97]),
        "=r"(__out_var[98]),
        "=r"(__out_var[99]),
        "=r"(__out_var[100]),
        "=r"(__out_var[101]),
        "=r"(__out_var[102]),
        "=r"(__out_var[103]),
        "=r"(__out_var[104]),
        "=r"(__out_var[105]),
        "=r"(__out_var[106]),
        "=r"(__out_var[107]),
        "=r"(__out_var[108]),
        "=r"(__out_var[109]),
        "=r"(__out_var[110]),
        "=r"(__out_var[111]),
        "=r"(__out_var[112]),
        "=r"(__out_var[113]),
        "=r"(__out_var[114]),
        "=r"(__out_var[115]),
        "=r"(__out_var[116]),
        "=r"(__out_var[117]),
        "=r"(__out_var[118]),
        "=r"(__out_var[119]),
        "=r"(__out_var[120]),
        "=r"(__out_var[121]),
        "=r"(__out_var[122]),
        "=r"(__out_var[123]),
        "=r"(__out_var[124]),
        "=r"(__out_var[125]),
        "=r"(__out_var[126]),
        "=r"(__out_var[127]),
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
      : "=r"(__out_var[0]),
        "=r"(__out_var[1]),
        "=r"(__out_var[2]),
        "=r"(__out_var[3]),
        "=r"(__out_var[4]),
        "=r"(__out_var[5]),
        "=r"(__out_var[6]),
        "=r"(__out_var[7]),
        "=r"(__out_var[8]),
        "=r"(__out_var[9]),
        "=r"(__out_var[10]),
        "=r"(__out_var[11]),
        "=r"(__out_var[12]),
        "=r"(__out_var[13]),
        "=r"(__out_var[14]),
        "=r"(__out_var[15]),
        "=r"(__out_var[16]),
        "=r"(__out_var[17]),
        "=r"(__out_var[18]),
        "=r"(__out_var[19]),
        "=r"(__out_var[20]),
        "=r"(__out_var[21]),
        "=r"(__out_var[22]),
        "=r"(__out_var[23]),
        "=r"(__out_var[24]),
        "=r"(__out_var[25]),
        "=r"(__out_var[26]),
        "=r"(__out_var[27]),
        "=r"(__out_var[28]),
        "=r"(__out_var[29]),
        "=r"(__out_var[30]),
        "=r"(__out_var[31]),
        "=r"(__out_var[32]),
        "=r"(__out_var[33]),
        "=r"(__out_var[34]),
        "=r"(__out_var[35]),
        "=r"(__out_var[36]),
        "=r"(__out_var[37]),
        "=r"(__out_var[38]),
        "=r"(__out_var[39]),
        "=r"(__out_var[40]),
        "=r"(__out_var[41]),
        "=r"(__out_var[42]),
        "=r"(__out_var[43]),
        "=r"(__out_var[44]),
        "=r"(__out_var[45]),
        "=r"(__out_var[46]),
        "=r"(__out_var[47]),
        "=r"(__out_var[48]),
        "=r"(__out_var[49]),
        "=r"(__out_var[50]),
        "=r"(__out_var[51]),
        "=r"(__out_var[52]),
        "=r"(__out_var[53]),
        "=r"(__out_var[54]),
        "=r"(__out_var[55]),
        "=r"(__out_var[56]),
        "=r"(__out_var[57]),
        "=r"(__out_var[58]),
        "=r"(__out_var[59]),
        "=r"(__out_var[60]),
        "=r"(__out_var[61]),
        "=r"(__out_var[62]),
        "=r"(__out_var[63]),
        "=r"(__out_var[64]),
        "=r"(__out_var[65]),
        "=r"(__out_var[66]),
        "=r"(__out_var[67]),
        "=r"(__out_var[68]),
        "=r"(__out_var[69]),
        "=r"(__out_var[70]),
        "=r"(__out_var[71]),
        "=r"(__out_var[72]),
        "=r"(__out_var[73]),
        "=r"(__out_var[74]),
        "=r"(__out_var[75]),
        "=r"(__out_var[76]),
        "=r"(__out_var[77]),
        "=r"(__out_var[78]),
        "=r"(__out_var[79]),
        "=r"(__out_var[80]),
        "=r"(__out_var[81]),
        "=r"(__out_var[82]),
        "=r"(__out_var[83]),
        "=r"(__out_var[84]),
        "=r"(__out_var[85]),
        "=r"(__out_var[86]),
        "=r"(__out_var[87]),
        "=r"(__out_var[88]),
        "=r"(__out_var[89]),
        "=r"(__out_var[90]),
        "=r"(__out_var[91]),
        "=r"(__out_var[92]),
        "=r"(__out_var[93]),
        "=r"(__out_var[94]),
        "=r"(__out_var[95]),
        "=r"(__out_var[96]),
        "=r"(__out_var[97]),
        "=r"(__out_var[98]),
        "=r"(__out_var[99]),
        "=r"(__out_var[100]),
        "=r"(__out_var[101]),
        "=r"(__out_var[102]),
        "=r"(__out_var[103]),
        "=r"(__out_var[104]),
        "=r"(__out_var[105]),
        "=r"(__out_var[106]),
        "=r"(__out_var[107]),
        "=r"(__out_var[108]),
        "=r"(__out_var[109]),
        "=r"(__out_var[110]),
        "=r"(__out_var[111]),
        "=r"(__out_var[112]),
        "=r"(__out_var[113]),
        "=r"(__out_var[114]),
        "=r"(__out_var[115]),
        "=r"(__out_var[116]),
        "=r"(__out_var[117]),
        "=r"(__out_var[118]),
        "=r"(__out_var[119]),
        "=r"(__out_var[120]),
        "=r"(__out_var[121]),
        "=r"(__out_var[122]),
        "=r"(__out_var[123]),
        "=r"(__out_var[124]),
        "=r"(__out_var[125]),
        "=r"(__out_var[126]),
        "=r"(__out_var[127]),
        "=r"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x128.f32.op.abs out_var, redval, [taddr], immHalfSplitoff; // PTX ISA 88,
SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_16x32bx2_abs(
  cuda::ptx::op_t<Op> op,
  float (&out_var)[128],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_16x32bx2_abs(
  ::cuda::ptx::op_t<_Op> __op,
  float (&__out_var)[128],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
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
      : "=f"(__out_var[0]),
        "=f"(__out_var[1]),
        "=f"(__out_var[2]),
        "=f"(__out_var[3]),
        "=f"(__out_var[4]),
        "=f"(__out_var[5]),
        "=f"(__out_var[6]),
        "=f"(__out_var[7]),
        "=f"(__out_var[8]),
        "=f"(__out_var[9]),
        "=f"(__out_var[10]),
        "=f"(__out_var[11]),
        "=f"(__out_var[12]),
        "=f"(__out_var[13]),
        "=f"(__out_var[14]),
        "=f"(__out_var[15]),
        "=f"(__out_var[16]),
        "=f"(__out_var[17]),
        "=f"(__out_var[18]),
        "=f"(__out_var[19]),
        "=f"(__out_var[20]),
        "=f"(__out_var[21]),
        "=f"(__out_var[22]),
        "=f"(__out_var[23]),
        "=f"(__out_var[24]),
        "=f"(__out_var[25]),
        "=f"(__out_var[26]),
        "=f"(__out_var[27]),
        "=f"(__out_var[28]),
        "=f"(__out_var[29]),
        "=f"(__out_var[30]),
        "=f"(__out_var[31]),
        "=f"(__out_var[32]),
        "=f"(__out_var[33]),
        "=f"(__out_var[34]),
        "=f"(__out_var[35]),
        "=f"(__out_var[36]),
        "=f"(__out_var[37]),
        "=f"(__out_var[38]),
        "=f"(__out_var[39]),
        "=f"(__out_var[40]),
        "=f"(__out_var[41]),
        "=f"(__out_var[42]),
        "=f"(__out_var[43]),
        "=f"(__out_var[44]),
        "=f"(__out_var[45]),
        "=f"(__out_var[46]),
        "=f"(__out_var[47]),
        "=f"(__out_var[48]),
        "=f"(__out_var[49]),
        "=f"(__out_var[50]),
        "=f"(__out_var[51]),
        "=f"(__out_var[52]),
        "=f"(__out_var[53]),
        "=f"(__out_var[54]),
        "=f"(__out_var[55]),
        "=f"(__out_var[56]),
        "=f"(__out_var[57]),
        "=f"(__out_var[58]),
        "=f"(__out_var[59]),
        "=f"(__out_var[60]),
        "=f"(__out_var[61]),
        "=f"(__out_var[62]),
        "=f"(__out_var[63]),
        "=f"(__out_var[64]),
        "=f"(__out_var[65]),
        "=f"(__out_var[66]),
        "=f"(__out_var[67]),
        "=f"(__out_var[68]),
        "=f"(__out_var[69]),
        "=f"(__out_var[70]),
        "=f"(__out_var[71]),
        "=f"(__out_var[72]),
        "=f"(__out_var[73]),
        "=f"(__out_var[74]),
        "=f"(__out_var[75]),
        "=f"(__out_var[76]),
        "=f"(__out_var[77]),
        "=f"(__out_var[78]),
        "=f"(__out_var[79]),
        "=f"(__out_var[80]),
        "=f"(__out_var[81]),
        "=f"(__out_var[82]),
        "=f"(__out_var[83]),
        "=f"(__out_var[84]),
        "=f"(__out_var[85]),
        "=f"(__out_var[86]),
        "=f"(__out_var[87]),
        "=f"(__out_var[88]),
        "=f"(__out_var[89]),
        "=f"(__out_var[90]),
        "=f"(__out_var[91]),
        "=f"(__out_var[92]),
        "=f"(__out_var[93]),
        "=f"(__out_var[94]),
        "=f"(__out_var[95]),
        "=f"(__out_var[96]),
        "=f"(__out_var[97]),
        "=f"(__out_var[98]),
        "=f"(__out_var[99]),
        "=f"(__out_var[100]),
        "=f"(__out_var[101]),
        "=f"(__out_var[102]),
        "=f"(__out_var[103]),
        "=f"(__out_var[104]),
        "=f"(__out_var[105]),
        "=f"(__out_var[106]),
        "=f"(__out_var[107]),
        "=f"(__out_var[108]),
        "=f"(__out_var[109]),
        "=f"(__out_var[110]),
        "=f"(__out_var[111]),
        "=f"(__out_var[112]),
        "=f"(__out_var[113]),
        "=f"(__out_var[114]),
        "=f"(__out_var[115]),
        "=f"(__out_var[116]),
        "=f"(__out_var[117]),
        "=f"(__out_var[118]),
        "=f"(__out_var[119]),
        "=f"(__out_var[120]),
        "=f"(__out_var[121]),
        "=f"(__out_var[122]),
        "=f"(__out_var[123]),
        "=f"(__out_var[124]),
        "=f"(__out_var[125]),
        "=f"(__out_var[126]),
        "=f"(__out_var[127]),
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
      : "=f"(__out_var[0]),
        "=f"(__out_var[1]),
        "=f"(__out_var[2]),
        "=f"(__out_var[3]),
        "=f"(__out_var[4]),
        "=f"(__out_var[5]),
        "=f"(__out_var[6]),
        "=f"(__out_var[7]),
        "=f"(__out_var[8]),
        "=f"(__out_var[9]),
        "=f"(__out_var[10]),
        "=f"(__out_var[11]),
        "=f"(__out_var[12]),
        "=f"(__out_var[13]),
        "=f"(__out_var[14]),
        "=f"(__out_var[15]),
        "=f"(__out_var[16]),
        "=f"(__out_var[17]),
        "=f"(__out_var[18]),
        "=f"(__out_var[19]),
        "=f"(__out_var[20]),
        "=f"(__out_var[21]),
        "=f"(__out_var[22]),
        "=f"(__out_var[23]),
        "=f"(__out_var[24]),
        "=f"(__out_var[25]),
        "=f"(__out_var[26]),
        "=f"(__out_var[27]),
        "=f"(__out_var[28]),
        "=f"(__out_var[29]),
        "=f"(__out_var[30]),
        "=f"(__out_var[31]),
        "=f"(__out_var[32]),
        "=f"(__out_var[33]),
        "=f"(__out_var[34]),
        "=f"(__out_var[35]),
        "=f"(__out_var[36]),
        "=f"(__out_var[37]),
        "=f"(__out_var[38]),
        "=f"(__out_var[39]),
        "=f"(__out_var[40]),
        "=f"(__out_var[41]),
        "=f"(__out_var[42]),
        "=f"(__out_var[43]),
        "=f"(__out_var[44]),
        "=f"(__out_var[45]),
        "=f"(__out_var[46]),
        "=f"(__out_var[47]),
        "=f"(__out_var[48]),
        "=f"(__out_var[49]),
        "=f"(__out_var[50]),
        "=f"(__out_var[51]),
        "=f"(__out_var[52]),
        "=f"(__out_var[53]),
        "=f"(__out_var[54]),
        "=f"(__out_var[55]),
        "=f"(__out_var[56]),
        "=f"(__out_var[57]),
        "=f"(__out_var[58]),
        "=f"(__out_var[59]),
        "=f"(__out_var[60]),
        "=f"(__out_var[61]),
        "=f"(__out_var[62]),
        "=f"(__out_var[63]),
        "=f"(__out_var[64]),
        "=f"(__out_var[65]),
        "=f"(__out_var[66]),
        "=f"(__out_var[67]),
        "=f"(__out_var[68]),
        "=f"(__out_var[69]),
        "=f"(__out_var[70]),
        "=f"(__out_var[71]),
        "=f"(__out_var[72]),
        "=f"(__out_var[73]),
        "=f"(__out_var[74]),
        "=f"(__out_var[75]),
        "=f"(__out_var[76]),
        "=f"(__out_var[77]),
        "=f"(__out_var[78]),
        "=f"(__out_var[79]),
        "=f"(__out_var[80]),
        "=f"(__out_var[81]),
        "=f"(__out_var[82]),
        "=f"(__out_var[83]),
        "=f"(__out_var[84]),
        "=f"(__out_var[85]),
        "=f"(__out_var[86]),
        "=f"(__out_var[87]),
        "=f"(__out_var[88]),
        "=f"(__out_var[89]),
        "=f"(__out_var[90]),
        "=f"(__out_var[91]),
        "=f"(__out_var[92]),
        "=f"(__out_var[93]),
        "=f"(__out_var[94]),
        "=f"(__out_var[95]),
        "=f"(__out_var[96]),
        "=f"(__out_var[97]),
        "=f"(__out_var[98]),
        "=f"(__out_var[99]),
        "=f"(__out_var[100]),
        "=f"(__out_var[101]),
        "=f"(__out_var[102]),
        "=f"(__out_var[103]),
        "=f"(__out_var[104]),
        "=f"(__out_var[105]),
        "=f"(__out_var[106]),
        "=f"(__out_var[107]),
        "=f"(__out_var[108]),
        "=f"(__out_var[109]),
        "=f"(__out_var[110]),
        "=f"(__out_var[111]),
        "=f"(__out_var[112]),
        "=f"(__out_var[113]),
        "=f"(__out_var[114]),
        "=f"(__out_var[115]),
        "=f"(__out_var[116]),
        "=f"(__out_var[117]),
        "=f"(__out_var[118]),
        "=f"(__out_var[119]),
        "=f"(__out_var[120]),
        "=f"(__out_var[121]),
        "=f"(__out_var[122]),
        "=f"(__out_var[123]),
        "=f"(__out_var[124]),
        "=f"(__out_var[125]),
        "=f"(__out_var[126]),
        "=f"(__out_var[127]),
        "=f"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.sync.aligned.16x32bx2.x128.f32.op out_var, redval, [taddr], immHalfSplitoff; // PTX ISA 88, SM_103a,
SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .op        = { .min, .max }
template <int N32, cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_16x32bx2(
  cuda::ptx::op_t<Op> op,
  float (&out_var)[128],
  uint32_t taddr,
  cuda::ptx::n32_t<N32> immHalfSplitoff);
*/
#if __cccl_ptx_isa >= 880
template <int _N32, ::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_16x32bx2(
  ::cuda::ptx::op_t<_Op> __op,
  float (&__out_var)[128],
  ::cuda::std::uint32_t __taddr,
  ::cuda::ptx::n32_t<_N32> __immHalfSplitoff)
{
  static_assert(__op == op_min || __op == op_max, "");
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
      : "=f"(__out_var[0]),
        "=f"(__out_var[1]),
        "=f"(__out_var[2]),
        "=f"(__out_var[3]),
        "=f"(__out_var[4]),
        "=f"(__out_var[5]),
        "=f"(__out_var[6]),
        "=f"(__out_var[7]),
        "=f"(__out_var[8]),
        "=f"(__out_var[9]),
        "=f"(__out_var[10]),
        "=f"(__out_var[11]),
        "=f"(__out_var[12]),
        "=f"(__out_var[13]),
        "=f"(__out_var[14]),
        "=f"(__out_var[15]),
        "=f"(__out_var[16]),
        "=f"(__out_var[17]),
        "=f"(__out_var[18]),
        "=f"(__out_var[19]),
        "=f"(__out_var[20]),
        "=f"(__out_var[21]),
        "=f"(__out_var[22]),
        "=f"(__out_var[23]),
        "=f"(__out_var[24]),
        "=f"(__out_var[25]),
        "=f"(__out_var[26]),
        "=f"(__out_var[27]),
        "=f"(__out_var[28]),
        "=f"(__out_var[29]),
        "=f"(__out_var[30]),
        "=f"(__out_var[31]),
        "=f"(__out_var[32]),
        "=f"(__out_var[33]),
        "=f"(__out_var[34]),
        "=f"(__out_var[35]),
        "=f"(__out_var[36]),
        "=f"(__out_var[37]),
        "=f"(__out_var[38]),
        "=f"(__out_var[39]),
        "=f"(__out_var[40]),
        "=f"(__out_var[41]),
        "=f"(__out_var[42]),
        "=f"(__out_var[43]),
        "=f"(__out_var[44]),
        "=f"(__out_var[45]),
        "=f"(__out_var[46]),
        "=f"(__out_var[47]),
        "=f"(__out_var[48]),
        "=f"(__out_var[49]),
        "=f"(__out_var[50]),
        "=f"(__out_var[51]),
        "=f"(__out_var[52]),
        "=f"(__out_var[53]),
        "=f"(__out_var[54]),
        "=f"(__out_var[55]),
        "=f"(__out_var[56]),
        "=f"(__out_var[57]),
        "=f"(__out_var[58]),
        "=f"(__out_var[59]),
        "=f"(__out_var[60]),
        "=f"(__out_var[61]),
        "=f"(__out_var[62]),
        "=f"(__out_var[63]),
        "=f"(__out_var[64]),
        "=f"(__out_var[65]),
        "=f"(__out_var[66]),
        "=f"(__out_var[67]),
        "=f"(__out_var[68]),
        "=f"(__out_var[69]),
        "=f"(__out_var[70]),
        "=f"(__out_var[71]),
        "=f"(__out_var[72]),
        "=f"(__out_var[73]),
        "=f"(__out_var[74]),
        "=f"(__out_var[75]),
        "=f"(__out_var[76]),
        "=f"(__out_var[77]),
        "=f"(__out_var[78]),
        "=f"(__out_var[79]),
        "=f"(__out_var[80]),
        "=f"(__out_var[81]),
        "=f"(__out_var[82]),
        "=f"(__out_var[83]),
        "=f"(__out_var[84]),
        "=f"(__out_var[85]),
        "=f"(__out_var[86]),
        "=f"(__out_var[87]),
        "=f"(__out_var[88]),
        "=f"(__out_var[89]),
        "=f"(__out_var[90]),
        "=f"(__out_var[91]),
        "=f"(__out_var[92]),
        "=f"(__out_var[93]),
        "=f"(__out_var[94]),
        "=f"(__out_var[95]),
        "=f"(__out_var[96]),
        "=f"(__out_var[97]),
        "=f"(__out_var[98]),
        "=f"(__out_var[99]),
        "=f"(__out_var[100]),
        "=f"(__out_var[101]),
        "=f"(__out_var[102]),
        "=f"(__out_var[103]),
        "=f"(__out_var[104]),
        "=f"(__out_var[105]),
        "=f"(__out_var[106]),
        "=f"(__out_var[107]),
        "=f"(__out_var[108]),
        "=f"(__out_var[109]),
        "=f"(__out_var[110]),
        "=f"(__out_var[111]),
        "=f"(__out_var[112]),
        "=f"(__out_var[113]),
        "=f"(__out_var[114]),
        "=f"(__out_var[115]),
        "=f"(__out_var[116]),
        "=f"(__out_var[117]),
        "=f"(__out_var[118]),
        "=f"(__out_var[119]),
        "=f"(__out_var[120]),
        "=f"(__out_var[121]),
        "=f"(__out_var[122]),
        "=f"(__out_var[123]),
        "=f"(__out_var[124]),
        "=f"(__out_var[125]),
        "=f"(__out_var[126]),
        "=f"(__out_var[127]),
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
      : "=f"(__out_var[0]),
        "=f"(__out_var[1]),
        "=f"(__out_var[2]),
        "=f"(__out_var[3]),
        "=f"(__out_var[4]),
        "=f"(__out_var[5]),
        "=f"(__out_var[6]),
        "=f"(__out_var[7]),
        "=f"(__out_var[8]),
        "=f"(__out_var[9]),
        "=f"(__out_var[10]),
        "=f"(__out_var[11]),
        "=f"(__out_var[12]),
        "=f"(__out_var[13]),
        "=f"(__out_var[14]),
        "=f"(__out_var[15]),
        "=f"(__out_var[16]),
        "=f"(__out_var[17]),
        "=f"(__out_var[18]),
        "=f"(__out_var[19]),
        "=f"(__out_var[20]),
        "=f"(__out_var[21]),
        "=f"(__out_var[22]),
        "=f"(__out_var[23]),
        "=f"(__out_var[24]),
        "=f"(__out_var[25]),
        "=f"(__out_var[26]),
        "=f"(__out_var[27]),
        "=f"(__out_var[28]),
        "=f"(__out_var[29]),
        "=f"(__out_var[30]),
        "=f"(__out_var[31]),
        "=f"(__out_var[32]),
        "=f"(__out_var[33]),
        "=f"(__out_var[34]),
        "=f"(__out_var[35]),
        "=f"(__out_var[36]),
        "=f"(__out_var[37]),
        "=f"(__out_var[38]),
        "=f"(__out_var[39]),
        "=f"(__out_var[40]),
        "=f"(__out_var[41]),
        "=f"(__out_var[42]),
        "=f"(__out_var[43]),
        "=f"(__out_var[44]),
        "=f"(__out_var[45]),
        "=f"(__out_var[46]),
        "=f"(__out_var[47]),
        "=f"(__out_var[48]),
        "=f"(__out_var[49]),
        "=f"(__out_var[50]),
        "=f"(__out_var[51]),
        "=f"(__out_var[52]),
        "=f"(__out_var[53]),
        "=f"(__out_var[54]),
        "=f"(__out_var[55]),
        "=f"(__out_var[56]),
        "=f"(__out_var[57]),
        "=f"(__out_var[58]),
        "=f"(__out_var[59]),
        "=f"(__out_var[60]),
        "=f"(__out_var[61]),
        "=f"(__out_var[62]),
        "=f"(__out_var[63]),
        "=f"(__out_var[64]),
        "=f"(__out_var[65]),
        "=f"(__out_var[66]),
        "=f"(__out_var[67]),
        "=f"(__out_var[68]),
        "=f"(__out_var[69]),
        "=f"(__out_var[70]),
        "=f"(__out_var[71]),
        "=f"(__out_var[72]),
        "=f"(__out_var[73]),
        "=f"(__out_var[74]),
        "=f"(__out_var[75]),
        "=f"(__out_var[76]),
        "=f"(__out_var[77]),
        "=f"(__out_var[78]),
        "=f"(__out_var[79]),
        "=f"(__out_var[80]),
        "=f"(__out_var[81]),
        "=f"(__out_var[82]),
        "=f"(__out_var[83]),
        "=f"(__out_var[84]),
        "=f"(__out_var[85]),
        "=f"(__out_var[86]),
        "=f"(__out_var[87]),
        "=f"(__out_var[88]),
        "=f"(__out_var[89]),
        "=f"(__out_var[90]),
        "=f"(__out_var[91]),
        "=f"(__out_var[92]),
        "=f"(__out_var[93]),
        "=f"(__out_var[94]),
        "=f"(__out_var[95]),
        "=f"(__out_var[96]),
        "=f"(__out_var[97]),
        "=f"(__out_var[98]),
        "=f"(__out_var[99]),
        "=f"(__out_var[100]),
        "=f"(__out_var[101]),
        "=f"(__out_var[102]),
        "=f"(__out_var[103]),
        "=f"(__out_var[104]),
        "=f"(__out_var[105]),
        "=f"(__out_var[106]),
        "=f"(__out_var[107]),
        "=f"(__out_var[108]),
        "=f"(__out_var[109]),
        "=f"(__out_var[110]),
        "=f"(__out_var[111]),
        "=f"(__out_var[112]),
        "=f"(__out_var[113]),
        "=f"(__out_var[114]),
        "=f"(__out_var[115]),
        "=f"(__out_var[116]),
        "=f"(__out_var[117]),
        "=f"(__out_var[118]),
        "=f"(__out_var[119]),
        "=f"(__out_var[120]),
        "=f"(__out_var[121]),
        "=f"(__out_var[122]),
        "=f"(__out_var[123]),
        "=f"(__out_var[124]),
        "=f"(__out_var[125]),
        "=f"(__out_var[126]),
        "=f"(__out_var[127]),
        "=f"(__redval)
      : "r"(__taddr), "n"(__immHalfSplitoff.value)
      : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 880

/*
// tcgen05.ld.red.spcompress.sync.aligned.32x32b.x4.op.sp::2:4.abs.f32.b2 mdata, cdata, redval, [taddr]; // PTX ISA 94,
SM_107a
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_spcompress_32x32b_abs(
  cuda::ptx::op_t<Op> op,
  uint32_t (&mdata)[1],
  float (&cdata)[2],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_spcompress_32x32b_abs(
  ::cuda::ptx::op_t<_Op> __op, ::cuda::std::uint32_t (&__mdata)[1], float (&__cdata)[2], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.spcompress.sync.aligned.32x32b.x4.min.sp::2:4.abs.f32.b2 {%1}, {%2, %3}, %0, [%4];"
        : "=f"(__redval), "=r"(__mdata[0]), "=f"(__cdata[0]), "=f"(__cdata[1])
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.spcompress.sync.aligned.32x32b.x4.max.sp::2:4.abs.f32.b2 {%1}, {%2, %3}, %0, [%4];"
        : "=f"(__redval), "=r"(__mdata[0]), "=f"(__cdata[0]), "=f"(__cdata[1])
        : "r"(__taddr)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 940

/*
// tcgen05.ld.red.spcompress.sync.aligned.32x32b.x4.op.sp::2:4.f32.b2 mdata, cdata, redval, [taddr]; // PTX ISA 94,
SM_107a
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_spcompress_32x32b(
  cuda::ptx::op_t<Op> op,
  uint32_t (&mdata)[1],
  float (&cdata)[2],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_spcompress_32x32b(
  ::cuda::ptx::op_t<_Op> __op, ::cuda::std::uint32_t (&__mdata)[1], float (&__cdata)[2], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.spcompress.sync.aligned.32x32b.x4.min.sp::2:4.f32.b2 {%1}, {%2, %3}, %0, [%4];"
        : "=f"(__redval), "=r"(__mdata[0]), "=f"(__cdata[0]), "=f"(__cdata[1])
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.spcompress.sync.aligned.32x32b.x4.max.sp::2:4.f32.b2 {%1}, {%2, %3}, %0, [%4];"
        : "=f"(__redval), "=r"(__mdata[0]), "=f"(__cdata[0]), "=f"(__cdata[1])
        : "r"(__taddr)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 940

/*
// tcgen05.ld.red.spcompress.sync.aligned.32x32b.x8.op.sp::2:4.abs.f32.b2 mdata, cdata, redval, [taddr]; // PTX ISA 94,
SM_107a
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_spcompress_32x32b_abs(
  cuda::ptx::op_t<Op> op,
  uint32_t (&mdata)[1],
  float (&cdata)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_spcompress_32x32b_abs(
  ::cuda::ptx::op_t<_Op> __op, ::cuda::std::uint32_t (&__mdata)[1], float (&__cdata)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.spcompress.sync.aligned.32x32b.x8.min.sp::2:4.abs.f32.b2 {%1}, {%2, %3, %4, %5}, %0, [%6];"
        : "=f"(__redval), "=r"(__mdata[0]), "=f"(__cdata[0]), "=f"(__cdata[1]), "=f"(__cdata[2]), "=f"(__cdata[3])
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.spcompress.sync.aligned.32x32b.x8.max.sp::2:4.abs.f32.b2 {%1}, {%2, %3, %4, %5}, %0, [%6];"
        : "=f"(__redval), "=r"(__mdata[0]), "=f"(__cdata[0]), "=f"(__cdata[1]), "=f"(__cdata[2]), "=f"(__cdata[3])
        : "r"(__taddr)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 940

/*
// tcgen05.ld.red.spcompress.sync.aligned.32x32b.x8.op.sp::2:4.f32.b2 mdata, cdata, redval, [taddr]; // PTX ISA 94,
SM_107a
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_spcompress_32x32b(
  cuda::ptx::op_t<Op> op,
  uint32_t (&mdata)[1],
  float (&cdata)[4],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_spcompress_32x32b(
  ::cuda::ptx::op_t<_Op> __op, ::cuda::std::uint32_t (&__mdata)[1], float (&__cdata)[4], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.spcompress.sync.aligned.32x32b.x8.min.sp::2:4.f32.b2 {%1}, {%2, %3, %4, %5}, %0, [%6];"
        : "=f"(__redval), "=r"(__mdata[0]), "=f"(__cdata[0]), "=f"(__cdata[1]), "=f"(__cdata[2]), "=f"(__cdata[3])
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.spcompress.sync.aligned.32x32b.x8.max.sp::2:4.f32.b2 {%1}, {%2, %3, %4, %5}, %0, [%6];"
        : "=f"(__redval), "=r"(__mdata[0]), "=f"(__cdata[0]), "=f"(__cdata[1]), "=f"(__cdata[2]), "=f"(__cdata[3])
        : "r"(__taddr)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 940

/*
// tcgen05.ld.red.spcompress.sync.aligned.32x32b.x16.op.sp::2:4.abs.f32.b2 mdata, cdata, redval, [taddr]; // PTX ISA 94,
SM_107a
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_spcompress_32x32b_abs(
  cuda::ptx::op_t<Op> op,
  uint32_t (&mdata)[1],
  float (&cdata)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_spcompress_32x32b_abs(
  ::cuda::ptx::op_t<_Op> __op, ::cuda::std::uint32_t (&__mdata)[1], float (&__cdata)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.spcompress.sync.aligned.32x32b.x16.min.sp::2:4.abs.f32.b2 {%1}, {%2, %3, %4, %5, %6, %7, %8, "
        "%9}, %0, [%10];"
        : "=f"(__redval),
          "=r"(__mdata[0]),
          "=f"(__cdata[0]),
          "=f"(__cdata[1]),
          "=f"(__cdata[2]),
          "=f"(__cdata[3]),
          "=f"(__cdata[4]),
          "=f"(__cdata[5]),
          "=f"(__cdata[6]),
          "=f"(__cdata[7])
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.spcompress.sync.aligned.32x32b.x16.max.sp::2:4.abs.f32.b2 {%1}, {%2, %3, %4, %5, %6, %7, %8, "
        "%9}, %0, [%10];"
        : "=f"(__redval),
          "=r"(__mdata[0]),
          "=f"(__cdata[0]),
          "=f"(__cdata[1]),
          "=f"(__cdata[2]),
          "=f"(__cdata[3]),
          "=f"(__cdata[4]),
          "=f"(__cdata[5]),
          "=f"(__cdata[6]),
          "=f"(__cdata[7])
        : "r"(__taddr)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 940

/*
// tcgen05.ld.red.spcompress.sync.aligned.32x32b.x16.op.sp::2:4.f32.b2 mdata, cdata, redval, [taddr]; // PTX ISA 94,
SM_107a
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_spcompress_32x32b(
  cuda::ptx::op_t<Op> op,
  uint32_t (&mdata)[1],
  float (&cdata)[8],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_spcompress_32x32b(
  ::cuda::ptx::op_t<_Op> __op, ::cuda::std::uint32_t (&__mdata)[1], float (&__cdata)[8], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.spcompress.sync.aligned.32x32b.x16.min.sp::2:4.f32.b2 {%1}, {%2, %3, %4, %5, %6, %7, %8, %9}, "
        "%0, [%10];"
        : "=f"(__redval),
          "=r"(__mdata[0]),
          "=f"(__cdata[0]),
          "=f"(__cdata[1]),
          "=f"(__cdata[2]),
          "=f"(__cdata[3]),
          "=f"(__cdata[4]),
          "=f"(__cdata[5]),
          "=f"(__cdata[6]),
          "=f"(__cdata[7])
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.spcompress.sync.aligned.32x32b.x16.max.sp::2:4.f32.b2 {%1}, {%2, %3, %4, %5, %6, %7, %8, %9}, "
        "%0, [%10];"
        : "=f"(__redval),
          "=r"(__mdata[0]),
          "=f"(__cdata[0]),
          "=f"(__cdata[1]),
          "=f"(__cdata[2]),
          "=f"(__cdata[3]),
          "=f"(__cdata[4]),
          "=f"(__cdata[5]),
          "=f"(__cdata[6]),
          "=f"(__cdata[7])
        : "r"(__taddr)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 940

/*
// tcgen05.ld.red.spcompress.sync.aligned.32x32b.x32.op.sp::2:4.abs.f32.b2 mdata, cdata, redval, [taddr]; // PTX ISA 94,
SM_107a
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_spcompress_32x32b_abs(
  cuda::ptx::op_t<Op> op,
  uint32_t (&mdata)[1],
  float (&cdata)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_spcompress_32x32b_abs(
  ::cuda::ptx::op_t<_Op> __op, ::cuda::std::uint32_t (&__mdata)[1], float (&__cdata)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.spcompress.sync.aligned.32x32b.x32.min.sp::2:4.abs.f32.b2 {%1}, {%2, %3, %4, %5, %6, %7, %8, "
        "%9, %10, %11, %12, %13, %14, %15, %16, %17}, %0, [%18];"
        : "=f"(__redval),
          "=r"(__mdata[0]),
          "=f"(__cdata[0]),
          "=f"(__cdata[1]),
          "=f"(__cdata[2]),
          "=f"(__cdata[3]),
          "=f"(__cdata[4]),
          "=f"(__cdata[5]),
          "=f"(__cdata[6]),
          "=f"(__cdata[7]),
          "=f"(__cdata[8]),
          "=f"(__cdata[9]),
          "=f"(__cdata[10]),
          "=f"(__cdata[11]),
          "=f"(__cdata[12]),
          "=f"(__cdata[13]),
          "=f"(__cdata[14]),
          "=f"(__cdata[15])
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.spcompress.sync.aligned.32x32b.x32.max.sp::2:4.abs.f32.b2 {%1}, {%2, %3, %4, %5, %6, %7, %8, "
        "%9, %10, %11, %12, %13, %14, %15, %16, %17}, %0, [%18];"
        : "=f"(__redval),
          "=r"(__mdata[0]),
          "=f"(__cdata[0]),
          "=f"(__cdata[1]),
          "=f"(__cdata[2]),
          "=f"(__cdata[3]),
          "=f"(__cdata[4]),
          "=f"(__cdata[5]),
          "=f"(__cdata[6]),
          "=f"(__cdata[7]),
          "=f"(__cdata[8]),
          "=f"(__cdata[9]),
          "=f"(__cdata[10]),
          "=f"(__cdata[11]),
          "=f"(__cdata[12]),
          "=f"(__cdata[13]),
          "=f"(__cdata[14]),
          "=f"(__cdata[15])
        : "r"(__taddr)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 940

/*
// tcgen05.ld.red.spcompress.sync.aligned.32x32b.x32.op.sp::2:4.f32.b2 mdata, cdata, redval, [taddr]; // PTX ISA 94,
SM_107a
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_spcompress_32x32b(
  cuda::ptx::op_t<Op> op,
  uint32_t (&mdata)[1],
  float (&cdata)[16],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_spcompress_32x32b(
  ::cuda::ptx::op_t<_Op> __op, ::cuda::std::uint32_t (&__mdata)[1], float (&__cdata)[16], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm("tcgen05.ld.red.spcompress.sync.aligned.32x32b.x32.min.sp::2:4.f32.b2 {%1}, {%2, %3, %4, %5, %6, %7, %8, %9, "
        "%10, %11, %12, %13, %14, %15, %16, %17}, %0, [%18];"
        : "=f"(__redval),
          "=r"(__mdata[0]),
          "=f"(__cdata[0]),
          "=f"(__cdata[1]),
          "=f"(__cdata[2]),
          "=f"(__cdata[3]),
          "=f"(__cdata[4]),
          "=f"(__cdata[5]),
          "=f"(__cdata[6]),
          "=f"(__cdata[7]),
          "=f"(__cdata[8]),
          "=f"(__cdata[9]),
          "=f"(__cdata[10]),
          "=f"(__cdata[11]),
          "=f"(__cdata[12]),
          "=f"(__cdata[13]),
          "=f"(__cdata[14]),
          "=f"(__cdata[15])
        : "r"(__taddr)
        : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm("tcgen05.ld.red.spcompress.sync.aligned.32x32b.x32.max.sp::2:4.f32.b2 {%1}, {%2, %3, %4, %5, %6, %7, %8, %9, "
        "%10, %11, %12, %13, %14, %15, %16, %17}, %0, [%18];"
        : "=f"(__redval),
          "=r"(__mdata[0]),
          "=f"(__cdata[0]),
          "=f"(__cdata[1]),
          "=f"(__cdata[2]),
          "=f"(__cdata[3]),
          "=f"(__cdata[4]),
          "=f"(__cdata[5]),
          "=f"(__cdata[6]),
          "=f"(__cdata[7]),
          "=f"(__cdata[8]),
          "=f"(__cdata[9]),
          "=f"(__cdata[10]),
          "=f"(__cdata[11]),
          "=f"(__cdata[12]),
          "=f"(__cdata[13]),
          "=f"(__cdata[14]),
          "=f"(__cdata[15])
        : "r"(__taddr)
        : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 940

/*
// tcgen05.ld.red.spcompress.sync.aligned.32x32b.x64.op.sp::2:4.abs.f32.b2 mdata, cdata, redval, [taddr]; // PTX ISA 94,
SM_107a
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_spcompress_32x32b_abs(
  cuda::ptx::op_t<Op> op,
  uint32_t (&mdata)[2],
  float (&cdata)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_spcompress_32x32b_abs(
  ::cuda::ptx::op_t<_Op> __op, ::cuda::std::uint32_t (&__mdata)[2], float (&__cdata)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.spcompress.sync.aligned.32x32b.x64.min.sp::2:4.abs.f32.b2 {%1, %2}, {%3, %4, %5, %6, %7, %8, %9, "
      "%10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34}, %0, [%35];"
      : "=f"(__redval),
        "=r"(__mdata[0]),
        "=r"(__mdata[1]),
        "=f"(__cdata[0]),
        "=f"(__cdata[1]),
        "=f"(__cdata[2]),
        "=f"(__cdata[3]),
        "=f"(__cdata[4]),
        "=f"(__cdata[5]),
        "=f"(__cdata[6]),
        "=f"(__cdata[7]),
        "=f"(__cdata[8]),
        "=f"(__cdata[9]),
        "=f"(__cdata[10]),
        "=f"(__cdata[11]),
        "=f"(__cdata[12]),
        "=f"(__cdata[13]),
        "=f"(__cdata[14]),
        "=f"(__cdata[15]),
        "=f"(__cdata[16]),
        "=f"(__cdata[17]),
        "=f"(__cdata[18]),
        "=f"(__cdata[19]),
        "=f"(__cdata[20]),
        "=f"(__cdata[21]),
        "=f"(__cdata[22]),
        "=f"(__cdata[23]),
        "=f"(__cdata[24]),
        "=f"(__cdata[25]),
        "=f"(__cdata[26]),
        "=f"(__cdata[27]),
        "=f"(__cdata[28]),
        "=f"(__cdata[29]),
        "=f"(__cdata[30]),
        "=f"(__cdata[31])
      : "r"(__taddr)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.spcompress.sync.aligned.32x32b.x64.max.sp::2:4.abs.f32.b2 {%1, %2}, {%3, %4, %5, %6, %7, %8, %9, "
      "%10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34}, %0, [%35];"
      : "=f"(__redval),
        "=r"(__mdata[0]),
        "=r"(__mdata[1]),
        "=f"(__cdata[0]),
        "=f"(__cdata[1]),
        "=f"(__cdata[2]),
        "=f"(__cdata[3]),
        "=f"(__cdata[4]),
        "=f"(__cdata[5]),
        "=f"(__cdata[6]),
        "=f"(__cdata[7]),
        "=f"(__cdata[8]),
        "=f"(__cdata[9]),
        "=f"(__cdata[10]),
        "=f"(__cdata[11]),
        "=f"(__cdata[12]),
        "=f"(__cdata[13]),
        "=f"(__cdata[14]),
        "=f"(__cdata[15]),
        "=f"(__cdata[16]),
        "=f"(__cdata[17]),
        "=f"(__cdata[18]),
        "=f"(__cdata[19]),
        "=f"(__cdata[20]),
        "=f"(__cdata[21]),
        "=f"(__cdata[22]),
        "=f"(__cdata[23]),
        "=f"(__cdata[24]),
        "=f"(__cdata[25]),
        "=f"(__cdata[26]),
        "=f"(__cdata[27]),
        "=f"(__cdata[28]),
        "=f"(__cdata[29]),
        "=f"(__cdata[30]),
        "=f"(__cdata[31])
      : "r"(__taddr)
      : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 940

/*
// tcgen05.ld.red.spcompress.sync.aligned.32x32b.x64.op.sp::2:4.f32.b2 mdata, cdata, redval, [taddr]; // PTX ISA 94,
SM_107a
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_spcompress_32x32b(
  cuda::ptx::op_t<Op> op,
  uint32_t (&mdata)[2],
  float (&cdata)[32],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_spcompress_32x32b(
  ::cuda::ptx::op_t<_Op> __op, ::cuda::std::uint32_t (&__mdata)[2], float (&__cdata)[32], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.spcompress.sync.aligned.32x32b.x64.min.sp::2:4.f32.b2 {%1, %2}, {%3, %4, %5, %6, %7, %8, %9, "
      "%10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34}, %0, [%35];"
      : "=f"(__redval),
        "=r"(__mdata[0]),
        "=r"(__mdata[1]),
        "=f"(__cdata[0]),
        "=f"(__cdata[1]),
        "=f"(__cdata[2]),
        "=f"(__cdata[3]),
        "=f"(__cdata[4]),
        "=f"(__cdata[5]),
        "=f"(__cdata[6]),
        "=f"(__cdata[7]),
        "=f"(__cdata[8]),
        "=f"(__cdata[9]),
        "=f"(__cdata[10]),
        "=f"(__cdata[11]),
        "=f"(__cdata[12]),
        "=f"(__cdata[13]),
        "=f"(__cdata[14]),
        "=f"(__cdata[15]),
        "=f"(__cdata[16]),
        "=f"(__cdata[17]),
        "=f"(__cdata[18]),
        "=f"(__cdata[19]),
        "=f"(__cdata[20]),
        "=f"(__cdata[21]),
        "=f"(__cdata[22]),
        "=f"(__cdata[23]),
        "=f"(__cdata[24]),
        "=f"(__cdata[25]),
        "=f"(__cdata[26]),
        "=f"(__cdata[27]),
        "=f"(__cdata[28]),
        "=f"(__cdata[29]),
        "=f"(__cdata[30]),
        "=f"(__cdata[31])
      : "r"(__taddr)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.spcompress.sync.aligned.32x32b.x64.max.sp::2:4.f32.b2 {%1, %2}, {%3, %4, %5, %6, %7, %8, %9, "
      "%10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34}, %0, [%35];"
      : "=f"(__redval),
        "=r"(__mdata[0]),
        "=r"(__mdata[1]),
        "=f"(__cdata[0]),
        "=f"(__cdata[1]),
        "=f"(__cdata[2]),
        "=f"(__cdata[3]),
        "=f"(__cdata[4]),
        "=f"(__cdata[5]),
        "=f"(__cdata[6]),
        "=f"(__cdata[7]),
        "=f"(__cdata[8]),
        "=f"(__cdata[9]),
        "=f"(__cdata[10]),
        "=f"(__cdata[11]),
        "=f"(__cdata[12]),
        "=f"(__cdata[13]),
        "=f"(__cdata[14]),
        "=f"(__cdata[15]),
        "=f"(__cdata[16]),
        "=f"(__cdata[17]),
        "=f"(__cdata[18]),
        "=f"(__cdata[19]),
        "=f"(__cdata[20]),
        "=f"(__cdata[21]),
        "=f"(__cdata[22]),
        "=f"(__cdata[23]),
        "=f"(__cdata[24]),
        "=f"(__cdata[25]),
        "=f"(__cdata[26]),
        "=f"(__cdata[27]),
        "=f"(__cdata[28]),
        "=f"(__cdata[29]),
        "=f"(__cdata[30]),
        "=f"(__cdata[31])
      : "r"(__taddr)
      : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 940

/*
// tcgen05.ld.red.spcompress.sync.aligned.32x32b.x128.op.sp::2:4.abs.f32.b2 mdata, cdata, redval, [taddr]; // PTX ISA
94, SM_107a
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_spcompress_32x32b_abs(
  cuda::ptx::op_t<Op> op,
  uint32_t (&mdata)[4],
  float (&cdata)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_spcompress_32x32b_abs(
  ::cuda::ptx::op_t<_Op> __op, ::cuda::std::uint32_t (&__mdata)[4], float (&__cdata)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.spcompress.sync.aligned.32x32b.x128.min.sp::2:4.abs.f32.b2 {%1, %2, %3, %4}, {%5, %6, %7, %8, "
      "%9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, "
      "%31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, "
      "%53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68}, %0, [%69];"
      : "=f"(__redval),
        "=r"(__mdata[0]),
        "=r"(__mdata[1]),
        "=r"(__mdata[2]),
        "=r"(__mdata[3]),
        "=f"(__cdata[0]),
        "=f"(__cdata[1]),
        "=f"(__cdata[2]),
        "=f"(__cdata[3]),
        "=f"(__cdata[4]),
        "=f"(__cdata[5]),
        "=f"(__cdata[6]),
        "=f"(__cdata[7]),
        "=f"(__cdata[8]),
        "=f"(__cdata[9]),
        "=f"(__cdata[10]),
        "=f"(__cdata[11]),
        "=f"(__cdata[12]),
        "=f"(__cdata[13]),
        "=f"(__cdata[14]),
        "=f"(__cdata[15]),
        "=f"(__cdata[16]),
        "=f"(__cdata[17]),
        "=f"(__cdata[18]),
        "=f"(__cdata[19]),
        "=f"(__cdata[20]),
        "=f"(__cdata[21]),
        "=f"(__cdata[22]),
        "=f"(__cdata[23]),
        "=f"(__cdata[24]),
        "=f"(__cdata[25]),
        "=f"(__cdata[26]),
        "=f"(__cdata[27]),
        "=f"(__cdata[28]),
        "=f"(__cdata[29]),
        "=f"(__cdata[30]),
        "=f"(__cdata[31]),
        "=f"(__cdata[32]),
        "=f"(__cdata[33]),
        "=f"(__cdata[34]),
        "=f"(__cdata[35]),
        "=f"(__cdata[36]),
        "=f"(__cdata[37]),
        "=f"(__cdata[38]),
        "=f"(__cdata[39]),
        "=f"(__cdata[40]),
        "=f"(__cdata[41]),
        "=f"(__cdata[42]),
        "=f"(__cdata[43]),
        "=f"(__cdata[44]),
        "=f"(__cdata[45]),
        "=f"(__cdata[46]),
        "=f"(__cdata[47]),
        "=f"(__cdata[48]),
        "=f"(__cdata[49]),
        "=f"(__cdata[50]),
        "=f"(__cdata[51]),
        "=f"(__cdata[52]),
        "=f"(__cdata[53]),
        "=f"(__cdata[54]),
        "=f"(__cdata[55]),
        "=f"(__cdata[56]),
        "=f"(__cdata[57]),
        "=f"(__cdata[58]),
        "=f"(__cdata[59]),
        "=f"(__cdata[60]),
        "=f"(__cdata[61]),
        "=f"(__cdata[62]),
        "=f"(__cdata[63])
      : "r"(__taddr)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.spcompress.sync.aligned.32x32b.x128.max.sp::2:4.abs.f32.b2 {%1, %2, %3, %4}, {%5, %6, %7, %8, "
      "%9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, "
      "%31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, "
      "%53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68}, %0, [%69];"
      : "=f"(__redval),
        "=r"(__mdata[0]),
        "=r"(__mdata[1]),
        "=r"(__mdata[2]),
        "=r"(__mdata[3]),
        "=f"(__cdata[0]),
        "=f"(__cdata[1]),
        "=f"(__cdata[2]),
        "=f"(__cdata[3]),
        "=f"(__cdata[4]),
        "=f"(__cdata[5]),
        "=f"(__cdata[6]),
        "=f"(__cdata[7]),
        "=f"(__cdata[8]),
        "=f"(__cdata[9]),
        "=f"(__cdata[10]),
        "=f"(__cdata[11]),
        "=f"(__cdata[12]),
        "=f"(__cdata[13]),
        "=f"(__cdata[14]),
        "=f"(__cdata[15]),
        "=f"(__cdata[16]),
        "=f"(__cdata[17]),
        "=f"(__cdata[18]),
        "=f"(__cdata[19]),
        "=f"(__cdata[20]),
        "=f"(__cdata[21]),
        "=f"(__cdata[22]),
        "=f"(__cdata[23]),
        "=f"(__cdata[24]),
        "=f"(__cdata[25]),
        "=f"(__cdata[26]),
        "=f"(__cdata[27]),
        "=f"(__cdata[28]),
        "=f"(__cdata[29]),
        "=f"(__cdata[30]),
        "=f"(__cdata[31]),
        "=f"(__cdata[32]),
        "=f"(__cdata[33]),
        "=f"(__cdata[34]),
        "=f"(__cdata[35]),
        "=f"(__cdata[36]),
        "=f"(__cdata[37]),
        "=f"(__cdata[38]),
        "=f"(__cdata[39]),
        "=f"(__cdata[40]),
        "=f"(__cdata[41]),
        "=f"(__cdata[42]),
        "=f"(__cdata[43]),
        "=f"(__cdata[44]),
        "=f"(__cdata[45]),
        "=f"(__cdata[46]),
        "=f"(__cdata[47]),
        "=f"(__cdata[48]),
        "=f"(__cdata[49]),
        "=f"(__cdata[50]),
        "=f"(__cdata[51]),
        "=f"(__cdata[52]),
        "=f"(__cdata[53]),
        "=f"(__cdata[54]),
        "=f"(__cdata[55]),
        "=f"(__cdata[56]),
        "=f"(__cdata[57]),
        "=f"(__cdata[58]),
        "=f"(__cdata[59]),
        "=f"(__cdata[60]),
        "=f"(__cdata[61]),
        "=f"(__cdata[62]),
        "=f"(__cdata[63])
      : "r"(__taddr)
      : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 940

/*
// tcgen05.ld.red.spcompress.sync.aligned.32x32b.x128.op.sp::2:4.f32.b2 mdata, cdata, redval, [taddr]; // PTX ISA 94,
SM_107a
// .op        = { .min, .max }
template <cuda::ptx::dot_op Op>
__device__ static inline float tcgen05_ld_red_spcompress_32x32b(
  cuda::ptx::op_t<Op> op,
  uint32_t (&mdata)[4],
  float (&cdata)[64],
  uint32_t taddr);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_op _Op>
_CCCL_DEVICE static inline float tcgen05_ld_red_spcompress_32x32b(
  ::cuda::ptx::op_t<_Op> __op, ::cuda::std::uint32_t (&__mdata)[4], float (&__cdata)[64], ::cuda::std::uint32_t __taddr)
{
  static_assert(__op == op_min || __op == op_max, "");
  float __redval;
  if constexpr (__op == op_min)
  {
    asm(
      "tcgen05.ld.red.spcompress.sync.aligned.32x32b.x128.min.sp::2:4.f32.b2 {%1, %2, %3, %4}, {%5, %6, %7, %8, %9, "
      "%10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, "
      "%54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68}, %0, [%69];"
      : "=f"(__redval),
        "=r"(__mdata[0]),
        "=r"(__mdata[1]),
        "=r"(__mdata[2]),
        "=r"(__mdata[3]),
        "=f"(__cdata[0]),
        "=f"(__cdata[1]),
        "=f"(__cdata[2]),
        "=f"(__cdata[3]),
        "=f"(__cdata[4]),
        "=f"(__cdata[5]),
        "=f"(__cdata[6]),
        "=f"(__cdata[7]),
        "=f"(__cdata[8]),
        "=f"(__cdata[9]),
        "=f"(__cdata[10]),
        "=f"(__cdata[11]),
        "=f"(__cdata[12]),
        "=f"(__cdata[13]),
        "=f"(__cdata[14]),
        "=f"(__cdata[15]),
        "=f"(__cdata[16]),
        "=f"(__cdata[17]),
        "=f"(__cdata[18]),
        "=f"(__cdata[19]),
        "=f"(__cdata[20]),
        "=f"(__cdata[21]),
        "=f"(__cdata[22]),
        "=f"(__cdata[23]),
        "=f"(__cdata[24]),
        "=f"(__cdata[25]),
        "=f"(__cdata[26]),
        "=f"(__cdata[27]),
        "=f"(__cdata[28]),
        "=f"(__cdata[29]),
        "=f"(__cdata[30]),
        "=f"(__cdata[31]),
        "=f"(__cdata[32]),
        "=f"(__cdata[33]),
        "=f"(__cdata[34]),
        "=f"(__cdata[35]),
        "=f"(__cdata[36]),
        "=f"(__cdata[37]),
        "=f"(__cdata[38]),
        "=f"(__cdata[39]),
        "=f"(__cdata[40]),
        "=f"(__cdata[41]),
        "=f"(__cdata[42]),
        "=f"(__cdata[43]),
        "=f"(__cdata[44]),
        "=f"(__cdata[45]),
        "=f"(__cdata[46]),
        "=f"(__cdata[47]),
        "=f"(__cdata[48]),
        "=f"(__cdata[49]),
        "=f"(__cdata[50]),
        "=f"(__cdata[51]),
        "=f"(__cdata[52]),
        "=f"(__cdata[53]),
        "=f"(__cdata[54]),
        "=f"(__cdata[55]),
        "=f"(__cdata[56]),
        "=f"(__cdata[57]),
        "=f"(__cdata[58]),
        "=f"(__cdata[59]),
        "=f"(__cdata[60]),
        "=f"(__cdata[61]),
        "=f"(__cdata[62]),
        "=f"(__cdata[63])
      : "r"(__taddr)
      : "memory");
  }
  else if constexpr (__op == op_max)
  {
    asm(
      "tcgen05.ld.red.spcompress.sync.aligned.32x32b.x128.max.sp::2:4.f32.b2 {%1, %2, %3, %4}, {%5, %6, %7, %8, %9, "
      "%10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, "
      "%54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68}, %0, [%69];"
      : "=f"(__redval),
        "=r"(__mdata[0]),
        "=r"(__mdata[1]),
        "=r"(__mdata[2]),
        "=r"(__mdata[3]),
        "=f"(__cdata[0]),
        "=f"(__cdata[1]),
        "=f"(__cdata[2]),
        "=f"(__cdata[3]),
        "=f"(__cdata[4]),
        "=f"(__cdata[5]),
        "=f"(__cdata[6]),
        "=f"(__cdata[7]),
        "=f"(__cdata[8]),
        "=f"(__cdata[9]),
        "=f"(__cdata[10]),
        "=f"(__cdata[11]),
        "=f"(__cdata[12]),
        "=f"(__cdata[13]),
        "=f"(__cdata[14]),
        "=f"(__cdata[15]),
        "=f"(__cdata[16]),
        "=f"(__cdata[17]),
        "=f"(__cdata[18]),
        "=f"(__cdata[19]),
        "=f"(__cdata[20]),
        "=f"(__cdata[21]),
        "=f"(__cdata[22]),
        "=f"(__cdata[23]),
        "=f"(__cdata[24]),
        "=f"(__cdata[25]),
        "=f"(__cdata[26]),
        "=f"(__cdata[27]),
        "=f"(__cdata[28]),
        "=f"(__cdata[29]),
        "=f"(__cdata[30]),
        "=f"(__cdata[31]),
        "=f"(__cdata[32]),
        "=f"(__cdata[33]),
        "=f"(__cdata[34]),
        "=f"(__cdata[35]),
        "=f"(__cdata[36]),
        "=f"(__cdata[37]),
        "=f"(__cdata[38]),
        "=f"(__cdata[39]),
        "=f"(__cdata[40]),
        "=f"(__cdata[41]),
        "=f"(__cdata[42]),
        "=f"(__cdata[43]),
        "=f"(__cdata[44]),
        "=f"(__cdata[45]),
        "=f"(__cdata[46]),
        "=f"(__cdata[47]),
        "=f"(__cdata[48]),
        "=f"(__cdata[49]),
        "=f"(__cdata[50]),
        "=f"(__cdata[51]),
        "=f"(__cdata[52]),
        "=f"(__cdata[53]),
        "=f"(__cdata[54]),
        "=f"(__cdata[55]),
        "=f"(__cdata[56]),
        "=f"(__cdata[57]),
        "=f"(__cdata[58]),
        "=f"(__cdata[59]),
        "=f"(__cdata[60]),
        "=f"(__cdata[61]),
        "=f"(__cdata[62]),
        "=f"(__cdata[63])
      : "r"(__taddr)
      : "memory");
  }
  return __redval;
}
#endif // __cccl_ptx_isa >= 940

#endif // _CUDA_PTX_GENERATED_TCGEN05_LD_H_
