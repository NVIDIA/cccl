// This file was automatically generated. Do not edit.

namespace cuda_ptx
{

/*
// ld.global.b8 dest, [addr]; // PTX ISA 10, SM_50
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 100
extern "C" __device__ void __cuda_ptx_ld_global_is_not_supported_before_SM_50__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 500
  std::uint32_t dest;
  asm volatile("ld.global.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_is_not_supported_before_SM_50__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 100

/*
// ld.global.b16 dest, [addr]; // PTX ISA 10, SM_50
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 100
extern "C" __device__ void __cuda_ptx_ld_global_is_not_supported_before_SM_50__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 500
  std::uint16_t dest;
  asm volatile("ld.global.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_is_not_supported_before_SM_50__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 100

/*
// ld.global.b32 dest, [addr]; // PTX ISA 10, SM_50
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 100
extern "C" __device__ void __cuda_ptx_ld_global_is_not_supported_before_SM_50__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 500
  std::uint32_t dest;
  asm volatile("ld.global.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_is_not_supported_before_SM_50__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 100

/*
// ld.global.b64 dest, [addr]; // PTX ISA 10, SM_50
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 100
extern "C" __device__ void __cuda_ptx_ld_global_is_not_supported_before_SM_50__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 500
  std::uint64_t dest;
  asm volatile("ld.global.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_is_not_supported_before_SM_50__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 100

/*
// ld.global.b128 dest, [addr]; // PTX ISA 83, SM_70
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_is_not_supported_before_SM_70__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 700
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_is_not_supported_before_SM_70__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.L2::64B.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L2_64B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L2_64B_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L2_64B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L2::64B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L2_64B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L2::64B.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L2_64B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L2_64B_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L2_64B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.L2::64B.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L2_64B_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L2::64B.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L2_64B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L2_64B_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L2_64B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L2::64B.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L2_64B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L2::64B.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L2_64B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L2_64B_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L2_64B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.L2::64B.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L2_64B_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L2::64B.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L2_64B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_L2_64B_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L2_64B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.L2::64B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L2_64B_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.L2::128B.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L2_128B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L2_128B_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L2_128B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L2::128B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L2_128B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L2::128B.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L2_128B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L2_128B_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L2_128B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.L2::128B.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L2_128B_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L2::128B.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L2_128B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L2_128B_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L2_128B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L2::128B.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L2_128B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L2::128B.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L2_128B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L2_128B_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L2_128B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.L2::128B.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L2_128B_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L2::128B.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L2_128B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_L2_128B_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L2_128B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.L2::128B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L2_128B_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.L2::256B.b8 dest, [addr]; // PTX ISA 74, SM_80
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L2_256B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L2_256B_is_not_supported_before_SM_80__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L2_256B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint32_t dest;
  asm volatile("ld.global.L2::256B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L2_256B_is_not_supported_before_SM_80__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L2::256B.b16 dest, [addr]; // PTX ISA 74, SM_80
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L2_256B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L2_256B_is_not_supported_before_SM_80__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L2_256B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint16_t dest;
  asm volatile("ld.global.L2::256B.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L2_256B_is_not_supported_before_SM_80__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L2::256B.b32 dest, [addr]; // PTX ISA 74, SM_80
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L2_256B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L2_256B_is_not_supported_before_SM_80__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L2_256B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint32_t dest;
  asm volatile("ld.global.L2::256B.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L2_256B_is_not_supported_before_SM_80__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L2::256B.b64 dest, [addr]; // PTX ISA 74, SM_80
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L2_256B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L2_256B_is_not_supported_before_SM_80__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L2_256B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint64_t dest;
  asm volatile("ld.global.L2::256B.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L2_256B_is_not_supported_before_SM_80__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L2::256B.b128 dest, [addr]; // PTX ISA 83, SM_80
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L2_256B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_L2_256B_is_not_supported_before_SM_80__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L2_256B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.L2::256B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L2_256B_is_not_supported_before_SM_80__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.L1::evict_normal.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_normal(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_normal_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_normal(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_normal.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_normal_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_normal.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_normal(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_normal_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_normal(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.L1::evict_normal.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_normal_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_normal.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_normal(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_normal_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_normal(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_normal.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_normal_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_normal.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_normal(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_normal_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_normal(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.L1::evict_normal.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_normal_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_normal.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_normal(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_normal_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_normal(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.L1::evict_normal.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_normal_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.L1::evict_normal.L2::64B.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_normal_L2_64B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_normal_L2_64B_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_normal_L2_64B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_normal.L2::64B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_normal_L2_64B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_normal.L2::64B.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_normal_L2_64B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_normal_L2_64B_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_normal_L2_64B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.L1::evict_normal.L2::64B.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_normal_L2_64B_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_normal.L2::64B.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_normal_L2_64B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_normal_L2_64B_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_normal_L2_64B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_normal.L2::64B.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_normal_L2_64B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_normal.L2::64B.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_normal_L2_64B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_normal_L2_64B_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_normal_L2_64B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.L1::evict_normal.L2::64B.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_normal_L2_64B_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_normal.L2::64B.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_normal_L2_64B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_normal_L2_64B_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_normal_L2_64B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.L1::evict_normal.L2::64B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_normal_L2_64B_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.L1::evict_normal.L2::128B.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_normal_L2_128B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_normal_L2_128B_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_normal_L2_128B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_normal.L2::128B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_normal_L2_128B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_normal.L2::128B.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_normal_L2_128B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_normal_L2_128B_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_normal_L2_128B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.L1::evict_normal.L2::128B.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_normal_L2_128B_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_normal.L2::128B.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_normal_L2_128B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_normal_L2_128B_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_normal_L2_128B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_normal.L2::128B.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_normal_L2_128B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_normal.L2::128B.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_normal_L2_128B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_normal_L2_128B_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_normal_L2_128B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.L1::evict_normal.L2::128B.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_normal_L2_128B_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_normal.L2::128B.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_normal_L2_128B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_normal_L2_128B_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_normal_L2_128B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.L1::evict_normal.L2::128B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_normal_L2_128B_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.L1::evict_normal.L2::256B.b8 dest, [addr]; // PTX ISA 74, SM_80
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_normal_L2_256B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_normal_L2_256B_is_not_supported_before_SM_80__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_normal_L2_256B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_normal.L2::256B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_normal_L2_256B_is_not_supported_before_SM_80__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_normal.L2::256B.b16 dest, [addr]; // PTX ISA 74, SM_80
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_normal_L2_256B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_normal_L2_256B_is_not_supported_before_SM_80__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_normal_L2_256B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint16_t dest;
  asm volatile("ld.global.L1::evict_normal.L2::256B.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_normal_L2_256B_is_not_supported_before_SM_80__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_normal.L2::256B.b32 dest, [addr]; // PTX ISA 74, SM_80
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_normal_L2_256B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_normal_L2_256B_is_not_supported_before_SM_80__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_normal_L2_256B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_normal.L2::256B.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_normal_L2_256B_is_not_supported_before_SM_80__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_normal.L2::256B.b64 dest, [addr]; // PTX ISA 74, SM_80
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_normal_L2_256B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_normal_L2_256B_is_not_supported_before_SM_80__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_normal_L2_256B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint64_t dest;
  asm volatile("ld.global.L1::evict_normal.L2::256B.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_normal_L2_256B_is_not_supported_before_SM_80__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_normal.L2::256B.b128 dest, [addr]; // PTX ISA 83, SM_80
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_normal_L2_256B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_normal_L2_256B_is_not_supported_before_SM_80__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_normal_L2_256B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.L1::evict_normal.L2::256B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_normal_L2_256B_is_not_supported_before_SM_80__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.L1::evict_unchanged.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_unchanged(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_unchanged_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_unchanged(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_unchanged.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_unchanged_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_unchanged.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_unchanged(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_unchanged_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_unchanged(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.L1::evict_unchanged.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_unchanged_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_unchanged.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_unchanged(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_unchanged_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_unchanged(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_unchanged.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_unchanged_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_unchanged.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_unchanged(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_unchanged_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_unchanged(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.L1::evict_unchanged.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_unchanged_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_unchanged.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_unchanged(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_unchanged_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_unchanged(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.L1::evict_unchanged.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_unchanged_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.L1::evict_unchanged.L2::64B.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_unchanged_L2_64B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_unchanged_L2_64B_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_unchanged_L2_64B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_unchanged.L2::64B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_unchanged_L2_64B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_unchanged.L2::64B.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_unchanged_L2_64B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_unchanged_L2_64B_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_unchanged_L2_64B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.L1::evict_unchanged.L2::64B.b16 %0, [%1];"
               : "=h"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_unchanged_L2_64B_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_unchanged.L2::64B.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_unchanged_L2_64B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_unchanged_L2_64B_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_unchanged_L2_64B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_unchanged.L2::64B.b32 %0, [%1];"
               : "=r"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_unchanged_L2_64B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_unchanged.L2::64B.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_unchanged_L2_64B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_unchanged_L2_64B_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_unchanged_L2_64B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.L1::evict_unchanged.L2::64B.b64 %0, [%1];"
               : "=l"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_unchanged_L2_64B_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_unchanged.L2::64B.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_unchanged_L2_64B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_unchanged_L2_64B_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_unchanged_L2_64B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.L1::evict_unchanged.L2::64B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_unchanged_L2_64B_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.L1::evict_unchanged.L2::128B.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_unchanged_L2_128B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_unchanged_L2_128B_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_unchanged_L2_128B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_unchanged.L2::128B.b8 %0, [%1];"
               : "=r"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_unchanged_L2_128B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_unchanged.L2::128B.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_unchanged_L2_128B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_unchanged_L2_128B_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_unchanged_L2_128B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.L1::evict_unchanged.L2::128B.b16 %0, [%1];"
               : "=h"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_unchanged_L2_128B_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_unchanged.L2::128B.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_unchanged_L2_128B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_unchanged_L2_128B_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_unchanged_L2_128B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_unchanged.L2::128B.b32 %0, [%1];"
               : "=r"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_unchanged_L2_128B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_unchanged.L2::128B.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_unchanged_L2_128B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_unchanged_L2_128B_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_unchanged_L2_128B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.L1::evict_unchanged.L2::128B.b64 %0, [%1];"
               : "=l"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_unchanged_L2_128B_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_unchanged.L2::128B.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_unchanged_L2_128B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_unchanged_L2_128B_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_unchanged_L2_128B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.L1::evict_unchanged.L2::128B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_unchanged_L2_128B_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.L1::evict_unchanged.L2::256B.b8 dest, [addr]; // PTX ISA 74, SM_80
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_unchanged_L2_256B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_unchanged_L2_256B_is_not_supported_before_SM_80__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_unchanged_L2_256B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_unchanged.L2::256B.b8 %0, [%1];"
               : "=r"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_unchanged_L2_256B_is_not_supported_before_SM_80__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_unchanged.L2::256B.b16 dest, [addr]; // PTX ISA 74, SM_80
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_unchanged_L2_256B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_unchanged_L2_256B_is_not_supported_before_SM_80__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_unchanged_L2_256B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint16_t dest;
  asm volatile("ld.global.L1::evict_unchanged.L2::256B.b16 %0, [%1];"
               : "=h"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_unchanged_L2_256B_is_not_supported_before_SM_80__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_unchanged.L2::256B.b32 dest, [addr]; // PTX ISA 74, SM_80
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_unchanged_L2_256B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_unchanged_L2_256B_is_not_supported_before_SM_80__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_unchanged_L2_256B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_unchanged.L2::256B.b32 %0, [%1];"
               : "=r"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_unchanged_L2_256B_is_not_supported_before_SM_80__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_unchanged.L2::256B.b64 dest, [addr]; // PTX ISA 74, SM_80
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_unchanged_L2_256B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_unchanged_L2_256B_is_not_supported_before_SM_80__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_unchanged_L2_256B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint64_t dest;
  asm volatile("ld.global.L1::evict_unchanged.L2::256B.b64 %0, [%1];"
               : "=l"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_unchanged_L2_256B_is_not_supported_before_SM_80__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_unchanged.L2::256B.b128 dest, [addr]; // PTX ISA 83, SM_80
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_unchanged_L2_256B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_unchanged_L2_256B_is_not_supported_before_SM_80__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_unchanged_L2_256B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.L1::evict_unchanged.L2::256B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_unchanged_L2_256B_is_not_supported_before_SM_80__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.L1::evict_first.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_first(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_first_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_first(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_first.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_first_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_first.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_first(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_first_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_first(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.L1::evict_first.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_first_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_first.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_first(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_first_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_first(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_first.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_first_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_first.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_first(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_first_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_first(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.L1::evict_first.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_first_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_first.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_first(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_first_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_first(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.L1::evict_first.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_first_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.L1::evict_first.L2::64B.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_first_L2_64B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_first_L2_64B_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_first_L2_64B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_first.L2::64B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_first_L2_64B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_first.L2::64B.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_first_L2_64B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_first_L2_64B_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_first_L2_64B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.L1::evict_first.L2::64B.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_first_L2_64B_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_first.L2::64B.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_first_L2_64B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_first_L2_64B_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_first_L2_64B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_first.L2::64B.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_first_L2_64B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_first.L2::64B.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_first_L2_64B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_first_L2_64B_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_first_L2_64B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.L1::evict_first.L2::64B.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_first_L2_64B_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_first.L2::64B.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_first_L2_64B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_first_L2_64B_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_first_L2_64B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.L1::evict_first.L2::64B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_first_L2_64B_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.L1::evict_first.L2::128B.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_first_L2_128B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_first_L2_128B_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_first_L2_128B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_first.L2::128B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_first_L2_128B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_first.L2::128B.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_first_L2_128B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_first_L2_128B_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_first_L2_128B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.L1::evict_first.L2::128B.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_first_L2_128B_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_first.L2::128B.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_first_L2_128B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_first_L2_128B_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_first_L2_128B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_first.L2::128B.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_first_L2_128B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_first.L2::128B.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_first_L2_128B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_first_L2_128B_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_first_L2_128B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.L1::evict_first.L2::128B.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_first_L2_128B_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_first.L2::128B.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_first_L2_128B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_first_L2_128B_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_first_L2_128B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.L1::evict_first.L2::128B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_first_L2_128B_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.L1::evict_first.L2::256B.b8 dest, [addr]; // PTX ISA 74, SM_80
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_first_L2_256B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_first_L2_256B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_first.L2::256B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_first.L2::256B.b16 dest, [addr]; // PTX ISA 74, SM_80
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_first_L2_256B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_first_L2_256B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint16_t dest;
  asm volatile("ld.global.L1::evict_first.L2::256B.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_first.L2::256B.b32 dest, [addr]; // PTX ISA 74, SM_80
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_first_L2_256B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_first_L2_256B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_first.L2::256B.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_first.L2::256B.b64 dest, [addr]; // PTX ISA 74, SM_80
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_first_L2_256B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_first_L2_256B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint64_t dest;
  asm volatile("ld.global.L1::evict_first.L2::256B.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_first.L2::256B.b128 dest, [addr]; // PTX ISA 83, SM_80
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_first_L2_256B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_first_L2_256B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.L1::evict_first.L2::256B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.L1::evict_last.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_last(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_last_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_last(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_last.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_last_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_last.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_last(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_last_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_last(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.L1::evict_last.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_last_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_last.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_last(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_last_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_last(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_last.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_last_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_last.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_last(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_last_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_last(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.L1::evict_last.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_last_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_last.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_last(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_last_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_last(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.L1::evict_last.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_last_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.L1::evict_last.L2::64B.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_last_L2_64B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_last_L2_64B_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_last_L2_64B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_last.L2::64B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_last_L2_64B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_last.L2::64B.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_last_L2_64B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_last_L2_64B_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_last_L2_64B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.L1::evict_last.L2::64B.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_last_L2_64B_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_last.L2::64B.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_last_L2_64B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_last_L2_64B_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_last_L2_64B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_last.L2::64B.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_last_L2_64B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_last.L2::64B.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_last_L2_64B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_last_L2_64B_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_last_L2_64B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.L1::evict_last.L2::64B.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_last_L2_64B_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_last.L2::64B.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_last_L2_64B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_last_L2_64B_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_last_L2_64B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.L1::evict_last.L2::64B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_last_L2_64B_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.L1::evict_last.L2::128B.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_last_L2_128B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_last_L2_128B_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_last_L2_128B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_last.L2::128B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_last_L2_128B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_last.L2::128B.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_last_L2_128B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_last_L2_128B_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_last_L2_128B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.L1::evict_last.L2::128B.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_last_L2_128B_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_last.L2::128B.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_last_L2_128B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_last_L2_128B_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_last_L2_128B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_last.L2::128B.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_last_L2_128B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_last.L2::128B.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_last_L2_128B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_last_L2_128B_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_last_L2_128B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.L1::evict_last.L2::128B.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_last_L2_128B_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_last.L2::128B.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_last_L2_128B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_last_L2_128B_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_last_L2_128B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.L1::evict_last.L2::128B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_last_L2_128B_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.L1::evict_last.L2::256B.b8 dest, [addr]; // PTX ISA 74, SM_80
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_last_L2_256B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_evict_last_L2_256B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_last.L2::256B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_last.L2::256B.b16 dest, [addr]; // PTX ISA 74, SM_80
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_last_L2_256B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_evict_last_L2_256B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint16_t dest;
  asm volatile("ld.global.L1::evict_last.L2::256B.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_last.L2::256B.b32 dest, [addr]; // PTX ISA 74, SM_80
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_last_L2_256B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_evict_last_L2_256B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint32_t dest;
  asm volatile("ld.global.L1::evict_last.L2::256B.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_last.L2::256B.b64 dest, [addr]; // PTX ISA 74, SM_80
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_last_L2_256B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_evict_last_L2_256B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint64_t dest;
  asm volatile("ld.global.L1::evict_last.L2::256B.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::evict_last.L2::256B.b128 dest, [addr]; // PTX ISA 83, SM_80
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_last_L2_256B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_evict_last_L2_256B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.L1::evict_last.L2::256B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.L1::no_allocate.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_no_allocate(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_no_allocate_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_no_allocate(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::no_allocate.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_no_allocate_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::no_allocate.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_no_allocate(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_no_allocate_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_no_allocate(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.L1::no_allocate.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_no_allocate_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::no_allocate.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_no_allocate(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_no_allocate_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_no_allocate(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::no_allocate.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_no_allocate_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::no_allocate.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_no_allocate(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_no_allocate_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_no_allocate(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.L1::no_allocate.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_no_allocate_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::no_allocate.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_no_allocate(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_L1_no_allocate_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_no_allocate(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.L1::no_allocate.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_no_allocate_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.L1::no_allocate.L2::64B.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_no_allocate_L2_64B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_no_allocate_L2_64B_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_no_allocate_L2_64B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::no_allocate.L2::64B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_no_allocate_L2_64B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::no_allocate.L2::64B.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_no_allocate_L2_64B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_no_allocate_L2_64B_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_no_allocate_L2_64B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.L1::no_allocate.L2::64B.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_no_allocate_L2_64B_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::no_allocate.L2::64B.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_no_allocate_L2_64B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_no_allocate_L2_64B_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_no_allocate_L2_64B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::no_allocate.L2::64B.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_no_allocate_L2_64B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::no_allocate.L2::64B.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_no_allocate_L2_64B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_no_allocate_L2_64B_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_no_allocate_L2_64B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.L1::no_allocate.L2::64B.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_no_allocate_L2_64B_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::no_allocate.L2::64B.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_no_allocate_L2_64B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_L1_no_allocate_L2_64B_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_no_allocate_L2_64B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.L1::no_allocate.L2::64B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_no_allocate_L2_64B_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.L1::no_allocate.L2::128B.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_no_allocate_L2_128B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_no_allocate_L2_128B_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_no_allocate_L2_128B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::no_allocate.L2::128B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_no_allocate_L2_128B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::no_allocate.L2::128B.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_no_allocate_L2_128B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_no_allocate_L2_128B_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_no_allocate_L2_128B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.L1::no_allocate.L2::128B.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_no_allocate_L2_128B_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::no_allocate.L2::128B.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_no_allocate_L2_128B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_no_allocate_L2_128B_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_no_allocate_L2_128B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.L1::no_allocate.L2::128B.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_no_allocate_L2_128B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::no_allocate.L2::128B.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_no_allocate_L2_128B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_no_allocate_L2_128B_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_no_allocate_L2_128B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.L1::no_allocate.L2::128B.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_no_allocate_L2_128B_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::no_allocate.L2::128B.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_no_allocate_L2_128B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_L1_no_allocate_L2_128B_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_no_allocate_L2_128B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.L1::no_allocate.L2::128B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_no_allocate_L2_128B_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.L1::no_allocate.L2::256B.b8 dest, [addr]; // PTX ISA 74, SM_80
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_no_allocate_L2_256B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_L1_no_allocate_L2_256B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint32_t dest;
  asm volatile("ld.global.L1::no_allocate.L2::256B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::no_allocate.L2::256B.b16 dest, [addr]; // PTX ISA 74, SM_80
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_no_allocate_L2_256B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_L1_no_allocate_L2_256B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint16_t dest;
  asm volatile("ld.global.L1::no_allocate.L2::256B.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::no_allocate.L2::256B.b32 dest, [addr]; // PTX ISA 74, SM_80
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_no_allocate_L2_256B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_L1_no_allocate_L2_256B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint32_t dest;
  asm volatile("ld.global.L1::no_allocate.L2::256B.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::no_allocate.L2::256B.b64 dest, [addr]; // PTX ISA 74, SM_80
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_no_allocate_L2_256B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_L1_no_allocate_L2_256B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint64_t dest;
  asm volatile("ld.global.L1::no_allocate.L2::256B.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.L1::no_allocate.L2::256B.b128 dest, [addr]; // PTX ISA 83, SM_80
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_no_allocate_L2_256B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_L1_no_allocate_L2_256B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.L1::no_allocate.L2::256B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.nc.b8 dest, [addr]; // PTX ISA 10, SM_50
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 100
extern "C" __device__ void __cuda_ptx_ld_global_nc_is_not_supported_before_SM_50__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 500
  std::uint32_t dest;
  asm volatile("ld.global.nc.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_is_not_supported_before_SM_50__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 100

/*
// ld.global.nc.b16 dest, [addr]; // PTX ISA 10, SM_50
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 100
extern "C" __device__ void __cuda_ptx_ld_global_nc_is_not_supported_before_SM_50__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 500
  std::uint16_t dest;
  asm volatile("ld.global.nc.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_is_not_supported_before_SM_50__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 100

/*
// ld.global.nc.b32 dest, [addr]; // PTX ISA 10, SM_50
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 100
extern "C" __device__ void __cuda_ptx_ld_global_nc_is_not_supported_before_SM_50__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 500
  std::uint32_t dest;
  asm volatile("ld.global.nc.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_is_not_supported_before_SM_50__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 100

/*
// ld.global.nc.b64 dest, [addr]; // PTX ISA 10, SM_50
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 100
extern "C" __device__ void __cuda_ptx_ld_global_nc_is_not_supported_before_SM_50__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 500
  std::uint64_t dest;
  asm volatile("ld.global.nc.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_is_not_supported_before_SM_50__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 100

/*
// ld.global.nc.b128 dest, [addr]; // PTX ISA 83, SM_70
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_nc_is_not_supported_before_SM_70__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 700
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.nc.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_is_not_supported_before_SM_70__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.nc.L2::64B.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L2_64B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L2_64B_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L2_64B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L2::64B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L2_64B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L2::64B.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L2_64B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L2_64B_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L2_64B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.nc.L2::64B.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L2_64B_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L2::64B.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L2_64B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L2_64B_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L2_64B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L2::64B.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L2_64B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L2::64B.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L2_64B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L2_64B_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L2_64B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.nc.L2::64B.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L2_64B_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L2::64B.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L2_64B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_nc_L2_64B_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L2_64B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.nc.L2::64B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L2_64B_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.nc.L2::128B.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L2_128B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L2_128B_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L2_128B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L2::128B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L2_128B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L2::128B.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L2_128B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L2_128B_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L2_128B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.nc.L2::128B.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L2_128B_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L2::128B.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L2_128B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L2_128B_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L2_128B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L2::128B.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L2_128B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L2::128B.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L2_128B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L2_128B_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L2_128B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.nc.L2::128B.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L2_128B_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L2::128B.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L2_128B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_nc_L2_128B_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L2_128B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.nc.L2::128B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L2_128B_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.nc.L2::256B.b8 dest, [addr]; // PTX ISA 74, SM_80
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L2_256B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L2_256B_is_not_supported_before_SM_80__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L2_256B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint32_t dest;
  asm volatile("ld.global.nc.L2::256B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L2_256B_is_not_supported_before_SM_80__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L2::256B.b16 dest, [addr]; // PTX ISA 74, SM_80
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L2_256B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L2_256B_is_not_supported_before_SM_80__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L2_256B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint16_t dest;
  asm volatile("ld.global.nc.L2::256B.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L2_256B_is_not_supported_before_SM_80__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L2::256B.b32 dest, [addr]; // PTX ISA 74, SM_80
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L2_256B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L2_256B_is_not_supported_before_SM_80__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L2_256B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint32_t dest;
  asm volatile("ld.global.nc.L2::256B.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L2_256B_is_not_supported_before_SM_80__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L2::256B.b64 dest, [addr]; // PTX ISA 74, SM_80
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L2_256B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L2_256B_is_not_supported_before_SM_80__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L2_256B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint64_t dest;
  asm volatile("ld.global.nc.L2::256B.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L2_256B_is_not_supported_before_SM_80__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L2::256B.b128 dest, [addr]; // PTX ISA 83, SM_80
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L2_256B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_nc_L2_256B_is_not_supported_before_SM_80__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L2_256B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.nc.L2::256B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L2_256B_is_not_supported_before_SM_80__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.nc.L1::evict_normal.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_normal(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_normal_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_normal(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_normal.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_normal_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_normal.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_normal(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_normal_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_normal(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.nc.L1::evict_normal.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_normal_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_normal.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_normal(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_normal_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_normal(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_normal.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_normal_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_normal.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_normal(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_normal_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_normal(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.nc.L1::evict_normal.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_normal_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_normal.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_normal(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_normal_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_normal(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.nc.L1::evict_normal.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_normal_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.nc.L1::evict_normal.L2::64B.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_normal_L2_64B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_normal_L2_64B_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_normal_L2_64B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_normal.L2::64B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_normal_L2_64B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_normal.L2::64B.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_normal_L2_64B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_normal_L2_64B_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_normal_L2_64B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.nc.L1::evict_normal.L2::64B.b16 %0, [%1];"
               : "=h"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_normal_L2_64B_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_normal.L2::64B.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_normal_L2_64B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_normal_L2_64B_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_normal_L2_64B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_normal.L2::64B.b32 %0, [%1];"
               : "=r"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_normal_L2_64B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_normal.L2::64B.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_normal_L2_64B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_normal_L2_64B_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_normal_L2_64B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.nc.L1::evict_normal.L2::64B.b64 %0, [%1];"
               : "=l"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_normal_L2_64B_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_normal.L2::64B.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_normal_L2_64B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_normal_L2_64B_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_normal_L2_64B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.nc.L1::evict_normal.L2::64B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_normal_L2_64B_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.nc.L1::evict_normal.L2::128B.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_normal_L2_128B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_normal_L2_128B_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_normal_L2_128B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_normal.L2::128B.b8 %0, [%1];"
               : "=r"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_normal_L2_128B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_normal.L2::128B.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_normal_L2_128B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_normal_L2_128B_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_normal_L2_128B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.nc.L1::evict_normal.L2::128B.b16 %0, [%1];"
               : "=h"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_normal_L2_128B_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_normal.L2::128B.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_normal_L2_128B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_normal_L2_128B_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_normal_L2_128B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_normal.L2::128B.b32 %0, [%1];"
               : "=r"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_normal_L2_128B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_normal.L2::128B.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_normal_L2_128B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_normal_L2_128B_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_normal_L2_128B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.nc.L1::evict_normal.L2::128B.b64 %0, [%1];"
               : "=l"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_normal_L2_128B_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_normal.L2::128B.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_normal_L2_128B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_normal_L2_128B_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_normal_L2_128B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.nc.L1::evict_normal.L2::128B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_normal_L2_128B_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.nc.L1::evict_normal.L2::256B.b8 dest, [addr]; // PTX ISA 74, SM_80
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_normal_L2_256B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_normal_L2_256B_is_not_supported_before_SM_80__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_normal_L2_256B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_normal.L2::256B.b8 %0, [%1];"
               : "=r"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_normal_L2_256B_is_not_supported_before_SM_80__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_normal.L2::256B.b16 dest, [addr]; // PTX ISA 74, SM_80
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_normal_L2_256B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_normal_L2_256B_is_not_supported_before_SM_80__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_normal_L2_256B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint16_t dest;
  asm volatile("ld.global.nc.L1::evict_normal.L2::256B.b16 %0, [%1];"
               : "=h"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_normal_L2_256B_is_not_supported_before_SM_80__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_normal.L2::256B.b32 dest, [addr]; // PTX ISA 74, SM_80
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_normal_L2_256B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_normal_L2_256B_is_not_supported_before_SM_80__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_normal_L2_256B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_normal.L2::256B.b32 %0, [%1];"
               : "=r"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_normal_L2_256B_is_not_supported_before_SM_80__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_normal.L2::256B.b64 dest, [addr]; // PTX ISA 74, SM_80
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_normal_L2_256B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_normal_L2_256B_is_not_supported_before_SM_80__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_normal_L2_256B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint64_t dest;
  asm volatile("ld.global.nc.L1::evict_normal.L2::256B.b64 %0, [%1];"
               : "=l"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_normal_L2_256B_is_not_supported_before_SM_80__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_normal.L2::256B.b128 dest, [addr]; // PTX ISA 83, SM_80
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_normal_L2_256B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_normal_L2_256B_is_not_supported_before_SM_80__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_normal_L2_256B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.nc.L1::evict_normal.L2::256B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_normal_L2_256B_is_not_supported_before_SM_80__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.nc.L1::evict_unchanged.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_unchanged(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_unchanged_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_unchanged(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_unchanged.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_unchanged_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_unchanged.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_unchanged(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_unchanged_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_unchanged(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.nc.L1::evict_unchanged.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_unchanged_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_unchanged.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_unchanged(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_unchanged_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_unchanged(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_unchanged.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_unchanged_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_unchanged.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_unchanged(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_unchanged_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_unchanged(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.nc.L1::evict_unchanged.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_unchanged_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_unchanged.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_unchanged(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_unchanged_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_unchanged(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.nc.L1::evict_unchanged.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_unchanged_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.nc.L1::evict_unchanged.L2::64B.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_unchanged_L2_64B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_64B_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_unchanged_L2_64B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_unchanged.L2::64B.b8 %0, [%1];"
               : "=r"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_64B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_unchanged.L2::64B.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_unchanged_L2_64B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_64B_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_unchanged_L2_64B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.nc.L1::evict_unchanged.L2::64B.b16 %0, [%1];"
               : "=h"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_64B_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_unchanged.L2::64B.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_unchanged_L2_64B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_64B_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_unchanged_L2_64B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_unchanged.L2::64B.b32 %0, [%1];"
               : "=r"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_64B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_unchanged.L2::64B.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_unchanged_L2_64B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_64B_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_unchanged_L2_64B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.nc.L1::evict_unchanged.L2::64B.b64 %0, [%1];"
               : "=l"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_64B_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_unchanged.L2::64B.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_unchanged_L2_64B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_64B_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_unchanged_L2_64B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.nc.L1::evict_unchanged.L2::64B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_64B_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.nc.L1::evict_unchanged.L2::128B.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_unchanged_L2_128B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_128B_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_unchanged_L2_128B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_unchanged.L2::128B.b8 %0, [%1];"
               : "=r"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_128B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_unchanged.L2::128B.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_unchanged_L2_128B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_128B_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_unchanged_L2_128B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.nc.L1::evict_unchanged.L2::128B.b16 %0, [%1];"
               : "=h"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_128B_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_unchanged.L2::128B.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_unchanged_L2_128B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_128B_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_unchanged_L2_128B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_unchanged.L2::128B.b32 %0, [%1];"
               : "=r"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_128B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_unchanged.L2::128B.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_unchanged_L2_128B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_128B_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_unchanged_L2_128B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.nc.L1::evict_unchanged.L2::128B.b64 %0, [%1];"
               : "=l"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_128B_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_unchanged.L2::128B.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_unchanged_L2_128B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_128B_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_unchanged_L2_128B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.nc.L1::evict_unchanged.L2::128B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_128B_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.nc.L1::evict_unchanged.L2::256B.b8 dest, [addr]; // PTX ISA 74, SM_80
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_unchanged_L2_256B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_256B_is_not_supported_before_SM_80__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_unchanged_L2_256B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_unchanged.L2::256B.b8 %0, [%1];"
               : "=r"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_256B_is_not_supported_before_SM_80__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_unchanged.L2::256B.b16 dest, [addr]; // PTX ISA 74, SM_80
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_unchanged_L2_256B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_256B_is_not_supported_before_SM_80__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_unchanged_L2_256B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint16_t dest;
  asm volatile("ld.global.nc.L1::evict_unchanged.L2::256B.b16 %0, [%1];"
               : "=h"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_256B_is_not_supported_before_SM_80__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_unchanged.L2::256B.b32 dest, [addr]; // PTX ISA 74, SM_80
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_unchanged_L2_256B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_256B_is_not_supported_before_SM_80__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_unchanged_L2_256B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_unchanged.L2::256B.b32 %0, [%1];"
               : "=r"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_256B_is_not_supported_before_SM_80__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_unchanged.L2::256B.b64 dest, [addr]; // PTX ISA 74, SM_80
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_unchanged_L2_256B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_256B_is_not_supported_before_SM_80__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_unchanged_L2_256B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint64_t dest;
  asm volatile("ld.global.nc.L1::evict_unchanged.L2::256B.b64 %0, [%1];"
               : "=l"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_256B_is_not_supported_before_SM_80__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_unchanged.L2::256B.b128 dest, [addr]; // PTX ISA 83, SM_80
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_unchanged_L2_256B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_256B_is_not_supported_before_SM_80__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_unchanged_L2_256B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.nc.L1::evict_unchanged.L2::256B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_unchanged_L2_256B_is_not_supported_before_SM_80__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.nc.L1::evict_first.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_first(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_first_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_first(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_first.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_first_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_first.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_first(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_first_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_first(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.nc.L1::evict_first.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_first_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_first.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_first(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_first_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_first(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_first.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_first_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_first.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_first(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_first_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_first(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.nc.L1::evict_first.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_first_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_first.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_first(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_first_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_first(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.nc.L1::evict_first.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_first_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.nc.L1::evict_first.L2::64B.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_first_L2_64B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_first_L2_64B_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_first_L2_64B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_first.L2::64B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_first_L2_64B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_first.L2::64B.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_first_L2_64B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_first_L2_64B_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_first_L2_64B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.nc.L1::evict_first.L2::64B.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_first_L2_64B_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_first.L2::64B.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_first_L2_64B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_first_L2_64B_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_first_L2_64B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_first.L2::64B.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_first_L2_64B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_first.L2::64B.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_first_L2_64B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_first_L2_64B_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_first_L2_64B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.nc.L1::evict_first.L2::64B.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_first_L2_64B_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_first.L2::64B.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_first_L2_64B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_first_L2_64B_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_first_L2_64B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.nc.L1::evict_first.L2::64B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_first_L2_64B_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.nc.L1::evict_first.L2::128B.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_first_L2_128B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_first_L2_128B_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_first_L2_128B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_first.L2::128B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_first_L2_128B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_first.L2::128B.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_first_L2_128B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_first_L2_128B_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_first_L2_128B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.nc.L1::evict_first.L2::128B.b16 %0, [%1];"
               : "=h"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_first_L2_128B_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_first.L2::128B.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_first_L2_128B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_first_L2_128B_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_first_L2_128B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_first.L2::128B.b32 %0, [%1];"
               : "=r"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_first_L2_128B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_first.L2::128B.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_first_L2_128B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_first_L2_128B_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_first_L2_128B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.nc.L1::evict_first.L2::128B.b64 %0, [%1];"
               : "=l"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_first_L2_128B_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_first.L2::128B.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_first_L2_128B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_first_L2_128B_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_first_L2_128B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.nc.L1::evict_first.L2::128B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_first_L2_128B_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.nc.L1::evict_first.L2::256B.b8 dest, [addr]; // PTX ISA 74, SM_80
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_first_L2_256B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_first_L2_256B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_first.L2::256B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_first.L2::256B.b16 dest, [addr]; // PTX ISA 74, SM_80
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_first_L2_256B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_first_L2_256B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint16_t dest;
  asm volatile("ld.global.nc.L1::evict_first.L2::256B.b16 %0, [%1];"
               : "=h"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_first.L2::256B.b32 dest, [addr]; // PTX ISA 74, SM_80
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_first_L2_256B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_first_L2_256B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_first.L2::256B.b32 %0, [%1];"
               : "=r"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_first.L2::256B.b64 dest, [addr]; // PTX ISA 74, SM_80
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_first_L2_256B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_first_L2_256B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint64_t dest;
  asm volatile("ld.global.nc.L1::evict_first.L2::256B.b64 %0, [%1];"
               : "=l"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_first.L2::256B.b128 dest, [addr]; // PTX ISA 83, SM_80
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_first_L2_256B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_first_L2_256B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.nc.L1::evict_first.L2::256B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_first_L2_256B_is_not_supported_before_SM_80__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.nc.L1::evict_last.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_last(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_last_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_last(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_last.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_last_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_last.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_last(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_last_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_last(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.nc.L1::evict_last.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_last_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_last.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_last(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_last_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_last(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_last.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_last_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_last.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_last(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_last_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_last(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.nc.L1::evict_last.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_last_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_last.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_last(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_last_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_last(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.nc.L1::evict_last.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_last_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.nc.L1::evict_last.L2::64B.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_last_L2_64B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_last_L2_64B_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_last_L2_64B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_last.L2::64B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_last_L2_64B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_last.L2::64B.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_last_L2_64B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_last_L2_64B_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_last_L2_64B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.nc.L1::evict_last.L2::64B.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_last_L2_64B_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_last.L2::64B.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_last_L2_64B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_last_L2_64B_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_last_L2_64B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_last.L2::64B.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_last_L2_64B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_last.L2::64B.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_last_L2_64B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_last_L2_64B_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_last_L2_64B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.nc.L1::evict_last.L2::64B.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_last_L2_64B_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_last.L2::64B.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_last_L2_64B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_last_L2_64B_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_last_L2_64B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.nc.L1::evict_last.L2::64B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_last_L2_64B_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.nc.L1::evict_last.L2::128B.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_last_L2_128B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_last_L2_128B_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_last_L2_128B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_last.L2::128B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_last_L2_128B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_last.L2::128B.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_last_L2_128B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_last_L2_128B_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_last_L2_128B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.nc.L1::evict_last.L2::128B.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_last_L2_128B_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_last.L2::128B.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_last_L2_128B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_last_L2_128B_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_last_L2_128B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_last.L2::128B.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_last_L2_128B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_last.L2::128B.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_last_L2_128B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_last_L2_128B_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_last_L2_128B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.nc.L1::evict_last.L2::128B.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_last_L2_128B_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_last.L2::128B.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_last_L2_128B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_last_L2_128B_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_last_L2_128B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.nc.L1::evict_last.L2::128B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_last_L2_128B_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.nc.L1::evict_last.L2::256B.b8 dest, [addr]; // PTX ISA 74, SM_80
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_last_L2_256B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_evict_last_L2_256B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_last.L2::256B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_last.L2::256B.b16 dest, [addr]; // PTX ISA 74, SM_80
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_last_L2_256B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_evict_last_L2_256B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint16_t dest;
  asm volatile("ld.global.nc.L1::evict_last.L2::256B.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_last.L2::256B.b32 dest, [addr]; // PTX ISA 74, SM_80
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_last_L2_256B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_evict_last_L2_256B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::evict_last.L2::256B.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_last.L2::256B.b64 dest, [addr]; // PTX ISA 74, SM_80
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_last_L2_256B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_evict_last_L2_256B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint64_t dest;
  asm volatile("ld.global.nc.L1::evict_last.L2::256B.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::evict_last.L2::256B.b128 dest, [addr]; // PTX ISA 83, SM_80
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_last_L2_256B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_evict_last_L2_256B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.nc.L1::evict_last.L2::256B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_evict_last_L2_256B_is_not_supported_before_SM_80__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.nc.L1::no_allocate.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_no_allocate(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_no_allocate_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_no_allocate(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::no_allocate.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_no_allocate_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::no_allocate.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_no_allocate(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_no_allocate_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_no_allocate(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.nc.L1::no_allocate.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_no_allocate_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::no_allocate.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_no_allocate(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_no_allocate_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_no_allocate(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::no_allocate.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_no_allocate_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::no_allocate.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_no_allocate(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_no_allocate_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_no_allocate(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.nc.L1::no_allocate.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_no_allocate_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::no_allocate.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_no_allocate(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_no_allocate_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_no_allocate(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.nc.L1::no_allocate.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_no_allocate_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.nc.L1::no_allocate.L2::64B.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_no_allocate_L2_64B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_no_allocate_L2_64B_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_no_allocate_L2_64B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::no_allocate.L2::64B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_no_allocate_L2_64B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::no_allocate.L2::64B.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_no_allocate_L2_64B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_no_allocate_L2_64B_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_no_allocate_L2_64B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.nc.L1::no_allocate.L2::64B.b16 %0, [%1];" : "=h"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_no_allocate_L2_64B_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::no_allocate.L2::64B.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_no_allocate_L2_64B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_no_allocate_L2_64B_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_no_allocate_L2_64B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::no_allocate.L2::64B.b32 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_no_allocate_L2_64B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::no_allocate.L2::64B.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_no_allocate_L2_64B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_no_allocate_L2_64B_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_no_allocate_L2_64B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.nc.L1::no_allocate.L2::64B.b64 %0, [%1];" : "=l"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_no_allocate_L2_64B_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::no_allocate.L2::64B.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_no_allocate_L2_64B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_no_allocate_L2_64B_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_no_allocate_L2_64B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.nc.L1::no_allocate.L2::64B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_no_allocate_L2_64B_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.nc.L1::no_allocate.L2::128B.b8 dest, [addr]; // PTX ISA 74, SM_75
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_no_allocate_L2_128B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_no_allocate_L2_128B_is_not_supported_before_SM_75__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_no_allocate_L2_128B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::no_allocate.L2::128B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_no_allocate_L2_128B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::no_allocate.L2::128B.b16 dest, [addr]; // PTX ISA 74, SM_75
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_no_allocate_L2_128B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_no_allocate_L2_128B_is_not_supported_before_SM_75__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_no_allocate_L2_128B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint16_t dest;
  asm volatile("ld.global.nc.L1::no_allocate.L2::128B.b16 %0, [%1];"
               : "=h"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_no_allocate_L2_128B_is_not_supported_before_SM_75__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::no_allocate.L2::128B.b32 dest, [addr]; // PTX ISA 74, SM_75
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_no_allocate_L2_128B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_no_allocate_L2_128B_is_not_supported_before_SM_75__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_no_allocate_L2_128B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::no_allocate.L2::128B.b32 %0, [%1];"
               : "=r"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_no_allocate_L2_128B_is_not_supported_before_SM_75__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::no_allocate.L2::128B.b64 dest, [addr]; // PTX ISA 74, SM_75
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_no_allocate_L2_128B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_no_allocate_L2_128B_is_not_supported_before_SM_75__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_no_allocate_L2_128B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  std::uint64_t dest;
  asm volatile("ld.global.nc.L1::no_allocate.L2::128B.b64 %0, [%1];"
               : "=l"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_no_allocate_L2_128B_is_not_supported_before_SM_75__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::no_allocate.L2::128B.b128 dest, [addr]; // PTX ISA 83, SM_75
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_no_allocate_L2_128B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_no_allocate_L2_128B_is_not_supported_before_SM_75__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_no_allocate_L2_128B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 750
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.nc.L1::no_allocate.L2::128B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_no_allocate_L2_128B_is_not_supported_before_SM_75__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

/*
// ld.global.nc.L1::no_allocate.L2::256B.b8 dest, [addr]; // PTX ISA 74, SM_80
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_no_allocate_L2_256B(
  const B8* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
template <typename B8, std::enable_if_t<sizeof(B8) == 1, bool> = true>
__device__ static inline B8 ld_global_nc_L1_no_allocate_L2_256B(const B8* addr)
{
  static_assert(sizeof(B8) == 1, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.b8 %0, [%1];" : "=r"(dest) : "l"(__as_ptr_gmem(addr)) : "memory");
  return __u32_as_b8<B8>(dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B8*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::no_allocate.L2::256B.b16 dest, [addr]; // PTX ISA 74, SM_80
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_no_allocate_L2_256B(
  const B16* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
template <typename B16, std::enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline B16 ld_global_nc_L1_no_allocate_L2_256B(const B16* addr)
{
  static_assert(sizeof(B16) == 2, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint16_t dest;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.b16 %0, [%1];"
               : "=h"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B16*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
  std::uint16_t err_out_var = 0;
  return *reinterpret_cast<B16*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::no_allocate.L2::256B.b32 dest, [addr]; // PTX ISA 74, SM_80
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_no_allocate_L2_256B(
  const B32* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
template <typename B32, std::enable_if_t<sizeof(B32) == 4, bool> = true>
__device__ static inline B32 ld_global_nc_L1_no_allocate_L2_256B(const B32* addr)
{
  static_assert(sizeof(B32) == 4, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint32_t dest;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.b32 %0, [%1];"
               : "=r"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B32*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
  std::uint32_t err_out_var = 0;
  return *reinterpret_cast<B32*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::no_allocate.L2::256B.b64 dest, [addr]; // PTX ISA 74, SM_80
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_no_allocate_L2_256B(
  const B64* addr);
*/
#if __libcuda_ptx_isa >= 740
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
template <typename B64, std::enable_if_t<sizeof(B64) == 8, bool> = true>
__device__ static inline B64 ld_global_nc_L1_no_allocate_L2_256B(const B64* addr)
{
  static_assert(sizeof(B64) == 8, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  std::uint64_t dest;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.b64 %0, [%1];"
               : "=l"(dest)
               : "l"(__as_ptr_gmem(addr))
               : "memory");
  return *reinterpret_cast<B64*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
  std::uint64_t err_out_var = 0;
  return *reinterpret_cast<B64*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 740

/*
// ld.global.nc.L1::no_allocate.L2::256B.b128 dest, [addr]; // PTX ISA 83, SM_80
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_no_allocate_L2_256B(
  const B128* addr);
*/
#if __libcuda_ptx_isa >= 830
extern "C" __device__ void __cuda_ptx_ld_global_nc_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
template <typename B128, std::enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline B128 ld_global_nc_L1_no_allocate_L2_256B(const B128* addr)
{
  static_assert(sizeof(B128) == 16, "");
#  if defined(_NVHPC_CUDA) || __CUDA_ARCH__ >= 800
  long2 dest;
  asm volatile(
    "{\n\t .reg .b128 B128_dest; \n\t"
    "ld.global.nc.L1::no_allocate.L2::256B.b128 B128_dest, [%2];\n\t"
    "mov.b128 {%0, %1}, B128_dest; \n"
    "}"
    : "=l"(dest.x), "=l"(dest.y)
    : "l"(__as_ptr_gmem(addr))
    : "memory");
  return *reinterpret_cast<B128*>(&dest);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_ld_global_nc_L1_no_allocate_L2_256B_is_not_supported_before_SM_80__();
  long2 err_out_var{0, 0};
  return *reinterpret_cast<B128*>(&err_out_var);
#  endif
}
#endif // __libcuda_ptx_isa >= 830

} // namespace cuda_ptx
