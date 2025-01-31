// This file was automatically generated. Do not edit.

// We use a special strategy to force the generation of the PTX. This is mainly
// a fight against dead-code-elimination in the NVVM layer.
//
// The reason we need this strategy is because certain older versions of ptxas
// segfault when a non-sensical sequence of PTX is generated. So instead, we try
// to force the instantiation and compilation to PTX of all the overloads of the
// PTX wrapping functions.
//
// We do this by writing a function pointer of each overload to the kernel
// parameter `fn_ptr`.
//
// Because `fn_ptr` is possibly visible outside this translation unit, the
// compiler must compile all the functions which are stored.

__global__ void test_bfind(void** fn_ptr)
{
#if __cccl_ptx_isa >= 200
  NV_IF_TARGET(NV_PROVIDES_SM_50,
               (
                   // bfind.u32 dest, a_reg;
                   * fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)(uint32_t)>(cuda::ptx::bfind));));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  NV_IF_TARGET(
    NV_PROVIDES_SM_50,
    (
        // bfind.shiftamt.u32 dest, a_reg;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)(uint32_t)>(cuda::ptx::bfind_shiftamt));));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  NV_IF_TARGET(NV_PROVIDES_SM_50,
               (
                   // bfind.u64 dest, a_reg;
                   * fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)(uint64_t)>(cuda::ptx::bfind));));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  NV_IF_TARGET(
    NV_PROVIDES_SM_50,
    (
        // bfind.shiftamt.u64 dest, a_reg;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)(uint64_t)>(cuda::ptx::bfind_shiftamt));));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  NV_IF_TARGET(NV_PROVIDES_SM_50,
               (
                   // bfind.s32 dest, a_reg;
                   * fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)(int32_t)>(cuda::ptx::bfind));));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  NV_IF_TARGET(
    NV_PROVIDES_SM_50,
    (
        // bfind.shiftamt.s32 dest, a_reg;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)(int32_t)>(cuda::ptx::bfind_shiftamt));));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  NV_IF_TARGET(NV_PROVIDES_SM_50,
               (
                   // bfind.s64 dest, a_reg;
                   * fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)(int64_t)>(cuda::ptx::bfind));));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  NV_IF_TARGET(
    NV_PROVIDES_SM_50,
    (
        // bfind.shiftamt.s64 dest, a_reg;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<uint32_t (*)(int64_t)>(cuda::ptx::bfind_shiftamt));));
#endif // __cccl_ptx_isa >= 200
}
