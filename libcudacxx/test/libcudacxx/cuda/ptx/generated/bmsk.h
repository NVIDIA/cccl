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

__global__ void test_bmsk(void** fn_ptr)
{
#if __cccl_ptx_isa >= 760
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // bmsk.clamp.b32 dest, a_reg, b_reg;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<uint32_t (*)(cuda::std::uint32_t, cuda::std::uint32_t)>(cuda::ptx::bmsk_clamp));));
#endif // __cccl_ptx_isa >= 760

#if __cccl_ptx_isa >= 760
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (
                   // bmsk.wrap.b32 dest, a_reg, b_reg;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<uint32_t (*)(cuda::std::uint32_t, cuda::std::uint32_t)>(cuda::ptx::bmsk_wrap));));
#endif // __cccl_ptx_isa >= 760
}
