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

__global__ void test_prmt(void** fn_ptr)
{
#if __cccl_ptx_isa >= 200
  NV_IF_TARGET(
    NV_PROVIDES_SM_50,
    (
        // prmt.b32 dest, a_reg, b_reg, c_reg;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<uint32_t (*)(cuda::std::int32_t, cuda::std::int32_t, cuda::std::uint32_t)>(cuda::ptx::prmt));));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  NV_IF_TARGET(
    NV_PROVIDES_SM_50,
    (
        // prmt.b32.f4e dest, a_reg, b_reg, c_reg;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<uint32_t (*)(cuda::std::int32_t, cuda::std::int32_t, cuda::std::uint32_t)>(
            cuda::ptx::prmt_f4e));));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  NV_IF_TARGET(
    NV_PROVIDES_SM_50,
    (
        // prmt.b32.b4e dest, a_reg, b_reg, c_reg;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<uint32_t (*)(cuda::std::int32_t, cuda::std::int32_t, cuda::std::uint32_t)>(
            cuda::ptx::prmt_b4e));));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  NV_IF_TARGET(
    NV_PROVIDES_SM_50,
    (
        // prmt.b32.rc8 dest, a_reg, b_reg, c_reg;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<uint32_t (*)(cuda::std::int32_t, cuda::std::int32_t, cuda::std::uint32_t)>(
            cuda::ptx::prmt_rc8));));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  NV_IF_TARGET(
    NV_PROVIDES_SM_50,
    (
        // prmt.b32.ecl dest, a_reg, b_reg, c_reg;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<uint32_t (*)(cuda::std::int32_t, cuda::std::int32_t, cuda::std::uint32_t)>(
            cuda::ptx::prmt_ecl));));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  NV_IF_TARGET(
    NV_PROVIDES_SM_50,
    (
        // prmt.b32.ecr dest, a_reg, b_reg, c_reg;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<uint32_t (*)(cuda::std::int32_t, cuda::std::int32_t, cuda::std::uint32_t)>(
            cuda::ptx::prmt_ecr));));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  NV_IF_TARGET(
    NV_PROVIDES_SM_50,
    (
        // prmt.b32.rc16 dest, a_reg, b_reg, c_reg;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<uint32_t (*)(cuda::std::int32_t, cuda::std::int32_t, cuda::std::uint32_t)>(
            cuda::ptx::prmt_rc16));));
#endif // __cccl_ptx_isa >= 200
}
