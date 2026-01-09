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

__global__ void test_setmaxnreg(void** fn_ptr)
{
#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // setmaxnreg.inc.sync.aligned.u32 imm_reg_count;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::n32_t<32>)>(cuda::ptx::setmaxnreg_inc));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // setmaxnreg.inc.sync.aligned.u32 imm_reg_count;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::n32_t<32>)>(cuda::ptx::setmaxnreg_inc));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // setmaxnreg.inc.sync.aligned.u32 imm_reg_count;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::n32_t<32>)>(cuda::ptx::setmaxnreg_inc));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // setmaxnreg.inc.sync.aligned.u32 imm_reg_count;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::n32_t<32>)>(cuda::ptx::setmaxnreg_inc));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_120a,
    (
        // setmaxnreg.inc.sync.aligned.u32 imm_reg_count;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::n32_t<32>)>(cuda::ptx::setmaxnreg_inc));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_121a,
    (
        // setmaxnreg.inc.sync.aligned.u32 imm_reg_count;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::n32_t<32>)>(cuda::ptx::setmaxnreg_inc));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // setmaxnreg.inc.sync.aligned.u32 imm_reg_count;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::n32_t<32>)>(cuda::ptx::setmaxnreg_inc));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // setmaxnreg.inc.sync.aligned.u32 imm_reg_count;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::n32_t<32>)>(cuda::ptx::setmaxnreg_inc));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // setmaxnreg.inc.sync.aligned.u32 imm_reg_count;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::n32_t<32>)>(cuda::ptx::setmaxnreg_inc));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_120f,
    (
        // setmaxnreg.inc.sync.aligned.u32 imm_reg_count;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::n32_t<32>)>(cuda::ptx::setmaxnreg_inc));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_121f,
    (
        // setmaxnreg.inc.sync.aligned.u32 imm_reg_count;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::n32_t<32>)>(cuda::ptx::setmaxnreg_inc));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // setmaxnreg.dec.sync.aligned.u32 imm_reg_count;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::n32_t<32>)>(cuda::ptx::setmaxnreg_dec));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // setmaxnreg.dec.sync.aligned.u32 imm_reg_count;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::n32_t<32>)>(cuda::ptx::setmaxnreg_dec));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // setmaxnreg.dec.sync.aligned.u32 imm_reg_count;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::n32_t<32>)>(cuda::ptx::setmaxnreg_dec));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // setmaxnreg.dec.sync.aligned.u32 imm_reg_count;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::n32_t<32>)>(cuda::ptx::setmaxnreg_dec));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_120a,
    (
        // setmaxnreg.dec.sync.aligned.u32 imm_reg_count;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::n32_t<32>)>(cuda::ptx::setmaxnreg_dec));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_121a,
    (
        // setmaxnreg.dec.sync.aligned.u32 imm_reg_count;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::n32_t<32>)>(cuda::ptx::setmaxnreg_dec));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // setmaxnreg.dec.sync.aligned.u32 imm_reg_count;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::n32_t<32>)>(cuda::ptx::setmaxnreg_dec));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // setmaxnreg.dec.sync.aligned.u32 imm_reg_count;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::n32_t<32>)>(cuda::ptx::setmaxnreg_dec));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // setmaxnreg.dec.sync.aligned.u32 imm_reg_count;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::n32_t<32>)>(cuda::ptx::setmaxnreg_dec));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_120f,
    (
        // setmaxnreg.dec.sync.aligned.u32 imm_reg_count;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::n32_t<32>)>(cuda::ptx::setmaxnreg_dec));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_121f,
    (
        // setmaxnreg.dec.sync.aligned.u32 imm_reg_count;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::n32_t<32>)>(cuda::ptx::setmaxnreg_dec));));
#endif // __cccl_ptx_isa >= 800
}
