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

__global__ void test_tcgen05_st(void** fn_ptr)
{
#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x64b.x1.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[1])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x64b.x1.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[1])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x64b.x1.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[1])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x64b.x1.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[1])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x64b.x1.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[1])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x64b.x1.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[1])>(cuda::ptx::tcgen05_st_16x64b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x64b.x1.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[1])>(
          cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x64b.x1.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[1])>(
          cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x64b.x1.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[1])>(
          cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x64b.x1.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[1])>(
          cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x64b.x1.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[1])>(
          cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x64b.x1.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[1])>(
          cuda::ptx::tcgen05_st_16x64b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x64b.x2.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x64b.x2.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x64b.x2.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x64b.x2.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x64b.x2.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x64b.x2.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(cuda::ptx::tcgen05_st_16x64b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x64b.x2.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(
          cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x64b.x2.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(
          cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x64b.x2.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(
          cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x64b.x2.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(
          cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x64b.x2.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(
          cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x64b.x2.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(
          cuda::ptx::tcgen05_st_16x64b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x64b.x4.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x64b.x4.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x64b.x4.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x64b.x4.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x64b.x4.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x64b.x4.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(cuda::ptx::tcgen05_st_16x64b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x64b.x4.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(
          cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x64b.x4.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(
          cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x64b.x4.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(
          cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x64b.x4.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(
          cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x64b.x4.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(
          cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x64b.x4.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(
          cuda::ptx::tcgen05_st_16x64b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x64b.x8.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x64b.x8.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x64b.x8.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x64b.x8.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x64b.x8.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x64b.x8.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(cuda::ptx::tcgen05_st_16x64b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x64b.x8.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(
          cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x64b.x8.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(
          cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x64b.x8.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(
          cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x64b.x8.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(
          cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x64b.x8.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(
          cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x64b.x8.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(
          cuda::ptx::tcgen05_st_16x64b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x64b.x16.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x64b.x16.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x64b.x16.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x64b.x16.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x64b.x16.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x64b.x16.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(cuda::ptx::tcgen05_st_16x64b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x64b.x16.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x64b.x16.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x64b.x16.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x64b.x16.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x64b.x16.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x64b.x16.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x64b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x64b.x32.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x64b.x32.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x64b.x32.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x64b.x32.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x64b.x32.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x64b.x32.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(cuda::ptx::tcgen05_st_16x64b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x64b.x32.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x64b.x32.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x64b.x32.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x64b.x32.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x64b.x32.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x64b.x32.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x64b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x64b.x64.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x64b.x64.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x64b.x64.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x64b.x64.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x64b.x64.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x64b.x64.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(cuda::ptx::tcgen05_st_16x64b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x64b.x64.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x64b.x64.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x64b.x64.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x64b.x64.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x64b.x64.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x64b.x64.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x64b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x64b.x128.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x64b.x128.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x64b.x128.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x64b.x128.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x64b.x128.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(cuda::ptx::tcgen05_st_16x64b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x64b.x128.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x64b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x64b.x128.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x64b.x128.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x64b.x128.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x64b.x128.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x64b.x128.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x64b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x64b.x128.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x64b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x128b.x1.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x128b.x1.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x128b.x1.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x128b.x1.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x128b.x1.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x128b.x1.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(cuda::ptx::tcgen05_st_16x128b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x128b.x1.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(
          cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x128b.x1.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(
          cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x128b.x1.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(
          cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x128b.x1.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(
          cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x128b.x1.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(
          cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x128b.x1.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(
          cuda::ptx::tcgen05_st_16x128b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x128b.x2.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x128b.x2.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x128b.x2.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x128b.x2.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x128b.x2.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x128b.x2.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(cuda::ptx::tcgen05_st_16x128b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x128b.x2.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(
          cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x128b.x2.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(
          cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x128b.x2.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(
          cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x128b.x2.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(
          cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x128b.x2.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(
          cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x128b.x2.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(
          cuda::ptx::tcgen05_st_16x128b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x128b.x4.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x128b.x4.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x128b.x4.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x128b.x4.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x128b.x4.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x128b.x4.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(cuda::ptx::tcgen05_st_16x128b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x128b.x4.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(
          cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x128b.x4.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(
          cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x128b.x4.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(
          cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x128b.x4.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(
          cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x128b.x4.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(
          cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x128b.x4.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(
          cuda::ptx::tcgen05_st_16x128b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x128b.x8.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x128b.x8.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x128b.x8.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x128b.x8.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x128b.x8.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x128b.x8.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x128b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x128b.x8.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x128b.x8.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x128b.x8.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x128b.x8.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x128b.x8.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x128b.x8.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x128b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x128b.x16.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x128b.x16.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x128b.x16.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x128b.x16.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x128b.x16.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x128b.x16.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x128b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x128b.x16.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x128b.x16.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x128b.x16.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x128b.x16.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x128b.x16.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x128b.x16.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x128b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x128b.x32.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x128b.x32.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x128b.x32.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x128b.x32.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x128b.x32.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x128b.x32.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x128b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x128b.x32.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x128b.x32.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x128b.x32.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x128b.x32.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x128b.x32.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x128b.x32.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x128b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x128b.x64.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x128b.x64.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x128b.x64.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x128b.x64.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x128b.x64.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x128b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x128b.x64.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x128b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x128b.x64.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x128b.x64.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x128b.x64.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x128b.x64.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x128b.x64.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x128b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x128b.x64.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x128b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x256b.x1.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x256b.x1.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x256b.x1.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x256b.x1.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x256b.x1.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x256b.x1.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(cuda::ptx::tcgen05_st_16x256b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x256b.x1.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(
          cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x256b.x1.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(
          cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x256b.x1.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(
          cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x256b.x1.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(
          cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x256b.x1.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(
          cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x256b.x1.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(
          cuda::ptx::tcgen05_st_16x256b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x256b.x2.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x256b.x2.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x256b.x2.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x256b.x2.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x256b.x2.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x256b.x2.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(cuda::ptx::tcgen05_st_16x256b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x256b.x2.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(
          cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x256b.x2.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(
          cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x256b.x2.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(
          cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x256b.x2.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(
          cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x256b.x2.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(
          cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x256b.x2.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(
          cuda::ptx::tcgen05_st_16x256b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x256b.x4.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x256b.x4.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x256b.x4.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x256b.x4.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x256b.x4.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x256b.x4.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x256b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x256b.x4.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x256b.x4.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x256b.x4.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x256b.x4.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x256b.x4.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x256b.x4.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x256b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x256b.x8.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x256b.x8.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x256b.x8.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x256b.x8.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x256b.x8.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x256b.x8.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x256b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x256b.x8.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x256b.x8.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x256b.x8.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x256b.x8.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x256b.x8.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x256b.x8.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x256b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x256b.x16.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x256b.x16.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x256b.x16.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x256b.x16.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x256b.x16.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x256b.x16.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x256b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x256b.x16.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x256b.x16.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x256b.x16.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x256b.x16.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x256b.x16.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x256b.x16.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x256b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x256b.x32.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x256b.x32.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x256b.x32.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x256b.x32.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x256b.x32.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x256b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x256b.x32.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x256b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x256b.x32.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x256b.x32.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x256b.x32.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x256b.x32.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x256b.x32.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x256b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x256b.x32.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x256b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.32x32b.x1.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[1])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.32x32b.x1.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[1])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.32x32b.x1.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[1])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.32x32b.x1.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[1])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.32x32b.x1.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[1])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.32x32b.x1.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[1])>(cuda::ptx::tcgen05_st_32x32b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.32x32b.x1.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[1])>(
          cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.32x32b.x1.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[1])>(
          cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.32x32b.x1.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[1])>(
          cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.32x32b.x1.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[1])>(
          cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.32x32b.x1.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[1])>(
          cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.32x32b.x1.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[1])>(
          cuda::ptx::tcgen05_st_32x32b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.32x32b.x2.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.32x32b.x2.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.32x32b.x2.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.32x32b.x2.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.32x32b.x2.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.32x32b.x2.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(cuda::ptx::tcgen05_st_32x32b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.32x32b.x2.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(
          cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.32x32b.x2.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(
          cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.32x32b.x2.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(
          cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.32x32b.x2.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(
          cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.32x32b.x2.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(
          cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.32x32b.x2.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[2])>(
          cuda::ptx::tcgen05_st_32x32b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.32x32b.x4.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.32x32b.x4.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.32x32b.x4.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.32x32b.x4.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.32x32b.x4.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.32x32b.x4.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(cuda::ptx::tcgen05_st_32x32b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.32x32b.x4.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(
          cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.32x32b.x4.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(
          cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.32x32b.x4.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(
          cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.32x32b.x4.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(
          cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.32x32b.x4.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(
          cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.32x32b.x4.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[4])>(
          cuda::ptx::tcgen05_st_32x32b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.32x32b.x8.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.32x32b.x8.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.32x32b.x8.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.32x32b.x8.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.32x32b.x8.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.32x32b.x8.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(cuda::ptx::tcgen05_st_32x32b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.32x32b.x8.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(
          cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.32x32b.x8.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(
          cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.32x32b.x8.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(
          cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.32x32b.x8.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(
          cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.32x32b.x8.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(
          cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.32x32b.x8.unpack::16b.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[8])>(
          cuda::ptx::tcgen05_st_32x32b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.32x32b.x16.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.32x32b.x16.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.32x32b.x16.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.32x32b.x16.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.32x32b.x16.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.32x32b.x16.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(cuda::ptx::tcgen05_st_32x32b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.32x32b.x16.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.32x32b.x16.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.32x32b.x16.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.32x32b.x16.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.32x32b.x16.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.32x32b.x16.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_32x32b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.32x32b.x32.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.32x32b.x32.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.32x32b.x32.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.32x32b.x32.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.32x32b.x32.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.32x32b.x32.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(cuda::ptx::tcgen05_st_32x32b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.32x32b.x32.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.32x32b.x32.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.32x32b.x32.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.32x32b.x32.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.32x32b.x32.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.32x32b.x32.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_32x32b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.32x32b.x64.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.32x32b.x64.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.32x32b.x64.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.32x32b.x64.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.32x32b.x64.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.32x32b.x64.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(cuda::ptx::tcgen05_st_32x32b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.32x32b.x64.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.32x32b.x64.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.32x32b.x64.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.32x32b.x64.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.32x32b.x64.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.32x32b.x64.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_32x32b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.32x32b.x128.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.32x32b.x128.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.32x32b.x128.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.32x32b.x128.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.32x32b.x128.b32 [taddr], values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(cuda::ptx::tcgen05_st_32x32b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.32x32b.x128.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_32x32b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.32x32b.x128.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.32x32b.x128.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.32x32b.x128.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.32x32b.x128.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.32x32b.x128.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_32x32b_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.32x32b.x128.unpack::16b.b32 [taddr], values;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::uint32_t, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_32x32b_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x1.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[1])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x1.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[1])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x1.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[1])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x1.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[1])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x1.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[1])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x1.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[1])>(
            cuda::ptx::tcgen05_st_16x32bx2));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x1.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[1])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x1.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[1])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x1.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[1])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x1.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[1])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x1.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[1])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x1.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[1])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x2.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[2])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x2.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[2])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x2.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[2])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x2.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[2])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x2.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[2])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x2.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[2])>(
            cuda::ptx::tcgen05_st_16x32bx2));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x2.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[2])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x2.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[2])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x2.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[2])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x2.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[2])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x2.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[2])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x2.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[2])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x4.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[4])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x4.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[4])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x4.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[4])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x4.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[4])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x4.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[4])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x4.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[4])>(
            cuda::ptx::tcgen05_st_16x32bx2));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x4.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[4])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x4.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[4])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x4.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[4])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x4.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[4])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x4.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[4])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x4.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[4])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x8.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[8])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x8.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[8])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x8.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[8])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x8.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[8])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x8.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[8])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x8.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[8])>(
            cuda::ptx::tcgen05_st_16x32bx2));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x8.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[8])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x8.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[8])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x8.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[8])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x8.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[8])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x8.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[8])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x8.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[8])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x16.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x16.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x16.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x16.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x16.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x16.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x32bx2));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x16.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x16.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x16.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x16.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x16.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x16.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[16])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x32.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x32.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x32.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x32.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x32.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x32.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x32bx2));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x32.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x32.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x32.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x32.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x32.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x32.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[32])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x64.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x64.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x64.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x64.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x64.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x64.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x32bx2));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x64.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x64.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x64.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x64.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x64.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x64.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[64])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x128.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x128.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x128.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x128.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x128.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x32bx2));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x128.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x32bx2));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x128.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x128.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x128.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x128.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x128.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.st.sync.aligned.16x32bx2.x128.unpack::16b.b32 [taddr], immHalfSplitoff, values;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::uint32_t, cuda::ptx::n32_t<0>, const cuda::std::int32_t (&)[128])>(
            cuda::ptx::tcgen05_st_16x32bx2_unpack_16b));));
#endif // __cccl_ptx_isa >= 860
}
