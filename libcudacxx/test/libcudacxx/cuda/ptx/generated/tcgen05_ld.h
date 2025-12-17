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

__global__ void test_tcgen05_ld(void** fn_ptr)
{
#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x1.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x1.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x1.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x1.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x1.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x1.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x1.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x1.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x1.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x1.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x1.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x1.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x2.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x2.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x2.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x2.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x2.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x2.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x2.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x2.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x2.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x2.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x2.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x2.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x4.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x4.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x4.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x4.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x4.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x4.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x4.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x4.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x4.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x4.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x4.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x4.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x8.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x8.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x8.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x8.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x8.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x8.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x8.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x8.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x8.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x8.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x8.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x8.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x16.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x16.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x16.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x16.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x16.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x16.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x16.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x16.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x16.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x16.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x16.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x16.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x32.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x32.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x32.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x32.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x32.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x32.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x32.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x32.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x32.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x32.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x32.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x32.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x64.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x64.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x64.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x64.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x64.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x64.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x64.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x64.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x64.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x64.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x64.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x64.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x128.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x128.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x128.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x128.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x128.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x128.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x64b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x128.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x128.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x64b.x128.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x128.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x128.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x64b.x128.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x64b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x1.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x1.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x1.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x1.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x1.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x1.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x1.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x1.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x1.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x1.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x1.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x1.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x2.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x2.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x2.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x2.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x2.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x2.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x2.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x2.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x2.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x2.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x2.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x2.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x4.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x4.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x4.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x4.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x4.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x4.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x8.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x8.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x8.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x8.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x8.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x8.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x8.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x8.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x8.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x8.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x8.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x8.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x16.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x16.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x16.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x16.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x16.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x16.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x16.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x16.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x16.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x16.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x16.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x16.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x32.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x32.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x32.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x32.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x32.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x32.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x32.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x32.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x32.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x32.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x32.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x32.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x64.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x64.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x64.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x64.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x64.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x64.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x128b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x64.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x64.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x128b.x64.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x64.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x64.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x128b.x64.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x128b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x1.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x1.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x1.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x1.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x1.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x1.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x1.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x1.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x1.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x1.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x1.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x1.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x2.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x2.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x2.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x2.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x2.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x2.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x2.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x2.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x2.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x2.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x2.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x2.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x4.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x4.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x4.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x4.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x4.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x4.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x4.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x4.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x4.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x4.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x4.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x4.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x8.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x8.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x8.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x8.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x8.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x8.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x8.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x8.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x8.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x8.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x8.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x8.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x16.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x16.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x16.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x16.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x16.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x16.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x16.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x16.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x16.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x16.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x16.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x16.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x32.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x32.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x32.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x32.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x32.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x32.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_16x256b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x32.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x32.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x256b.x32.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x32.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x32.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x256b.x32.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_16x256b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x1.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x1.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x1.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x1.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x1.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x1.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x1.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x1.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x1.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x1.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x1.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x1.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x2.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x2.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x2.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x2.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x2.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x2.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x2.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x2.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x2.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x2.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x2.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x2.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x4.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x4.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x4.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x4.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x4.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x4.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x4.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x4.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x4.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x4.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x4.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x4.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x8.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x8.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x8.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x8.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x8.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x8.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x8.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x8.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x8.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x8.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x8.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x8.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x16.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x16.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x16.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x16.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x16.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x16.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x16.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x16.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x16.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x16.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x16.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x16.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x32.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x32.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x32.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x32.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x32.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x32.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x32.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x32.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x32.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x32.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x32.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x32.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x64.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x64.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x64.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x64.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x64.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x64.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x64.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x64.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x64.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x64.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x64.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x64.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x128.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x128.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x128.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x128.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x128.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x128.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(cuda::ptx::tcgen05_ld_32x32b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x128.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x128.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.32x32b.x128.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x128.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x128.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.32x32b.x128.pack::16b.b32 out, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
          cuda::ptx::tcgen05_ld_32x32b_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x1.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x1.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x1.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x1.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x1.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x1.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x1.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x1.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x1.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x1.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x1.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x1.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[1], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x2.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x2.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x2.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x2.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x2.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x2.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x2.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x2.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x2.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x2.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x2.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x2.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x4.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x4.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x4.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x4.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x4.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x4.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x4.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x4.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x4.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x4.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x4.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x4.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x8.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x8.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x8.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x8.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x8.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x8.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x8.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x8.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x8.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x8.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x8.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x8.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x16.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x16.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x16.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x16.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x16.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x16.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x16.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x16.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x16.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x16.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x16.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x16.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x32.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x32.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x32.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x32.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x32.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x32.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x32.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x32.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x32.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x32.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x32.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x32.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x64.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x64.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x64.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x64.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x64.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x64.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x64.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x64.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x64.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x64.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x64.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x64.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x128.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x128.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x128.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x128.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x128.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x128.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x128.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x128.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x128.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x128.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x128.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.sync.aligned.16x32bx2.x128.pack::16b.b32 out, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_16x32bx2_pack_16b));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.u32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.u32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.u32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.u32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.u32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.u32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.u32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.u32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.s32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_min_t, cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.s32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_min_t, cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.s32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_min_t, cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.s32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_min_t, cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.s32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_max_t, cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.s32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_max_t, cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.s32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_max_t, cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.s32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_max_t, cuda::std::int32_t (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.f32.min.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.f32.min.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.f32.min.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.f32.min.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.f32.max.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.f32.max.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.f32.max.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.f32.max.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.f32.min out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.f32.min out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.f32.min out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.f32.min out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.f32.max out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.f32.max out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.f32.max out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x2.f32.max out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[2], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.u32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.u32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.u32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.u32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.u32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.u32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.u32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.u32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.s32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_min_t, cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.s32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_min_t, cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.s32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_min_t, cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.s32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_min_t, cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.s32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_max_t, cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.s32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_max_t, cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.s32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_max_t, cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.s32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_max_t, cuda::std::int32_t (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.f32.min.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.f32.min.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.f32.min.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.f32.min.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.f32.max.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.f32.max.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.f32.max.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.f32.max.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.f32.min out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.f32.min out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.f32.min out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.f32.min out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.f32.max out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.f32.max out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.f32.max out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x4.f32.max out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[4], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.u32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.u32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.u32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.u32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.u32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.u32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.u32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.u32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.s32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_min_t, cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.s32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_min_t, cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.s32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_min_t, cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.s32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_min_t, cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.s32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_max_t, cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.s32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_max_t, cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.s32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_max_t, cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.s32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_max_t, cuda::std::int32_t (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.f32.min.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.f32.min.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.f32.min.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.f32.min.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.f32.max.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.f32.max.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.f32.max.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.f32.max.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.f32.min out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.f32.min out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.f32.min out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.f32.min out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.f32.max out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.f32.max out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.f32.max out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x8.f32.max out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[8], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.u32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.u32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.u32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.u32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.u32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.u32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.u32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.u32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.s32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_min_t, cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.s32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_min_t, cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.s32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_min_t, cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.s32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_min_t, cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.s32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_max_t, cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.s32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_max_t, cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.s32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_max_t, cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.s32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_max_t, cuda::std::int32_t (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.f32.min.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.f32.min.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.f32.min.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.f32.min.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.f32.max.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.f32.max.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.f32.max.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.f32.max.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.f32.min out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.f32.min out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.f32.min out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.f32.min out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.f32.max out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.f32.max out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.f32.max out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x16.f32.max out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[16], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.u32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.u32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.u32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.u32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.u32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.u32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.u32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.u32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.s32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_min_t, cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.s32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_min_t, cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.s32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_min_t, cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.s32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_min_t, cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.s32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_max_t, cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.s32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_max_t, cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.s32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_max_t, cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.s32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_max_t, cuda::std::int32_t (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.f32.min.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.f32.min.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.f32.min.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.f32.min.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.f32.max.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.f32.max.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.f32.max.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.f32.max.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.f32.min out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.f32.min out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.f32.min out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.f32.min out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.f32.max out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.f32.max out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.f32.max out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x32.f32.max out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[32], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.u32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.u32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.u32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.u32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.u32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.u32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.u32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.u32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.s32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_min_t, cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.s32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_min_t, cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.s32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_min_t, cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.s32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_min_t, cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.s32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_max_t, cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.s32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_max_t, cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.s32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_max_t, cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.s32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_max_t, cuda::std::int32_t (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.f32.min.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.f32.min.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.f32.min.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.f32.min.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.f32.max.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.f32.max.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.f32.max.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.f32.max.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.f32.min out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.f32.min out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.f32.min out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.f32.min out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.f32.max out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.f32.max out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.f32.max out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x64.f32.max out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[64], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.u32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.u32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.u32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.u32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.u32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.u32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.u32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.u32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.s32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_min_t, cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.s32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_min_t, cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.s32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_min_t, cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.s32.min out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_min_t, cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.s32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_max_t, cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.s32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_max_t, cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.s32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_max_t, cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.s32.max out, redval, [taddr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(cuda::ptx::op_max_t, cuda::std::int32_t (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.f32.min.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.f32.min.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.f32.min.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.f32.min.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.f32.max.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.f32.max.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.f32.max.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.f32.max.abs out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b_abs));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.f32.min out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.f32.min out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.f32.min out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.f32.min out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.f32.max out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.f32.max out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.f32.max out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.32x32b.x128.f32.max out, redval, [taddr];
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[128], cuda::std::uint32_t)>(
            cuda::ptx::tcgen05_ld_red_32x32b));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.u32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.u32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.u32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.u32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.u32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.u32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.u32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.u32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.s32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.s32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.s32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.s32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.s32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.s32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.s32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.s32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.f32.min.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.f32.min.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.f32.min.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.f32.min.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.f32.max.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.f32.max.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.f32.max.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.f32.max.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.f32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.f32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.f32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.f32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.f32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.f32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.f32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x2.f32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[2], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.u32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.u32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.u32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.u32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.u32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.u32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.u32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.u32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.s32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.s32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.s32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.s32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.s32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.s32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.s32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.s32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.min.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.min.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.min.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.min.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.max.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.max.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.max.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.max.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[4], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.u32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.u32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.u32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.u32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.u32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.u32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.u32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.u32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.s32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.s32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.s32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.s32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.s32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.s32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.s32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.s32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.f32.min.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.f32.min.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.f32.min.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.f32.min.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.f32.max.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.f32.max.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.f32.max.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.f32.max.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.f32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.f32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.f32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.f32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.f32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.f32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.f32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x8.f32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[8], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.u32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.u32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.u32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.u32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.u32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.u32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.u32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.u32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.s32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.s32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.s32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.s32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.s32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.s32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.s32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.s32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.f32.min.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.f32.min.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.f32.min.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.f32.min.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.f32.max.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.f32.max.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.f32.max.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.f32.max.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.f32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.f32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.f32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.f32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.f32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.f32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.f32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x16.f32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[16], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.u32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.u32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.u32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.u32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.u32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.u32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.u32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.u32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.s32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.s32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.s32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.s32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.s32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.s32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.s32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.s32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.min.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.min.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.min.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.min.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.max.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.max.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.max.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.max.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x32.f32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[32], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.u32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.u32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.u32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.u32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.u32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.u32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.u32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.u32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.s32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.s32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.s32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.s32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.s32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.s32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.s32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.s32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.f32.min.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.f32.min.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.f32.min.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.f32.min.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.f32.max.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.f32.max.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.f32.max.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.f32.max.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.f32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.f32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.f32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.f32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.f32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.f32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.f32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x64.f32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[64], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.u32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.u32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.u32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.u32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_min_t, cuda::std::uint32_t (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.u32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.u32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.u32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.u32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::uint32_t (*)(
            cuda::ptx::op_max_t, cuda::std::uint32_t (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.s32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.s32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.s32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.s32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_min_t, cuda::std::int32_t (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.s32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.s32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.s32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.s32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<cuda::std::int32_t (*)(
            cuda::ptx::op_max_t, cuda::std::int32_t (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.f32.min.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.f32.min.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.f32.min.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.f32.min.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.f32.max.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.f32.max.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.f32.max.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.f32.max.abs out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2_abs));));
#endif // __cccl_ptx_isa >= 880

#if __cccl_ptx_isa >= 880
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.f32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.f32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.f32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.f32.min out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_min_t, float (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.f32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.f32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.f32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.ld.red.sync.aligned.16x32bx2.x128.f32.max out, redval, [taddr], immHalfSplitoff;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<float (*)(cuda::ptx::op_max_t, float (&out)[128], cuda::std::uint32_t, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tcgen05_ld_red_16x32bx2));));
#endif // __cccl_ptx_isa >= 880
}
