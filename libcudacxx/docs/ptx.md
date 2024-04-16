---
title: PTX
has_children: true
has_toc: false
nav_order: 4
---

# PTX

The `cuda::ptx` namespace contains functions that map one-to-one to
[PTX instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html). These can be used for maximal control of the generated code, or to
experiment with new hardware features before a high-level C++ API is available.

### Versions and compatibility

The `cuda/ptx` header is intended to present a stable API (not ABI) within one
major version of the CTK on a best effort basis. This means that:

- All functions are marked static inline.

- The type of a function parameter can be changed to be more generic if
  that means that code that called the original version can still be
  compiled.

- Good exposure of the PTX should be high priority. If, at a new major
  version, we face a difficult choice between breaking backward-compatibility
  and an improvement of the PTX exposure, we will tend to the latter option
  more easily than in other parts of libcu++.

API stability is not taken to the extreme. Call functions like below to ensure
forward-compatibility:

```cuda
// Use arguments to drive overload resolution:
cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared, &bar, 1);

// Specifying templates directly is not forward-compatible, as order and number
// of template parameters may change in a minor release:
cuda::ptx::mbarrier_arrive_expect_tx<cuda::ptx::sem_release_t>(
  cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared, &bar, 1
);
```

**PTX ISA version and compute capability.** Each binding notes under which PTX
ISA version and SM version it may be used. Example:

```cuda
// mbarrier.arrive.shared::cta.b64 state, [addr]; // 1.  PTX ISA 70, SM_80
__device__ inline uint64_t mbarrier_arrive(
  cuda::ptx::sem_release_t sem,
  cuda::ptx::scope_cta_t scope,
  cuda::ptx::space_shared_t space,
  uint64_t* addr);
```

To check if the current compiler is recent enough, use:
```cuda
#if __cccl_ptx_isa >= 700
cuda::ptx::mbarrier_arrive(cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared, &bar, 1);
#endif
```

Ensure that you only call the function when compiling for a recent enough
compute capability (SM version), like this:
```cuda
NV_IF_TARGET(NV_PROVIDES_SM_80,(
  cuda::ptx::mbarrier_arrive(cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared, &bar, 1);
));
```

For more information on which compilers correspond to which PTX ISA, see the
[PTX ISA release
notes](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#release-notes).

## Instructions by section
### [Integer Arithmetic Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions)

| Instruction                              | Available in libcu++ |
|------------------------------------------|----------------------|
| [`sad`]                                  | No                   |
| [`div`]                                  | No                   |
| [`rem`]                                  | No                   |
| [`abs`]                                  | No                   |
| [`neg`]                                  | No                   |
| [`min`]                                  | No                   |
| [`max`]                                  | No                   |
| [`popc`]                                 | No                   |
| [`clz`]                                  | No                   |
| [`bfind`]                                | No                   |
| [`fns`]                                  | No                   |
| [`brev`]                                 | No                   |
| [`bfe`]                                  | No                   |
| [`bfi`]                                  | No                   |
| [`szext`]                                | No                   |
| [`bmsk`]                                 | No                   |
| [`dp4a`]                                 | No                   |
| [`dp2a`]                                 | No                   |

[`sad`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-sad
[`div`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-div
[`rem`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-rem
[`abs`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-abs
[`neg`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-neg
[`min`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-min
[`max`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-max
[`popc`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-popc
[`clz`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-clz
[`bfind`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfind
[`fns`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-fns
[`brev`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-brev
[`bfe`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfe
[`bfi`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfi
[`szext`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-szext
[`bmsk`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bmsk
[`dp4a`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-dp4a
[`dp2a`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-dp2a

### [Extended-Precision Integer Arithmetic Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-integer-arithmetic-instructions)

| Instruction                              | Available in libcu++ |
|------------------------------------------|----------------------|
| [`add.cc`]                               | No                   |
| [`addc`]                                 | No                   |
| [`sub.cc`]                               | No                   |
| [`subc`]                                 | No                   |
| [`mad.cc`]                               | No                   |
| [`madc`]                                 | No                   |

[`add.cc`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-add-cc
[`addc`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-addc
[`sub.cc`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-sub-cc
[`subc`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-subc
[`mad.cc`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-mad-cc
[`madc`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-madc

### [Floating-Point Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions)

| Instruction                              | Available in libcu++ |
|------------------------------------------|----------------------|
| [`testp`]                                | No                   |
| [`copysign`]                             | No                   |
| [`add`]                                  | No                   |
| [`sub`]                                  | No                   |
| [`mul`]                                  | No                   |
| [`fma`]                                  | No                   |
| [`mad`]                                  | No                   |
| [`div`]                                  | No                   |
| [`abs`]                                  | No                   |
| [`neg`]                                  | No                   |
| [`min`]                                  | No                   |
| [`max`]                                  | No                   |
| [`rcp`]                                  | No                   |
| [`rcp.approx.ftz.f64`]                   | No                   |
| [`sqrt`]                                 | No                   |
| [`rsqrt`]                                | No                   |
| [`rsqrt.approx.ftz.f64`]                 | No                   |
| [`sin`]                                  | No                   |
| [`cos`]                                  | No                   |
| [`lg2`]                                  | No                   |
| [`ex2`]                                  | No                   |
| [`tanh`]                                 | No                   |

[`testp`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-testp
[`copysign`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-copysign
[`add`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-add
[`sub`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-sub
[`mul`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-mul
[`fma`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-fma
[`mad`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-mad
[`div`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-div
[`abs`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-abs
[`neg`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-neg
[`min`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-min
[`max`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-max
[`rcp`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rcp
[`rcp.approx.ftz.f64`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rcp-approx-ftz-f64
[`sqrt`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-sqrt
[`rsqrt`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rsqrt
[`rsqrt.approx.ftz.f64`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rsqrt-approx-ftz-f64
[`sin`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-sin
[`cos`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-cos
[`lg2`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-lg2
[`ex2`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-ex2
[`tanh`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-tanh

### [Half Precision Floating-Point Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions)

| Instruction                              | Available in libcu++ |
|------------------------------------------|----------------------|
| [`add`]                                  | No                   |
| [`sub`]                                  | No                   |
| [`mul`]                                  | No                   |
| [`fma`]                                  | No                   |
| [`neg`]                                  | No                   |
| [`abs`]                                  | No                   |
| [`min`]                                  | No                   |
| [`max`]                                  | No                   |
| [`tanh`]                                 | No                   |
| [`ex2`]                                  | No                   |

[`add`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-add
[`sub`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-sub
[`mul`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-mul
[`fma`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-fma
[`neg`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-neg
[`abs`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-abs
[`min`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-min
[`max`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-max
[`tanh`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-tanh
[`ex2`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-ex2

### [Comparison and Selection Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions)

| Instruction                              | Available in libcu++ |
|------------------------------------------|----------------------|
| [`set`]                                  | No                   |
| [`setp`]                                 | No                   |
| [`selp`]                                 | No                   |
| [`slct`]                                 | No                   |

[`set`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-set
[`setp`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp
[`selp`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-selp
[`slct`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-slct

### [Half Precision Comparison Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-comparison-instructions)

| Instruction                              | Available in libcu++ |
|------------------------------------------|----------------------|
| [`set`]                                  | No                   |
| [`setp`]                                 | No                   |

[`set`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-comparison-instructions-set
[`setp`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-comparison-instructions-setp

### [Logic and Shift Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions)

| Instruction                              | Available in libcu++ |
|------------------------------------------|----------------------|
| [`and`]                                  | No                   |
| [`or`]                                   | No                   |
| [`xor`]                                  | No                   |
| [`not`]                                  | No                   |
| [`cnot`]                                 | No                   |
| [`lop3`]                                 | No                   |
| [`shf`]                                  | No                   |
| [`shl`]                                  | No                   |
| [`shr`]                                  | No                   |

[`and`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-and
[`or`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-or
[`xor`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-xor
[`not`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-not
[`cnot`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-cnot
[`lop3`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-lop3
[`shf`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-shf
[`shl`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-shl
[`shr`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-shr

### [Data Movement and Conversion Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions)

| Instruction                                       | Available in libcu++    |
|---------------------------------------------------|-------------------------|
| [`mov`]                                           | No                      |
| [`mov`]                                           | No                      |
| [`shfl (deprecated)`]                             | No                      |
| [`shfl.sync`]                                     | No                      |
| [`prmt`]                                          | No                      |
| [`ld`]                                            | No                      |
| [`ld.global.nc`]                                  | No                      |
| [`ldu`]                                           | No                      |
| [`st`]                                            | No                      |
| [`st.async`]                                      | [CTK 12.4, CCCL v2.3.0] |
| [`multimem.ld_reduce, multimem.st, multimem.red`] | No                      |
| [`prefetch, prefetchu`]                           | No                      |
| [`applypriority`]                                 | No                      |
| [`discard`]                                       | No                      |
| [`createpolicy`]                                  | No                      |
| [`isspacep`]                                      | No                      |
| [`cvta`]                                          | No                      |
| [`cvt`]                                           | No                      |
| [`cvt.pack`]                                      | No                      |
| [`mapa`]                                          | No                      |
| [`getctarank`]                                    | CTK-FUTURE, CCCL v2.4.0 |

[`mov`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mov-2
[`shfl (deprecated)`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-deprecated
[`shfl.sync`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync
[`prmt`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt
[`ld`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld
[`ld.global.nc`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld-global-nc
[`ldu`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ldu
[`st`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st
[`st.async`]: ptx/instructions/st.async.md
[`multimem.ld_reduce, multimem.st, multimem.red`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem-ld-reduce-multimem-st-multimem-red
[`prefetch, prefetchu`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prefetch-prefetchu
[`applypriority`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-applypriority
[`discard`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-discard
[`createpolicy`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-createpolicy
[`isspacep`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-isspacep
[`cvta`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvta
[`cvt`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt
[`cvt.pack`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt-pack
[`mapa`]: ptx/instructions/mapa.md
[`getctarank`]: ptx/instructions/getctarank.md

### [Data Movement and Conversion Instructions: Asynchronous copy](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy)

| Instruction                                 | Available in libcu++    |
|---------------------------------------------|-------------------------|
| [`cp.async`]                                | No                      |
| [`cp.async.commit_group`]                   | No                      |
| [`cp.async.wait_group / cp.async.wait_all`] | No                      |
| [`cp.async.bulk`]                           | CTK-FUTURE, CCCL v2.4.0 |
| [`cp.reduce.async.bulk`]                    | CTK-FUTURE, CCCL v2.4.0 |
| [`cp.async.bulk.prefetch`]                  | No                      |
| [`cp.async.bulk.tensor`]                    | CTK-FUTURE, CCCL v2.4.0 |
| [`cp.reduce.async.bulk.tensor`]             | CTK-FUTURE, CCCL v2.4.0 |
| [`cp.async.bulk.prefetch.tensor`]           | No                      |
| [`cp.async.bulk.commit_group`]              | CTK-FUTURE, CCCL v2.4.0 |
| [`cp.async.bulk.wait_group`]                | CTK-FUTURE, CCCL v2.4.0 |
| [`tensormap.replace`]                       | CTK-FUTURE, CCCL v2.4.0 |

[`cp.async`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async
[`cp.async.commit_group`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-commit-group
[`cp.async.wait_group / cp.async.wait_all`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group-cp-async-wait-all
[`cp.reduce.async.bulk`]: ptx/instructions/cp.reduce.async.bulk.md
[`cp.async.bulk`]: ptx/instructions/cp.async.bulk.md
[`cp.async.bulk.prefetch`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch
[`cp.async.bulk.tensor`]: ptx/instructions/cp.async.bulk.tensor.md
[`cp.reduce.async.bulk.tensor`]: ptx/instructions/cp.reduce.async.bulk.tensor.md
[`cp.async.bulk.prefetch.tensor`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch-tensor
[`cp.async.bulk.commit_group`]: ptx/instructions/cp.async.bulk.commit_group.md
[`cp.async.bulk.wait_group`]: ptx/instructions/cp.async.bulk.wait_group.md
[`tensormap.replace`]: ptx/instructions/tensormap.replace.md

### [Texture Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions)

| Instruction                              | Available in libcu++ |
|------------------------------------------|----------------------|
| [`tex`]                                  | No                   |
| [`tld4`]                                 | No                   |
| [`txq`]                                  | No                   |
| [`istypep`]                              | No                   |

[`tex`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions-tex
[`tld4`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions-tld4
[`txq`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions-txq
[`istypep`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions-istypep

### [Surface Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions)

| Instruction                              | Available in libcu++ |
|------------------------------------------|----------------------|
| [`suld`]                                 | No                   |
| [`sust`]                                 | No                   |
| [`sured`]                                | No                   |
| [`suq`]                                  | No                   |

[`suld`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-suld
[`sust`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-sust
[`sured`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-sured
[`suq`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-suq

### [Control Flow Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions)

| Instruction                              | Available in libcu++ |
|------------------------------------------|----------------------|
| [`{}`]                                   | No                   |
| [`@`]                                    | No                   |
| [`bra`]                                  | No                   |
| [`brx.idx`]                              | No                   |
| [`call`]                                 | No                   |
| [`ret`]                                  | No                   |
| [`exit`]                                 | No                   |

[`{}`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-curly-braces
[`@`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-at
[`bra`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-bra
[`brx.idx`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-brx-idx
[`call`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-call
[`ret`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-ret
[`exit`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-exit

### [Parallel Synchronization and Communication Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions)

| Instruction           | Available in libcu++    |
|-----------------------|-------------------------|
| [`bar, barrier`]      | No                      |
| [`bar.warp.sync`]     | No                      |
| [`barrier.cluster`]   | CTK-FUTURE, CCCL v2.4.0 |
| [`membar`]            | No                      |
| [`fence`]             | CTK-FUTURE, CCCL v2.4.0 |
| [`atom`]              | No                      |
| [`red`]               | No                      |
| [`red.async`]         | [CTK 12.4, CCCL v2.3.0] |
| [`vote (deprecated)`] | No                      |
| [`vote.sync`]         | No                      |
| [`match.sync`]        | No                      |
| [`activemask`]        | No                      |
| [`redux.sync`]        | No                      |
| [`griddepcontrol`]    | No                      |
| [`elect.sync`]        | No                      |

[`bar, barrier`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-barrier
[`bar.warp.sync`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-warp-sync
[`barrier.cluster`]: ptx/instructions/barrier.cluster.md
[`membar`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence
[`fence`]: ptx/instructions/fence.md
[`atom`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom
[`red`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-red
[`red.async`]: ptx/instructions/red.async.md
[`vote (deprecated)`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-vote-deprecated
[`vote.sync`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-vote-sync
[`match.sync`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-match-sync
[`activemask`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-activemask
[`redux.sync`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-redux-sync
[`griddepcontrol`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol
[`elect.sync`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-elect-sync

### [Parallel Synchronization and Communication Instructions: mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier)

| Instruction                  | Available in libcu++    |
|------------------------------|-------------------------|
| [`mbarrier.init`]            | CTK-FUTURE, CCCL v2.5.0 |
| [`mbarrier.inval`]           | No                      |
| [`mbarrier.expect_tx`]       | No                      |
| [`mbarrier.complete_tx`]     | No                      |
| [`mbarrier.arrive`]          | [CTK 12.4, CCCL v2.3.0] |
| [`mbarrier.arrive_drop`]     | No                      |
| [`cp.async.mbarrier.arrive`] | No                      |
| [`mbarrier.test_wait`]       | [CTK 12.4, CCCL v2.3.0] |
| [`mbarrier.try_wait`]        | [CTK 12.4, CCCL v2.3.0] |
| [`mbarrier.pending_count`]   | No                      |
| [`tensormap.cp_fenceproxy`]  | CTK-FUTURE, CCCL v2.4.0 |

[`mbarrier.init`]: ptx/instructions/mbarrier.init.md
[`mbarrier.inval`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-inval
[`mbarrier.expect_tx`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx
[`mbarrier.complete_tx`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx
[`mbarrier.arrive`]: ptx/instructions/mbarrier.arrive.md
[`mbarrier.arrive_drop`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive-drop
[`cp.async.mbarrier.arrive`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive
[`mbarrier.test_wait`]: ptx/instructions/mbarrier.test_wait.md
[`mbarrier.try_wait`]: ptx/instructions/mbarrier.try_wait.md
[`mbarrier.pending_count`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-pending-count
[`tensormap.cp_fenceproxy`]: ptx/instructions/tensormap.cp_fenceproxy.md

### [Warp Level Matrix Multiply-Accumulate Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-multiply-accumulate-instructions)

| Instruction                              | Available in libcu++ |
|------------------------------------------|----------------------|
| [`wmma.load`]                            | No                   |
| [`wmma.store`]                           | No                   |
| [`wmma.mma`]                             | No                   |
| [`mma`]                                  | No                   |
| [`ldmatrix`]                             | No                   |
| [`stmatrix`]                             | No                   |
| [`movmatrix`]                            | No                   |
| [`mma.sp`]                               | No                   |

[`wmma.load`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-load-instruction-wmma-load
[`wmma.store`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-store-instruction-wmma-store
[`wmma.mma`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-multiply-and-accumulate-instruction-wmma-mma
[`mma`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#multiply-and-accumulate-instruction-mma
[`ldmatrix`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-load-instruction-ldmatrix
[`stmatrix`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-store-instruction-stmatrix
[`movmatrix`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-transpose-instruction-movmatrix
[`mma.sp`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#multiply-and-accumulate-instruction-mma-sp

### [Asynchronous Warpgroup Level Matrix Multiply-Accumulate Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions)

| Instruction                              | Available in libcu++ |
|------------------------------------------|----------------------|
| [`wgmma.mma_async`]                      | No                   |
| [`wgmma.mma_async.sp`]                   | No                   |
| [`wgmma.fence`]                          | No                   |
| [`wgmma.commit_group`]                   | No                   |
| [`wgmma.wait_group`]                     | No                   |

[`wgmma.mma_async`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-mma-async
[`wgmma.mma_async.sp`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-mma-async-sp
[`wgmma.fence`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-fence
[`wgmma.commit_group`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-commit-group
[`wgmma.wait_group`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-wait-group

### [Stack Manipulation Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#stack-manipulation-instructions)

| Instruction                              | Available in libcu++ |
|------------------------------------------|----------------------|
| [`stacksave`]                            | No                   |
| [`stackrestore`]                         | No                   |
| [`alloca`]                               | No                   |

[`stacksave`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#stack-manipulation-instructions-stacksave
[`stackrestore`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#stack-manipulation-instructions-stackrestore
[`alloca`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#stack-manipulation-instructions-alloca

### [Video Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#video-instructions)

| Instruction                              | Available in libcu++ |
|------------------------------------------|----------------------|
| [`vadd, vsub, vabsdiff, vmin, vmax`]     | No                   |
| [`vshl, vshr`]                           | No                   |
| [`vmad`]                                 | No                   |
| [`vset`]                                 | No                   |

[`vadd, vsub, vabsdiff, vmin, vmax`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scalar-video-instructions-vadd-vsub-vabsdiff-vmin-vmax
[`vshl, vshr`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scalar-video-instructions-vshl-vshr
[`vmad`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scalar-video-instructions-vmad
[`vset`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scalar-video-instructions-vset

### [SIMD Video Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions)

| Instruction                              | Available in libcu++ |
|------------------------------------------|----------------------|
| [`vadd2, vsub2, vavrg2, vabsdiff2, vmin2, vmax2`] | No                   |
| [`vset2`]                                | No                   |
| [`vadd4, vsub4, vavrg4, vabsdiff4, vmin4, vmax4`] | No                   |
| [`vset4`]                                | No                   |

[`vadd2, vsub2, vavrg2, vabsdiff2, vmin2, vmax2`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions-vadd2-vsub2-vavrg2-vabsdiff2-vmin2-vmax2
[`vset2`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions-vset2
[`vadd4, vsub4, vavrg4, vabsdiff4, vmin4, vmax4`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions-vadd4-vsub4-vavrg4-vabsdiff4-vmin4-vmax4
[`vset4`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions-vset4

### [Miscellaneous Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions)

| Instruction                              | Available in libcu++ |
|------------------------------------------|----------------------|
| [`brkpt`]                                | No                   |
| [`nanosleep`]                            | No                   |
| [`pmevent`]                              | No                   |
| [`trap`]                                 | No                   |
| [`setmaxnreg`]                           | No                   |

[`brkpt`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-brkpt
[`nanosleep`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-nanosleep
[`pmevent`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-pmevent
[`trap`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-trap
[`setmaxnreg`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg

### [Special registers](ptx/instructions/special_registers.md)

| Register                       | PTX ISA | SM Version | Available in libcu++    |
|--------------------------------|---------|------------|-------------------------|
| [`tid`]                        | 20      | All        | CTK-FUTURE, CCCL v2.4.0 |
| [`ntid`]                       | 20      | All        | CTK-FUTURE, CCCL v2.4.0 |
| [`laneid`]                     | 13      | All        | CTK-FUTURE, CCCL v2.4.0 |
| [`warpid`]                     | 13      | All        | CTK-FUTURE, CCCL v2.4.0 |
| [`nwarpid`]                    | 20      | 20         | CTK-FUTURE, CCCL v2.4.0 |
| [`ctaid`]                      | 20      | All        | CTK-FUTURE, CCCL v2.4.0 |
| [`nctaid`]                     | 20      | All        | CTK-FUTURE, CCCL v2.4.0 |
| [`smid`]                       | 13      | All        | CTK-FUTURE, CCCL v2.4.0 |
| [`nsmid`]                      | 20      | 20         | CTK-FUTURE, CCCL v2.4.0 |
| [`gridid`]                     | 30      | 30         | CTK-FUTURE, CCCL v2.4.0 |
| [`is_explicit_cluster`]        | 78      | 90         | CTK-FUTURE, CCCL v2.4.0 |
| [`clusterid`]                  | 78      | 90         | CTK-FUTURE, CCCL v2.4.0 |
| [`nclusterid`]                 | 78      | 90         | CTK-FUTURE, CCCL v2.4.0 |
| [`cluster_ctaid`]              | 78      | 90         | CTK-FUTURE, CCCL v2.4.0 |
| [`cluster_nctaid`]             | 78      | 90         | CTK-FUTURE, CCCL v2.4.0 |
| [`cluster_ctarank`]            | 78      | 90         | CTK-FUTURE, CCCL v2.4.0 |
| [`cluster_nctarank`]           | 78      | 90         | CTK-FUTURE, CCCL v2.4.0 |
| [`lanemask_eq`]                | 20      | 20         | CTK-FUTURE, CCCL v2.4.0 |
| [`lanemask_le`]                | 20      | 20         | CTK-FUTURE, CCCL v2.4.0 |
| [`lanemask_lt`]                | 20      | 20         | CTK-FUTURE, CCCL v2.4.0 |
| [`lanemask_ge`]                | 20      | 20         | CTK-FUTURE, CCCL v2.4.0 |
| [`lanemask_gt`]                | 20      | 20         | CTK-FUTURE, CCCL v2.4.0 |
| [`clock`]                      | 10      | All        | CTK-FUTURE, CCCL v2.4.0 |
| [`clock_hi`]                   | 50      | 20         | CTK-FUTURE, CCCL v2.4.0 |
| [`clock64`]                    | 20      | 20         | CTK-FUTURE, CCCL v2.4.0 |
| [`pm0`]                        |         |            | No                      |
| [`pm0_64`]                     |         |            | No                      |
| [`envreg`]                     |         |            | No                      |
| [`globaltimer`]                | 31      | 30         | CTK-FUTURE, CCCL v2.4.0 |
| [`globaltimer_lo`]             | 31      | 30         | CTK-FUTURE, CCCL v2.4.0 |
| [`globaltimer_hi`]             | 31      | 30         | CTK-FUTURE, CCCL v2.4.0 |
| [`reserved_smem_offset_begin`] |         |            | No                      |
| [`reserved_smem_offset_end`]   |         |            | No                      |
| [`reserved_smem_offset_cap`]   |         |            | No                      |
| [`reserved_smem_offset_2`]     |         |            | No                      |
| [`total_smem_size`]            | 41      | 20         | CTK-FUTURE, CCCL v2.4.0 |
| [`aggr_smem_size`]             | 81      | 90         | CTK-FUTURE, CCCL v2.4.0 |
| [`dynamic_smem_size`]          | 41      | 20         | CTK-FUTURE, CCCL v2.4.0 |
| [`current_graph_exec`]         | 80      | 50         | CTK-FUTURE, CCCL v2.4.0 |

[`tid`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-tid
[`ntid`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-ntid
[`laneid`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-laneid
[`warpid`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-warpid
[`nwarpid`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-nwarpid
[`ctaid`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-ctaid
[`nctaid`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-nctaid
[`smid`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-smid
[`nsmid`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-nsmid
[`gridid`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-gridid
[`is_explicit_cluster`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-is-explicit-cluster
[`clusterid`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-clusterid
[`nclusterid`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-nclusterid
[`cluster_ctaid`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-cluster-ctaid
[`cluster_nctaid`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-cluster-nctaid
[`cluster_ctarank`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-cluster-ctarank
[`cluster_nctarank`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-cluster-nctarank
[`lanemask_eq`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-lanemask-eq
[`lanemask_le`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-lanemask-le
[`lanemask_lt`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-lanemask-lt
[`lanemask_ge`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-lanemask-ge
[`lanemask_gt`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-lanemask-gt
[`clock`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-clock-clock-hi
[`clock_hi`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-clock-clock-hi
[`clock64`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-clock64
[`pm0`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-pm0-pm7
[`pm0_64`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-pm0-64-pm7-64
[`envreg`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-envreg-32
[`globaltimer`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer-globaltimer-lo-globaltimer-hi
[`globaltimer_lo`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer-globaltimer-lo-globaltimer-hi
[`globaltimer_hi`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer-globaltimer-lo-globaltimer-hi
[`reserved_smem_offset_begin`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-reserved-smem-offset-begin-reserved-smem-offset-end-reserved-smem-offset-cap-reserved-smem-offset-2
[`reserved_smem_offset_end`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-reserved-smem-offset-begin-reserved-smem-offset-end-reserved-smem-offset-cap-reserved-smem-offset-2
[`reserved_smem_offset_cap`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-reserved-smem-offset-begin-reserved-smem-offset-end-reserved-smem-offset-cap-reserved-smem-offset-2
[`reserved_smem_offset_2`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-reserved-smem-offset-begin-reserved-smem-offset-end-reserved-smem-offset-cap-reserved-smem-offset-2
[`total_smem_size`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-total-smem-size
[`aggr_smem_size`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-aggr-smem-size
[`dynamic_smem_size`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-dynamic-smem-size
[`current_graph_exec`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-current-graph-exec




[CTK 12.4, CCCL v2.3.0]: https://github.com/NVIDIA/cccl/releases/tag/v2.3.0
