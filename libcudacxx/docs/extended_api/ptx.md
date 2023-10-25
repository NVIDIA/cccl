## PTX instructions

The `cuda::ptx` namespace contains functions that map one-to-one to
[PTX instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html). These can be used for maximal control of the generated code, or to
experiment with new hardware features before a high-level C++ API is available.

### [9.7.1. Integer Arithmetic Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions)

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

### [9.7.2. Extended-Precision Integer Arithmetic Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-integer-arithmetic-instructions)

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

### [9.7.3. Floating-Point Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions)

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

### [9.7.4. Half Precision Floating-Point Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions)

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

### [9.7.5. Comparison and Selection Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions)

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

### [9.7.6. Half Precision Comparison Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-comparison-instructions)

| Instruction                              | Available in libcu++ |
|------------------------------------------|----------------------|
| [`set`]                                  | No                   |
| [`setp`]                                 | No                   |

[`set`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-comparison-instructions-set
[`setp`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-comparison-instructions-setp

### [9.7.7. Logic and Shift Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions)

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

### [9.7.8. Data Movement and Conversion Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions)

| Instruction                              | Available in libcu++ |
|------------------------------------------|----------------------|
| [`mov`]                                  | No                   |
| [`mov`]                                  | No                   |
| [`shfl (deprecated)`]                    | No                   |
| [`shfl.sync`]                            | No                   |
| [`prmt`]                                 | No                   |
| [`ld`]                                   | No                   |
| [`ld.global.nc`]                         | No                   |
| [`ldu`]                                  | No                   |
| [`st`]                                   | No                   |
| [`st.async`]                             | No                   |
| [`multimem.ld_reduce, multimem.st, multimem.red`] | No                   |
| [`prefetch, prefetchu`]                  | No                   |
| [`applypriority`]                        | No                   |
| [`discard`]                              | No                   |
| [`createpolicy`]                         | No                   |
| [`isspacep`]                             | No                   |
| [`cvta`]                                 | No                   |
| [`cvt`]                                  | No                   |
| [`cvt.pack`]                             | No                   |
| [`mapa`]                                 | No                   |
| [`getctarank`]                           | No                   |

[`mov`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mov-2
[`shfl (deprecated)`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-deprecated
[`shfl.sync`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync
[`prmt`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt
[`ld`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld
[`ld.global.nc`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld-global-nc
[`ldu`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ldu
[`st`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st
[`st.async`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st-async
[`multimem.ld_reduce, multimem.st, multimem.red`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem-ld-reduce-multimem-st-multimem-red
[`prefetch, prefetchu`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prefetch-prefetchu
[`applypriority`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-applypriority
[`discard`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-discard
[`createpolicy`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-createpolicy
[`isspacep`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-isspacep
[`cvta`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvta
[`cvt`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt
[`cvt.pack`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt-pack
[`mapa`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mapa
[`getctarank`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-getctarank

### [9.7.8.24. Data Movement and Conversion Instructions: Asynchronous copy](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy)

| Instruction                              | Available in libcu++ |
|------------------------------------------|----------------------|
| [`cp.async`]                             | No                   |
| [`cp.async.commit_group`]                | No                   |
| [`cp.async.wait_group / cp.async.wait_all`] | No                   |
| [`cp.async.bulk`]                        | No                   |
| [`cp.reduce.async.bulk`]                 | No                   |
| [`cp.async.bulk.prefetch`]               | No                   |
| [`cp.async.bulk.tensor`]                 | No                   |
| [`cp.reduce.async.bulk.tensor`]          | No                   |
| [`cp.async.bulk.prefetch.tensor`]        | No                   |
| [`cp.async.bulk.commit_group`]           | No                   |
| [`cp.async.bulk.wait_group`]             | No                   |
| [`tensormap.replace`]                    | No                   |

[`cp.async`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async
[`cp.async.commit_group`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-commit-group
[`cp.async.wait_group / cp.async.wait_all`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group-cp-async-wait-all
[`cp.async.bulk`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk
[`cp.reduce.async.bulk`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk
[`cp.async.bulk.prefetch`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch
[`cp.async.bulk.tensor`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
[`cp.reduce.async.bulk.tensor`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor
[`cp.async.bulk.prefetch.tensor`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch-tensor
[`cp.async.bulk.commit_group`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-commit-group
[`cp.async.bulk.wait_group`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group
[`tensormap.replace`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-tensormap-replace

### [9.7.9. Texture Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions)

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

### [9.7.10. Surface Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions)

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

### [9.7.11. Control Flow Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions)

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

### [9.7.12. Parallel Synchronization and Communication Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions)

| Instruction                              | Available in libcu++ |
|------------------------------------------|----------------------|
| [`bar, barrier`]                         | No                   |
| [`bar.warp.sync`]                        | No                   |
| [`barrier.cluster`]                      | No                   |
| [`membar/fence`]                         | No                   |
| [`atom`]                                 | No                   |
| [`red`]                                  | No                   |
| [`red.async`]                            | No                   |
| [`vote (deprecated)`]                    | No                   |
| [`vote.sync`]                            | No                   |
| [`match.sync`]                           | No                   |
| [`activemask`]                           | No                   |
| [`redux.sync`]                           | No                   |
| [`griddepcontrol`]                       | No                   |
| [`elect.sync`]                           | No                   |

[`bar, barrier`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-barrier
[`bar.warp.sync`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-warp-sync
[`barrier.cluster`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster
[`membar/fence`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence
[`atom`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom
[`red`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-red
[`red.async`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-red-async
[`vote (deprecated)`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-vote-deprecated
[`vote.sync`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-vote-sync
[`match.sync`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-match-sync
[`activemask`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-activemask
[`redux.sync`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-redux-sync
[`griddepcontrol`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol
[`elect.sync`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-elect-sync

### [9.7.12.15. Parallel Synchronization and Communication Instructions: mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier)

| Instruction                              | Available in libcu++ |
|------------------------------------------|----------------------|
| [`mbarrier.init`]                        | No                   |
| [`mbarrier.inval`]                       | No                   |
| [`mbarrier.expect_tx`]                   | No                   |
| [`mbarrier.complete_tx`]                 | No                   |
| [`mbarrier.arrive`]                      | CTK-FUTURE, CCCL v2.3.0 |
| [`mbarrier.arrive_drop`]                 | No                   |
| [`cp.async.mbarrier.arrive`]             | No                   |
| [`mbarrier.test_wait/mbarrier.try_wait`] | No                   |
| [`mbarrier.pending_count`]               | No                   |
| [`tensormap.cp_fenceproxy`]              | No                   |

[`mbarrier.init`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-init
[`mbarrier.inval`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-inval
[`mbarrier.expect_tx`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx
[`mbarrier.complete_tx`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx
[`mbarrier.arrive`]: #mbarrierarrive
[`mbarrier.arrive_drop`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive-drop
[`cp.async.mbarrier.arrive`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive
[`mbarrier.test_wait/mbarrier.try_wait`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait
[`mbarrier.pending_count`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-pending-count
[`tensormap.cp_fenceproxy`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tensormap-cp-fenceproxy


#### `mbarrier.arrive`

-  PTX ISA: [mbarrier.arrive](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive)

 
```cuda
template <dot_scope _Sco>
__device__ inline
uint64_t mbarrier_arrive_expect_tx(sem_release_t sem, scope_t<_Sco> scope, space_shared_t spc, uint64_t* addr, uint32_t tx_count);

template <dot_scope _Sco>
__device__ inline
void mbarrier_arrive_expect_tx(sem_release_t sem, scope_t<_Sco> scope, space_shared_cluster_t spc, uint64_t* addr, uint32_t tx_count);
```

Usage:

```cuda
#include <cuda/ptx>
#include <cuda/barrier>
#include <cooperative_groups.h>

__global__ void kernel() {
    using cuda::ptx::sem_release;
    using cuda::ptx::space_shared_cluster;
    using cuda::ptx::space_shared;
    using cuda::ptx::scope_cluster;
    using cuda::ptx::scope_cta;

    using barrier_t = cuda::barrier<cuda::thread_scope_block>;
    __shared__ barrier_t bar;
    init(&bar, blockDim.x);
    __syncthreads();

    NV_IF_TARGET(NV_PROVIDES_SM_90, (
        // Arrive on local shared memory barrier:
        uint64_t token;
        token = cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cta, space_shared, &bar, 1);
        token = cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cluster, space_shared, &bar, 1);

        // Get address of remote cluster barrier:
        namespace cg = cooperative_groups;
        cg::cluster_group cluster = cg::this_cluster();
        unsigned int other_block_rank = cluster.block_rank() ^ 1;
        uint64_t * remote_bar = cluster.map_shared_rank(&bar, other_block_rank);

        // Sync cluster to ensure remote barrier is initialized.
        cluster.sync();

        // Arrive on remote cluster barrier:
        cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cta, space_shared_cluster, remote_bar, 1);
        cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cluster, space_shared_cluster, remote_bar, 1);
    )
}
```
### [9.7.13. Warp Level Matrix Multiply-Accumulate Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-multiply-accumulate-instructions)

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

### [9.7.14. Asynchronous Warpgroup Level Matrix Multiply-Accumulate Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions)

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

### [9.7.15. Stack Manipulation Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#stack-manipulation-instructions)

| Instruction                              | Available in libcu++ |
|------------------------------------------|----------------------|
| [`stacksave`]                            | No                   |
| [`stackrestore`]                         | No                   |
| [`alloca`]                               | No                   |

[`stacksave`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#stack-manipulation-instructions-stacksave
[`stackrestore`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#stack-manipulation-instructions-stackrestore
[`alloca`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#stack-manipulation-instructions-alloca

### [9.7.16. Video Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#video-instructions)

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

### [9.7.16.2. SIMD Video Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions)

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

### [9.7.17. Miscellaneous Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions)

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











### Shared memory barrier (mbarrier)

| Instruction | Compute capability | CUDA Toolkit |
|----------------------------------------|--------------------|--------------|
| `cuda::ptx::mbarrier_arrive_expect_tx` | 9.0                | CTK 12.4     |





