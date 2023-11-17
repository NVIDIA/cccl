## PTX instructions

The `cuda::ptx` namespace contains functions that map one-to-one to
[PTX instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html). These can be used for maximal control of the generated code, or to
experiment with new hardware features before a high-level C++ API is available.

### Variants

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
[`st.async`]: #stasync
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

#### `st.async`

-  PTX ISA: [`st.async`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st-async)

**NOTE.** Alignment of `addr` must be a multiple of vector size. For instance,
the `addr` supplied to the `v2.b32` variant must be aligned to `2 x 4 = 8` bytes.

**st_async**:
```cuda
// st.async.weak.shared::cluster.mbarrier::complete_tx::bytes{.type} [addr], value, [remote_bar];    // 1.  PTX ISA 81, SM_90
// .type      = { .b32, .b64 }
template <typename Type>
__device__ static inline void st_async(
  Type* addr,
  const Type& value,
  uint64_t* remote_bar);

// st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2{.type} [addr], value, [remote_bar]; // 2.  PTX ISA 81, SM_90
// .type      = { .b32, .b64 }
template <typename Type>
__device__ static inline void st_async(
  Type* addr,
  const Type (&value)[2],
  uint64_t* remote_bar);

// st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [addr], value, [remote_bar];    // 3.  PTX ISA 81, SM_90
template <typename B32>
__device__ static inline void st_async(
  B32* addr,
  const B32 (&value)[4],
  uint64_t* remote_bar);
```

**Usage**:
```cuda
#include <cstdio>
#include <cuda/ptx>
#include <cuda/barrier>
#include <cooperative_groups.h>

__global__ void __cluster_dims__(8, 1, 1) kernel()
{
  using cuda::ptx::sem_release;
  using cuda::ptx::sem_acquire;
  using cuda::ptx::space_cluster;
  using cuda::ptx::space_shared;
  using cuda::ptx::scope_cluster;

  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();

  using barrier_t = cuda::barrier<cuda::thread_scope_block>;

  __shared__ int receive_buffer[4];
  __shared__ barrier_t bar;
  init(&bar, blockDim.x);

  // Sync cluster to ensure remote barrier is initialized.
  cluster.sync();

  // Get address of remote cluster barrier:
  unsigned int other_block_rank = cluster.block_rank() ^ 1;
  uint64_t * remote_bar = cluster.map_shared_rank(cuda::device::barrier_native_handle(bar), other_block_rank);
  // int * remote_buffer = cluster.map_shared_rank(&receive_buffer, other_block_rank);
  int * remote_buffer = cluster.map_shared_rank(&receive_buffer[0], other_block_rank);

  // Arrive on local barrier:
  uint64_t arrival_token;
  if (threadIdx.x == 0) {
    // Thread 0 arrives and indicates it expects to receive a certain number of bytes as well
    arrival_token = cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cluster, space_shared, cuda::device::barrier_native_handle(bar), sizeof(receive_buffer));
  } else {
    arrival_token = cuda::ptx::mbarrier_arrive(sem_release, scope_cluster, space_shared, cuda::device::barrier_native_handle(bar));
  }

  if (threadIdx.x == 0) {
    printf("[block %d] arrived with expected tx count = %llu\n", cluster.block_rank(), sizeof(receive_buffer));
  }

  // Send bytes to remote buffer, arriving on remote barrier
  if (threadIdx.x == 0) {
    cuda::ptx::st_async(remote_buffer, {int(cluster.block_rank()), 2, 3, 4}, remote_bar);
  }

  if (threadIdx.x == 0) {
    printf("[block %d] st_async to %p, %p\n",
           cluster.block_rank(),
           remote_buffer,
           remote_bar
    );
  }

  // Wait on local barrier:
  while(!cuda::ptx::mbarrier_try_wait(sem_acquire, scope_cluster, cuda::device::barrier_native_handle(bar), arrival_token)) {}

  // Print received values:
  if (threadIdx.x == 0) {
    printf(
      "[block %d] receive_buffer = {%d, %d, %d, %d}\n",
      cluster.block_rank(),
      receive_buffer[0], receive_buffer[1], receive_buffer[2], receive_buffer[3]
    );
  }

}

int main() {
  kernel<<<8, 128>>>();
  cudaDeviceSynchronize();
}
```
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

| Instruction           | Available in libcu++    |
|-----------------------|-------------------------|
| [`bar, barrier`]      | No                      |
| [`bar.warp.sync`]     | No                      |
| [`barrier.cluster`]   | No                      |
| [`membar/fence`]      | No                      |
| [`atom`]              | No                      |
| [`red`]               | No                      |
| [`red.async`]         | CTK-FUTURE, CCCL v2.3.0 |
| [`vote (deprecated)`] | No                      |
| [`vote.sync`]         | No                      |
| [`match.sync`]        | No                      |
| [`activemask`]        | No                      |
| [`redux.sync`]        | No                      |
| [`griddepcontrol`]    | No                      |
| [`elect.sync`]        | No                      |

[`bar, barrier`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-barrier
[`bar.warp.sync`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-warp-sync
[`barrier.cluster`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster
[`membar/fence`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence
[`atom`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom
[`red`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-red
[`red.async`]: #redasync
[`vote (deprecated)`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-vote-deprecated
[`vote.sync`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-vote-sync
[`match.sync`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-match-sync
[`activemask`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-activemask
[`redux.sync`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-redux-sync
[`griddepcontrol`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol
[`elect.sync`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-elect-sync

#### `red.async`

-  PTX ISA: [`red.async`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-red-async)

PTX does not currently (CTK 12.3) expose `red.async.add.s64`. This exposure is emulated in `cuda::ptx` using

```cuda
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}.u64  [dest], value, [remote_bar]; // .u64 intentional PTX ISA 81, SM_90
// .op        = { .add }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_add_t,
  int64_t* dest,
  const int64_t& value,
  int64_t* remote_bar);
```

**red_async**:
```cuda
// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .u32 }
// .op        = { .inc }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_inc_t,
  uint32_t* dest,
  const uint32_t& value,
  uint64_t* remote_bar);

// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .u32 }
// .op        = { .dec }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_dec_t,
  uint32_t* dest,
  const uint32_t& value,
  uint64_t* remote_bar);

// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .u32 }
// .op        = { .min }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_min_t,
  uint32_t* dest,
  const uint32_t& value,
  uint64_t* remote_bar);

// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .u32 }
// .op        = { .max }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_max_t,
  uint32_t* dest,
  const uint32_t& value,
  uint64_t* remote_bar);

// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .u32 }
// .op        = { .add }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_add_t,
  uint32_t* dest,
  const uint32_t& value,
  uint64_t* remote_bar);

// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .s32 }
// .op        = { .min }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_min_t,
  int32_t* dest,
  const int32_t& value,
  uint64_t* remote_bar);

// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .s32 }
// .op        = { .max }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_max_t,
  int32_t* dest,
  const int32_t& value,
  uint64_t* remote_bar);

// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .s32 }
// .op        = { .add }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_add_t,
  int32_t* dest,
  const int32_t& value,
  uint64_t* remote_bar);

// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .b32 }
// .op        = { .and }
template <typename B32>
__device__ static inline void red_async(
  cuda::ptx::op_and_op_t,
  B32* dest,
  const B32& value,
  uint64_t* remote_bar);

// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .b32 }
// .op        = { .or }
template <typename B32>
__device__ static inline void red_async(
  cuda::ptx::op_or_op_t,
  B32* dest,
  const B32& value,
  uint64_t* remote_bar);

// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .b32 }
// .op        = { .xor }
template <typename B32>
__device__ static inline void red_async(
  cuda::ptx::op_xor_op_t,
  B32* dest,
  const B32& value,
  uint64_t* remote_bar);

// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}{.type}  [dest], value, [remote_bar];  // PTX ISA 81, SM_90
// .type      = { .u64 }
// .op        = { .add }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_add_t,
  uint64_t* dest,
  const uint64_t& value,
  uint64_t* remote_bar);

// red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}.u64  [dest], value, [remote_bar]; // .u64 intentional PTX ISA 81, SM_90
// .op        = { .add }
template <typename=void>
__device__ static inline void red_async(
  cuda::ptx::op_add_t,
  int64_t* dest,
  const int64_t& value,
  int64_t* remote_bar);
```

### [9.7.12.15. Parallel Synchronization and Communication Instructions: mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier)

| Instruction                              | Available in libcu++    |
|------------------------------------------|-------------------------|
| [`mbarrier.init`]                        | No                      |
| [`mbarrier.inval`]                       | No                      |
| [`mbarrier.expect_tx`]                   | No                      |
| [`mbarrier.complete_tx`]                 | No                      |
| [`mbarrier.arrive`]                      | CTK-FUTURE, CCCL v2.3.0 |
| [`mbarrier.arrive_drop`]                 | No                      |
| [`cp.async.mbarrier.arrive`]             | No                      |
| [`mbarrier.test_wait/mbarrier.try_wait`] | CTK-FUTURE, CCCL v2.3.0 |
| [`mbarrier.pending_count`]               | No                      |
| [`tensormap.cp_fenceproxy`]              | No                      |

[`mbarrier.init`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-init
[`mbarrier.inval`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-inval
[`mbarrier.expect_tx`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx
[`mbarrier.complete_tx`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx
[`mbarrier.arrive`]: #mbarrierarrive
[`mbarrier.arrive_drop`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive-drop
[`cp.async.mbarrier.arrive`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive
[`mbarrier.test_wait/mbarrier.try_wait`]: #mbarriertest_waitmbarriertry_wait
[`mbarrier.pending_count`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-pending-count
[`tensormap.cp_fenceproxy`]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tensormap-cp-fenceproxy



#### `mbarrier.arrive`

-  PTX ISA: [`mbarrier.arrive`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive)

```cuda
// mbarrier.arrive.shared.b64                                  state,  [addr];           // 1.  PTX ISA 70, SM_80
template <typename=void>
__device__ static inline uint64_t mbarrier_arrive(
  uint64_t* addr);

// mbarrier.arrive.shared::cta.b64                             state,  [addr], count;    // 2.  PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint64_t mbarrier_arrive(
  uint64_t* addr,
  const uint32_t& count);

// mbarrier.arrive{.sem}{.scope}{.space}.b64                   state,  [addr];           // 3a.  PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cta, .cluster }
// .space     = { .shared::cta }
template <cuda::ptx::dot_scope Scope>
__device__ static inline uint64_t mbarrier_arrive(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::space_shared_t,
  uint64_t* addr);

// mbarrier.arrive{.sem}{.scope}{.space}.b64                   state,  [addr], count;    // 3b.  PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cta, .cluster }
// .space     = { .shared::cta }
template <cuda::ptx::dot_scope Scope>
__device__ static inline uint64_t mbarrier_arrive(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::space_shared_t,
  uint64_t* addr,
  const uint32_t& count);

// mbarrier.arrive{.sem}{.scope}{.space}.b64                   _, [addr];                // 4a.  PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cluster }
// .space     = { .shared::cluster }
template <typename=void>
__device__ static inline void mbarrier_arrive(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_cluster_t,
  cuda::ptx::space_cluster_t,
  uint64_t* addr);

// mbarrier.arrive{.sem}{.scope}{.space}.b64                   _, [addr], count;         // 4b.  PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cluster }
// .space     = { .shared::cluster }
template <typename=void>
__device__ static inline void mbarrier_arrive(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_cluster_t,
  cuda::ptx::space_cluster_t,
  uint64_t* addr,
  const uint32_t& count);
```

```cuda
// mbarrier.arrive.noComplete.shared.b64                       state,  [addr], count;    // 5.  PTX ISA 70, SM_80
template <typename=void>
__device__ static inline uint64_t mbarrier_arrive_no_complete(
  uint64_t* addr,
  const uint32_t& count);
```

```cuda
// mbarrier.arrive.expect_tx{.sem}{.scope}{.space}.b64 state, [addr], tx_count; // 8.  PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cta, .cluster }
// .space     = { .shared::cta }
template <cuda::ptx::dot_scope Scope>
__device__ static inline uint64_t mbarrier_arrive_expect_tx(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::space_shared_t,
  uint64_t* addr,
  const uint32_t& tx_count);

// mbarrier.arrive.expect_tx{.sem}{.scope}{.space}.b64   _, [addr], tx_count; // 9.  PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cluster }
// .space     = { .shared::cluster }
template <typename=void>
__device__ static inline void mbarrier_arrive_expect_tx(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_cluster_t,
  cuda::ptx::space_cluster_t,
  uint64_t* addr,
  const uint32_t& tx_count);
```

Usage:
```cuda
#include <cuda/ptx>
#include <cuda/barrier>
#include <cooperative_groups.h>

__global__ void kernel() {
    using cuda::ptx::sem_release;
    using cuda::ptx::space_cluster;
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
        cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cta, space_cluster, remote_bar, 1);
        cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cluster, space_cluster, remote_bar, 1);
    )
}
```

#### `mbarrier.test_wait/mbarrier.try_wait`

- PTX ISA: [`mbarrier.test_wait/mbarrier.try_wait`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait)

**mbarrier_test_wait**:
```cuda
// mbarrier.test_wait.shared.b64 waitComplete, [addr], state;                                                  // 1.  PTX ISA 70, SM_80
template <typename=void>
__device__ static inline bool mbarrier_test_wait(
  uint64_t* addr,
  const uint64_t& state);

// mbarrier.test_wait{.sem}{.scope}.shared::cta.b64        waitComplete, [addr], state;                        // 2.   PTX ISA 80, SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_test_wait(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint64_t& state);
```

**mbarrier_test_wait_parity**:
```cuda
// mbarrier.test_wait.parity.shared.b64 waitComplete, [addr], phaseParity;                                     // 3.  PTX ISA 71, SM_80
template <typename=void>
__device__ static inline bool mbarrier_test_wait_parity(
  uint64_t* addr,
  const uint32_t& phaseParity);

// mbarrier.test_wait.parity{.sem}{.scope}.shared::cta.b64 waitComplete, [addr], phaseParity;                  // 4.  PTX ISA 80, SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_test_wait_parity(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint32_t& phaseParity);
```

**mbarrier_try_wait**:
```cuda
// mbarrier.try_wait.shared::cta.b64         waitComplete, [addr], state;                                      // 5a.  PTX ISA 78, SM_90
template <typename=void>
__device__ static inline bool mbarrier_try_wait(
  uint64_t* addr,
  const uint64_t& state);

// mbarrier.try_wait.shared::cta.b64         waitComplete, [addr], state, suspendTimeHint;                    // 5b.  PTX ISA 78, SM_90
template <typename=void>
__device__ static inline bool mbarrier_try_wait(
  uint64_t* addr,
  const uint64_t& state,
  const uint32_t& suspendTimeHint);

// mbarrier.try_wait{.sem}{.scope}.shared::cta.b64         waitComplete, [addr], state;                        // 6a.  PTX ISA 80, SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_try_wait(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint64_t& state);

// mbarrier.try_wait{.sem}{.scope}.shared::cta.b64         waitComplete, [addr], state , suspendTimeHint;      // 6b.  PTX ISA 80, SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_try_wait(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint64_t& state,
  const uint32_t& suspendTimeHint);
```

**mbarrier_try_wait_parity**:
```cuda
// mbarrier.try_wait.parity.shared::cta.b64  waitComplete, [addr], phaseParity;                                // 7a.  PTX ISA 78, SM_90
template <typename=void>
__device__ static inline bool mbarrier_try_wait_parity(
  uint64_t* addr,
  const uint32_t& phaseParity);

// mbarrier.try_wait.parity.shared::cta.b64  waitComplete, [addr], phaseParity, suspendTimeHint;               // 7b.  PTX ISA 78, SM_90
template <typename=void>
__device__ static inline bool mbarrier_try_wait_parity(
  uint64_t* addr,
  const uint32_t& phaseParity,
  const uint32_t& suspendTimeHint);

// mbarrier.try_wait.parity{.sem}{.scope}.shared::cta.b64  waitComplete, [addr], phaseParity;                  // 8a.  PTX ISA 80, SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_try_wait_parity(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint32_t& phaseParity);

// mbarrier.try_wait.parity{.sem}{.scope}.shared::cta.b64  waitComplete, [addr], phaseParity, suspendTimeHint; // 8b.  PTX ISA 80, SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_try_wait_parity(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint32_t& phaseParity,
  const uint32_t& suspendTimeHint);
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
