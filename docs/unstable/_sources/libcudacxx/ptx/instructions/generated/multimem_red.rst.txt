..
   This file was automatically generated. Do not edit.

multimem.red.relaxed.cta.global.min.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     uint32_t* addr,
     uint32_t val);

multimem.red.relaxed.cluster.global.min.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     uint32_t* addr,
     uint32_t val);

multimem.red.relaxed.gpu.global.min.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     uint32_t* addr,
     uint32_t val);

multimem.red.relaxed.sys.global.min.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     uint32_t* addr,
     uint32_t val);

multimem.red.release.cta.global.min.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     uint32_t* addr,
     uint32_t val);

multimem.red.release.cluster.global.min.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     uint32_t* addr,
     uint32_t val);

multimem.red.release.gpu.global.min.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     uint32_t* addr,
     uint32_t val);

multimem.red.release.sys.global.min.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     uint32_t* addr,
     uint32_t val);

multimem.red.relaxed.cta.global.min.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     uint64_t* addr,
     uint64_t val);

multimem.red.relaxed.cluster.global.min.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     uint64_t* addr,
     uint64_t val);

multimem.red.relaxed.gpu.global.min.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     uint64_t* addr,
     uint64_t val);

multimem.red.relaxed.sys.global.min.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     uint64_t* addr,
     uint64_t val);

multimem.red.release.cta.global.min.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     uint64_t* addr,
     uint64_t val);

multimem.red.release.cluster.global.min.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     uint64_t* addr,
     uint64_t val);

multimem.red.release.gpu.global.min.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     uint64_t* addr,
     uint64_t val);

multimem.red.release.sys.global.min.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     uint64_t* addr,
     uint64_t val);

multimem.red.relaxed.cta.global.min.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     int32_t* addr,
     int32_t val);

multimem.red.relaxed.cluster.global.min.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     int32_t* addr,
     int32_t val);

multimem.red.relaxed.gpu.global.min.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     int32_t* addr,
     int32_t val);

multimem.red.relaxed.sys.global.min.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     int32_t* addr,
     int32_t val);

multimem.red.release.cta.global.min.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     int32_t* addr,
     int32_t val);

multimem.red.release.cluster.global.min.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     int32_t* addr,
     int32_t val);

multimem.red.release.gpu.global.min.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     int32_t* addr,
     int32_t val);

multimem.red.release.sys.global.min.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     int32_t* addr,
     int32_t val);

multimem.red.relaxed.cta.global.min.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     int64_t* addr,
     int64_t val);

multimem.red.relaxed.cluster.global.min.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     int64_t* addr,
     int64_t val);

multimem.red.relaxed.gpu.global.min.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     int64_t* addr,
     int64_t val);

multimem.red.relaxed.sys.global.min.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     int64_t* addr,
     int64_t val);

multimem.red.release.cta.global.min.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     int64_t* addr,
     int64_t val);

multimem.red.release.cluster.global.min.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     int64_t* addr,
     int64_t val);

multimem.red.release.gpu.global.min.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     int64_t* addr,
     int64_t val);

multimem.red.release.sys.global.min.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     int64_t* addr,
     int64_t val);

multimem.red.relaxed.cta.global.max.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     uint32_t* addr,
     uint32_t val);

multimem.red.relaxed.cluster.global.max.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     uint32_t* addr,
     uint32_t val);

multimem.red.relaxed.gpu.global.max.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     uint32_t* addr,
     uint32_t val);

multimem.red.relaxed.sys.global.max.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     uint32_t* addr,
     uint32_t val);

multimem.red.release.cta.global.max.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     uint32_t* addr,
     uint32_t val);

multimem.red.release.cluster.global.max.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     uint32_t* addr,
     uint32_t val);

multimem.red.release.gpu.global.max.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     uint32_t* addr,
     uint32_t val);

multimem.red.release.sys.global.max.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     uint32_t* addr,
     uint32_t val);

multimem.red.relaxed.cta.global.max.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     uint64_t* addr,
     uint64_t val);

multimem.red.relaxed.cluster.global.max.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     uint64_t* addr,
     uint64_t val);

multimem.red.relaxed.gpu.global.max.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     uint64_t* addr,
     uint64_t val);

multimem.red.relaxed.sys.global.max.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     uint64_t* addr,
     uint64_t val);

multimem.red.release.cta.global.max.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     uint64_t* addr,
     uint64_t val);

multimem.red.release.cluster.global.max.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     uint64_t* addr,
     uint64_t val);

multimem.red.release.gpu.global.max.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     uint64_t* addr,
     uint64_t val);

multimem.red.release.sys.global.max.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     uint64_t* addr,
     uint64_t val);

multimem.red.relaxed.cta.global.max.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     int32_t* addr,
     int32_t val);

multimem.red.relaxed.cluster.global.max.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     int32_t* addr,
     int32_t val);

multimem.red.relaxed.gpu.global.max.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     int32_t* addr,
     int32_t val);

multimem.red.relaxed.sys.global.max.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     int32_t* addr,
     int32_t val);

multimem.red.release.cta.global.max.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     int32_t* addr,
     int32_t val);

multimem.red.release.cluster.global.max.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     int32_t* addr,
     int32_t val);

multimem.red.release.gpu.global.max.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     int32_t* addr,
     int32_t val);

multimem.red.release.sys.global.max.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     int32_t* addr,
     int32_t val);

multimem.red.relaxed.cta.global.max.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     int64_t* addr,
     int64_t val);

multimem.red.relaxed.cluster.global.max.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     int64_t* addr,
     int64_t val);

multimem.red.relaxed.gpu.global.max.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     int64_t* addr,
     int64_t val);

multimem.red.relaxed.sys.global.max.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     int64_t* addr,
     int64_t val);

multimem.red.release.cta.global.max.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     int64_t* addr,
     int64_t val);

multimem.red.release.cluster.global.max.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     int64_t* addr,
     int64_t val);

multimem.red.release.gpu.global.max.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     int64_t* addr,
     int64_t val);

multimem.red.release.sys.global.max.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     int64_t* addr,
     int64_t val);

multimem.red.relaxed.cta.global.add.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     uint32_t* addr,
     uint32_t val);

multimem.red.relaxed.cluster.global.add.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     uint32_t* addr,
     uint32_t val);

multimem.red.relaxed.gpu.global.add.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     uint32_t* addr,
     uint32_t val);

multimem.red.relaxed.sys.global.add.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     uint32_t* addr,
     uint32_t val);

multimem.red.release.cta.global.add.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     uint32_t* addr,
     uint32_t val);

multimem.red.release.cluster.global.add.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     uint32_t* addr,
     uint32_t val);

multimem.red.release.gpu.global.add.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     uint32_t* addr,
     uint32_t val);

multimem.red.release.sys.global.add.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     uint32_t* addr,
     uint32_t val);

multimem.red.relaxed.cta.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     uint64_t* addr,
     uint64_t val);

multimem.red.relaxed.cluster.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     uint64_t* addr,
     uint64_t val);

multimem.red.relaxed.gpu.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     uint64_t* addr,
     uint64_t val);

multimem.red.relaxed.sys.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     uint64_t* addr,
     uint64_t val);

multimem.red.release.cta.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     uint64_t* addr,
     uint64_t val);

multimem.red.release.cluster.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     uint64_t* addr,
     uint64_t val);

multimem.red.release.gpu.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     uint64_t* addr,
     uint64_t val);

multimem.red.release.sys.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     uint64_t* addr,
     uint64_t val);

multimem.red.relaxed.cta.global.add.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     int32_t* addr,
     int32_t val);

multimem.red.relaxed.cluster.global.add.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     int32_t* addr,
     int32_t val);

multimem.red.relaxed.gpu.global.add.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     int32_t* addr,
     int32_t val);

multimem.red.relaxed.sys.global.add.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     int32_t* addr,
     int32_t val);

multimem.red.release.cta.global.add.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     int32_t* addr,
     int32_t val);

multimem.red.release.cluster.global.add.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     int32_t* addr,
     int32_t val);

multimem.red.release.gpu.global.add.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     int32_t* addr,
     int32_t val);

multimem.red.release.sys.global.add.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.s32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     int32_t* addr,
     int32_t val);

multimem.red.relaxed.cta.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     int64_t* addr,
     int64_t val);

multimem.red.relaxed.cluster.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     int64_t* addr,
     int64_t val);

multimem.red.relaxed.gpu.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     int64_t* addr,
     int64_t val);

multimem.red.relaxed.sys.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     int64_t* addr,
     int64_t val);

multimem.red.release.cta.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     int64_t* addr,
     int64_t val);

multimem.red.release.cluster.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     int64_t* addr,
     int64_t val);

multimem.red.release.gpu.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     int64_t* addr,
     int64_t val);

multimem.red.release.sys.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.u64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     int64_t* addr,
     int64_t val);

multimem.red.relaxed.cta.global.and.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     B32* addr,
     B32 val);

multimem.red.relaxed.cluster.global.and.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     B32* addr,
     B32 val);

multimem.red.relaxed.gpu.global.and.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     B32* addr,
     B32 val);

multimem.red.relaxed.sys.global.and.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     B32* addr,
     B32 val);

multimem.red.release.cta.global.and.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     B32* addr,
     B32 val);

multimem.red.release.cluster.global.and.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     B32* addr,
     B32 val);

multimem.red.release.gpu.global.and.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     B32* addr,
     B32 val);

multimem.red.release.sys.global.and.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     B32* addr,
     B32 val);

multimem.red.relaxed.cta.global.or.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     B32* addr,
     B32 val);

multimem.red.relaxed.cluster.global.or.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     B32* addr,
     B32 val);

multimem.red.relaxed.gpu.global.or.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     B32* addr,
     B32 val);

multimem.red.relaxed.sys.global.or.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     B32* addr,
     B32 val);

multimem.red.release.cta.global.or.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     B32* addr,
     B32 val);

multimem.red.release.cluster.global.or.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     B32* addr,
     B32 val);

multimem.red.release.gpu.global.or.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     B32* addr,
     B32 val);

multimem.red.release.sys.global.or.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     B32* addr,
     B32 val);

multimem.red.relaxed.cta.global.xor.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     B32* addr,
     B32 val);

multimem.red.relaxed.cluster.global.xor.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     B32* addr,
     B32 val);

multimem.red.relaxed.gpu.global.xor.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     B32* addr,
     B32 val);

multimem.red.relaxed.sys.global.xor.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     B32* addr,
     B32 val);

multimem.red.release.cta.global.xor.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     B32* addr,
     B32 val);

multimem.red.release.cluster.global.xor.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     B32* addr,
     B32 val);

multimem.red.release.gpu.global.xor.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     B32* addr,
     B32 val);

multimem.red.release.sys.global.xor.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     B32* addr,
     B32 val);

multimem.red.relaxed.cta.global.and.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     B64* addr,
     B64 val);

multimem.red.relaxed.cluster.global.and.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     B64* addr,
     B64 val);

multimem.red.relaxed.gpu.global.and.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     B64* addr,
     B64 val);

multimem.red.relaxed.sys.global.and.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     B64* addr,
     B64 val);

multimem.red.release.cta.global.and.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     B64* addr,
     B64 val);

multimem.red.release.cluster.global.and.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     B64* addr,
     B64 val);

multimem.red.release.gpu.global.and.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     B64* addr,
     B64 val);

multimem.red.release.sys.global.and.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     B64* addr,
     B64 val);

multimem.red.relaxed.cta.global.or.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     B64* addr,
     B64 val);

multimem.red.relaxed.cluster.global.or.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     B64* addr,
     B64 val);

multimem.red.relaxed.gpu.global.or.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     B64* addr,
     B64 val);

multimem.red.relaxed.sys.global.or.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     B64* addr,
     B64 val);

multimem.red.release.cta.global.or.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     B64* addr,
     B64 val);

multimem.red.release.cluster.global.or.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     B64* addr,
     B64 val);

multimem.red.release.gpu.global.or.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     B64* addr,
     B64 val);

multimem.red.release.sys.global.or.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     B64* addr,
     B64 val);

multimem.red.relaxed.cta.global.xor.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     B64* addr,
     B64 val);

multimem.red.relaxed.cluster.global.xor.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     B64* addr,
     B64 val);

multimem.red.relaxed.gpu.global.xor.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     B64* addr,
     B64 val);

multimem.red.relaxed.sys.global.xor.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     B64* addr,
     B64 val);

multimem.red.release.cta.global.xor.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     B64* addr,
     B64 val);

multimem.red.release.cluster.global.xor.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     B64* addr,
     B64 val);

multimem.red.release.gpu.global.xor.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     B64* addr,
     B64 val);

multimem.red.release.sys.global.xor.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.red.sem.scope.global.op.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_red(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     B64* addr,
     B64 val);
