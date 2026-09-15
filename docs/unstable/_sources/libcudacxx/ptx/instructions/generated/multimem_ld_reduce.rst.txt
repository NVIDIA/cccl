..
   This file was automatically generated. Do not edit.

multimem.ld_reduce.weak.global.min.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .weak }
   // .op        = { .min }
   template <typename = void>
   __device__ static inline uint32_t multimem_ld_reduce(
     cuda::ptx::sem_weak_t,
     cuda::ptx::op_min_t,
     const uint32_t* addr);

multimem.ld_reduce.relaxed.cta.global.min.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const uint32_t* addr);

multimem.ld_reduce.relaxed.cluster.global.min.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const uint32_t* addr);

multimem.ld_reduce.relaxed.gpu.global.min.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const uint32_t* addr);

multimem.ld_reduce.relaxed.sys.global.min.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const uint32_t* addr);

multimem.ld_reduce.acquire.cta.global.min.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const uint32_t* addr);

multimem.ld_reduce.acquire.cluster.global.min.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const uint32_t* addr);

multimem.ld_reduce.acquire.gpu.global.min.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const uint32_t* addr);

multimem.ld_reduce.acquire.sys.global.min.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const uint32_t* addr);

multimem.ld_reduce.weak.global.min.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .weak }
   // .op        = { .min }
   template <typename = void>
   __device__ static inline uint64_t multimem_ld_reduce(
     cuda::ptx::sem_weak_t,
     cuda::ptx::op_min_t,
     const uint64_t* addr);

multimem.ld_reduce.relaxed.cta.global.min.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const uint64_t* addr);

multimem.ld_reduce.relaxed.cluster.global.min.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const uint64_t* addr);

multimem.ld_reduce.relaxed.gpu.global.min.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const uint64_t* addr);

multimem.ld_reduce.relaxed.sys.global.min.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const uint64_t* addr);

multimem.ld_reduce.acquire.cta.global.min.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const uint64_t* addr);

multimem.ld_reduce.acquire.cluster.global.min.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const uint64_t* addr);

multimem.ld_reduce.acquire.gpu.global.min.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const uint64_t* addr);

multimem.ld_reduce.acquire.sys.global.min.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const uint64_t* addr);

multimem.ld_reduce.weak.global.min.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .weak }
   // .op        = { .min }
   template <typename = void>
   __device__ static inline int32_t multimem_ld_reduce(
     cuda::ptx::sem_weak_t,
     cuda::ptx::op_min_t,
     const int32_t* addr);

multimem.ld_reduce.relaxed.cta.global.min.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const int32_t* addr);

multimem.ld_reduce.relaxed.cluster.global.min.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const int32_t* addr);

multimem.ld_reduce.relaxed.gpu.global.min.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const int32_t* addr);

multimem.ld_reduce.relaxed.sys.global.min.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const int32_t* addr);

multimem.ld_reduce.acquire.cta.global.min.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const int32_t* addr);

multimem.ld_reduce.acquire.cluster.global.min.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const int32_t* addr);

multimem.ld_reduce.acquire.gpu.global.min.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const int32_t* addr);

multimem.ld_reduce.acquire.sys.global.min.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const int32_t* addr);

multimem.ld_reduce.weak.global.min.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.global.op.s64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .weak }
   // .op        = { .min }
   template <typename = void>
   __device__ static inline int64_t multimem_ld_reduce(
     cuda::ptx::sem_weak_t,
     cuda::ptx::op_min_t,
     const int64_t* addr);

multimem.ld_reduce.relaxed.cta.global.min.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const int64_t* addr);

multimem.ld_reduce.relaxed.cluster.global.min.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const int64_t* addr);

multimem.ld_reduce.relaxed.gpu.global.min.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const int64_t* addr);

multimem.ld_reduce.relaxed.sys.global.min.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const int64_t* addr);

multimem.ld_reduce.acquire.cta.global.min.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const int64_t* addr);

multimem.ld_reduce.acquire.cluster.global.min.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const int64_t* addr);

multimem.ld_reduce.acquire.gpu.global.min.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const int64_t* addr);

multimem.ld_reduce.acquire.sys.global.min.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .min }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_min_t,
     const int64_t* addr);

multimem.ld_reduce.weak.global.max.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .weak }
   // .op        = { .max }
   template <typename = void>
   __device__ static inline uint32_t multimem_ld_reduce(
     cuda::ptx::sem_weak_t,
     cuda::ptx::op_max_t,
     const uint32_t* addr);

multimem.ld_reduce.relaxed.cta.global.max.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const uint32_t* addr);

multimem.ld_reduce.relaxed.cluster.global.max.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const uint32_t* addr);

multimem.ld_reduce.relaxed.gpu.global.max.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const uint32_t* addr);

multimem.ld_reduce.relaxed.sys.global.max.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const uint32_t* addr);

multimem.ld_reduce.acquire.cta.global.max.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const uint32_t* addr);

multimem.ld_reduce.acquire.cluster.global.max.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const uint32_t* addr);

multimem.ld_reduce.acquire.gpu.global.max.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const uint32_t* addr);

multimem.ld_reduce.acquire.sys.global.max.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const uint32_t* addr);

multimem.ld_reduce.weak.global.max.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .weak }
   // .op        = { .max }
   template <typename = void>
   __device__ static inline uint64_t multimem_ld_reduce(
     cuda::ptx::sem_weak_t,
     cuda::ptx::op_max_t,
     const uint64_t* addr);

multimem.ld_reduce.relaxed.cta.global.max.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const uint64_t* addr);

multimem.ld_reduce.relaxed.cluster.global.max.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const uint64_t* addr);

multimem.ld_reduce.relaxed.gpu.global.max.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const uint64_t* addr);

multimem.ld_reduce.relaxed.sys.global.max.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const uint64_t* addr);

multimem.ld_reduce.acquire.cta.global.max.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const uint64_t* addr);

multimem.ld_reduce.acquire.cluster.global.max.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const uint64_t* addr);

multimem.ld_reduce.acquire.gpu.global.max.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const uint64_t* addr);

multimem.ld_reduce.acquire.sys.global.max.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const uint64_t* addr);

multimem.ld_reduce.weak.global.max.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .weak }
   // .op        = { .max }
   template <typename = void>
   __device__ static inline int32_t multimem_ld_reduce(
     cuda::ptx::sem_weak_t,
     cuda::ptx::op_max_t,
     const int32_t* addr);

multimem.ld_reduce.relaxed.cta.global.max.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const int32_t* addr);

multimem.ld_reduce.relaxed.cluster.global.max.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const int32_t* addr);

multimem.ld_reduce.relaxed.gpu.global.max.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const int32_t* addr);

multimem.ld_reduce.relaxed.sys.global.max.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const int32_t* addr);

multimem.ld_reduce.acquire.cta.global.max.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const int32_t* addr);

multimem.ld_reduce.acquire.cluster.global.max.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const int32_t* addr);

multimem.ld_reduce.acquire.gpu.global.max.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const int32_t* addr);

multimem.ld_reduce.acquire.sys.global.max.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const int32_t* addr);

multimem.ld_reduce.weak.global.max.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.global.op.s64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .weak }
   // .op        = { .max }
   template <typename = void>
   __device__ static inline int64_t multimem_ld_reduce(
     cuda::ptx::sem_weak_t,
     cuda::ptx::op_max_t,
     const int64_t* addr);

multimem.ld_reduce.relaxed.cta.global.max.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const int64_t* addr);

multimem.ld_reduce.relaxed.cluster.global.max.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const int64_t* addr);

multimem.ld_reduce.relaxed.gpu.global.max.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const int64_t* addr);

multimem.ld_reduce.relaxed.sys.global.max.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const int64_t* addr);

multimem.ld_reduce.acquire.cta.global.max.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const int64_t* addr);

multimem.ld_reduce.acquire.cluster.global.max.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const int64_t* addr);

multimem.ld_reduce.acquire.gpu.global.max.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const int64_t* addr);

multimem.ld_reduce.acquire.sys.global.max.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .max }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_max_t,
     const int64_t* addr);

multimem.ld_reduce.weak.global.add.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .weak }
   // .op        = { .add }
   template <typename = void>
   __device__ static inline uint32_t multimem_ld_reduce(
     cuda::ptx::sem_weak_t,
     cuda::ptx::op_add_t,
     const uint32_t* addr);

multimem.ld_reduce.relaxed.cta.global.add.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const uint32_t* addr);

multimem.ld_reduce.relaxed.cluster.global.add.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const uint32_t* addr);

multimem.ld_reduce.relaxed.gpu.global.add.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const uint32_t* addr);

multimem.ld_reduce.relaxed.sys.global.add.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const uint32_t* addr);

multimem.ld_reduce.acquire.cta.global.add.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const uint32_t* addr);

multimem.ld_reduce.acquire.cluster.global.add.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const uint32_t* addr);

multimem.ld_reduce.acquire.gpu.global.add.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const uint32_t* addr);

multimem.ld_reduce.acquire.sys.global.add.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const uint32_t* addr);

multimem.ld_reduce.weak.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .weak }
   // .op        = { .add }
   template <typename = void>
   __device__ static inline uint64_t multimem_ld_reduce(
     cuda::ptx::sem_weak_t,
     cuda::ptx::op_add_t,
     const uint64_t* addr);

multimem.ld_reduce.relaxed.cta.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const uint64_t* addr);

multimem.ld_reduce.relaxed.cluster.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const uint64_t* addr);

multimem.ld_reduce.relaxed.gpu.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const uint64_t* addr);

multimem.ld_reduce.relaxed.sys.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const uint64_t* addr);

multimem.ld_reduce.acquire.cta.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const uint64_t* addr);

multimem.ld_reduce.acquire.cluster.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const uint64_t* addr);

multimem.ld_reduce.acquire.gpu.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const uint64_t* addr);

multimem.ld_reduce.acquire.sys.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline uint64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const uint64_t* addr);

multimem.ld_reduce.weak.global.add.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .weak }
   // .op        = { .add }
   template <typename = void>
   __device__ static inline int32_t multimem_ld_reduce(
     cuda::ptx::sem_weak_t,
     cuda::ptx::op_add_t,
     const int32_t* addr);

multimem.ld_reduce.relaxed.cta.global.add.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const int32_t* addr);

multimem.ld_reduce.relaxed.cluster.global.add.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const int32_t* addr);

multimem.ld_reduce.relaxed.gpu.global.add.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const int32_t* addr);

multimem.ld_reduce.relaxed.sys.global.add.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const int32_t* addr);

multimem.ld_reduce.acquire.cta.global.add.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const int32_t* addr);

multimem.ld_reduce.acquire.cluster.global.add.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const int32_t* addr);

multimem.ld_reduce.acquire.gpu.global.add.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const int32_t* addr);

multimem.ld_reduce.acquire.sys.global.add.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.s32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int32_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const int32_t* addr);

multimem.ld_reduce.weak.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .weak }
   // .op        = { .add }
   template <typename = void>
   __device__ static inline int64_t multimem_ld_reduce(
     cuda::ptx::sem_weak_t,
     cuda::ptx::op_add_t,
     const int64_t* addr);

multimem.ld_reduce.relaxed.cta.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const int64_t* addr);

multimem.ld_reduce.relaxed.cluster.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const int64_t* addr);

multimem.ld_reduce.relaxed.gpu.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const int64_t* addr);

multimem.ld_reduce.relaxed.sys.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const int64_t* addr);

multimem.ld_reduce.acquire.cta.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const int64_t* addr);

multimem.ld_reduce.acquire.cluster.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const int64_t* addr);

multimem.ld_reduce.acquire.gpu.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const int64_t* addr);

multimem.ld_reduce.acquire.sys.global.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.u64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .add }
   template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline int64_t multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_add_t,
     const int64_t* addr);

multimem.ld_reduce.weak.global.and.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .weak }
   // .op        = { .and }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 multimem_ld_reduce(
     cuda::ptx::sem_weak_t,
     cuda::ptx::op_and_op_t,
     const B32* addr);

multimem.ld_reduce.relaxed.cta.global.and.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B32 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     const B32* addr);

multimem.ld_reduce.relaxed.cluster.global.and.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B32 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     const B32* addr);

multimem.ld_reduce.relaxed.gpu.global.and.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B32 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     const B32* addr);

multimem.ld_reduce.relaxed.sys.global.and.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B32 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     const B32* addr);

multimem.ld_reduce.acquire.cta.global.and.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B32 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     const B32* addr);

multimem.ld_reduce.acquire.cluster.global.and.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B32 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     const B32* addr);

multimem.ld_reduce.acquire.gpu.global.and.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B32 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     const B32* addr);

multimem.ld_reduce.acquire.sys.global.and.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B32 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     const B32* addr);

multimem.ld_reduce.weak.global.or.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .weak }
   // .op        = { .or }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 multimem_ld_reduce(
     cuda::ptx::sem_weak_t,
     cuda::ptx::op_or_op_t,
     const B32* addr);

multimem.ld_reduce.relaxed.cta.global.or.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B32 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     const B32* addr);

multimem.ld_reduce.relaxed.cluster.global.or.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B32 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     const B32* addr);

multimem.ld_reduce.relaxed.gpu.global.or.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B32 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     const B32* addr);

multimem.ld_reduce.relaxed.sys.global.or.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B32 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     const B32* addr);

multimem.ld_reduce.acquire.cta.global.or.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B32 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     const B32* addr);

multimem.ld_reduce.acquire.cluster.global.or.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B32 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     const B32* addr);

multimem.ld_reduce.acquire.gpu.global.or.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B32 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     const B32* addr);

multimem.ld_reduce.acquire.sys.global.or.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B32 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     const B32* addr);

multimem.ld_reduce.weak.global.xor.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .weak }
   // .op        = { .xor }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline B32 multimem_ld_reduce(
     cuda::ptx::sem_weak_t,
     cuda::ptx::op_xor_op_t,
     const B32* addr);

multimem.ld_reduce.relaxed.cta.global.xor.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B32 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     const B32* addr);

multimem.ld_reduce.relaxed.cluster.global.xor.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B32 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     const B32* addr);

multimem.ld_reduce.relaxed.gpu.global.xor.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B32 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     const B32* addr);

multimem.ld_reduce.relaxed.sys.global.xor.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B32 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     const B32* addr);

multimem.ld_reduce.acquire.cta.global.xor.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B32 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     const B32* addr);

multimem.ld_reduce.acquire.cluster.global.xor.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B32 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     const B32* addr);

multimem.ld_reduce.acquire.gpu.global.xor.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B32 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     const B32* addr);

multimem.ld_reduce.acquire.sys.global.xor.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b32 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B32 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     const B32* addr);

multimem.ld_reduce.weak.global.and.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .weak }
   // .op        = { .and }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 multimem_ld_reduce(
     cuda::ptx::sem_weak_t,
     cuda::ptx::op_and_op_t,
     const B64* addr);

multimem.ld_reduce.relaxed.cta.global.and.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B64 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     const B64* addr);

multimem.ld_reduce.relaxed.cluster.global.and.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B64 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     const B64* addr);

multimem.ld_reduce.relaxed.gpu.global.and.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B64 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     const B64* addr);

multimem.ld_reduce.relaxed.sys.global.and.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B64 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     const B64* addr);

multimem.ld_reduce.acquire.cta.global.and.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B64 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     const B64* addr);

multimem.ld_reduce.acquire.cluster.global.and.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B64 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     const B64* addr);

multimem.ld_reduce.acquire.gpu.global.and.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B64 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     const B64* addr);

multimem.ld_reduce.acquire.sys.global.and.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .and }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B64 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_and_op_t,
     const B64* addr);

multimem.ld_reduce.weak.global.or.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .weak }
   // .op        = { .or }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 multimem_ld_reduce(
     cuda::ptx::sem_weak_t,
     cuda::ptx::op_or_op_t,
     const B64* addr);

multimem.ld_reduce.relaxed.cta.global.or.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B64 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     const B64* addr);

multimem.ld_reduce.relaxed.cluster.global.or.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B64 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     const B64* addr);

multimem.ld_reduce.relaxed.gpu.global.or.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B64 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     const B64* addr);

multimem.ld_reduce.relaxed.sys.global.or.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B64 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     const B64* addr);

multimem.ld_reduce.acquire.cta.global.or.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B64 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     const B64* addr);

multimem.ld_reduce.acquire.cluster.global.or.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B64 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     const B64* addr);

multimem.ld_reduce.acquire.gpu.global.or.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B64 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     const B64* addr);

multimem.ld_reduce.acquire.sys.global.or.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .or }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B64 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_or_op_t,
     const B64* addr);

multimem.ld_reduce.weak.global.xor.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .weak }
   // .op        = { .xor }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline B64 multimem_ld_reduce(
     cuda::ptx::sem_weak_t,
     cuda::ptx::op_xor_op_t,
     const B64* addr);

multimem.ld_reduce.relaxed.cta.global.xor.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B64 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     const B64* addr);

multimem.ld_reduce.relaxed.cluster.global.xor.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B64 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     const B64* addr);

multimem.ld_reduce.relaxed.gpu.global.xor.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B64 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     const B64* addr);

multimem.ld_reduce.relaxed.sys.global.xor.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B64 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     const B64* addr);

multimem.ld_reduce.acquire.cta.global.xor.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B64 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     const B64* addr);

multimem.ld_reduce.acquire.cluster.global.xor.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B64 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     const B64* addr);

multimem.ld_reduce.acquire.gpu.global.xor.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B64 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     const B64* addr);

multimem.ld_reduce.acquire.sys.global.xor.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.ld_reduce.sem.scope.global.op.b64 dest, [addr]; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .acquire }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   // .op        = { .xor }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline B64 multimem_ld_reduce(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     cuda::ptx::op_xor_op_t,
     const B64* addr);
