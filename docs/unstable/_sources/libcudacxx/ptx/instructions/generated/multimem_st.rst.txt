..
   This file was automatically generated. Do not edit.

multimem.st.weak.global.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.st.sem.global.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .weak }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void multimem_st(
     cuda::ptx::sem_weak_t,
     B32* addr,
     B32 val);

multimem.st.relaxed.cta.global.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.st.sem.scope.global.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_st(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     B32* addr,
     B32 val);

multimem.st.relaxed.cluster.global.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.st.sem.scope.global.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_st(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     B32* addr,
     B32 val);

multimem.st.relaxed.gpu.global.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.st.sem.scope.global.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_st(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     B32* addr,
     B32 val);

multimem.st.relaxed.sys.global.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.st.sem.scope.global.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_st(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     B32* addr,
     B32 val);

multimem.st.release.cta.global.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.st.sem.scope.global.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_st(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     B32* addr,
     B32 val);

multimem.st.release.cluster.global.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.st.sem.scope.global.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_st(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     B32* addr,
     B32 val);

multimem.st.release.gpu.global.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.st.sem.scope.global.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_st(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     B32* addr,
     B32 val);

multimem.st.release.sys.global.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.st.sem.scope.global.b32 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_st(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     B32* addr,
     B32 val);

multimem.st.weak.global.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.st.sem.global.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .weak }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline void multimem_st(
     cuda::ptx::sem_weak_t,
     B64* addr,
     B64 val);

multimem.st.relaxed.cta.global.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.st.sem.scope.global.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_st(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     B64* addr,
     B64 val);

multimem.st.relaxed.cluster.global.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.st.sem.scope.global.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_st(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     B64* addr,
     B64 val);

multimem.st.relaxed.gpu.global.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.st.sem.scope.global.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_st(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     B64* addr,
     B64 val);

multimem.st.relaxed.sys.global.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.st.sem.scope.global.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_st(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     B64* addr,
     B64 val);

multimem.st.release.cta.global.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.st.sem.scope.global.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_st(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     B64* addr,
     B64 val);

multimem.st.release.cluster.global.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.st.sem.scope.global.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_st(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     B64* addr,
     B64 val);

multimem.st.release.gpu.global.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.st.sem.scope.global.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_st(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     B64* addr,
     B64 val);

multimem.st.release.sys.global.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // multimem.st.sem.scope.global.b64 [addr], val; // PTX ISA 81, SM_90
   // .sem       = { .relaxed, .release }
   // .scope     = { .cta, .cluster, .gpu, .sys }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true, cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
   __device__ static inline void multimem_st(
     cuda::ptx::sem_t<Sem> sem,
     cuda::ptx::scope_t<Scope> scope,
     B64* addr,
     B64 val);
