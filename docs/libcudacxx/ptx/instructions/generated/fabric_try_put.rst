..
   This file was automatically generated. Do not edit.

fabric.try_put.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_put.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.b128 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_put(
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const void* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_put.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.cp_mask.relaxed.sys.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_put.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.cp_mask.sem.scope.b128 [dstLeId, dstDataOff], [srcMem], size, [smem_bar], bytemask; // PTX ISA 93, SM_100
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_put_cp_mask(
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const void* srcMem,
     uint32_t size,
     uint64_t* smem_bar,
     uint16_t bytemask);

fabric.try_put.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_put.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.b128 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_put_counted(
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const void* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_put.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_put.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.b128 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_put_multimem(
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const void* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_put.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.cp_mask.relaxed.sys.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_put.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.cp_mask.sem.scope.b128 [dstLeId, dstDataOff], [srcMem], size, [smem_bar], bytemask; // PTX ISA 93, SM_100
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_put_multimem_cp_mask(
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const void* srcMem,
     uint32_t size,
     uint64_t* smem_bar,
     uint16_t bytemask);

fabric.try_put.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.b128
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_put.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.b128 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_put_multimem_counted(
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const void* srcMem,
     uint32_t size,
     uint64_t* smem_bar);
