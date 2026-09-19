..
   This file was automatically generated. Do not edit.

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.and.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b32 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .and }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void fabric_try_red(
     cuda::ptx::op_and_op_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const B32* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.and.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.b32 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .and }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void fabric_try_red_counted(
     cuda::ptx::op_and_op_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const B32* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.and.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b32 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .and }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void fabric_try_red_multimem(
     cuda::ptx::op_and_op_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const B32* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.and.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.b32 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .and }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void fabric_try_red_multimem_counted(
     cuda::ptx::op_and_op_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const B32* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.xor.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b32 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .xor }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void fabric_try_red(
     cuda::ptx::op_xor_op_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const B32* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.xor.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.b32 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .xor }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void fabric_try_red_counted(
     cuda::ptx::op_xor_op_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const B32* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.xor.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b32 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .xor }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void fabric_try_red_multimem(
     cuda::ptx::op_xor_op_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const B32* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.xor.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.b32 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .xor }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void fabric_try_red_multimem_counted(
     cuda::ptx::op_xor_op_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const B32* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.or.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b32 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .or }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void fabric_try_red(
     cuda::ptx::op_or_op_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const B32* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.or.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.b32 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .or }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void fabric_try_red_counted(
     cuda::ptx::op_or_op_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const B32* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.or.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b32 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .or }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void fabric_try_red_multimem(
     cuda::ptx::op_or_op_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const B32* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.or.b32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.b32 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .or }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void fabric_try_red_multimem_counted(
     cuda::ptx::op_or_op_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const B32* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.and.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b64 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .and }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline void fabric_try_red(
     cuda::ptx::op_and_op_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const B64* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.and.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.b64 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .and }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline void fabric_try_red_counted(
     cuda::ptx::op_and_op_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const B64* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.and.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b64 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .and }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline void fabric_try_red_multimem(
     cuda::ptx::op_and_op_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const B64* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.and.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.b64 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .and }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline void fabric_try_red_multimem_counted(
     cuda::ptx::op_and_op_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const B64* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.xor.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b64 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .xor }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline void fabric_try_red(
     cuda::ptx::op_xor_op_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const B64* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.xor.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.b64 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .xor }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline void fabric_try_red_counted(
     cuda::ptx::op_xor_op_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const B64* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.xor.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b64 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .xor }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline void fabric_try_red_multimem(
     cuda::ptx::op_xor_op_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const B64* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.xor.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.b64 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .xor }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline void fabric_try_red_multimem_counted(
     cuda::ptx::op_xor_op_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const B64* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.or.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b64 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .or }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline void fabric_try_red(
     cuda::ptx::op_or_op_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const B64* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.or.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.b64 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .or }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline void fabric_try_red_counted(
     cuda::ptx::op_or_op_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const B64* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.or.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.b64 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .or }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline void fabric_try_red_multimem(
     cuda::ptx::op_or_op_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const B64* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.or.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.b64 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .or }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename B64, enable_if_t<sizeof(B64) == 8, bool> = true>
   __device__ static inline void fabric_try_red_multimem_counted(
     cuda::ptx::op_or_op_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const B64* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u32 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .min }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red(
     cuda::ptx::op_min_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const uint32_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.min.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.u32 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .min }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_counted(
     cuda::ptx::op_min_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const uint32_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u32 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .min }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem(
     cuda::ptx::op_min_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const uint32_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.min.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.u32 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .min }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem_counted(
     cuda::ptx::op_min_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const uint32_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u32 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .max }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red(
     cuda::ptx::op_max_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const uint32_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.max.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.u32 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .max }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_counted(
     cuda::ptx::op_max_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const uint32_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u32 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .max }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem(
     cuda::ptx::op_max_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const uint32_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.max.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.u32 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .max }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem_counted(
     cuda::ptx::op_max_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const uint32_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.s32 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .min }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red(
     cuda::ptx::op_min_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const int32_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.min.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.s32 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .min }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_counted(
     cuda::ptx::op_min_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const int32_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.s32 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .min }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem(
     cuda::ptx::op_min_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const int32_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.min.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.s32 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .min }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem_counted(
     cuda::ptx::op_min_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const int32_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.s32 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .max }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red(
     cuda::ptx::op_max_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const int32_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.max.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.s32 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .max }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_counted(
     cuda::ptx::op_max_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const int32_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.s32 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .max }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem(
     cuda::ptx::op_max_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const int32_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.max.s32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.s32 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .max }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem_counted(
     cuda::ptx::op_max_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const int32_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u64 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .min }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red(
     cuda::ptx::op_min_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const uint64_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.min.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.u64 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .min }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_counted(
     cuda::ptx::op_min_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const uint64_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u64 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .min }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem(
     cuda::ptx::op_min_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const uint64_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.min.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.u64 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .min }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem_counted(
     cuda::ptx::op_min_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const uint64_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u64 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .max }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red(
     cuda::ptx::op_max_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const uint64_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.max.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.u64 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .max }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_counted(
     cuda::ptx::op_max_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const uint64_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u64 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .max }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem(
     cuda::ptx::op_max_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const uint64_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.max.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.u64 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .max }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem_counted(
     cuda::ptx::op_max_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const uint64_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.s64 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .min }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red(
     cuda::ptx::op_min_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const int64_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.min.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.s64 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .min }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_counted(
     cuda::ptx::op_min_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const int64_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.s64 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .min }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem(
     cuda::ptx::op_min_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const int64_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.min.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.s64 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .min }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem_counted(
     cuda::ptx::op_min_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const int64_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.s64 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .max }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red(
     cuda::ptx::op_max_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const int64_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.max.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.s64 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .max }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_counted(
     cuda::ptx::op_max_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const int64_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.s64 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .max }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem(
     cuda::ptx::op_max_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const int64_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.max.s64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.s64 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .max }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem_counted(
     cuda::ptx::op_max_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const int64_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.f16 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .min }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red(
     cuda::ptx::op_min_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const __half* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.min.f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.f16 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .min }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_counted(
     cuda::ptx::op_min_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const __half* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.f16 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .min }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem(
     cuda::ptx::op_min_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const __half* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.min.f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.f16 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .min }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem_counted(
     cuda::ptx::op_min_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const __half* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.f16 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .max }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red(
     cuda::ptx::op_max_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const __half* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.max.f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.f16 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .max }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_counted(
     cuda::ptx::op_max_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const __half* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.f16 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .max }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem(
     cuda::ptx::op_max_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const __half* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.max.f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.f16 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .max }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem_counted(
     cuda::ptx::op_max_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const __half* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.bf16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.bf16 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .min }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red(
     cuda::ptx::op_min_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const __nv_bfloat16* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.min.bf16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.bf16 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .min }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_counted(
     cuda::ptx::op_min_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const __nv_bfloat16* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.min.bf16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.bf16 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .min }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem(
     cuda::ptx::op_min_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const __nv_bfloat16* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.min.bf16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.bf16 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .min }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem_counted(
     cuda::ptx::op_min_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const __nv_bfloat16* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.bf16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.bf16 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .max }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red(
     cuda::ptx::op_max_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const __nv_bfloat16* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.max.bf16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.bf16 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .max }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_counted(
     cuda::ptx::op_max_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const __nv_bfloat16* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.bf16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.bf16 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .max }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem(
     cuda::ptx::op_max_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const __nv_bfloat16* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.max.bf16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.bf16 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .max }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem_counted(
     cuda::ptx::op_max_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const __nv_bfloat16* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u32 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .add }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red(
     cuda::ptx::op_add_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const uint32_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.add.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.u32 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .add }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_counted(
     cuda::ptx::op_add_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const uint32_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u32 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .add }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem(
     cuda::ptx::op_add_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const uint32_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.add.u32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.u32 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .add }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem_counted(
     cuda::ptx::op_add_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const uint32_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u64 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .add }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red(
     cuda::ptx::op_add_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const uint64_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.u64 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .add }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_counted(
     cuda::ptx::op_add_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const uint64_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.u64 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .add }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem(
     cuda::ptx::op_add_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const uint64_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.add.u64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.u64 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .add }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem_counted(
     cuda::ptx::op_add_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const uint64_t* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.f16 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .add }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red(
     cuda::ptx::op_add_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const __half* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.add.f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.f16 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .add }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_counted(
     cuda::ptx::op_add_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const __half* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.f16 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .add }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem(
     cuda::ptx::op_add_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const __half* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.add.f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.f16 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .add }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem_counted(
     cuda::ptx::op_add_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const __half* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.bf16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.bf16 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .add }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red(
     cuda::ptx::op_add_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const __nv_bfloat16* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.add.bf16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.bf16 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .add }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_counted(
     cuda::ptx::op_add_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const __nv_bfloat16* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.bf16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.bf16 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .add }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem(
     cuda::ptx::op_add_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const __nv_bfloat16* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.add.bf16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.bf16 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .add }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem_counted(
     cuda::ptx::op_add_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const __nv_bfloat16* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.f32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.f32 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .add }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red(
     cuda::ptx::op_add_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const float* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.add.f32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.f32 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .add }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_counted(
     cuda::ptx::op_add_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const float* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.f32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.f32 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .add }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem(
     cuda::ptx::op_add_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const float* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.add.f32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.f32 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .add }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem_counted(
     cuda::ptx::op_add_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const float* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.f64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.f64 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .add }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red(
     cuda::ptx::op_add_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const double* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.add.f64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.f64 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .add }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_counted(
     cuda::ptx::op_add_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const double* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.f64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.sem.scope.op.f64 [dstLeId, dstDataOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .add }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem(
     cuda::ptx::op_add_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     const double* srcMem,
     uint32_t size,
     uint64_t* smem_bar);

fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.add.f64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // fabric.try_red.async.multimem.src.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.sem.scope.op.f64 [dstLeId, dstDataOff, dstCounterOff], [srcMem], size, [smem_bar]; // PTX ISA 93, SM_100
   // .op        = { .add }
   // .src       = { .shared::cta }
   // .sem       = { .relaxed }
   // .scope     = { .sys }
   template <typename = void>
   __device__ static inline void fabric_try_red_multimem_counted(
     cuda::ptx::op_add_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::sem_relaxed_t,
     cuda::ptx::scope_sys_t,
     uint32_t dstLeId,
     uint64_t dstDataOff,
     uint64_t dstCounterOff,
     const double* srcMem,
     uint32_t size,
     uint64_t* smem_bar);
