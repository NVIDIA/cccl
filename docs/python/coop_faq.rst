.. _cccl-python-coop-faq:

FAQ
===

1. I keep seeing the terms single-phase and two-phase; what do they mean?

   - **Single-phase** means calling a primitive directly inside a kernel. The
     primitive is specialized on demand based on its call arguments.
   - **Two-phase** means pre-creating a primitive on the host, then invoking the
     instance inside a kernel. This still works, but it is now a more advanced
     interface rather than the default path.

   Single-phase example:

   .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_load_store_api.py
      :language: python
      :dedent:
      :start-after: example-begin load-store-single-phase
      :end-before: example-end load-store-single-phase

   Two-phase example:

   .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_load_store_api.py
      :language: python
      :dedent:
      :start-after: example-begin load-store-two-phase
      :end-before: example-end load-store-two-phase

2. Which one should I use?

   - **Single-phase** unless you know you need the two-phase interface.

3. When would I need to use the two-phase interface?

   - Much less often than before. In most kernels, single-phase plus
     :class:`coop.TempStorage` and :class:`coop.ThreadData` is enough.
   - Reach for two-phase when you specifically need host-side primitive objects,
     such as:
       - Querying temp-storage size and alignment before JIT-compiling a kernel.
       - Building trait-style helper objects with :func:`coop.gpu_dataclass`.
       - Reusing a pre-created primitive instance across several kernels or call
         sites.
   - You do not need two-phase just to share temporary storage, to use
     ``foo[temp_storage](...)`` sugar, or to let cuda.coop infer size,
     alignment, dtype, and synchronization behavior.

4. How do I share temp storage across primitives?

   - ``cuda.coop`` supports three common patterns:
       - **Pre-computed size/alignment**: pre-create primitives, compute a
         common size/alignment (or use :func:`coop.gpu_dataclass`), then pass a
         single :class:`coop.TempStorage` to each primitive.
       - **Inference from omitted size/alignment**: use
         ``temp_storage = coop.TempStorage()`` and let rewrite infer required
         size/alignment from the participating primitive calls.
       - **Getitem sugar**: use ``primitive[temp_storage](...)`` when you want
         to bind temp storage at the call site without repeating
         ``temp_storage=...``.

5. Is temp storage just another name for shared memory?

   - Usually, yes. For the current cuda.coop block and warp primitives, temp
     storage maps to the temporary on-chip storage that CUB refers to as
     ``TempStorage``, which is generally shared memory.
   - We keep the name ``TempStorage`` because it matches the established CUB C++
     terminology and API shape.
   - ``coop.TempStorage()`` is higher-level than writing ``cuda.shared.array()``
     by hand: it can infer size and alignment, enforce sharing policy, and
     inject synchronization when ``auto_sync=True``.
   - Manual ``cuda.shared.array()`` is still available when you need complete
     control over layout or are interfacing with non-coop code.

   Single-phase temp-storage inference:

   .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_load_store_api.py
      :language: python
      :dedent:
      :start-after: example-begin load_store_single_phase_implicit_temp_storage_kernel
      :end-before: example-end load_store_single_phase_implicit_temp_storage_usage

6. Why is my kernel compiling multiple times?

   - ``cuda.coop`` batches LTO-IR by default. If you disable bundling, each
     primitive may trigger NVRTC. Ensure
     ``CUDA_COOP_BUNDLE_LTOIR=1`` and use
     ``CUDA_COOP_NVRTC_COMPILE_COUNT=1`` to verify.
