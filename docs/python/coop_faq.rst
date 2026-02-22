.. _cccl-python-coop-faq:

FAQ
===

1. I keep seeing the terms single-phase and two-phase; what do they mean?

   - **Single-phase** means calling a primitive directly inside a kernel. The
     primitive is specialized on demand based on its call arguments.
   - **Two-phase** means pre-creating a primitive on the host, then invoking the
     instance inside a kernel. This lets you query temp storage sizes and share
     shared memory across primitives.

2. Which one should I use?

   - **Single-phase** unless you know you need the two-phase interface.

3. When would I need to use the two-phase interface?

   - Use it when you need tighter shared-memory coordination across multiple
     primitives. Typical cases:
       - Reusing one shared-memory allocation across a pipeline (for example,
         Mamba-style load/scan/reduce/store chains).
       - Querying temp-storage requirements up front.
       - Controlling synchronization and sharing policy explicitly
         (``auto_sync`` and ``sharing``).

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

5. Why is my kernel compiling multiple times?

   - ``cuda.coop`` batches LTO-IR by default. If you disable bundling, each
     primitive may trigger NVRTC. Ensure
     ``NUMBA_CCCL_COOP_BUNDLE_LTOIR=1`` and use
     ``NUMBA_CCCL_COOP_NVRTC_COMPILE_COUNT=1`` to verify.
