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

   - Use it when you need to **share shared memory across multiple primitives**
     or explicitly manage temp storage and synchronization. The Mamba selective
     scan example is a good reference: it pre-creates multiple primitives and
     uses a single shared-memory buffer for the pipeline.

4. How do I share temp storage across primitives?

   - Pre-create the primitives, compute a common size/alignment (or use
     :func:`coop.gpu_dataclass`), allocate a single
     :class:`coop.TempStorage`, and pass it to each primitive.

5. Why is my kernel compiling multiple times?

   - ``cuda.coop`` batches LTO-IR by default. If you disable bundling, each
     primitive may trigger NVRTC. Ensure
     ``NUMBA_CCCL_COOP_BUNDLE_LTOIR=1`` and use
     ``NUMBA_CCCL_COOP_NVRTC_COMPILE_COUNT=1`` to verify.
