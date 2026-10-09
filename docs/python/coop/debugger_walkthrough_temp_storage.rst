.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _cuda.coop.debugger_walkthrough_temp_storage:

Temporary Storage Debugger Walkthrough
======================================

Follow temporary storage from a kernel's primitive calls through layout,
synchronization, and the launch's dynamic shared-memory requirement. This
continues the :ref:`tile-copy walkthrough <cuda.coop.debugger_walkthrough>`
with several storage-consuming primitives. The breakpoints stop in the host
Python compiler and launcher; GPU execution is checked by the example's
output assertions.

Open :download:`debugger_walkthrough_temp_storage.py
<debugger_walkthrough_temp_storage.py>` in the same checkout as this guide.
Each case runs a small, deterministic kernel and checks its output. For a
one-hour review, use this order:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Minutes
     - Case
     - Review focus
   * - 0--5
     - Setup
     - Confirm the interpreter and source checkout; set the dynamic stops.
   * - 5--30
     - ``dynamic``
     - Follow an allocation beyond the static limit into launch metadata.
   * - 30--45
     - ``shared-auto``, ``shared-manual``
     - Compare one shared region with automatic and caller-placed barriers.
   * - 45--55
     - ``exclusive``, ``independent``
     - Compare per-call slices with separate descriptor regions.
   * - 55--60
     - Wrap-up
     - Record layout, barrier, and launch questions requiring follow-up.

Setup
-----

Open this CCCL checkout as the VS Code folder. Select a Python interpreter
with the :ref:`Numba-CUDA-MLIR dependencies <coop-numba-requirements>` and
the Python Debugger extension installed. Choose **cuda.coop: Temp storage
(dynamic first)** in Run and Debug. **cuda.coop: Temp storage (choose case)**
offers the remaining cases.

Both configurations disable the provider disk cache, allow stepping into
dependencies with ``justMyCode: false``, and write generated C++ under
``build/coop-temp-storage-sources`` in this checkout. ``subProcess: false``
keeps debugpy from attaching to CUDA library-discovery probes, which can
otherwise time out. Restart debugging between cases to avoid reusing an
in-process specialization.

Set a host breakpoint on ``kernel[BLOCKS, BLOCK_THREADS](*args)`` in
``run_case()``. At that stop, evaluate ``coop.__file__``. It must point into
this checkout's ``python/cuda_coop/cuda/coop`` directory. An editable install
from another checkout can override ``PYTHONPATH``; select an environment
installed from the intended checkout if that happens.

For a smoke run outside the debugger, from the repository root:

.. code-block:: bash

   export PYTHONPATH="$PWD/python/cuda_coop${PYTHONPATH:+:$PYTHONPATH}"
   export CUDA_COOP_CCCL_ROOT="$PWD"
   export CUDA_COOP_ENABLE_CACHE=0
   export CUDA_COOP_SOURCE_DUMP_DIR="$PWD/build/coop-temp-storage-sources"
   python docs/python/coop/debugger_walkthrough_temp_storage.py --case all

The small cases default to ``--items-per-thread 4``. Every kernel uses two
blocks and processes two tiles per block, so scratch reuse across loop
iterations is visible too. The dynamic case selects its workload using
the current device's shared-memory limits.

Dynamic backing: the first 25 minutes
------------------------------------

The ``dynamic`` case first runs ``exclusive`` with enough ``int64`` items
per thread for its combined Load, Merge Sort, and Store scratch to exceed
the static limit. It then runs ``shared-auto`` with the same tile size.
Both sort the same data. The script checks that the exclusive version
requires dynamic bytes and that the shared version fits statically.
It reports an error if the current device has insufficient opt-in headroom
for this comparison.

Set these breakpoints in
``python/cuda_coop/cuda/coop/numba_mlir/_compiler/_rewrite_storage.py``.
Use the function names and statements below so line-number changes do not
move the intended stops.

1. In ``_compute_func_temp_storage_requirements()``, stop at
   ``summary.uses.append(...)``. Inspect ``ctor_key``, ``size_in_bytes``,
   ``alignment``, and ``match.lowering_plan``. These byte counts come from
   compiled CUB types; the size of ``ThreadData`` alone does not determine
   the scratch requirement.
2. In ``_ensure_temp_storage_global_plan()``, stop at
   ``uses_dynamic_smem = total_size > max_default``. Inspect ``total_size``,
   ``max_alignment``, ``max_default``, ``max_optin``, and
   ``self._temp_storage_plans``. Step through the alignment and opt-in
   checks to ``set_required_dynamic_shared_memory(...)``. The required
   dynamic bytes cover the entire backing, including padding.
3. In ``_emit_temp_storage_backing()``, stop at
   ``alloc_size = 0 if plan.uses_dynamic_smem else int(plan.total_size)``.
   Inspect ``plan``. A zero-length shared array selects the dynamic shared
   window; ``plan.dynamic_shared_bytes`` supplies its launch requirement.
4. Step into ``set_required_dynamic_shared_memory()`` in the installed
   ``numba_cuda_mlir`` package to inspect the compiler metadata. In that
   package's ``descriptor.py``, search for
   ``effective_sharedmem = max(configured_sharedmem, required_dynamic_shared_memory)``
   and break there. Observe a zero configured byte count becoming the
   compiler's required byte count before the adjusted launch is created.
   This compiler-side statement can move between versions.

At step 2, inspect each region's ``base_offset``, ``size_in_bytes``,
``alignment``, and ``slices_by_call_id``. Distinct descriptors occupy aligned
regions in one backing allocation. The dynamic switch applies to their
combined size. In ``_emit_temp_storage_slice_for_call()``, the
``static_start = int(base_offset) + int(slice_info.offset)`` statement
connects a primitive's view to its planned offset.

Questions worth resolving here: Does every consumer fit its slice? Does
the total include alignment gaps? Do the compiler metadata and launch use
the same byte count? Is the device opt-in limit checked before launch?

The current integration rejects user ``cuda.shared.array`` allocations
alongside dynamic cooperative backing. The walkthrough keeps all scratch
under the cooperative planner. The compatibility guard is in
``_reject_conflicting_user_shared_arrays()``; inspect it if discussing
extensions to these kernels.

Shared storage: automatic and manual synchronization
---------------------------------------------------

Run ``shared-auto``, then restart with ``shared-manual``. Compare their
kernel bodies and region plans. Both reuse one descriptor for transpose
Load, Merge Sort, and transpose Store. In ``_rewrite_provenance.py``, stop
at the return of ``_layout_temp_storage_uses()``: compare ``required_size``
and the offsets in ``slices_by_call_id``. The reused region covers its
largest consumer, with alignment accounted for.

For ``shared-auto``, stop at ``sync_args = []`` in
``_emit_temp_storage_auto_sync()``. ``synchronization_scope`` is ``BLOCK``
and ``sync_attr`` is ``syncthreads``. The compiler inserts a trailing reuse
barrier after each storage-consuming call. In ``shared-manual``, the
kernel's explicit ``block.sync()`` calls establish those reuse points,
including the transition to the next loop iteration.

Review every transition that overwrites shared scratch. All block threads
must reach the corresponding barrier. Internal CUB barriers do not generally
establish safe reuse by the next primitive. Compare synchronization emitted
by the rewrite separately from CUB's own synchronization in generated code.

Exclusive slices and independent descriptors
--------------------------------------------

Run ``exclusive`` and ``independent`` last. Keep the layout and allocation
breakpoints enabled:

* ``exclusive`` uses one descriptor with ``sharing="exclusive"``. Each
  distinct primitive call gets a separate slice within that descriptor's
  region. Compare the slice offsets with ``shared-auto``. A block barrier
  at the end of the loop makes the next iteration's reuse safe.
* ``independent`` gives Exclusive Sum and Merge Sort separate descriptors.
  Inspect their distinct ``base_offset`` values in
  ``self._temp_storage_plans``. Separate descriptors still share one
  compiler-generated backing allocation. Direct Load and Store need no
  cooperative scratch, and the loop ends with a block barrier.

Sharing and synchronization are separate choices. ``sharing="exclusive"``
does not itself insert a barrier. Reaching the same call site again in a
loop reuses that call site's slice, so exclusive storage also needs safe
reuse when the kernel loops.
