.. _cccl-python-coop-limitations:

Limitations and Deferred Features
=================================

The following items are intentionally deferred or not yet supported:

* **BlockRadixSort decomposer for user-defined types** — CUB expects a tuple-of-
  references adapter. A C++ shim or alternate lowering is required.
* **Multi-channel BlockHistogram outputs** — the current CUB BlockHistogram API
  does not expose multi-output overloads in the same style as other collectives.

Additional notes:

* ``TempStorage`` sizes must be compile-time constants for a given kernel
  specialization. Changing sizes triggers a new specialization.
