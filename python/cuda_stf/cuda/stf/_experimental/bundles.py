# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bundles: non-owning groups of logical data used as a single dependency.

A :class:`bundle` ties several logical data together behind one object (the
three arrays of a CSR matrix, a graph's topology) so tasks can depend on the
whole group with a single argument, while every constituent ("field") remains
an ordinary logical data usable on its own. A bundle owns no data and no
dependency-tracking state: a bundle dependency expands into one ordinary
dependency per field before reaching the task, and ``task.get``
reassembles the per-field views with the bundle counting as ONE slot.

This mirrors the C++ ``bundle`` / ``field`` / ``constant`` feature
(see ``cudax/include/cuda/experimental/__stf/internal/bundle.cuh``); the two
front ends share no code but implement the same semantics: whole-bundle modes
distribute per field as the strongest mode the field admits (``rw()`` on a
bundle clamps ``constant`` fields to read), explicitly requesting more than a
field's ceiling raises, unspecified fields in per-field spellings default to
read, and one submitted dependency is one ``get`` slot.

When to use a bundle — and when not to: bundles model data that is one
OBJECT at the level users reason about (a sparse matrix's invariant-bound
arrays, a graph's topology, a mesh's coordinates and connectivity, a library
descriptor's constituents). They are not a dependency-count reducer for
loosely related arrays: a solver workspace whose tasks touch different
subsets with different modes each time (e.g. Krylov vectors R/P/V/S/T)
should keep bare per-array dependencies. Bundling adds no synchronization
of its own (a bundle dependency flattens to exactly the leaf dependencies
you would write by hand) — but the whole-object spellings and the
read-default acquire every field, which for a workspace means transfers
and ordering against fields the task never touches, for no modeling gain.
Use ``AccessMode.NONE`` to exclude a field where object-level access is
otherwise right.

``constant`` is a promise about *users of this bundle*, not global
immutability: other views or bare handles may legitimately write the field,
and the ordinary read dependencies the bundle generates are what serialize
against those writers.

Example
-------
>>> from cuda.stf._experimental import constant
>>> A = ctx.bundle(vals=vals_array, colind=constant(colind_array),
...                rowptr=constant(rowptr_array))
>>> with ctx.task(A.rw(), ly.rw()) as t:
...     a = t.get(0)      # namespace: a.vals, a.colind, a.rowptr (CAI views)
...     y = t.get(1)      # ordinary dependency: one slot each
"""

from cuda.stf._experimental import _stf_bindings as _b

__all__ = ["bundle", "bundle_dep", "constant"]

_NONE = _b.AccessMode.NONE.value
_READ = _b.AccessMode.READ.value
_RW = _b.AccessMode.RW.value
_WRITE = _b.AccessMode.WRITE.value


class constant:
    """Marks a bundle field as read-only through this bundle (a view ceiling)."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


def _check_field(name, value):
    """Bundle fields must be logical data: bundles group handles, they do not
    register data (that is ``ctx.logical_data``'s job, with its explicit data
    place policy). Duck-typed so stackable logical data qualify too."""
    if not (hasattr(value, "read") and hasattr(value, "rw")):
        raise TypeError(
            f"bundle field {name!r} must be a logical data (register the array "
            "first with ctx.logical_data(...))"
        )
    return value


class bundle_dep:
    """A group of ordinary deps submitted as a single argument (one slot).

    ``names`` covers every field of the bundle; ``acquired`` marks which of
    them contribute a dependency (fields excluded with mode ``NONE`` are not
    acquired at all — their view is ``None``).
    """

    _stf_bundle_dep = True

    __slots__ = ("deps", "names", "acquired")

    def __init__(self, deps, names, acquired):
        self.deps = list(deps)
        self.names = list(names)
        self.acquired = list(acquired)


class bundle:
    """Non-owning group of logical data with named fields and ceilings.

    Field values must be logical data (handles are shared, nothing is copied
    or registered — register arrays first with ``ctx.logical_data``).
    Wrapping a value in :class:`constant` gives the field a read-only
    ceiling. Fields remain first-class, reachable as attributes (``b.vals``).
    """

    def __init__(self, ctx, **fields):
        if not fields:
            raise ValueError("a bundle needs at least one field")
        self._names = []
        self._lds = {}
        self._ceiling_read = set()
        for name, value in fields.items():
            if isinstance(value, constant):
                self._ceiling_read.add(name)
                value = value.value
            self._lds[name] = _check_field(name, value)
            self._names.append(name)

    def __getattr__(self, name):
        try:
            return self._lds[name]
        except KeyError:
            raise AttributeError(name) from None

    def __len__(self):
        return len(self._names)

    def read(self, dplace=None):
        """Depend on every field with read access."""
        return self._make(dict.fromkeys(self._names, _READ), dplace)

    def rw(self, dplace=None):
        """Depend on the bundle read-write: constant fields clamp to read."""
        return self._make(dict.fromkeys(self._names, _RW), dplace)

    def write(self, dplace=None):
        """Depend on the bundle write-mode: constant fields clamp to read.

        Mutable fields are written (previous content discarded, not fetched);
        constant fields are still fetched, since a writer typically needs
        them to interpret what it writes.
        """
        return self._make(dict.fromkeys(self._names, _WRITE), dplace)

    def dep(self, dplace=None, **modes):
        """Per-field access modes; unspecified fields default to read.

        Explicitly requesting more than a constant field's ceiling raises
        ``ValueError`` (distribution clamps; explicit excess is an error).
        A field set to ``AccessMode.NONE`` is not acquired at all: no
        dependency, no transfer, and its view is ``None``.
        """
        m = dict.fromkeys(self._names, _READ)
        for name, mode in modes.items():
            if name not in self._lds:
                raise KeyError(f"bundle has no field {name!r}")
            mode = int(mode)
            if name in self._ceiling_read and mode not in (_READ, _NONE):
                raise ValueError(
                    f"field {name!r} is constant in this bundle (read-only ceiling)"
                )
            m[name] = mode
        return self._make(m, dplace)

    def _make(self, modes, dplace):
        deps = []
        acquired = []
        for name in self._names:
            mode = modes[name]
            if mode != _NONE and name in self._ceiling_read:
                mode = _READ
            if mode == _NONE:
                acquired.append(False)
                continue
            acquired.append(True)
            deps.append(_b.dep(self._lds[name], mode, dplace))
        return bundle_dep(deps, self._names, acquired)
