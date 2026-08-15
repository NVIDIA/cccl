# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bundles: non-owning groups of logical data used as a single dependency.

A :class:`bundle` ties several logical data together behind one object (the
three arrays of a CSR matrix, a graph's topology) so tasks can depend on the
whole group with a single argument, while every constituent ("field") remains
an ordinary logical data usable on its own. A bundle owns no data and no
dependency-tracking state: a bundle dependency expands into one ordinary
dependency per field before reaching the task, and :meth:`bundle_task.get`
reassembles the per-field views with the bundle counting as ONE slot.

This mirrors the C++ ``bundle`` / ``field`` / ``constant`` feature
(see ``cudax/include/cuda/experimental/__stf/internal/bundle.cuh``); the two
front ends share no code but implement the same semantics: whole-bundle modes
distribute per field as the strongest mode the field admits (``rw()`` on a
bundle clamps ``constant`` fields to read), explicitly requesting more than a
field's ceiling raises, unspecified fields in per-field spellings default to
read, and one submitted dependency is one ``get`` slot.

``constant`` is a promise about *users of this bundle*, not global
immutability: other views or bare handles may legitimately write the field,
and the ordinary read dependencies the bundle generates are what serialize
against those writers.

Example
-------
>>> from cuda.stf._experimental.bundles import bundle, bundle_task, constant
>>> A = bundle(ctx, vals=vals_array, colind=constant(colind_array),
...            rowptr=constant(rowptr_array))
>>> with bundle_task(ctx, A.rw(), ly.rw()) as t:
...     a = t.get(0)      # namespace: a.vals, a.colind, a.rowptr (CAI views)
...     y = t.get(1)      # ordinary dependency: one slot each
"""

from types import SimpleNamespace

from cuda.stf._experimental import _stf_bindings as _b

__all__ = ["bundle", "bundle_dep", "bundle_task", "constant"]

_READ = _b.AccessMode.READ.value
_RW = _b.AccessMode.RW.value
_WRITE = _b.AccessMode.WRITE.value


class constant:
    """Marks a bundle field as read-only through this bundle (a view ceiling)."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


def _register(ctx, value):
    """Register an array-like, inferring the data place for device memory.

    CUDA Array Interface objects live on a device; registering them without a
    device data place trips an opaque host-pinning assertion, so infer the
    device from the pointer attributes.
    """
    if isinstance(value, _b.logical_data):
        return value
    cai = getattr(value, "__cuda_array_interface__", None)
    if cai is not None:
        from cuda.bindings import runtime as _rt

        err, attr = _rt.cudaPointerGetAttributes(cai["data"][0])
        if int(err) == 0 and attr.type == _rt.cudaMemoryType.cudaMemoryTypeDevice:
            return ctx.logical_data(value, _b.data_place.device(attr.device))
    return ctx.logical_data(value)


class bundle_dep:
    """A group of ordinary deps submitted as a single argument (one slot)."""

    __slots__ = ("deps", "names")

    def __init__(self, deps, names):
        self.deps = list(deps)
        self.names = list(names)


class bundle:
    """Non-owning group of logical data with named fields and ceilings.

    Field values may be existing :class:`logical_data` (adopted: handles are
    shared, nothing is copied) or array-likes (registered). Wrapping a value
    in :class:`constant` gives the field a read-only ceiling. Fields remain
    first-class logical data, reachable as attributes (``b.vals``).
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
            self._lds[name] = _register(ctx, value)
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
        """
        m = dict.fromkeys(self._names, _READ)
        for name, mode in modes.items():
            if name not in self._lds:
                raise KeyError(f"bundle has no field {name!r}")
            mode = int(mode)
            if name in self._ceiling_read and mode != _READ:
                raise ValueError(
                    f"field {name!r} is constant in this bundle (read-only ceiling)"
                )
            m[name] = mode
        return self._make(m, dplace)

    def _make(self, modes, dplace):
        deps = []
        for name in self._names:
            mode = _READ if name in self._ceiling_read else modes[name]
            deps.append(_b.dep(self._lds[name], mode, dplace))
        return bundle_dep(deps, self._names)


class bundle_task:
    """Context manager wrapping ``ctx.task``: flattens bundle dependencies and
    regroups per-slot access, with each bundle counting as one slot.

    Non-bundle arguments pass through unchanged; ``get(i)`` returns the plain
    CUDA Array Interface view for them, and a :class:`types.SimpleNamespace`
    of per-field views for bundle slots. Other task methods (``stream_ptr``,
    ``get_arg_cai``, ...) are forwarded to the underlying task.
    """

    def __init__(self, ctx, *args, **kwargs):
        self._slots = []  # (arity, names or None)
        flat = []
        for a in args:
            if isinstance(a, bundle_dep):
                self._slots.append((len(a.deps), a.names))
                flat.extend(a.deps)
            else:
                self._slots.append((1, None))
                flat.append(a)
        self._task = ctx.task(*flat, **kwargs)

    def __enter__(self):
        self._task.__enter__()
        return self

    def __exit__(self, *exc):
        return self._task.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._task, name)

    def get(self, slot):
        """Per-slot view access; a bundle is one slot."""
        arity, names = self._slots[slot]
        base = sum(s[0] for s in self._slots[:slot])
        if names is None:
            return self._task.get_arg_cai(base)
        views = [self._task.get_arg_cai(base + k) for k in range(arity)]
        return SimpleNamespace(**dict(zip(names, views)))
