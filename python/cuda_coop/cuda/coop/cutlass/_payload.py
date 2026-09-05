# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Payload frontend selectors for the CUTLASS API."""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from enum import Enum


class Payload(str, Enum):
    """Explicit CUTLASS load/store payload selector."""

    PRIMS = "prims"

    def __str__(self) -> str:
        return self.value


def normalize_payload_selector(
    payload: object,
    *,
    scope: str,
    primitive_name: str,
    allowed: Iterable[Payload],
    choices_text: str,
) -> Payload | None:
    """Normalize a load/store selector and validate allowed choices."""
    if payload is None:
        return None

    normalized = payload if isinstance(payload, Payload) else None
    if isinstance(payload, str) and payload == Payload.PRIMS.value:
        normalized = Payload.PRIMS

    allowed_payloads = frozenset(allowed)
    if normalized in allowed_payloads:
        return normalized
    raise ValueError(f"{scope}.{primitive_name} payload must be {choices_text}")


def reject_payload_selector_keyword(
    kwargs: MutableMapping[str, object],
    *,
    scope: str,
    primitive_name: str,
) -> None:
    """Reject payload selectors on primitives that do not consume them."""
    if "payload" not in kwargs:
        return
    if kwargs["payload"] is None:
        kwargs.pop("payload", None)
        return
    raise TypeError(
        f"{scope}.{primitive_name} does not accept payload=; payload selectors "
        "are only supported by load/store and make_load/make_store. Load "
        "cutlass.Array values or other memory operands through the Prims array "
        "path into per-thread register payloads before calling cooperative "
        "primitives."
    )


__all__ = ["Payload"]
