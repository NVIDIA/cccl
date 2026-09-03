# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe explicit shared-memory storage for cooperative operations."""


class TempStorage:
    """Shared-memory requirements for cooperative operations in one kernel."""

    def __init__(
        self,
        size_in_bytes=None,
        alignment=None,
        auto_sync=None,
        sharing="shared",
    ):
        if size_in_bytes is not None:
            if not isinstance(size_in_bytes, int) or isinstance(size_in_bytes, bool):
                raise TypeError("TempStorage size_in_bytes must be an integer or None.")
            if size_in_bytes <= 0:
                raise ValueError(
                    "TempStorage size_in_bytes must be a positive integer."
                )

        if alignment is not None:
            if not isinstance(alignment, int) or isinstance(alignment, bool):
                raise TypeError("TempStorage alignment must be an integer or None.")
            if alignment <= 0:
                raise ValueError("TempStorage alignment must be a positive integer.")
            if alignment & (alignment - 1):
                raise ValueError("TempStorage alignment must be a power of 2.")

        if not isinstance(sharing, str):
            raise TypeError(
                "TempStorage sharing must be a string: 'shared' or 'exclusive'."
            )
        sharing_value = sharing.strip().lower()
        if sharing_value not in {"shared", "exclusive"}:
            raise ValueError("TempStorage sharing must be 'shared' or 'exclusive'.")

        if auto_sync is not None and not isinstance(auto_sync, bool):
            raise TypeError("TempStorage auto_sync must be None/True/False.")
        if sharing_value == "exclusive" and auto_sync is True:
            raise ValueError(
                "TempStorage with sharing='exclusive' does not support auto_sync=True."
            )

        self.size_in_bytes = size_in_bytes
        self.alignment = alignment
        self.sharing = sharing_value
        self.auto_sync = (
            False
            if sharing_value == "exclusive"
            else (True if auto_sync is None else auto_sync)
        )


__all__ = ["TempStorage"]
