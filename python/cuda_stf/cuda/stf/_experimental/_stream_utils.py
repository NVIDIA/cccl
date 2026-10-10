# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Stream-resolution helpers for the STF Python bindings."""


def get_stream_pointer(stream) -> int:
    """Resolve a user stream to a raw CUstream pointer (0 == null stream).

    Accepts None, a raw integer pointer, or any object implementing the
    __cuda_stream__ protocol.
    """
    if stream is None:
        return 0
    cuda_stream = getattr(stream, "__cuda_stream__", None)
    if cuda_stream is not None:
        try:
            version, handle, *_ = cuda_stream()
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"could not obtain __cuda_stream__ protocol version and handle from {stream}"
            ) from e
        if version != 0:
            raise TypeError(f"unsupported __cuda_stream__ version {version}")
        if not isinstance(handle, int):
            raise TypeError(f"invalid stream handle {handle}")
        return handle
    if isinstance(stream, int):
        return int(stream)
    raise TypeError(
        f"stream argument {stream!r} does not implement the '__cuda_stream__' "
        "protocol and is not an int pointer"
    )
