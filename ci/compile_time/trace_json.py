# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read Chrome trace events without materializing the complete JSON document."""

import json
from typing import TextIO


class JSONStream:
    def __init__(self, source: TextIO, chunk_size: int):
        self.source = source
        self.chunk_size = chunk_size
        self.buffer = ""
        self.position = 0
        self.eof = False
        self.decoder = json.JSONDecoder()

    def fill(self, size=None):
        chunk = self.source.read(size or self.chunk_size)
        self.buffer = self.buffer[self.position :] + chunk
        self.position = 0
        self.eof = not chunk

    def peek(self):
        while True:
            while (
                self.position < len(self.buffer)
                and self.buffer[self.position] in " \t\r\n"
            ):
                self.position += 1
            if self.position < len(self.buffer):
                return self.buffer[self.position]
            if self.eof:
                return ""
            self.fill()

    def expect(self, token):
        if self.peek() != token:
            raise ValueError(f"Expected {token!r} in trace JSON")
        self.position += 1

    def value(self):
        if not self.peek():
            raise ValueError("Unexpected end of trace JSON")
        read_size = self.chunk_size
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
                # Numbers can end at a chunk boundary (e.g. 1 then .25e+2).
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if end == len(self.buffer) or self.buffer[end] not in ",]} \t\r\n":
                        if not self.eof:
                            self.fill(read_size)
                            read_size = min(read_size * 2, 4 * 1024 * 1024)
                            continue
                        if end != len(self.buffer):
                            raise ValueError("Invalid number in trace JSON")
                self.position = end
                return value
            except json.JSONDecodeError as error:
                # Incomplete strings report their opening quote. Other partial
                # tokens (including a Unicode escape) fail near the buffer end;
                # a syntax error earlier in the buffer cannot be fixed by reading
                # the rest of a multi-gigabyte file.
                if self.eof or (
                    not error.msg.startswith("Unterminated string")
                    and error.pos < len(self.buffer) - 8
                ):
                    raise
                # Large individual NVCC events may span many chunks. Grow reads
                # up to 4 MiB to avoid retrying their prefix once per 64 KiB.
                self.fill(read_size)
                read_size = min(read_size * 2, 4 * 1024 * 1024)


def iter_trace_events(source: TextIO, metadata: dict, *, chunk_size: int = 64 * 1024):
    """Yield array entries, preserving other top-level fields in metadata.

    Event boundaries are JSON boundaries, not lines: NVCC can put several
    events on one line and split an event across lines or read chunks.
    """
    reader = JSONStream(source, chunk_size)
    reader.expect("{")
    if reader.peek() != "}":
        while True:
            key = reader.value()
            if not isinstance(key, str):
                raise ValueError("Expected a trace JSON object key")
            reader.expect(":")
            if key == "traceEvents":
                reader.expect("[")
                if reader.peek() != "]":
                    while True:
                        yield reader.value()
                        if reader.peek() == "]":
                            break
                        reader.expect(",")
                reader.expect("]")
            else:
                metadata[key] = reader.value()
            if reader.peek() == "}":
                break
            reader.expect(",")
    reader.expect("}")
    if reader.peek():
        raise ValueError("Trailing data after trace JSON")
