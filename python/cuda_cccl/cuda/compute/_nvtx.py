# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
NVTX annotation utilities for cuda.compute module.
Uses NVIDIA green (76B900) color and cuda.compute domain.
"""

import functools

import nvtx

# NVIDIA green color hex value (76B900)
NVIDIA_GREEN = 0x76B900

# Domain name for cuda.compute annotations
COMPUTE_DOMAIN = "cuda.compute"


def annotate(message=None, domain=None, category=None, color=None):
    """
    Decorator to annotate functions with NVTX markers.

    Args:
        message: Optional message to display. If None, uses the function name.
        domain: Optional NVTX domain string. Defaults to "cuda.compute".
        category: Optional category for the annotation.
        color: Optional color in hexadecimal format (0xRRGGBB). Defaults to NVIDIA green (0x76B900).

    Returns:
        Decorated function with NVTX annotations.
    """

    def decorator(func):
        # Use function name if no message is provided
        annotation_message = message if message is not None else func.__name__
        annotation_domain = domain if domain is not None else COMPUTE_DOMAIN
        annotation_color = color if color is not None else NVIDIA_GREEN

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with nvtx.annotate(
                annotation_message,
                domain=annotation_domain,
                color=annotation_color,
                category=category,
            ):
                return func(*args, **kwargs)

        return wrapper

    return decorator
