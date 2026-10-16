# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe what call resolution decided for one kernel.

A ``KernelPlan`` is plain data: the cooperative call sites in program order,
the request each one completed to, the scratch regions and their backing,
the barriers, and the launch facts the resolution read. It does not refer to
Numba IR, so it is the part of the resolution a second host compiler could
share.

``format_plan`` renders a plan as stable text. Tests check that text with
FileCheck, and ``CUDA_COOP_NUMBA_MLIR_PIPELINE_PRINT=plan`` prints it for
each compiled kernel. The format names sites by program order and source
line, never by object identity or symbol digest, so it does not change
between runs, processes or targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Backing:
    """The one shared-memory allocation behind every scratch region."""

    kind: str
    size: int
    alignment: int


@dataclass(frozen=True)
class Region:
    """A byte range of the backing used by one or more sites."""

    name: str
    offset: int
    size: int
    alignment: int


@dataclass
class SitePlan:
    """One cooperative call and the decisions taken for it.

    ``request`` holds the completed specialization's parameters in the
    order they are printed, for example ``(("algorithm", "transpose"),)``.
    """

    index: int
    operation: str
    line: int | None
    request: tuple[tuple[str, object], ...] = ()
    region: str | None = None
    barrier_before: bool = False
    barrier_after: bool = False


@dataclass
class KernelPlan:
    """Everything call resolution decided for one kernel.

    ``bundle`` is the number of distinct provider specializations compiled
    together, in one translation unit, for all of the kernel's sites.
    """

    name: str
    block: tuple[int, int, int] | None = None
    block_source: str = "not requested"
    launch_bounds: int | None = None
    bundle: int | None = None
    backing: Backing | None = None
    regions: list[Region] = field(default_factory=list)
    sites: list[SitePlan] = field(default_factory=list)


def _site_line(site: SitePlan) -> str:
    text = f"site #{site.index} {site.operation}"
    if site.line is not None:
        text += f" @L{site.line}"
    for key, value in site.request:
        text += f" {key}={value}"
    if site.region is not None:
        text += f" region={site.region}"
    return text


def format_plan(plan: KernelPlan) -> str:
    """Render ``plan`` as the text that FileCheck tests match."""

    lines = [f"coop-plan {plan.name}"]
    if plan.block is None:
        lines.append(f"  block {plan.block_source}")
    else:
        lines.append(f"  block {plan.block} from {plan.block_source}")
    if plan.launch_bounds is not None:
        lines.append(f"  launch_bounds {plan.launch_bounds}")
    if not plan.sites:
        lines.append("  no coop sites")
        return "\n".join(lines) + "\n"
    if plan.bundle is not None:
        lines.append(
            f"  bundle sites={len(plan.sites)} specializations={plan.bundle}"
        )
    if plan.backing is not None:
        b = plan.backing
        lines.append(f"  backing {b.kind} size={b.size} align={b.alignment}")
    for r in plan.regions:
        lines.append(
            f"  region {r.name} offset={r.offset} size={r.size} "
            f"align={r.alignment}"
        )
    for site in plan.sites:
        if site.barrier_before:
            lines.append(f"  barrier before #{site.index}")
        lines.append(f"  {_site_line(site)}")
        if site.barrier_after:
            lines.append(f"  barrier after #{site.index}")
    return "\n".join(lines) + "\n"


__all__ = [
    "Backing",
    "KernelPlan",
    "Region",
    "SitePlan",
    "format_plan",
]
