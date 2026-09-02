# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS launch-fact adaptation for block collectives."""

from __future__ import annotations

from typing import Any

from cuda.coop._core import LaunchFactOrigin, LaunchFacts
from cuda.coop._core.root_api import CoopCompilerContextRequiredError

from ._runtime import is_current_cutlass_environment, validate_cutlass_runtime


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _to_core_launch_facts(exact_block_dim: Any) -> LaunchFacts:
    return LaunchFacts(
        exact_block_dim=exact_block_dim,
        provenance=(
            ()
            if exact_block_dim is None
            else (
                LaunchFactOrigin(
                    fact="exact_block_dim",
                    source="cutlass_provider_api",
                    detail="cute._get_launch_facts()",
                    verified=True,
                ),
            )
        ),
    )


def current_launch_facts(*, feature: str) -> LaunchFacts:
    """Return exact CUTLASS facts for the kernel currently being compiled."""

    try:
        if not is_current_cutlass_environment():
            raise CoopCompilerContextRequiredError(feature)
        runtime = validate_cutlass_runtime()
        provider_facts = runtime.cute._get_launch_facts()
        exact_block_dim = _field(provider_facts, "exact_block_dim")
    except CoopCompilerContextRequiredError:
        raise
    except Exception as error:
        raise CoopCompilerContextRequiredError(feature) from error
    try:
        facts = _to_core_launch_facts(exact_block_dim)
    except (TypeError, ValueError) as error:
        raise NotImplementedError(
            f"cuda.coop.cutlass {feature} requires exact static block dimensions"
        ) from error
    except Exception as error:
        raise CoopCompilerContextRequiredError(feature) from error
    if facts.exact_block_dim is None:
        raise NotImplementedError(
            f"cuda.coop.cutlass {feature} requires exact static block dimensions"
        )
    return facts


__all__ = ["current_launch_facts"]
