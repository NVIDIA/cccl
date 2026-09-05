# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections.abc import Callable

import pytest

from ...support.toolchains.cutlass import find_ptxas_with_ptx_93


@pytest.fixture(scope="session")
def cutlass_cuda_available(
    backend_prerequisite: Callable[[str, bool, str], None],
) -> None:
    """Check CUTLASS runtime GPU availability during test setup."""

    torch = pytest.importorskip("torch")
    backend_prerequisite(
        "cutlass",
        torch.cuda.is_available(),
        "requires a CUDA-capable PyTorch runtime",
    )


@pytest.fixture(scope="session")
def cutlass_ptxas_93_available(
    backend_prerequisite: Callable[[str, bool, str], None],
) -> None:
    """Check the provider's required PTX assembler during test setup."""

    ptxas, _ = find_ptxas_with_ptx_93()
    backend_prerequisite(
        "cutlass",
        ptxas is not None,
        "requires ptxas support for PTX .version 9.3",
    )


@pytest.fixture(scope="session")
def cutlass_runtime_available(
    cutlass_cuda_available: None,
    cutlass_ptxas_93_available: None,
) -> None:
    """Require the common CUTLASS runtime prerequisites."""


@pytest.fixture
def set_cutlass_launch_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[object], None]:
    """Set exact CUTLASS launch facts for out-of-trace frontend tests."""

    from cuda.coop._core import LaunchFactOrigin, LaunchFacts
    from cuda.coop.cutlass._dsl import _launch

    def set_facts(block_dim: object) -> None:
        monkeypatch.setattr(
            _launch,
            "current_kernel_launch_facts",
            lambda: LaunchFacts(
                exact_block_dim=block_dim,
                provenance=(
                    ()
                    if block_dim is None
                    else LaunchFactOrigin(
                        fact="exact_block_dim",
                        source="test_compiler",
                        verified=True,
                    )
                ),
            ),
        )

    return set_facts


@pytest.fixture(scope="session")
def cutlass_cluster_runtime_available(cutlass_cuda_available: None) -> None:
    """Require a GPU capable of launching thread-block clusters."""

    torch = pytest.importorskip("torch")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("thread-block clusters require compute capability 9.0 or newer")
