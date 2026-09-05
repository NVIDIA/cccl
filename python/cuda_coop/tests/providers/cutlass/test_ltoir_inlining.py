# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from ...support.toolchains.cutlass import find_cuda_tool
from ._ltoir_support import (
    _assert_ltoir_inlined,
    _configure_dump_environment,
    _find_one,
    _require_runtime,
    _run_example_subprocess,
)

pytestmark = [pytest.mark.gpu, pytest.mark.link]


def _require_sm100_or_sm103_runtime() -> None:
    _require_runtime()
    torch = pytest.importorskip("torch")
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip(
            "requires an SM100 or SM103 GPU for tcgen05 TMEM-to-RMEM codegen; "
            f"found compute capability {capability[0]}.{capability[1]}"
        )


@pytest.mark.parametrize(
    ("module_name", "expected_symbols", "expected_sass_tokens"),
    [
        (
            "cute_kmeans_assign_topk",
            (
                "cuda_coop_cutlass_topk_min_pair_keys_ki32_vi32_bt32_x2",
                "cuda_coop_cutlass_topk_min_pair_values_ki32_vi32_bt32_x2",
            ),
            (),
        ),
        (
            "cute_run_length_decode_window",
            ("cuda_coop_cutlass_cub_run_length_decode_b32_vu32_lu32_r2_x4_offsets",),
            (),
        ),
        (
            "cute_scheduler_prefix",
            (
                "cuda_coop_cutlass_cub_scan_block_b32_"
                "exclusivesum_sum_i32_scalar_raking_noinit_value",
            ),
            ("SHFL",),
        ),
        (
            "_group_exchange_codegen_probe",
            (
                "cuda_coop_cutlass_cub_exchange_block_b8x4x2_blockedtostriped_i32_x5",
                "cuda_coop_cutlass_cub_exchange_block_b8x4x2_stripedtoblocked_i32_x5",
                "cuda_coop_cutlass_cub_exchange_warp_b8x4x2_blockedtostriped_i32_x5",
                "cuda_coop_cutlass_cub_exchange_warp_b8x4x2_stripedtoblocked_i32_x5",
            ),
            (),
        ),
        (
            "_discontinuity_shuffle_codegen_probe",
            (
                "cuda_coop_cutlass_shuffle_b8x4x2_offset_i32",
                "cuda_coop_cutlass_shuffle_b8x4x2_rotate_i32",
                "cuda_coop_cutlass_shuffle_b8x4x2_up_i32_x2_suffix",
                (
                    "cuda_coop_cutlass_discontinuity_b8x4x2_"
                    "heads_and_tails_i32_x2_predecessor_successor"
                ),
            ),
            (),
        ),
        (
            "_group_merge_sort_codegen_probe",
            (
                (
                    "cuda_coop_cutlass_cub_merge_sort_block_b8x4x2_"
                    "pairs_descending_ki32_vi32_x2_full"
                ),
                (
                    "cuda_coop_cutlass_cub_merge_sort_block_b8x4x2_"
                    "keys_ascending_ki32_x2_partial"
                ),
                (
                    "cuda_coop_cutlass_cub_merge_sort_warp_b8x4x2_w32_"
                    "pairs_ascending_ki32_vi32_x2_full"
                ),
            ),
            (),
        ),
        (
            "_group_radix_codegen_probe",
            (
                "cuda_coop_cutlass_radix_sort_pairs_i32_b8x4x2_i32_asc_x2",
                "cuda_coop_cutlass_radix_rank_b8x4x2_i32_asc_b0_4_x2_prefix1",
            ),
            (),
        ),
        (
            "cute_warp_prefix_reduce",
            (
                (
                    "cuda_coop_cutlass_cub_scan_warp_b64_"
                    "exclusivesum_sum_i32_scalar_warp_noinit_value"
                ),
                "cuda_coop_cutlass_cudax_reduce_warp_b64_sum_i32",
                "cuda_coop_cutlass_cub_exchange_warp_b64_stripedtoblocked_i32_x2",
            ),
            ("SHFL.UP", "REDUX.SUM.S32"),
        ),
        (
            "cute_warp_merge_sort",
            (
                (
                    "cuda_coop_cutlass_cub_merge_sort_warp_b32_w32_"
                    "keys_descending_ki32_x2_full"
                ),
                (
                    "cuda_coop_cutlass_cub_merge_sort_warp_b32_w32_"
                    "pairs_ascending_ki32_vi32_x2_full"
                ),
            ),
            (),
        ),
        (
            "cute_sort_and_segment",
            (
                "cuda_coop_cutlass_radix_sort_pairs_i32_b32_i32_asc",
                "cuda_coop_cutlass_discontinuity_b32_heads_i32",
                (
                    "cuda_coop_cutlass_cub_scan_block_b32_"
                    "exclusivesum_sum_i32_x1_raking_noinit_value"
                ),
            ),
            (),
        ),
        (
            "cute_sort_and_segment_thread_data",
            (
                "cuda_coop_cutlass_radix_sort_pairs_i32_b32_i32_asc_x2",
                "cuda_coop_cutlass_discontinuity_b32_heads_i32_x2",
                (
                    "cuda_coop_cutlass_cub_scan_block_b32_"
                    "exclusivesum_sum_i32_x2_raking_noinit_value"
                ),
            ),
            (),
        ),
        (
            "cute_thread_group_reduce",
            (
                "cuda_coop_cutlass_cudax_reduce_block_b8x4x2_sum_i32",
                "cuda_coop_cutlass_cudax_reduce_block_b8x4x2_sum_i32_x2",
                "cuda_coop_cutlass_cudax_reduce_warp_b8x4x2_max_i32",
            ),
            (),
        ),
        (
            "_group_reduce_codegen_probe",
            (
                "cuda_coop_cutlass_cudax_reduce_block_b64_sum_i32_root",
                "cuda_coop_cutlass_cub_reduce_block_b64_sum_i32_raking_valid_r",
                (
                    "cuda_coop_cutlass_cub_reduce_block_b64_sum_i32_x2_"
                    "warp_reductions_full"
                ),
                "cuda_coop_cutlass_cub_reduce_warp_b64_sum_i32_warp_valid_s24",
            ),
            (),
        ),
        (
            "cute_thread_group_descriptor_reduce",
            (
                "cuda_coop_cutlass_cudax_reduce_block_b64_sum_i32",
                "cuda_coop_cutlass_cudax_reduce_block_b64_sum_i32_x2",
                "cuda_coop_cutlass_cudax_reduce_warp_b64_max_i32",
            ),
            (),
        ),
        (
            "cute_thread_hierarchy_reduce",
            (
                "cuda_coop_cutlass_cudax_reduce_block_b64_sum_i32",
                "cuda_coop_cutlass_cudax_reduce_block_b64_sum_i32_x2",
                "cuda_coop_cutlass_cudax_reduce_warp_b64_max_i32",
            ),
            (),
        ),
        (
            "cute_thread_group_query",
            (
                "cuda_coop_cutlass_cudax_group_block_b64_sync",
                "cuda_coop_cutlass_cudax_group_block_b64_rank_thread_i32",
                "cuda_coop_cutlass_cudax_group_block_b64_count_thread_i32",
                "cuda_coop_cutlass_cudax_group_warp_b64_rank_block_i32",
                "cuda_coop_cutlass_cudax_group_warp_b64_count_block_i32",
            ),
            (),
        ),
        (
            "cute_legacy_reduce_compare",
            (
                "cuda_coop_cutlass_cudax_reduce_block_b64_sum_i32",
                "cuda_coop_cutlass_cudax_reduce_block_b64_sum_i32_x2",
                "cuda_coop_cutlass_cudax_reduce_warp_b64_max_i32",
            ),
            (),
        ),
        (
            "cute_topk_score_window",
            (
                "cuda_coop_cutlass_topk_max_keys_i32_bt32_x3",
                "cuda_coop_cutlass_topk_min_pair_keys_ki32_vi32_bt32_x3",
                "cuda_coop_cutlass_topk_min_pair_values_ki32_vi32_bt32_x3",
            ),
            (),
        ),
    ],
)
def test_cute_provider_ltoir_inlines_into_final_cubin(
    module_name: str,
    expected_symbols: tuple[str, ...],
    expected_sass_tokens: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _configure_dump_environment(monkeypatch, tmp_path)
    _require_runtime()

    _run_example_subprocess(module_name)

    sass = _assert_ltoir_inlined(
        tmp_path=tmp_path,
        expected_symbols=expected_symbols,
        expected_sass_tokens=expected_sass_tokens,
    )
    if module_name in {
        "_discontinuity_shuffle_codegen_probe",
        "_group_exchange_codegen_probe",
        "_group_merge_sort_codegen_probe",
        "_group_radix_codegen_probe",
        "cute_run_length_decode_window",
        "cute_warp_merge_sort",
    }:
        assert re.search(r"\b(?:CALL|LDL|STL)(?:\.[A-Z0-9_]+)*\b", sass) is None
    if module_name == "_group_merge_sort_codegen_probe":
        cubin_path = _find_one("dsl/*.cubin", tmp_path=tmp_path)
        cuobjdump = find_cuda_tool("cuobjdump")
        assert cuobjdump is not None
        resources = subprocess.run(
            [str(cuobjdump), "--dump-resource-usage", str(cubin_path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert re.search(r"\bREG:\d+\b", resources)
        assert re.search(r"\bSHARED:\d+\b", resources)
        assert "STACK:0" in resources
        assert "LOCAL:0" in resources


def _normalized_sass_instructions(sass: str) -> tuple[str, ...]:
    instructions = []
    for line in sass.splitlines():
        match = re.search(r"/\*[0-9a-fA-F]+\*/\s*(.*?)\s*;", line)
        if match is None:
            continue
        instruction = re.sub(r"\.L_[A-Za-z0-9_.$]+", ".L", match.group(1))
        instructions.append(" ".join(instruction.split()))
    return tuple(instructions)


def _assert_sm100_mma_consumer_modes_match_codegen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    module_name: str,
    expected_symbols: tuple[str, ...],
) -> None:
    _require_sm100_or_sm103_runtime()

    cuobjdump = find_cuda_tool("cuobjdump")
    assert cuobjdump is not None
    mode_sass: dict[str, str] = {}
    mode_resources: dict[str, tuple[int, ...]] = {}
    resource_fields = ("REG", "SHARED", "STACK", "LOCAL")

    for mode in ("post_t2r", "tmem_loader"):
        mode_output = tmp_path / mode
        mode_output.mkdir()
        _configure_dump_environment(monkeypatch, mode_output)
        _run_example_subprocess(module_name, mode=mode)

        # DenseGemm selects the same LDTM policy in both modes. In tmem_loader,
        # ThreadData.load triggers the producer source hook; in post_t2r,
        # DenseGemm triggers the copy before the register-source adapter runs.
        sass = _assert_ltoir_inlined(
            tmp_path=mode_output,
            expected_symbols=expected_symbols,
            expected_sass_tokens=("UTCHMMA", "LDTM"),
        )
        assert len(re.findall(r"\bUTCHMMA\b", sass)) == 4
        assert len(re.findall(r"\bLDTM\b", sass)) == 1
        mode_sass[mode] = sass

        cubin_path = _find_one("dsl/*.cubin", tmp_path=mode_output)
        resources = subprocess.run(
            [str(cuobjdump), "--dump-resource-usage", str(cubin_path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        parsed_resources = []
        for field in resource_fields:
            match = re.search(rf"\b{field}:(\d+)\b", resources)
            assert match is not None, resources
            parsed_resources.append(int(match.group(1)))
        assert parsed_resources[-1] == 0
        mode_resources[mode] = tuple(parsed_resources)

    assert mode_resources["tmem_loader"] == mode_resources["post_t2r"]
    assert _normalized_sass_instructions(
        mode_sass["tmem_loader"]
    ) == _normalized_sass_instructions(mode_sass["post_t2r"])
    for opcode in ("CALL", "LDL", "STL"):
        assert len(
            re.findall(rf"\b{opcode}(?:\.[A-Z0-9_]+)*\b", mode_sass["tmem_loader"])
        ) == len(re.findall(rf"\b{opcode}(?:\.[A-Z0-9_]+)*\b", mode_sass["post_t2r"]))


@pytest.mark.requires_sm100
def test_sm100_mma_topk_tmem_loader_matches_post_t2r_codegen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _assert_sm100_mma_consumer_modes_match_codegen(
        monkeypatch,
        tmp_path,
        module_name="cute_mma_topk_sm100",
        expected_symbols=(
            "cuda_coop_cutlass_topk_max_keys_f32_bt128_x8",
            "cuda_coop_cutlass_topk_max_keys_f32_bt128",
        ),
    )


@pytest.mark.requires_sm100
def test_sm100_mma_amax_tmem_loader_matches_post_t2r_codegen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _assert_sm100_mma_consumer_modes_match_codegen(
        monkeypatch,
        tmp_path,
        module_name="cute_mma_amax_sm100",
        expected_symbols=("cuda_coop_cutlass_cudax_reduce_block_b128_max_f32",),
    )
