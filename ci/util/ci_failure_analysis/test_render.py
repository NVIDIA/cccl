# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run offline: python3 -m unittest discover -s ci/util/ci_failure_analysis."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

RENDER = Path(__file__).with_name("render.py")
ACTION = (
    Path(__file__).resolve().parents[3]
    / ".github/actions/ci-failure-analysis/action.yml"
)
FORMATS = ("github-comment", "github-verbose-summary", "slack")
JOBS = [{"id": 1, "name": "job 1"}, {"id": 3, "name": "job 3"}]
GROUP = {
    "title": "Build failed",
    "evidence": ["error: boom"],
    "explanation": "The build broke.",
    "agent_prompt": "Fix the build.",
    "job_ids": [1, 3],
}
SKIPPED = "AI analysis skipped: no job logs or failure annotations were available."
UNKNOWN = (
    "CI could not get the list of failed jobs from GitHub, so this report may be "
    "incomplete."
)
NOTICE_MARKERS = ("skipped", "could not get", "Collected")


def render(output_format, **overrides):
    """Return (exit status, rendered text, stderr) for a complete analysis plus overrides."""
    context = {
        "run": {"id": 123, "number": 44},
        "jobs": JOBS,
        "groups": [GROUP],
        "missing_log_job_count": 0,
        "unlisted_failed_job_count": 0,
        "analysis_skipped": False,
        **overrides,
    }
    context = {key: value for key, value in context.items() if value != "absent"}
    with tempfile.TemporaryDirectory() as temporary:
        analysis = Path(temporary) / "analysis.json"
        analysis.write_text(json.dumps(context))
        return render_file(analysis, output_format)


def render_file(analysis, output_format):
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "rendered"
        result = subprocess.run(
            [sys.executable, str(RENDER), "--repository", "NVIDIA/cccl"]
            + ["--analysis-file", str(analysis), "--format", output_format]
            + ["--output", str(output)],
            capture_output=True,
            text=True,
        )
        text = output.read_text() if output.exists() else ""
    if output_format == "slack" and text:
        text = json.loads(text)["overview"]
    return result.returncode, text, result.stderr


class RenderNoticeTest(unittest.TestCase):
    def test_collection_notices(self):
        skipped = dict(groups=[], analysis_skipped=True, missing_log_job_count=2)
        cases = {
            "complete": ({}, []),
            "one unlisted job": (
                dict(unlisted_failed_job_count=1),
                ["Collected 2 of 3 failed jobs."],
            ),
            "list not retrieved": (dict(unlisted_failed_job_count=None), [UNKNOWN]),
            "one log missing": (
                dict(missing_log_job_count=1),
                ["Collected 2 of 2 failed jobs (1 without logs)."],
            ),
            "unlisted jobs and a missing log": (
                dict(unlisted_failed_job_count=4, missing_log_job_count=1),
                ["Collected 2 of 6 failed jobs (1 without logs)."],
            ),
            "list not retrieved and a missing log": (
                dict(unlisted_failed_job_count=None, missing_log_job_count=1),
                [UNKNOWN, "Collected 2 of 2 failed jobs (1 without logs)."],
            ),
            "the only job has no log": (
                dict(
                    jobs=JOBS[:1],
                    groups=[dict(GROUP, job_ids=[1])],
                    missing_log_job_count=1,
                ),
                ["Collected 1 of 1 failed job (1 without logs)."],
            ),
            "skipped without evidence": (
                skipped,
                [SKIPPED, "Collected 2 of 2 failed jobs (2 without logs)."],
            ),
            "skipped without evidence, with unlisted jobs": (
                dict(skipped, unlisted_failed_job_count=4),
                [SKIPPED, "Collected 2 of 6 failed jobs (2 without logs)."],
            ),
            "skipped because nothing was listed": (
                dict(
                    skipped,
                    jobs=[],
                    missing_log_job_count=0,
                    unlisted_failed_job_count=None,
                ),
                [SKIPPED, UNKNOWN],
            ),
            "skipped with no failed jobs": (
                dict(skipped, jobs=[], missing_log_job_count=0),
                [SKIPPED],
            ),
        }
        for name, (overrides, notices) in cases.items():
            for output_format in FORMATS:
                with self.subTest(name=name, format=output_format):
                    status, text, stderr = render(output_format, **overrides)
                    self.assertEqual(status, 0, stderr)
                    if notices:
                        self.assertIn("\n\n".join(notices) + "\n", text)
                    self.assertEqual(
                        sum(text.count(marker) for marker in NOTICE_MARKERS),
                        len(notices),
                        text,
                    )

    def test_unlisted_failed_job_count_is_validated(self):
        for value in (-1, True, 1.5, "3", "absent"):
            with self.subTest(value=value):
                status, text, stderr = render(
                    "github-comment", unlisted_failed_job_count=value
                )
                self.assertEqual(status, 1)
                self.assertEqual(text, "")
                self.assertRegex(
                    stderr,
                    "^error: could not render CI triage output: "
                    ".*unlisted_failed_job_count",
                )


class AssembleTest(unittest.TestCase):
    """The action's assemble step carries the collector's outputs into the report."""

    def test_collector_outputs_reach_the_report(self):
        lines = ACTION.read_text().splitlines()
        start = lines.index("      id: assemble-analysis")
        run = lines.index("      run: |", start)
        header = "\n".join(lines[start:run])
        script = "\n".join(line[8:] for line in lines[run + 1 :])
        for variable, output in (
            ("HAS_EVIDENCE", "has_evidence"),
            ("UNLISTED_FAILED_JOB_COUNT", "unlisted_failed_job_count"),
        ):
            self.assertIn(
                f"        {variable}: ${{{{ steps.collect-logs.outputs.{output} }}}}",
                header,
            )

        job = {"conclusion": "failure", "annotations": []}
        cases = {
            "two unlisted jobs and a missing log": dict(
                jobs=[
                    dict(job, id=1, name="job 1", log_path="/logs/job-1.log"),
                    dict(job, id=3, name="job 3", log_path=""),
                ],
                has_evidence="true",
                unlisted="2",
                analysis=dict(
                    jobs=JOBS,
                    groups=[GROUP],
                    missing_log_job_count=1,
                    unlisted_failed_job_count=2,
                    analysis_skipped=False,
                ),
                notice="Collected 2 of 4 failed jobs (1 without logs).\n",
            ),
            "nothing listed": dict(
                jobs=[],
                has_evidence="false",
                unlisted="null",
                analysis=dict(
                    jobs=[],
                    groups=[],
                    missing_log_job_count=0,
                    unlisted_failed_job_count=None,
                    analysis_skipped=True,
                ),
                notice=f"{SKIPPED}\n\n{UNKNOWN}\n",
            ),
        }
        for name, case in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "run.json").write_text('{"id": 123, "run_number": 44}')
                (root / "jobs.json").write_text(json.dumps(case["jobs"]))
                (root / "model-output.json").write_text(json.dumps({"groups": [GROUP]}))
                analysis = root / "analysis.json"
                result = subprocess.run(
                    ["bash", "--noprofile", "--norc", "-euo", "pipefail", "-c"]
                    + [script.replace("/tmp/ci-logs", str(root))],
                    env={
                        **os.environ,
                        "HAS_EVIDENCE": case["has_evidence"],
                        "UNLISTED_FAILED_JOB_COUNT": case["unlisted"],
                        "MODEL_OUTPUT_FILE": str(root / "model-output.json"),
                        "ANALYSIS_FILE": str(analysis),
                        "GITHUB_OUTPUT": str(root / "step.output"),
                    },
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    json.loads(analysis.read_text()),
                    {"run": {"id": 123, "number": 44}, **case["analysis"]},
                )
                status, text, stderr = render_file(analysis, "github-comment")
                self.assertEqual(status, 0, stderr)
                self.assertIn(case["notice"], text)


if __name__ == "__main__":
    unittest.main()
