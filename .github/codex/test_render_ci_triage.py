import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("render_ci_triage.py")
SPEC = importlib.util.spec_from_file_location("render_ci_triage", MODULE_PATH)
RENDERER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDERER)
SCHEMA_PATH = Path(__file__).with_name("ci-triage-output.schema.json")


class RenderCiTriageTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.workspace = self.root / "workspace"
        self.logs = self.root / "logs"
        self.workspace.mkdir()
        self.logs.mkdir()
        (self.workspace / "src").mkdir()
        (self.workspace / "src" / "example.cpp").write_text(
            "first line\nroot cause\n",
            encoding="utf-8",
        )
        self.jobs = {
            101: {
                "id": 101,
                "name": "CUDA <build> @team",
                "conclusion": "failure",
                "steps": [{"number": 4, "name": "Compile [tests]"}],
            },
            102: {
                "id": 102,
                "name": "Thrust build",
                "conclusion": "failure",
                "steps": [{"number": 4, "name": "Compile tests"}],
            },
            103: {
                "id": 103,
                "name": "CI gate",
                "conclusion": "failure",
                "steps": [{"number": 2, "name": "Check results"}],
            },
            104: {
                "id": 104,
                "name": "Cancelled matrix job",
                "conclusion": "cancelled",
                "steps": [],
            },
            105: {
                "id": 105,
                "name": "Successful job",
                "conclusion": "success",
                "steps": [],
            },
        }
        self.job_pages = [
            {
                "total_count": len(self.jobs),
                "jobs": list(self.jobs.values()),
            }
        ]
        for job_id, message in {
            101: "nvcc fatal: Unsupported gpu architecture 'compute_999'",
            102: "nvcc fatal: Unsupported gpu architecture 'compute_999'",
            103: "One or more required jobs failed",
        }.items():
            (self.logs / f"ci-triage-{job_id}.log").write_text(
                f"setup\n{message}\n",
                encoding="utf-8",
            )

        self.analysis = {
            "status": "ok",
            "error": "",
            "summary": (
                "Both builds reject compute_999; do not visit https://evil.example/a "
                "or notify @team."
            ),
            "start_here": "Remove compute_999 from src/example.cpp before #123.",
            "groups": [
                {
                    "title": "Unsupported compute_999 target </summary>",
                    "classification": "PR-related",
                    "confidence": "high",
                    "explanation": "nvcc rejects the requested target.",
                    "evidence": [
                        {
                            "job_id": 101,
                            "step_number": 4,
                            "lines": [
                                "nvcc fatal: Unsupported gpu architecture 'compute_999'"
                            ],
                        },
                        {
                            "job_id": 102,
                            "step_number": 4,
                            "lines": [
                                "nvcc fatal: Unsupported gpu architecture 'compute_999'"
                            ],
                        },
                    ],
                    "root_cause_status": "confirmed",
                    "root_cause": "The source requests an unsupported architecture.",
                    "source_locations": [{"path": "src/example.cpp", "line": 2}],
                    "next_steps": "Use a supported architecture and rerun the focused jobs.",
                    "agent_prompt": (
                        "Independently verify the compute_999 diagnosis, inspect "
                        "src/example.cpp, implement the smallest fix, and run focused "
                        "validation."
                    ),
                    "job_ids": [101, 102],
                }
            ],
            "downstream_failures": [
                {
                    "job_id": 103,
                    "reason": "The gate reports the primary compile failures.",
                }
            ],
            "cancelled_job_ids": [104],
            "inspected_paths": ["src/example.cpp"],
        }

    def tearDown(self):
        self.temporary_directory.cleanup()

    def validate_and_render(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        RENDERER.validate_schema(self.analysis, schema)
        jobs_by_id = RENDERER.flatten_jobs(self.job_pages)
        failed_job_ids = RENDERER.validate_analysis(
            self.analysis,
            jobs_by_id,
            self.logs,
            self.workspace.resolve(),
        )
        return RENDERER.render_report(
            self.analysis,
            jobs_by_id,
            failed_job_ids,
            "NVIDIA/cccl",
            "12345",
            "a" * 40,
        )

    def test_renders_visible_summary_and_validated_links(self):
        report = self.validate_and_render()

        self.assertTrue(report.startswith("**Summary:**"))
        self.assertLess(report.index("**Start here:**"), report.index("<details open>"))
        self.assertIn("<details open>", report)
        self.assertEqual(report.count("<details open>"), 1)
        self.assertIn(
            "https://github.com/NVIDIA/cccl/actions/runs/12345/job/101#step:4:1",
            report,
        )
        self.assertIn(
            "https://github.com/NVIDIA/cccl/blob/" + "a" * 40 + "/src/example.cpp#L2",
            report,
        )
        self.assertIn("3 of 3 failure logs retrieved", report)

    def test_sanitizes_model_controlled_markdown_links_and_html(self):
        report = self.validate_and_render()

        self.assertNotIn("https://evil.example/a", report)
        visible_report = report.split(
            "<summary><strong>Prompt for an agent</strong></summary>",
            1,
        )[0]
        self.assertNotIn("@team", visible_report)
        self.assertNotIn(
            "</summary>", report.split("<details open>", 1)[1].split("\n", 1)[0]
        )
        self.assertIn("https:", report)
        self.assertIn("@", report)
        self.assertIn("&lt;/summary&gt;", report)

    def test_rejects_duplicate_failure_assignment(self):
        self.analysis["downstream_failures"].append(
            {"job_id": 101, "reason": "Duplicate assignment."}
        )

        with self.assertRaisesRegex(
            RENDERER.ValidationError,
            "all failure job assignments",
        ):
            self.validate_and_render()

    def test_rejects_evidence_not_found_in_the_job_log(self):
        self.analysis["groups"][0]["evidence"][0]["lines"] = ["invented failure"]

        with self.assertRaisesRegex(RENDERER.ValidationError, "not found verbatim"):
            self.validate_and_render()

    def test_rejects_more_than_three_evidence_lines(self):
        self.analysis["groups"][0]["evidence"][0]["lines"] = [
            "setup",
            "nvcc fatal: Unsupported gpu architecture 'compute_999'",
        ]
        self.analysis["groups"][0]["evidence"][1]["lines"] = [
            "setup",
            "nvcc fatal: Unsupported gpu architecture 'compute_999'",
        ]

        with self.assertRaisesRegex(RENDERER.ValidationError, "at most three"):
            self.validate_and_render()


if __name__ == "__main__":
    unittest.main()
