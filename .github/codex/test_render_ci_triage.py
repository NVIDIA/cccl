import importlib.util
import json
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("render_ci_triage.py")
SPEC = importlib.util.spec_from_file_location("render_ci_triage", MODULE_PATH)
RENDERER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDERER)
SCHEMA = json.loads(
    Path(__file__).with_name("ci-triage-output.schema.json").read_text(encoding="utf-8")
)


def sample_analysis():
    return {
        "status": "ok",
        "error": "",
        "summary": (
            "Both builds reject compute_999; do not visit https://evil.example/a "
            "or notify @team."
        ),
        "start_here": "Remove compute_999 from src/example.cpp before #123.",
        "jobs": [
            {"id": 101, "name": "CUDA <build> @team"},
            {"id": 102, "name": "Thrust build"},
            {"id": 103, "name": "CI gate"},
            {"id": 104, "name": "Cancelled matrix job"},
        ],
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
                "next_steps": "Edit src/example.cpp and run test_script.py.",
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


class RenderCiTriageTest(unittest.TestCase):
    def render(self, analysis=None):
        analysis = analysis or sample_analysis()
        RENDERER.validate_schema(analysis, SCHEMA)
        return RENDERER.render_report(
            analysis,
            "NVIDIA/cccl",
            "12345",
            "a" * 40,
        )

    def test_keeps_summary_visible_and_constructs_links(self):
        report = self.render()

        self.assertTrue(report.startswith("**Summary:**"))
        self.assertLess(report.index("**Start here:**"), report.index("<details open>"))
        self.assertEqual(report.count("<details open>"), 1)
        self.assertIn(
            "https://github.com/NVIDIA/cccl/actions/runs/12345/job/101#step:4:1",
            report,
        )
        self.assertIn(
            "https://github.com/NVIDIA/cccl/blob/" + "a" * 40 + "/src/example.cpp#L2",
            report,
        )

    def test_sanitizes_markdown_without_corrupting_file_names(self):
        report = self.render()

        self.assertNotIn("https://evil.example/a", report)
        self.assertNotIn("@team", report.split("```text", 1)[0])
        self.assertIn("&lt;/summary&gt;", report)
        self.assertIn("src/example.cpp", report)
        self.assertIn("test\\_script.py", report)
        self.assertNotIn(chr(0x200B), report)

    def test_missing_job_metadata_degrades_gracefully(self):
        analysis = sample_analysis()
        analysis["jobs"] = []

        report = self.render(analysis)

        self.assertIn("[Job 101]", report)
        self.assertIn("/job/101", report)

    def test_log_retrieval_failure_renders_a_safe_notice(self):
        analysis = {
            "status": "log_retrieval_failed",
            "error": "GitHub returned <b>403</b> @team",
            "summary": "",
            "start_here": "",
            "jobs": [],
            "groups": [],
            "downstream_failures": [],
            "cancelled_job_ids": [],
            "inspected_paths": [],
        }

        report = self.render(analysis)

        self.assertIn("analysis was unavailable", report)
        self.assertIn("&lt;b&gt;403&lt;/b&gt;", report)
        self.assertNotIn("@team", report)

    def test_rejects_fields_outside_the_schema(self):
        analysis = sample_analysis()
        analysis["markdown"] = "<script>alert(1)</script>"

        with self.assertRaisesRegex(RENDERER.ValidationError, "unexpected fields"):
            self.render(analysis)


if __name__ == "__main__":
    unittest.main()
