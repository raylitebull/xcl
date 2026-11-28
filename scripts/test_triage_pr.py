"""Unit tests for scripts/triage_pr.py."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import triage_pr as tp


class TestTriageClassification(unittest.TestCase):
    def test_docs_only_pr(self):
        files = ["AGENTS.md", "TESTING.md", ".github/workflows/stale.yml"]
        self.assertEqual(tp.classify_areas(files), ["area:docs-ci"])
        self.assertEqual(tp.classify_gpu_targets(files, "Docs: add PR testing guide"), [])
        self.assertEqual(tp.classify_test_tier(files, "Docs: add PR testing guide"), 0)

    def test_nvidia_wave32_recompiler_pr(self):
        files = ["crates/xcl/src/gpu/recomp/upper_lanes.rs", "crates/xcl/src/gpu/recomp/tests.rs"]
        areas = tp.classify_areas(files)
        self.assertIn("area:recompiler", areas)
        self.assertIn("area:tests", areas)
        self.assertEqual(
            tp.classify_gpu_targets(files, "Recompiler: fix split-wave64 upper lane mask on RTX 5090"),
            ["nvidia"],
        )
        self.assertEqual(tp.classify_test_tier(files, "Recompiler: fix split-wave64"), 1)

    def test_amd_wave64_d3d12_pr(self):
        files = ["crates/xcl/src/hle/d3d12x/device_x.rs"]
        self.assertEqual(tp.classify_areas(files), ["area:d3d12"])
        self.assertEqual(
            tp.classify_gpu_targets(files, "D3D12.X: RDNA4 native wave64 barrier hint"),
            ["amd"],
        )
        self.assertEqual(tp.classify_test_tier(files, "D3D12.X: RDNA4 native wave64 barrier hint"), 2)

    def test_shader_world_load_is_tier_3_both_gpus(self):
        files = ["crates/xcl/src/shaders/pipeline_library.rs", "crates/xcl/src/hle/d3d12x/vram.rs"]
        areas = tp.classify_areas(files)
        self.assertIn("area:shaders", areas)
        self.assertIn("area:memory", areas)
        self.assertEqual(tp.classify_gpu_targets(files, "Shaders: bounded startup pipeline library"), ["nvidia", "amd"])
        self.assertEqual(tp.classify_test_tier(files, "Shaders: bounded startup pipeline library"), 3)

    def test_non_gpu_hle_pr(self):
        files = ["crates/xcl/src/hle/dstorage.rs", "crates/xcl/src/vfs.rs"]
        body = (
            "Preserve queue order.\n\n"
            "**Tested**\n- AMD Ryzen 7 9800X3D, RTX 5090\n\n"
            "**Not tested**\n- AMD GPU\n"
        )
        self.assertEqual(tp.classify_areas(files), ["area:directstorage", "area:hle"])
        self.assertEqual(tp.classify_gpu_targets(files, "DirectStorage: preserve queue ordering", body), [])
        self.assertEqual(tp.classify_test_tier(files, "DirectStorage: preserve queue ordering"), 1)


class TestMergeCheckParsing(unittest.TestCase):
    def test_parses_linked_sha_and_deduplicates_and_excludes_author(self):
        head = "d338db7a8e24aeb0f42c4c96d98bacd651550f29"
        comments = [
            {
                "author": "pr_author",
                "body": f"merge-check\nhead: {head}\nverdict: safe to merge\nhardware / tool: RTX 5090",
            },
            {
                "author": "hollow",
                "body": f"merge-check\nhead: [`d338db7`](https://example.com)\nverdict: not yet: missing test\nhardware / tool: RTX 5070 Ti",
            },
            {
                "author": "hollow",
                "body": f"merge-check\nhead: [`d338db7`](https://example.com)\nverdict: safe to merge\nhardware / tool: AMD Ryzen 7 9800X3D, RTX 5070 Ti",
            },
            {
                "author": "throny",
                "body": f"merge-check\nhead: {head[:12]}\nverdict: safe to merge\nhardware / tool: RX 9070 XT (RDNA4)",
            },
            {
                "author": "stale_checker",
                "body": "merge-check\nhead: 000000000000\nverdict: safe to merge",
            },
        ]
        mc = tp.parse_merge_checks(comments, "pr_author", head)
        self.assertEqual(mc["safe_count"], 2)
        self.assertEqual(mc["not_yet_count"], 0)
        self.assertEqual(mc["safe_users"], ["hollow", "throny"])
        self.assertTrue(mc["tested_nvidia"])
        self.assertTrue(mc["tested_amd"])

    def test_managed_labels_needs_2nd_check_and_amd_needed(self):
        mc = {
            "safe_count": 1,
            "not_yet_count": 0,
            "tested_nvidia": True,
            "tested_amd": False,
        }
        labels = tp.compute_managed_labels(
            files=["crates/xcl/src/hle/d3d12x/sparse.rs"],
            title="D3D12.X: sparse tile mapping",
            body="",
            existing_labels={"goal:world-load"},
            is_draft=False,
            is_conflicting=False,
            ci_state="SUCCESS",
            pr_check_state="SUCCESS",
            merge_checks=mc,
        )
        self.assertIn("area:d3d12", labels)
        self.assertIn("area:memory", labels)
        self.assertIn("gpu:both", labels)
        self.assertIn("needs-2nd-check", labels)
        self.assertIn("needs:amd-test", labels)
        self.assertNotIn("needs:nvidia-test", labels)
        self.assertNotIn("ready-to-merge", labels)

    def test_managed_labels_conflict_and_failed_pr_check(self):
        mc = {"safe_count": 0, "not_yet_count": 0, "tested_nvidia": False, "tested_amd": False}
        labels = tp.compute_managed_labels(
            files=["crates/xcl/src/hle/d3d12x/sparse.rs"],
            title="D3D12.X: sparse tile mapping",
            body="",
            existing_labels=set(),
            is_draft=False,
            is_conflicting=True,
            ci_state="SUCCESS",
            pr_check_state="FAILURE",
            merge_checks=mc,
        )
        self.assertIn("needs-rebase", labels)
        self.assertIn("needs-pr-check", labels)
        self.assertNotIn("needs-1st-check", labels)
        self.assertNotIn("needs:nvidia-test", labels)


if __name__ == "__main__":
    unittest.main()
