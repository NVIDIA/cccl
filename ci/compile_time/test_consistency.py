#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.

import math
import unittest

from ci.compile_time.consistency import consistency_pvalue, holm_adjust


class ConsistencyTest(unittest.TestCase):
    def test_exact_tails(self):
        for n in (0, 1, 4, 16, 64, 559):
            for k in (0, n // 2, n):
                numerator = sum(math.comb(n, j) * 3**j for j in range(k, n + 1))
                self.assertEqual(consistency_pvalue(n, k), numerator / 4**n)

    def test_null_rejection_probability(self):
        # Exhaust the sufficient statistic's exact null distribution. Coverage
        # is conservative at the 75% boundary, and also below that boundary.
        for n in (8, 16, 32, 64):
            for probability in (0.25, 0.5, 0.75):
                rejection_probability = sum(
                    math.comb(n, k) * probability**k * (1 - probability) ** (n - k)
                    for k in range(n + 1)
                    if consistency_pvalue(n, k) <= 0.05
                )
                self.assertLessEqual(rejection_probability, 0.05 + 1e-14)

    def test_occurrences_do_not_change_trial_count(self):
        before = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        after = [11, 11, 11, 11, 11, 11, 11, 11, 9, 9, 9, 9]
        for scale in (1, 100, 138533):
            deltas = [(c - b) * scale for b, c in zip(before, after)]
            self.assertEqual(
                consistency_pvalue(len(deltas), sum(d > 0 for d in deltas)),
                consistency_pvalue(12, 8),
            )

    def test_holm_order(self):
        self.assertEqual(
            holm_adjust([0.04, 0.001, 0.03, 0.2]), [0.09, 0.004, 0.09, 0.2]
        )
        self.assertEqual(holm_adjust([]), [])

    def test_family_includes_unselected_hypotheses(self):
        p = consistency_pvalue(24, 24)
        self.assertLessEqual(holm_adjust([p, 1.0])[0], 0.05)
        self.assertGreater(holm_adjust([p] + [1.0] * 100)[0], 0.05)

    def test_small_context_sets_cannot_establish_consistency(self):
        self.assertEqual(consistency_pvalue(4, 4), 0.75**4)
        self.assertGreater(holm_adjust([consistency_pvalue(4, 4), 1.0])[0], 0.05)
        self.assertGreater(consistency_pvalue(559, 387), 0.99)
        self.assertLess(holm_adjust([consistency_pvalue(100, 95), 1.0])[0], 0.05)


if __name__ == "__main__":
    unittest.main()
