from __future__ import annotations

import unittest

from perbacco.bounds import exact_bin_count, full_queries_and_remainder, phi_bounds


class BoundTests(unittest.TestCase):
    def test_recursion_examples(self) -> None:
        self.assertEqual(full_queries_and_remainder(1, 10), (0, 1))
        self.assertEqual(full_queries_and_remainder(10, 10), (1, 1))
        self.assertEqual(full_queries_and_remainder(64, 10), (7, 1))

    def test_exact_bin_packing_beats_first_fit_counterexample(self) -> None:
        self.assertEqual(exact_bin_count([6, 5, 3, 2, 2, 2], 10), 2)

    def test_phi_bounds(self) -> None:
        self.assertEqual(phi_bounds([10, 5, 3, 2], 10), (2, 2))


if __name__ == "__main__":
    unittest.main()
