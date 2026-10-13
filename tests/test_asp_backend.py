import unittest
from unittest.mock import patch

import pytest

from boolmore.inference import asp_backend
from boolmore.inference.asp_backend import (
    ensure_asp_backend,
    reset_asp_backend_check,
)


class TestEnsureAspBackend(unittest.TestCase):

    def setUp(self):
        reset_asp_backend_check()

    def tearDown(self):
        reset_asp_backend_check()

    def test_probe_succeeds_when_asp_works(self):
        try:
            ensure_asp_backend()
        except RuntimeError as exc:
            pytest.skip(f"ASP backend unavailable in this environment: {exc}")

        # Cached success should be a no-op.
        ensure_asp_backend()

    def test_empty_probe_raises_and_is_cached(self):
        with patch.object(
            asp_backend,
            "compute_trap_spaces",
            return_value=[],
        ) as mock_compute:
            with self.assertRaises(RuntimeError) as ctx:
                ensure_asp_backend()
            self.assertIn("gringo/clasp", str(ctx.exception))
            self.assertIn("README", str(ctx.exception))
            self.assertEqual(mock_compute.call_count, 1)

            with self.assertRaises(RuntimeError):
                ensure_asp_backend()
            # Cached failure must not re-run the probe.
            self.assertEqual(mock_compute.call_count, 1)

    def test_exception_from_probe_raises_and_is_cached(self):
        with patch.object(
            asp_backend,
            "compute_trap_spaces",
            side_effect=OSError("boom"),
        ) as mock_compute:
            with self.assertRaises(RuntimeError) as ctx:
                ensure_asp_backend()
            self.assertIsInstance(ctx.exception.__cause__, OSError)
            self.assertEqual(mock_compute.call_count, 1)

            with self.assertRaises(RuntimeError):
                ensure_asp_backend()
            self.assertEqual(mock_compute.call_count, 1)


if __name__ == "__main__":
    unittest.main()
