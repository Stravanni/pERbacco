from __future__ import annotations

import unittest

from perbacco import _native


class AbiTests(unittest.TestCase):
    def test_public_symbols_are_exported(self) -> None:
        symbols = (
            "pb_config_init",
            "pb_engine_create",
            "pb_engine_create_borrowed",
            "pb_engine_create_subopt",
            "pb_engine_next_batch",
            "pb_engine_submit_partition",
            "pb_engine_group_members",
            "pb_engine_get_stats",
            "pb_engine_snapshot_size",
            "pb_engine_snapshot",
            "pb_engine_restore",
            "pb_engine_last_error",
            "pb_strerror",
            "pb_engine_free",
        )
        for symbol in symbols:
            with self.subTest(symbol=symbol):
                self.assertTrue(hasattr(_native.lib, symbol))


if __name__ == "__main__":
    unittest.main()
