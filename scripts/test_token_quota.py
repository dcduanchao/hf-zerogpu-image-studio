"""Unit tests for authenticated ZeroGPU quota display; no real tokens or network."""
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui-space"))
import token_quota


class TestTokenQuota(unittest.TestCase):
    def test_empty_token(self):
        self.assertIn("匿名", token_quota.token_quota_info(""))

    def test_invalid_token_format(self):
        self.assertIn("格式无效", token_quota.token_quota_info("invalid"))

    def test_success(self):
        api = SimpleNamespace(
            whoami=lambda: {"name": "tester"},
            get_zero_gpu_quota=lambda: SimpleNamespace(
                remaining=1810, base=2400,
                resets_at=datetime(2026, 11, 20, 0, 0, tzinfo=timezone.utc),
            ),
        )
        with patch.object(token_quota, "HfApi", return_value=api) as call:
            result = token_quota.token_quota_info("hf_test123")
            self.assertEqual(call.call_args.kwargs["token"], "hf_test123")
        self.assertIn("tester", result)
        self.assertIn("30 分 10 秒", result)
        self.assertIn("40 分 0 秒", result)
        self.assertIn("2026-11-20 00:00:00 UTC", result)

    def test_quota_failure_keeps_verified_identity(self):
        def fail_quota():
            raise PermissionError("Billing scope missing")
        api = SimpleNamespace(
            whoami=lambda: {"name": "tester"},
            get_zero_gpu_quota=fail_quota,
        )
        with patch.object(token_quota, "HfApi", return_value=api):
            result = token_quota.token_quota_info("hf_test123")
        self.assertIn("Token 有效：tester", result)
        self.assertIn("暂时无法查询", result)

    def test_identity_failure(self):
        def fail_auth():
            raise ValueError("invalid auth")
        api = SimpleNamespace(whoami=fail_auth)
        with patch.object(token_quota, "HfApi", return_value=api):
            self.assertIn("验证失败", token_quota.token_quota_info("hf_test123"))

    def test_none_reset(self):
        api = SimpleNamespace(
            whoami=lambda: {"name": "tester"},
            get_zero_gpu_quota=lambda: SimpleNamespace(remaining=120, base=180, resets_at=None),
        )
        with patch.object(token_quota, "HfApi", return_value=api):
            self.assertIn("暂未提供", token_quota.token_quota_info("hf_test123"))


if __name__ == "__main__":
    unittest.main()
