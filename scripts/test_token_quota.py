"""Offline quota tests; never send real HF credentials or call external services."""
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

    def test_success_and_official_quota_fields(self):
        fake_response = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"current": 1810, "base": 2400, "resetsAt": "2026-11-20T00:00:00Z"},
        )
        with patch.object(token_quota, "httpx") as http, patch.object(
            token_quota, "HfApi", return_value=SimpleNamespace(whoami=lambda: {"name": "tester"})
        ):
            http.get.return_value = fake_response
            result = token_quota.token_quota_info("hf_test123")
            self.assertEqual(http.get.call_args.kwargs["headers"]["Authorization"], "Bearer hf_test123")
            self.assertEqual(http.get.call_args.args[0], token_quota.QUOTA_ENDPOINT)
        self.assertIn("tester", result)
        self.assertIn("30 分 10 秒", result)
        self.assertIn("40 分 0 秒", result)
        self.assertIn("2026-11-20 00:00:00 UTC", result)

    def test_quota_failure_keeps_verified_identity(self):
        with patch.object(token_quota, "fetch_quota", side_effect=PermissionError()), patch.object(
            token_quota, "HfApi", return_value=SimpleNamespace(whoami=lambda: {"name": "tester"})
        ):
            result = token_quota.token_quota_info("hf_test123")
        self.assertIn("Token 有效：tester", result)
        self.assertIn("暂时无法查询", result)

    def test_identity_failure(self):
        with patch.object(token_quota, "HfApi", side_effect=ValueError()):
            self.assertIn("验证失败", token_quota.token_quota_info("hf_test123"))

    def test_none_reset(self):
        with patch.object(token_quota, "httpx") as http:
            http.get.return_value = SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"current": 120, "base": 180, "resetsAt": None},
            )
            self.assertIn("暂未提供", token_quota.fetch_quota("hf_test123")[2])


if __name__ == "__main__":
    unittest.main()
