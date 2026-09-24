"""Runs without a database: RBAC matrix, tokens, rate limiter (TRD 15, EC-43..45)."""
import time
import unittest

import jwt

from backend.app.security import rbac, tokens
from backend.app.security.ratelimit import RateLimiter

SECRET = "x" * 40


class Rbac(unittest.TestCase):
    def test_least_privilege_matrix(self):
        self.assertTrue(rbac.has_permission("inventory_manager", "recommendations:decide"))      # EC-43
        for role in ("analyst", "operations_user"):
            self.assertFalse(rbac.has_permission(role, "recommendations:decide"), role)          # EC-44
        self.assertFalse(rbac.has_permission("analyst", "sales:write"))
        self.assertTrue(rbac.has_permission("analyst", "audit:read"))
        self.assertFalse(rbac.has_permission("operations_user", "audit:read"))
        self.assertFalse(rbac.has_permission("automation_service", "recommendations:decide"))    # cannot approve manually
        self.assertTrue(rbac.has_permission("automation_service", "recommendations:auto_approve"))

    def test_admin_superset_and_unknown_role(self):
        for perms in rbac.PERMISSIONS.values():
            self.assertTrue(perms <= rbac.PERMISSIONS[rbac.Role.ADMIN])
        self.assertFalse(rbac.has_permission("hacker", "products:read"))                        # privilege escalation via payload
        self.assertFalse(rbac.has_permission("", "products:read"))


class Tokens(unittest.TestCase):
    def test_roundtrip(self):
        c = tokens.decode_token(SECRET, tokens.create_token(SECRET, 7, "analyst", "sid1"))
        self.assertEqual((c["sub"], c["role"], c["sid"]), ("7", "analyst", "sid1"))

    def test_expired_wrong_secret_and_alg_none(self):
        with self.assertRaises(jwt.ExpiredSignatureError):
            tokens.decode_token(SECRET, tokens.create_token(SECRET, 1, "admin", "s", minutes=-1))
        with self.assertRaises(jwt.InvalidTokenError):
            tokens.decode_token("y" * 40, tokens.create_token(SECRET, 1, "admin", "s"))
        forged = jwt.encode({"sub": "1", "sid": "s", "exp": time.time() + 99}, key=None, algorithm="none")
        with self.assertRaises(jwt.InvalidTokenError):
            tokens.decode_token(SECRET, forged)

    def test_required_claims(self):
        bad = jwt.encode({"sub": "1", "exp": time.time() + 99}, SECRET, algorithm="HS256")   # no sid
        with self.assertRaises(jwt.InvalidTokenError):
            tokens.decode_token(SECRET, bad)


class Limiter(unittest.TestCase):
    def test_window(self):
        now = [0.0]
        rl = RateLimiter(3, 60, clock=lambda: now[0])
        self.assertEqual([rl.allow("a") for _ in range(4)], [True, True, True, False])
        self.assertTrue(rl.allow("b"))
        now[0] = 61
        self.assertTrue(rl.allow("a"))


if __name__ == "__main__":
    unittest.main()
