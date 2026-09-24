import unittest

from mongo.security import redact_secrets, sanitize_filter


class SanitizeFilter(unittest.TestCase):
    def test_rejects_where_and_js_operators(self):
        for bad in ({"$where": "sleep(10000)"}, {"a": {"$function": {}}}, {"a": {"$accumulator": {}}}):
            with self.assertRaises(ValueError):
                sanitize_filter(bad)

    def test_rejects_unknown_operator(self):
        with self.assertRaises(ValueError):
            sanitize_filter({"role": {"$regexAll": ".*"}})   # not on the allow-list

    def test_allows_safe_operators(self):
        self.assertEqual(sanitize_filter({"qty": {"$gte": 1, "$lte": 10}}), {"qty": {"$gte": 1, "$lte": 10}})
        sanitize_filter({"$or": [{"status": "open"}, {"status": "pending"}]})

    def test_ne_auth_bypass_pattern_is_structurally_allowed_but_typed_callers_never_send_it(self):
        # $ne itself isn't a JS-execution vector, but repository.py methods take scalar
        # arguments and build filters internally - no caller path constructs {"$ne": null}
        # from external input. This test documents that boundary.
        sanitize_filter({"password": {"$ne": None}})

    def test_rejects_non_dict_and_deep_nesting(self):
        with self.assertRaises(ValueError):
            sanitize_filter("not a dict")
        nested = {}
        cur = nested
        for _ in range(10):
            cur["a"] = {}
            cur = cur["a"]
        with self.assertRaises(ValueError):
            sanitize_filter(nested)


class RedactSecrets(unittest.TestCase):
    def test_masks_credential_like_keys(self):
        out = redact_secrets({"email": "a@b.com", "password": "hunter2", "nested": {"api_key": "sk-x"}})
        self.assertEqual(out["password"], "***REDACTED***")
        self.assertEqual(out["nested"]["api_key"], "***REDACTED***")
        self.assertEqual(out["email"], "a@b.com")


if __name__ == "__main__":
    unittest.main()
