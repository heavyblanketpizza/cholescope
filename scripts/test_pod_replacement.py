"""Offline replacement transport tests; all provider responses are simulated."""

import base64
from io import BytesIO
import json
import unittest
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

import pod_replacement as replacement
import runpod_budget as guard


START = guard.timestamp("2026-10-23T18:00:00Z")
NAME = "cholescope-replacement-testnonce"
PUBLIC_KEY = "ssh-ed25519 " + base64.b64encode(
    b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20" + bytes(range(32))).decode() + " test-fixture"


def recipe():
    return {"gpu_id": replacement.GPU_ID, "data_center_id": "EU-TEST-1",
            "volume_id": "test-volume", "image": replacement.IMAGE,
            "public_key": PUBLIC_KEY, "max_hourly_rate": 2.49}


def gpu():
    return {"id": replacement.GPU_ID, "manufacturer": "NVIDIA", "secure": True,
            "memory": 96, "price": {"secure": 2.49, "community": 1.69},
            "availability": "LOW", "dataCenters": [{"id": "EU-TEST-1", "availability": "LOW"}]}


def volume():
    return {"id": "test-volume", "dataCenter": "EU-TEST-1", "size": 150, "type": "STANDARD"}


def created():
    return {"id": "pod-new", "name": NAME, "createdAt": guard.utc(START + 3),
            "gpu": {"id": replacement.GPU_ID, "count": 1, "memory": 188},
            "image": replacement.IMAGE, "disk": 30, "ports": ["22/tcp"],
            "cloud": "SECURE", "dataCenterId": "EU-TEST-1", "env": {"PUBLIC_KEY": PUBLIC_KEY},
            "args": "", "registry": None, "template": None,
            "mounts": {"network": [{"volumeId": "test-volume", "path": "/workspace"}]},
            "globalNetworking": {"enabled": False}, "locked": False, "cost": 2.49,
            "status": "PROVISIONING", "actions": ["stop", "terminate"]}


class Response(BytesIO):
    pass


class Opener:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        if not self.responses:
            raise AssertionError("Unexpected request or retry")
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return Response(value if isinstance(value, bytes) else json.dumps(value).encode())


def api(*responses):
    opener = Opener(*responses)
    return replacement.ReplacementAPI(guard.RunpodAPI("private-test-token", opener=opener)), opener


class RecipeTests(unittest.TestCase):
    def test_only_fixed_resources_and_public_ssh_key_are_sent(self):
        request = replacement.build_create_body(recipe(), NAME)
        self.assertEqual(request["gpu"], {"id": replacement.GPU_ID, "count": 1, "minCudaVersion": "13.2"})
        self.assertEqual(request["dataCenterIds"], ["EU-TEST-1"])
        self.assertEqual(request["mounts"], {"network": [{"volumeId": "test-volume", "path": "/workspace"}]})
        self.assertEqual(request["cloud"], "SECURE")
        self.assertEqual(request["disk"], 30)
        self.assertEqual(request["ports"], ["22/tcp"])
        self.assertEqual(request["env"], {"PUBLIC_KEY": PUBLIC_KEY})
        self.assertFalse(request["startJupyter"])
        self.assertFalse(request["globalNetworking"])
        self.assertNotIn("templateId", request)
        self.assertNotIn("args", request)

    def test_unapproved_recipe_cannot_reach_transport(self):
        for changes in ({"env": {"SECRET": "sensitive"}}, {"gpu_id": "other-gpu"},
                        {"data_center_id": "US-TEST-1"}, {"volume_id": "../other"},
                        {"image": "arbitrary/image"}, {"max_hourly_rate": 3.01},
                        {"max_hourly_rate": float("nan")}, {"max_hourly_rate": True},
                        {"public_key": PUBLIC_KEY + "\nother"}, {"public_key": "private-key"}):
            with self.subTest(changes=changes):
                client, opener = api()
                with self.assertRaises(replacement.ReplacementPreflightError):
                    client.create({**recipe(), **changes}, NAME)
                self.assertEqual(opener.calls, [])

    def test_bad_attempt_name_is_rejected(self):
        with self.assertRaises(replacement.ReplacementPreflightError):
            replacement.build_create_body(recipe(), "demo\nsecond-name")


class TransportTests(unittest.TestCase):
    def test_catalog_read_is_scoped_to_same_secure_gpu_and_cuda_floor(self):
        client, opener = api(gpu())
        client.get_gpu(recipe())
        request, timeout = opener.calls[0]
        url = urlsplit(request.full_url)
        self.assertEqual(url.netloc, "api.runpod.io")
        self.assertIn("NVIDIA%20RTX%20PRO%206000%20Blackwell%20Server%20Edition", url.path)
        self.assertEqual(parse_qs(url.query), {"include": ["AVAILABILITY"], "product": ["POD"],
                         "count": ["1"], "cloud": ["SECURE"], "minCudaVersion": ["13.2"]})
        self.assertEqual(request.get_header("Authorization"), "Bearer private-test-token")
        self.assertEqual(timeout, 30)

    def test_create_makes_exactly_one_post_even_when_response_is_lost(self):
        client, opener = api(URLError("private connection details"))
        with self.assertRaisesRegex(replacement.ReplacementError, "^Runpod API is unreachable") as caught:
            client.create(recipe(), NAME)
        self.assertNotIn("private", str(caught.exception))
        self.assertEqual(len(opener.calls), 1)
        request, _ = opener.calls[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.full_url, guard.API + "/pods")
        self.assertEqual(json.loads(request.data)["name"], NAME)

    def test_http_failures_are_typed_and_provider_body_is_not_exposed(self):
        for status in (400, 401, 404, 429, 500, 302):
            with self.subTest(status=status):
                failure = HTTPError(guard.API + "/pods", status, "sensitive reason",
                                    {"private-header": "secret"}, BytesIO(b'{"detail":"private-test-token"}'))
                client, opener = api(failure)
                with self.assertRaises(guard.RunpodHTTPError) as caught:
                    client.create(recipe(), NAME)
                self.assertEqual(caught.exception.status, status)
                self.assertEqual(str(caught.exception), f"Runpod HTTP {status}")
                self.assertIsNone(caught.exception.public_message)
                self.assertEqual(len(opener.calls), 1)

    def test_only_existing_allowlisted_capacity_phrase_crosses_transport(self):
        body = {"detail": "There are not enough free GPUs on the host machine to start this pod."}
        failure = HTTPError(guard.API + "/pods", 400, "Bad request", {}, BytesIO(json.dumps(body).encode()))
        client, _ = api(failure)
        with self.assertRaises(guard.RunpodHTTPError) as caught:
            client.create(recipe(), NAME)
        self.assertEqual(caught.exception.public_message, guard.GPU_UNAVAILABLE_MESSAGE)

    def test_pagination_follows_opaque_cursor_without_changing_origin(self):
        first = {"pods": [{"id": "first"}], "pagination": {"hasNextPage": True, "nextCursor": "https://other/?x=y&z=2"}}
        last = {"pods": [{"id": "last"}], "pagination": {"hasNextPage": False, "nextCursor": None}}
        client, opener = api(first, last)
        self.assertEqual([pod["id"] for pod in client.list_pods()], ["first", "last"])
        second_url = urlsplit(opener.calls[1][0].full_url)
        self.assertEqual(second_url.netloc, "api.runpod.io")
        self.assertEqual(parse_qs(second_url.query)["cursor"], ["https://other/?x=y&z=2"])

    def test_incomplete_or_repeating_listing_is_never_returned(self):
        cases = [
            [{"pods": [], "pagination": {"hasNextPage": True, "nextCursor": None}}],
            [{"pods": [], "pagination": {"hasNextPage": "false", "nextCursor": None}}],
            [{"pods": [{"id": "one"}], "pagination": {"hasNextPage": True, "nextCursor": "same"}},
             {"pods": [], "pagination": {"hasNextPage": True, "nextCursor": "same"}}],
            [{"pods": [{"id": "one"}], "pagination": {"hasNextPage": True, "nextCursor": "next"}},
             {"pods": [{"id": "one"}], "pagination": {"hasNextPage": False, "nextCursor": None}}],
        ]
        for pages in cases:
            with self.subTest(pages=pages):
                client, _ = api(*pages)
                with self.assertRaises(replacement.ReplacementProtocolError):
                    client.list_pods()

    def test_twenty_page_bound_fails_without_returning_partial_account(self):
        pages = [{"pods": [], "pagination": {"hasNextPage": True, "nextCursor": str(index)}} for index in range(20)]
        client, opener = api(*pages)
        with self.assertRaisesRegex(replacement.ReplacementProtocolError, "page limit"):
            client.list_pods()
        self.assertEqual(len(opener.calls), 20)

    def test_non_json_response_is_sanitized(self):
        client, _ = api(b"private-test-token")
        with self.assertRaisesRegex(replacement.ReplacementProtocolError, "invalid response") as caught:
            client.get_volume("test-volume")
        self.assertNotIn("private-test-token", str(caught.exception))


class PreflightTests(unittest.TestCase):
    def test_preflight_returns_safe_quote_without_key(self):
        client, _ = api(gpu(), volume())
        summary = client.preflight(recipe())
        self.assertEqual(summary["quoted_rate"], 2.49)
        self.assertEqual(summary["memory_gb"], 96)
        self.assertNotIn(PUBLIC_KEY, json.dumps(summary))

    def test_exact_region_stock_can_be_used_when_overall_stock_is_none(self):
        for overall in ("NONE", None):
            candidate = {**gpu(), "availability": overall}
            client, _ = api(candidate, volume())
            summary = client.preflight(recipe())
            self.assertEqual(summary["availability"], "LOW")
            self.assertEqual(summary["data_center_id"], "EU-TEST-1")

    def test_wrong_memory_price_or_regional_stock_prevents_create(self):
        candidates = [
            {**gpu(), "memory": 80}, {**gpu(), "secure": False},
            {**gpu(), "price": {"secure": 2.50}},
            {**gpu(), "price": {"secure": float("nan")}},
            {**gpu(), "dataCenters": [{"id": "EU-OTHER-1", "availability": "HIGH"}]},
            {**gpu(), "dataCenters": [{"id": "EU-TEST-1", "availability": "NONE"}]},
            {**gpu(), "availability": []},
            {**gpu(), "dataCenters": [{"id": "EU-TEST-1", "availability": []}]},
        ]
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                client, opener = api(candidate, volume())
                with self.assertRaises(replacement.ReplacementPreflightError):
                    client.preflight(recipe())
                self.assertTrue(all(request.method == "GET" for request, _ in opener.calls))

    def test_volume_must_match_both_identity_and_data_center(self):
        for candidate in ({**volume(), "id": "other-volume"}, {**volume(), "dataCenter": "EU-OTHER-1"}):
            client, _ = api(gpu(), candidate)
            with self.assertRaisesRegex(replacement.ReplacementPreflightError, "network volume"):
                client.preflight(recipe())

    def test_volume_must_remain_150_gb_standard_storage(self):
        for candidate in ({**volume(), "size": 600}, {**volume(), "type": "HIGH_PERFORMANCE"}):
            client, _ = api(gpu(), candidate)
            with self.assertRaisesRegex(replacement.ReplacementPreflightError, "150 GB standard"):
                client.preflight(recipe())


class VerificationTests(unittest.TestCase):
    def test_host_ram_is_not_mistaken_for_vram(self):
        result = replacement.verify_created(created(), recipe(), NAME, START)
        self.assertEqual(result["gpu"]["memory"], 188)

    def test_unrelated_old_or_wrong_name_pod_is_never_owned(self):
        for changes in ({"name": "unrelated"}, {"createdAt": guard.utc(START - 121)},
                        {"id": "../bad"}, {"createdAt": "invalid"}):
            with self.subTest(changes=changes):
                pod = {**created(), **changes}
                self.assertFalse(replacement.ownership_match(pod, NAME, START))
                with self.assertRaises(replacement.ReplacementVerificationError) as caught:
                    replacement.verify_created(pod, recipe(), NAME, START)
                self.assertFalse(caught.exception.owned_identity)

    def test_exact_nonce_allows_bounded_provider_clock_skew(self):
        self.assertTrue(replacement.ownership_match(
            {**created(), "createdAt": guard.utc(START - 120)}, NAME, START))

    def test_returned_cuda_must_meet_floor_when_known(self):
        for cuda in (None, "13.2", "13.10", "14.0"):
            replacement.verify_created({**created(), "cudaVersion": cuda}, recipe(), NAME, START)
        for cuda in ("13.0", "12.8", "unknown", ["13.2"]):
            with self.assertRaises(replacement.ReplacementVerificationError) as caught:
                replacement.verify_created({**created(), "cudaVersion": cuda}, recipe(), NAME, START)
            self.assertIn("CUDA version", caught.exception.findings)
            self.assertTrue(caught.exception.owned_identity)

    def test_late_queued_create_is_still_owned_after_session_deadline(self):
        pod = {**created(), "createdAt": guard.utc(START + 7200)}
        self.assertTrue(replacement.ownership_match(pod, NAME, START))

    def test_owned_but_mismatched_resource_reports_findings_for_stop(self):
        changes = [
            {"gpu": {"id": replacement.GPU_ID, "count": 2}},
            {"cloud": "COMMUNITY"}, {"dataCenterId": "EU-OTHER-1"},
            {"mounts": {"network": [{"volumeId": "other-volume", "path": "/workspace"}]}},
            {"image": "private-image-value"}, {"disk": 150}, {"cost": 3.99},
            {"ports": ["22/tcp", "8888/http"]}, {"env": {"SECRET": "private-secret-value"}},
            {"globalNetworking": {"enabled": True}}, {"locked": True},
            {"args": "private-command-value"},
            {"entrypoint": ["private-command-value"]},
        ]
        for mutation in changes:
            with self.subTest(mutation=mutation):
                with self.assertRaises(replacement.ReplacementVerificationError) as caught:
                    replacement.verify_created({**created(), **mutation}, recipe(), NAME, START)
                self.assertTrue(caught.exception.owned_identity)
                self.assertTrue(caught.exception.findings)
                self.assertNotIn("private-", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
