"""CPU-only fault-injection tests; no Runpod account or network is used."""
import io
import json
import unittest
from urllib.error import HTTPError

import runpod_budget as guard


class FakeClock:
    def __init__(self, now):
        self.now = now
        self.elapsed = 0

    def sleep(self, seconds):
        self.now += seconds
        self.elapsed += seconds
        if self.elapsed > 100000:
            raise AssertionError("Watchdog never terminated")


def lease(action="stop"):
    return guard.make_lease("pod-123", "cholescope-test", "2026-01-01T00:00:00Z",
                            2.49, hours=1, action=action)


def session_lease(**kwargs):
    return guard.build_session_lease(
        "pod-123", "cholescope-test", "2026-01-01T00:00:00Z", 2.49,
        session_start_epoch=guard.timestamp("2026-02-01T00:00:00Z"),
        spent=8.15, **kwargs)


class FakeAPI:
    def __init__(self, config, clock, *, status="RUNNING", cost=2.49, fail_shutdown=0):
        self.config = config
        self.clock = clock
        self.pod = {"id": "pod-123", "name": "cholescope-test", "status": status,
                    "createdAt": config["created_at"], "cost": cost}
        self.calls = []
        self.fail_shutdown = fail_shutdown
        self.confirm_lag = 0

    def get(self, pod_id):
        self.calls.append(("GET", pod_id, self.clock.now))
        if self.confirm_lag:
            self.confirm_lag -= 1
            if self.confirm_lag == 0:
                self.pod = None if self.config["action"] == "terminate" else {
                    **self.pod, "status": "EXITED", "cost": 0}
        return self.pod

    def shutdown(self, pod_id, action):
        self.calls.append((action, pod_id, self.clock.now))
        if self.fail_shutdown:
            self.fail_shutdown -= 1
            raise RuntimeError("temporary outage")
        if not self.confirm_lag:
            self.confirm_lag = 2


class BudgetTests(unittest.TestCase):
    def test_quote_counts_previous_spend_and_reserves_storage(self):
        plan = guard.quote(2.49, 12, 10)
        self.assertAlmostEqual(plan["projected_total_usd"], 44.88)
        with self.assertRaises(ValueError):
            guard.quote(6.25, 12)
        with self.assertRaises(ValueError):
            guard.quote(2.49, 12, 50)

    def test_bad_numeric_inputs_are_rejected(self):
        for bad in [float("nan"), float("inf"), -1, 0]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                guard.quote(bad)
        with self.assertRaises(ValueError):
            guard.quote(1, hours=12.01)
        with self.assertRaises(ValueError):
            guard.quote(1, reserve=4.99)

    def test_invalid_pod_identifier_or_timezone_is_rejected(self):
        for pod_id in ("", "../all", "pod?id=other", "pod/id"):
            with self.subTest(pod_id=pod_id), self.assertRaises(ValueError):
                guard.make_lease(pod_id, "test", "2026-01-01T00:00:00Z", 1)
        with self.assertRaises(ValueError):
            guard.timestamp("2026-01-01T00:00:00")

    def test_edited_budget_and_deadline_are_rejected(self):
        config = lease()
        for changed in ({"deadline_epoch": config["deadline_epoch"] + 1},
                        {"compute_ceiling_usd": 80}, {"action": "restart"}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                guard.validate_lease({**config, **changed})

    def run_watch(self, config, api, clock, **kwargs):
        guard.watch(config, api, clock=lambda: clock.now,
                    monotonic=lambda: clock.elapsed, sleep=clock.sleep,
                    log=lambda *args, **kwargs: None, **kwargs)

    def test_stop_only_exact_pod_and_verify_after_acknowledgment(self):
        config = lease()
        clock = FakeClock(config["deadline_epoch"] - 10)
        api = FakeAPI(config, clock)
        self.run_watch(config, api, clock)
        actions = [c for c in api.calls if c[0] != "GET"]
        self.assertTrue(actions)
        self.assertEqual({c[1] for c in api.calls}, {"pod-123"})
        self.assertEqual(actions[0][2], config["deadline_epoch"])
        self.assertEqual(api.calls[-1][0], "GET")
        self.assertEqual(api.pod["status"], "EXITED")

    def test_identity_mismatch_never_mutates(self):
        config = lease()
        clock = FakeClock(config["deadline_epoch"] + 1)
        for mutation in ({"id": "other"}, {"name": "other"},
                         {"createdAt": "2026-01-02T00:00:00Z"}):
            api = FakeAPI(config, clock)
            api.pod.update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.run_watch(config, api, clock)
            self.assertEqual([c[0] for c in api.calls], ["GET"])

    def test_failed_shutdown_is_retried_and_not_reported_as_success(self):
        config = lease("terminate")
        clock = FakeClock(config["deadline_epoch"])
        api = FakeAPI(config, clock, fail_shutdown=2)
        self.run_watch(config, api, clock)
        self.assertGreaterEqual(len([c for c in api.calls if c[0] == "terminate"]), 3)
        self.assertIsNone(api.pod)

    def test_terminate_removes_stopped_pod_too(self):
        config = lease("terminate")
        clock = FakeClock(config["deadline_epoch"])
        api = FakeAPI(config, clock, status="EXITED", cost=0)
        self.run_watch(config, api, clock)
        self.assertIsNone(api.pod)

    def test_rate_increase_shortens_deadline_and_decrease_never_extends(self):
        config = lease()
        created = guard.timestamp(config["created_at"])
        self.assertLess(guard.effective_deadline(config, 100), config["deadline_epoch"])
        self.assertEqual(guard.effective_deadline(config, 0.1), config["deadline_epoch"])
        clock = FakeClock(created + 2700)
        api = FakeAPI(config, clock, cost=100)
        self.run_watch(config, api, clock)
        self.assertEqual([c for c in api.calls if c[0] == "stop"][0][2], created + 2700)

    def test_signal_requests_immediate_shutdown(self):
        config = lease()
        clock = FakeClock(guard.timestamp(config["created_at"]))
        api = FakeAPI(config, clock)
        self.run_watch(config, api, clock, stop_requested=lambda: True)
        self.assertEqual([c for c in api.calls if c[0] == "stop"][0][2],
                         guard.timestamp(config["created_at"]))

    def test_missing_pod_is_already_terminated(self):
        config = lease()
        clock = FakeClock(config["deadline_epoch"])
        api = FakeAPI(config, clock)
        api.pod = None
        self.run_watch(config, api, clock)
        self.assertEqual([c[0] for c in api.calls], ["GET"])

    def test_inspection_outage_still_attempts_exact_pod_shutdown_at_deadline(self):
        config = lease()
        clock = FakeClock(config["deadline_epoch"])
        class UnreachableAPI(FakeAPI):
            def get(self, pod_id):
                if not self.calls:
                    self.calls.append(("GET_FAILED", pod_id, self.clock.now))
                    raise RuntimeError("offline")
                return super().get(pod_id)
        api = UnreachableAPI(config, clock)
        self.run_watch(config, api, clock)
        self.assertEqual(api.calls[1], ("stop", "pod-123", config["deadline_epoch"]))

    def test_backward_wall_clock_does_not_extend_running_lease(self):
        config = lease()
        clock = FakeClock(config["deadline_epoch"] - 10)
        api = FakeAPI(config, clock)
        def sleep(seconds):
            clock.elapsed += seconds
            clock.now -= 100
            if clock.elapsed > 100:
                raise AssertionError("Clock adjustment extended deadline")
        guard.watch(config, api, clock=lambda: clock.now,
                    monotonic=lambda: clock.elapsed, sleep=sleep,
                    log=lambda *args, **kwargs: None)
        self.assertTrue(any(c[0] == "stop" for c in api.calls))


class SessionBudgetTests(unittest.TestCase):
    """Resuming preserves identity and cumulative budget without deleting disk."""

    run_watch = BudgetTests.run_watch

    def test_resume_has_new_session_clock_and_original_pod_identity(self):
        config = session_lease()
        self.assertEqual(config["schema_version"], 2)
        self.assertEqual(config["action"], "stop")
        self.assertEqual(config["created_at"], "2026-01-01T00:00:00+00:00")
        self.assertEqual(config["deadline_epoch"] - config["session_start_epoch"], 3480)
        self.assertAlmostEqual(config["projected_total_usd"], 15.64)
        self.assertEqual(guard.effective_deadline(config, 2.49), config["deadline_epoch"])

    def test_resume_never_accepts_delete_or_excess_duration_or_rate(self):
        config = session_lease()
        for changed in ({"action": "terminate"}, {"hours": 1.01},
                        {"maximum_hourly_rate_usd": 3.01},
                        {"hourly_rate_usd": 3.01},
                        {"deadline_epoch": config["deadline_epoch"] + 1},
                        {"session_start_epoch": guard.timestamp(config["created_at"]) - 1}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                guard.validate_lease({**config, **changed})
        for kwargs in ({"minutes": 61}, {"maximum_hourly_rate": 2},
                       {"maximum_hourly_rate": 4}, {"minutes": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                session_lease(**kwargs)

    def test_prior_spend_is_charged_against_total_budget(self):
        with self.assertRaises(ValueError):
            guard.build_session_lease(
                "pod-123", "cholescope-test", "2026-01-01T00:00:00Z", 2.49,
                session_start_epoch=guard.timestamp("2026-02-01T00:00:00Z"),
                spent=73, reserve=5)

    def test_prearmed_watcher_waits_for_start_then_enforces_saved_deadline(self):
        config = session_lease()
        clock = FakeClock(config["session_start_epoch"])

        class StartingAPI(FakeAPI):
            def get(self, pod_id):
                if clock.elapsed >= 30 and not self.confirm_lag and not any(
                        call[0] == "stop" for call in self.calls):
                    self.pod["status"] = "RUNNING"
                return super().get(pod_id)

        api = StartingAPI(config, clock, status="EXITED")
        self.run_watch(config, api, clock)
        actions = [c for c in api.calls if c[0] != "GET"]
        self.assertTrue(actions)
        self.assertEqual(actions[0], ("stop", "pod-123", config["deadline_epoch"]))
        self.assertEqual(api.pod["status"], "EXITED")

    def test_never_started_guard_remains_armed_until_deadline_without_mutation(self):
        config = session_lease()
        clock = FakeClock(config["session_start_epoch"])
        api = FakeAPI(config, clock, status="EXITED")
        self.run_watch(config, api, clock)
        self.assertEqual(clock.now, config["deadline_epoch"])
        self.assertEqual({c[0] for c in api.calls}, {"GET"})

    def test_stopped_after_running_confirms_early_user_pause(self):
        config = session_lease()
        clock = FakeClock(config["session_start_epoch"])

        class PausedAPI(FakeAPI):
            def get(self, pod_id):
                if clock.elapsed >= 15:
                    self.pod["status"] = "EXITED"
                return super().get(pod_id)

        api = PausedAPI(config, clock)
        self.run_watch(config, api, clock)
        self.assertEqual(clock.elapsed, 15)
        self.assertEqual({c[0] for c in api.calls}, {"GET"})

    def test_rate_over_session_limit_stops_before_budget_is_exhausted(self):
        config = session_lease()
        clock = FakeClock(config["session_start_epoch"])
        api = FakeAPI(config, clock, cost=3.01)
        self.run_watch(config, api, clock)
        self.assertEqual([c for c in api.calls if c[0] == "stop"][0][2],
                         config["session_start_epoch"])

    def test_bad_observed_rate_stops_instead_of_crashing_guard(self):
        config = session_lease()
        for rate in (None, "invalid", -1, float("nan")):
            with self.subTest(rate=rate):
                clock = FakeClock(config["session_start_epoch"])
                api = FakeAPI(config, clock, cost=rate)
                self.run_watch(config, api, clock)
                self.assertTrue(any(c[0] == "stop" for c in api.calls))

    def test_signal_disarms_pending_start_without_delete(self):
        config = session_lease()
        clock = FakeClock(config["session_start_epoch"])
        api = FakeAPI(config, clock, status="EXITED")
        self.run_watch(config, api, clock, stop_requested=lambda: True)
        self.assertEqual(clock.elapsed, 0)
        self.assertEqual([c[0] for c in api.calls], ["GET"])

    def test_restarting_guard_cannot_reset_saved_session_deadline(self):
        config = session_lease()
        clock = FakeClock(config["deadline_epoch"] + 1)
        api = FakeAPI(config, clock)
        self.run_watch(config, api, clock)
        self.assertEqual([c for c in api.calls if c[0] == "stop"][0][2],
                         config["deadline_epoch"] + 1)


class TransportTests(unittest.TestCase):
    def test_v2_stop_contract_and_token_stays_in_header(self):
        calls = []
        class Opener:
            def open(self, request, timeout):
                calls.append(request)
                return io.BytesIO(b"{}")
        api = guard.RunpodAPI("secret-token", opener=Opener())
        api.shutdown("pod-123", "stop")
        request = calls[0]
        self.assertEqual(request.full_url, "https://api.runpod.io/v2/pods/pod-123/action")
        self.assertEqual(request.method, "POST")
        self.assertEqual(json.loads(request.data), {"action": "stop"})
        self.assertNotIn("secret-token", request.full_url)
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-token")

    def test_http_errors_are_redacted(self):
        class Opener:
            def open(self, request, timeout):
                raise HTTPError(request.full_url, 403, "secret-token", {}, None)
        api = guard.RunpodAPI("secret-token", opener=Opener())
        with self.assertRaisesRegex(RuntimeError, "^Runpod HTTP 403$"):
            api.get("pod-123")

    def test_capacity_error_has_only_allowlisted_public_message(self):
        class Opener:
            def open(self, request, timeout):
                body = json.dumps({"detail": "There are not enough free GPUs on the host machine to start this pod.",
                                   "private": "secret-token"}).encode()
                raise HTTPError(request.full_url, 400, "secret-token", {}, io.BytesIO(body))
        api = guard.RunpodAPI("secret-token", opener=Opener())
        with self.assertRaises(guard.RunpodHTTPError) as raised:
            api.request("POST", "pod-123", "start")
        self.assertEqual(raised.exception.status, 400)
        self.assertEqual(str(raised.exception), "Runpod HTTP 400")
        self.assertEqual(raised.exception.public_message, "GPU unavailable on this host. Try again later.")
        for body in [b'{"detail":"secret-token"}', b'not json', b'x' * 17000,
                     b'{"detail":"There are not enough free GPUs on the host machine to start this pod. secret-token"}']:
            self.assertIsNone(guard.safe_http_message(400, body))
        self.assertIsNone(guard.RunpodHTTPError(400, "secret-token").public_message)

    def test_action_404_is_typed_error_but_get_404_means_absent(self):
        class Opener:
            def open(self, request, timeout):
                raise HTTPError(request.full_url, 404, "private", {}, io.BytesIO(b"private"))
        api = guard.RunpodAPI("secret-token", opener=Opener())
        self.assertIsNone(api.get("pod-123"))
        with self.assertRaises(guard.RunpodHTTPError) as raised:
            api.request("POST", "pod-123", "start")
        self.assertEqual(raised.exception.status, 404)


if __name__ == "__main__":
    unittest.main()
