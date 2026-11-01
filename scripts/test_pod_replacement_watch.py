"""Exercise the independent allocation guard with an offline provider model."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pod_replacement_watch as watcher
import runpod_budget as guard
from test_pod_replacement import created, recipe


START = guard.timestamp("2026-10-23T18:00:00Z")
NAME = "cholescope-allocate-0123456789abcdef0123456789abcdef"


def intent():
    lease = guard.build_session_lease(
        "old-pod", "old-demo", guard.utc(START - 3600), 3,
        session_start_epoch=START, minutes=60, spent=20,
    )
    return {"schema_version": 1, "recipe": recipe(), "session_lease": lease,
            "deadline_epoch": lease["deadline_epoch"], "created_after": START + 1,
            "name": NAME, "attempt_state": "dispatched", "cancel_requested": False}


def owned_pod(**changes):
    return {**created(), "name": NAME, "status": "RUNNING", **changes}


class EndObservation(Exception):
    """The simulated watchdog remained alive until the test's observation limit."""


class FakeProvider:
    def __init__(self, *pods):
        self.pods = {pod["id"]: copy.deepcopy(pod) for pod in pods}
        self.calls = []
        self.changed_identity = set()
        self.fail_list = False

    def list_pods(self):
        self.calls.append(("list",))
        if self.fail_list:
            raise RuntimeError("provider temporarily unavailable")
        return copy.deepcopy(list(self.pods.values()))

    def get(self, pod_id):
        self.calls.append(("get", pod_id))
        value = copy.deepcopy(self.pods.get(pod_id))
        if value and pod_id in self.changed_identity:
            value["name"] = "unrelated-pod"
        return value

    def shutdown(self, pod_id, action):
        self.calls.append((action, pod_id))
        if action != "stop":
            raise AssertionError("Only stop is allowed")
        self.pods[pod_id]["status"] = "EXITED"

    def create(self, *_args, **_kwargs):
        raise AssertionError("The independent guard must never create a pod")


class WatcherTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "intent.json"
        self.value = intent()
        self.path.write_text(json.dumps(self.value))
        self.wall = START + 6
        self.monotonic = 100.0
        self.logs = []
        self.sleeps = 0

    def tearDown(self):
        self.directory.cleanup()

    def update(self, **changes):
        self.value.update(changes)
        self.path.write_text(json.dumps(self.value))

    def run_watch(self, provider, *, max_polls=4, after_poll=None, stop_requested=lambda: False, log=None):
        def sleep(seconds):
            self.sleeps += 1
            self.wall += seconds
            self.monotonic += seconds
            if after_poll:
                after_poll(self.sleeps)
            if self.sleeps >= max_polls:
                raise EndObservation()

        with patch.object(watcher, "ReplacementAPI", lambda api: api):
            watcher.watch(self.path, provider, clock=lambda: self.wall,
                          monotonic=lambda: self.monotonic, sleep=sleep,
                          stop_requested=stop_requested,
                          log=log or (lambda message, **_kwargs: self.logs.append(message)))

    def result(self):
        return json.loads(self.path.with_suffix(".result.json").read_text())

    def test_empty_unknown_allocation_keeps_watching_after_deadline(self):
        self.wall = self.value["deadline_epoch"] + 1
        provider = FakeProvider()
        with self.assertRaises(EndObservation):
            self.run_watch(provider)
        self.assertEqual(provider.calls, [("list",)] * 4)
        self.assertEqual(self.result()["state"], "stopping")
        self.assertEqual(self.result()["identities"], [])
        self.assertFalse(any("shutdown_verified" in line for line in self.logs))

    def test_cancelled_attempt_stops_a_pod_that_appears_late(self):
        self.wall = self.value["deadline_epoch"] + 1
        self.update(cancel_requested=True)
        provider = FakeProvider()

        def after_poll(number):
            if number == 2:
                provider.pods["pod-late"] = owned_pod(id="pod-late", createdAt=guard.utc(self.wall))

        self.run_watch(provider, max_polls=5, after_poll=after_poll)
        self.assertEqual([call for call in provider.calls if call[0] == "stop"], [("stop", "pod-late")])
        self.assertEqual(self.result()["state"], "stopped")
        self.assertEqual(self.result()["deadline_epoch"], self.value["deadline_epoch"])

    def test_mismatched_owned_pod_is_stopped_before_deadline(self):
        provider = FakeProvider(owned_pod(image="unapproved-image"))
        self.run_watch(provider)
        self.assertIn(("stop", "pod-new"), provider.calls)
        self.assertTrue(self.result()["recipe_mismatch"])
        self.assertEqual(self.result()["state"], "stopped")

    def test_owned_duplicates_are_both_stopped_before_deadline(self):
        provider = FakeProvider(owned_pod(id="first"), owned_pod(id="second"))
        self.run_watch(provider)
        self.assertEqual([call for call in provider.calls if call[0] == "stop"],
                         [("stop", "first"), ("stop", "second")])
        self.assertTrue(self.result()["recipe_mismatch"])

    def test_unrelated_name_is_untouched_at_deadline(self):
        self.wall = self.value["deadline_epoch"]
        provider = FakeProvider(owned_pod(), owned_pod(id="unrelated", name="other-demo"))
        self.run_watch(provider)
        self.assertEqual([call for call in provider.calls if call[0] == "stop"], [("stop", "pod-new")])
        self.assertEqual(provider.pods["unrelated"]["status"], "RUNNING")

    def test_identity_change_cannot_block_stopping_another_owned_duplicate(self):
        provider = FakeProvider(owned_pod(id="first"), owned_pod(id="second"))
        provider.changed_identity.add("first")
        with self.assertRaises(EndObservation):
            self.run_watch(provider, max_polls=2)
        self.assertNotIn(("stop", "first"), provider.calls)
        self.assertIn(("stop", "second"), provider.calls)
        self.assertEqual(self.result()["state"], "stopping")

    def test_missing_intent_cannot_disable_armed_guard(self):
        provider = FakeProvider()

        def after_poll(number):
            if number == 1:
                self.path.unlink()
                provider.pods["pod-new"] = owned_pod()

        self.run_watch(provider, after_poll=after_poll)
        self.assertIn(("stop", "pod-new"), provider.calls)
        self.assertEqual(self.result()["state"], "stopped")

    def test_malformed_json_shape_latches_stop_instead_of_crashing(self):
        provider = FakeProvider()

        def after_poll(number):
            if number == 1:
                self.path.write_text("[]")
                provider.pods["pod-new"] = owned_pod()

        self.run_watch(provider, after_poll=after_poll)
        self.assertIn(("stop", "pod-new"), provider.calls)

    def test_receipt_write_failure_does_not_block_deadline_stop(self):
        self.wall = self.value["deadline_epoch"]
        provider = FakeProvider(owned_pod())
        with patch.object(watcher, "private_json", side_effect=OSError("disk is full")):
            self.run_watch(provider)
        self.assertIn(("stop", "pod-new"), provider.calls)
        self.assertTrue(any("shutdown_verified" in line for line in self.logs))

    def test_log_failure_after_handshake_cannot_block_deadline_stop(self):
        self.wall = self.value["deadline_epoch"]
        provider = FakeProvider(owned_pod())
        calls = []

        def failing_log(message, **_kwargs):
            calls.append(message)
            if len(calls) > 1:
                raise OSError("log disk is full")

        with patch.object(watcher, "private_json", side_effect=OSError("disk is full")):
            self.run_watch(provider, log=failing_log)
        self.assertIn(("stop", "pod-new"), provider.calls)

    def test_failed_initial_handshake_does_not_claim_guard_is_armed(self):
        provider = FakeProvider(owned_pod())
        with self.assertRaises(OSError):
            self.run_watch(provider, log=lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("closed")))
        self.assertEqual(provider.calls, [])

    def test_changed_recipe_cannot_redirect_guard_or_disable_shutdown(self):
        provider = FakeProvider()

        def after_poll(number):
            if number == 1:
                self.update(recipe={**recipe(), "volume_id": "unrelated-volume"})
                provider.pods["pod-new"] = owned_pod()

        self.run_watch(provider, after_poll=after_poll)
        self.assertIn(("stop", "pod-new"), provider.calls)
        self.assertFalse(self.result()["recipe_mismatch"])

    def test_inspection_error_retains_unknown_allocation(self):
        self.wall = self.value["deadline_epoch"] + 1
        provider = FakeProvider()
        provider.fail_list = True
        with self.assertRaises(EndObservation):
            self.run_watch(provider, max_polls=2)
        self.assertEqual(provider.calls, [("list",), ("list",)])
        self.assertFalse(any("shutdown_verified" in line for line in self.logs))

    def test_signal_cancellation_is_sticky_even_if_signal_callback_resets(self):
        provider = FakeProvider()
        signals = iter([True, False, False, False])

        def after_poll(number):
            if number == 1:
                provider.pods["pod-new"] = owned_pod()

        self.run_watch(provider, stop_requested=lambda: next(signals), after_poll=after_poll)
        self.assertIn(("stop", "pod-new"), provider.calls)

    def test_monotonic_deadline_survives_wall_clock_rollback(self):
        self.wall = self.value["deadline_epoch"] - 1
        provider = FakeProvider(owned_pod())

        def after_poll(number):
            if number == 1:
                self.wall = START

        self.run_watch(provider, after_poll=after_poll)
        self.assertIn(("stop", "pod-new"), provider.calls)

    def test_definite_rejection_with_empty_listing_can_retire(self):
        self.update(attempt_state="rejected")
        provider = FakeProvider()
        self.run_watch(provider)
        self.assertEqual(self.sleeps, 0)
        self.assertEqual(self.result()["state"], "stopped")

    def test_fresh_stopped_record_is_not_prematurely_considered_confirmed(self):
        provider = FakeProvider(owned_pod(status="EXITED"))
        with self.assertRaises(EndObservation):
            self.run_watch(provider, max_polls=2)
        self.assertEqual(self.result()["state"], "watching")

    def test_never_active_exited_pod_at_deadline_stays_pending_until_real_stop(self):
        self.wall = self.value["deadline_epoch"] + 1
        self.update(cancel_requested=True)
        provider = FakeProvider(owned_pod(status="EXITED"))

        def after_poll(number):
            if number in (1, 2):
                self.assertEqual(self.result()["state"], "stopping")
                self.assertNotIn(("stop", "pod-new"), provider.calls)
            if number == 2:
                provider.pods["pod-new"]["status"] = "RUNNING"

        self.run_watch(provider, max_polls=5, after_poll=after_poll)
        self.assertEqual([call for call in provider.calls if call[0] == "stop"], [("stop", "pod-new")])
        self.assertEqual(self.result()["state"], "stopped")

    def test_restart_restores_exact_active_evidence_from_own_receipt(self):
        provider = FakeProvider(owned_pod())
        with self.assertRaises(EndObservation):
            self.run_watch(provider, max_polls=1)
        provider.pods["pod-new"]["status"] = "EXITED"
        self.sleeps = 0
        self.run_watch(provider)
        self.assertEqual(self.result()["state"], "stopped")

    def test_intent_accepted_stop_evidence_requires_exact_identity(self):
        evidence = watcher.identity_of(owned_pod())
        self.update(accepted_stop_identities=[evidence])
        provider = FakeProvider(owned_pod(status="EXITED"))
        self.run_watch(provider)
        self.assertEqual(self.result()["state"], "stopped")

    def test_wrong_identity_evidence_and_broad_boolean_do_not_confirm_exit(self):
        evidence = watcher.identity_of(owned_pod(id="other-pod"))
        self.update(observed_active=True, observed_active_identities=[evidence], accepted_stop_identities=[evidence])
        self.wall = self.value["deadline_epoch"]
        provider = FakeProvider(owned_pod(status="EXITED"))
        with self.assertRaises(EndObservation):
            self.run_watch(provider, max_polls=2)
        self.assertEqual(self.result()["state"], "stopping")


class ReplacementLeaseTests(unittest.TestCase):
    def test_replacement_retains_original_deadline_and_budget(self):
        original = intent()["session_lease"]
        untouched = copy.deepcopy(original)
        replacement = watcher.replacement_lease(original, owned_pod(createdAt=guard.utc(START + 300)))
        self.assertEqual(original, untouched)
        self.assertEqual(replacement["deadline_epoch"], original["deadline_epoch"])
        self.assertEqual(replacement["hours"], original["hours"])
        self.assertEqual(replacement["previous_spend_usd"], original["previous_spend_usd"])
        self.assertEqual(replacement["compute_ceiling_usd"], original["compute_ceiling_usd"])
        self.assertEqual(replacement["action"], "stop")
        self.assertEqual(replacement["pod_id"], "pod-new")
        self.assertEqual(replacement["session_start_epoch"], START + 300)
        self.assertLessEqual(replacement["deadline_epoch"], original["session_start_epoch"] + 3600)

    def test_changed_intent_deadline_is_rejected(self):
        value = intent()
        value["deadline_epoch"] += 100
        with self.assertRaisesRegex(ValueError, "deadline"):
            watcher.validate_intent(value)


if __name__ == "__main__":
    unittest.main()
