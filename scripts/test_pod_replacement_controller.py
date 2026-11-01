"""Offline full lifecycle tests for the optional one-attempt host replacement."""
import asyncio
import base64
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import pod_manager as manager
import pod_replacement as replacement
import pod_replacement_controller as controller
import runpod_budget as guard

NOW = guard.timestamp('2026-10-23T15:00:00Z')
KEY = 'ssh-ed25519 ' + base64.b64encode(b'\0\0\0\x0bssh-ed25519\0\0\0\x20' + b'x' * 32).decode()
RECIPE = {'gpu_id': replacement.GPU_ID, 'image': replacement.IMAGE, 'data_center_id': 'EU-RO-1',
          'volume_id': 'test-volume', 'public_key': KEY, 'max_hourly_rate': 2.49}


def new_pod(name):
    return {'id': 'new-pod', 'name': name, 'createdAt': guard.utc(NOW + 5), 'status': 'RUNNING',
            'cost': 2.49, 'locked': False, 'actions': ['stop', 'terminate'],
            'gpu': {'id': RECIPE['gpu_id'], 'count': 1, 'memory': 188}, 'cloud': 'SECURE',
            'dataCenterId': 'EU-RO-1', 'image': RECIPE['image'], 'disk': 30,
            'mounts': {'network': [{'volumeId': 'test-volume', 'path': '/workspace'}]},
            'ports': ['22/tcp'], 'env': {'PUBLIC_KEY': KEY}, 'globalNetworking': {'enabled': False},
            'cudaVersion': '13.2', 'args': ''}


class FakeAPI:
    def __init__(self):
        self.pods = {'old-pod': {'id': 'old-pod', 'name': 'old-demo',
             'createdAt': '2026-10-23T09:00:00Z', 'status': 'EXITED', 'cost': 2.49,
             'locked': False, 'actions': ['start', 'terminate'], 'dataCenterId': 'EU-RO-1',
             'mounts': {'network': [{'volumeId': 'test-volume', 'path': '/workspace'}]}}}
        self.actions = []
        self.wrong_identity = False

    def get(self, pod_id):
        value = copy.deepcopy(self.pods.get(pod_id))
        if self.wrong_identity and pod_id == 'new-pod' and value:
            value['name'] = 'unrelated-account-pod'
        return value

    def request(self, method, pod_id, action):
        self.actions.append((method, pod_id, action))
        if action == 'start':
            raise guard.RunpodHTTPError(400, guard.GPU_UNAVAILABLE_MESSAGE)
        if action == 'stop':
            self.pods[pod_id]['status'] = 'EXITED'
            self.pods[pod_id]['actions'] = ['start', 'terminate']
            return self.get(pod_id)
        raise AssertionError('Forbidden mutation')

    def shutdown(self, pod_id, action):
        assert action == 'stop'
        return self.request('POST', pod_id, action)


class FakeTransport:
    def __init__(self, api, events):
        self.api, self.events = api, events
        self.creates = 0
        self.mode = 'success'
        self.entered = threading.Event()
        self.release = threading.Event()
        self.name = None

    def preflight(self, recipe):
        if self.mode == 'no_capacity':
            raise replacement.ReplacementPreflightError('No capacity in the approved region.')
        return {'quoted_rate': 2.49}

    def create(self, recipe, name):
        assert self.events[-1] == 'allocation-guard'
        self.creates += 1
        self.name = name
        self.entered.set()
        if self.mode == 'block':
            assert self.release.wait(2)
        if self.mode == 'rejected':
            raise guard.RunpodHTTPError(400, guard.GPU_UNAVAILABLE_MESSAGE)
        if self.mode == 'unknown_absent':
            raise RuntimeError('connection lost')
        pod = new_pod(name)
        if self.mode == 'fresh_exited':
            pod['status'] = 'EXITED'
        if self.mode == 'wrong_volume':
            pod['mounts']['network'][0]['volumeId'] = 'wrong-volume'
        self.api.pods[pod['id']] = pod
        if self.mode == 'lost_response':
            raise RuntimeError('connection lost')
        return copy.deepcopy(pod)

    def list_pods(self):
        return copy.deepcopy(list(self.api.pods.values()))


class OfflineManager(manager.PodManager):
    async def _arm_local(self, lease_path, *, identity=None):
        self.events.append('pod-guard')
        self.guarded.append(json.loads(Path(lease_path).read_text()))

    async def _attach_worker(self):
        self._state, self._message = 'ready', 'Demo ready.'
        self.attached += 1

    async def _flush_app(self):
        pass


class ReplacementLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.api = FakeAPI()
        self.now = NOW
        for name in ('api-key', 'ssh-key'):
            p = self.root / name; p.write_text('dummy-secret'); p.chmod(0o600)
        config = {'pod_id': 'old-pod', 'pod_name': 'old-demo', 'created_at': '2026-10-23T09:00:00Z',
                  'key_file': str(self.root / 'api-key'), 'ssh_key': str(self.root / 'ssh-key'),
                  'known_hosts': str(self.root / 'known_hosts'), 'state_dir': str(self.root / 'state'),
                  'initial_spend_usd': 8.2, 'initial_spend_epoch': NOW - 3600,
                  'allow_replacement': True, 'replacement': RECIPE}
        self.config_path = self.root / 'config.json'
        self.config_path.write_text(json.dumps(config))
        self.m = self.make_manager()

    def make_manager(self):
        m = OfflineManager(self.config_path, api=self.api, clock=lambda: self.now)
        m.events, m.guarded, m.attached = [], [], 0
        m.replacements.transport = FakeTransport(self.api, m.events)
        class Alive:
            def poll(self): return None
        async def arm(path):
            m.events.append('allocation-guard')
            m.replacements.processes.append(Alive())
        m.replacements._arm = arm
        return m

    async def asyncTearDown(self):
        await self.m.close()
        self.temp.cleanup()

    async def start(self):
        result = await self.m.start()
        await self.m._start_task
        return result

    async def test_one_replacement_reuses_budget_deadline_and_changes_binding(self):
        await self.start()
        self.assertEqual(self.m.replacements.transport.creates, 1)
        self.assertEqual(self.api.actions, [('POST', 'old-pod', 'start')])
        self.assertEqual(self.api.pods['old-pod']['status'], 'EXITED')
        self.assertEqual(self.m.identity['pod_id'], 'new-pod')
        self.assertEqual(self.m._state, 'ready')
        self.assertAlmostEqual(self.m.ledger['spent_usd'], 11.25)
        self.assertEqual(self.m.guarded[0]['deadline_epoch'], self.m.guarded[1]['deadline_epoch'])
        self.assertEqual(self.m._lease['deadline_epoch'], NOW + 3480)
        self.assertEqual(len(self.m.ledger['previous_pods']), 1)
        self.assertEqual(json.loads(self.config_path.read_text())['pod_id'], 'new-pod')

    async def test_known_rejection_refunds_only_the_unused_session(self):
        self.m.replacements.transport.mode = 'rejected'
        await self.start()
        self.assertEqual(self.m._state, 'stopped')
        self.assertAlmostEqual(self.m.ledger['spent_usd'], 8.25)
        self.assertTrue(self.m.ledger['reservation_refunded'])
        self.assertEqual(self.m.replacements.transport.creates, 1)
        self.now += 9
        await self.m.status()
        self.assertAlmostEqual(self.m.ledger['spent_usd'], 8.25 + 9 * .05 / 3600)

    async def test_catalog_failure_never_creates_and_refunds_unused_reservation(self):
        self.m.replacements.transport.mode = 'no_capacity'
        await self.start()
        self.assertEqual(self.m.replacements.transport.creates, 0)
        self.assertAlmostEqual(self.m.ledger['spent_usd'], 8.25)
        self.assertEqual(self.m._state, 'stopped')

    async def test_actual_rejected_preflight_saves_private_bounded_catalog_history(self):
        catalog = {'id': replacement.GPU_ID, 'manufacturer': 'NVIDIA', 'secure': True,
                   'memory': 96, 'price': {'secure': 2.49, 'community': 1.69, 'secret': KEY},
                   'availability': 'LOW', 'dataCenters': [
                       {'id': 'EU-OTHER-1', 'availability': 'LOW', 'env': {'PUBLIC_KEY': KEY}}],
                   'key': 'dummy-secret', 'headers': {'Authorization': 'dummy-secret'}}
        transport = replacement.ReplacementAPI(self.api,
            preflight_diagnostic=self.m.replacements._record_preflight)
        self.m.replacements.transport = transport
        with patch.object(transport, 'get_gpu', return_value=catalog), \
                patch.object(transport, 'get_volume', return_value={}), \
                patch.object(transport, 'create') as create:
            await self.start()
            create.assert_not_called()
            path = self.m.state_dir / 'replacement-preflight-diagnostics.json'
            first = json.loads(path.read_text())['observations'][0]
            self.assertEqual(first['recipe_data_center_id'], 'EU-RO-1')
            self.assertEqual(first['observed_at_epoch'], NOW)
            self.assertTrue(first['catalog_received'])
            self.assertEqual(first['catalog']['dataCenters'],
                             [{'id': 'EU-OTHER-1', 'availability': 'LOW'}])
            self.assertEqual(first['catalog']['price'], {'secure': 2.49, 'community': 1.69})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(self.m._state, 'stopped')
            self.assertAlmostEqual(self.m.ledger['spent_usd'], 8.25)
            catalog['dataCenters'] *= 80
            catalog['dataCenters'][0]['id'] = 'x' * 1000
            for index in range(10):
                self.now += 1
                with self.assertRaises(replacement.ReplacementPreflightError):
                    transport.preflight(RECIPE)
            history = json.loads(path.read_text())['observations']
            self.assertEqual(len(history), 8)
            self.assertEqual(history[0]['observed_at_epoch'], NOW + 3)
            latest = history[-1]['catalog']
            self.assertEqual(len(latest['dataCenters']), 64)
            self.assertEqual(len(latest['dataCenters'][0]['id']), 128)
            self.assertEqual(latest['dataCenters_count'], 80)
            self.assertTrue(latest['dataCenters_truncated'])
            self.assertLess(path.stat().st_size, 512 * 1024)
            self.assertNotIn(KEY, path.read_text())
            self.assertNotIn('dummy-secret', path.read_text())

    async def test_preflight_diagnostic_disk_failure_does_not_replace_capacity_error(self):
        transport = replacement.ReplacementAPI(self.api,
            preflight_diagnostic=self.m.replacements._record_preflight)
        catalog = {'id': replacement.GPU_ID, 'manufacturer': 'NVIDIA', 'secure': True,
                   'memory': 96, 'price': {'secure': 2.49}, 'availability': 'NONE',
                   'dataCenters': []}
        with patch.object(transport, 'get_gpu', return_value=catalog), \
                patch.object(transport, 'get_volume', return_value={}), \
                patch.object(controller, 'private_json', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(replacement.ReplacementPreflightError,
                                        '^No GPU is available right now. Try again shortly.$'):
                transport.preflight(RECIPE)

    async def test_lost_create_response_reconciles_without_second_post(self):
        self.m.replacements.transport.mode = 'lost_response'
        await self.start()
        transport = self.m.replacements.transport
        self.assertTrue(self.m.replacements.pending())
        self.assertFalse(self.m._public()['can_start'])
        self.assertTrue(self.m._public()['can_stop'])
        await self.m.replacements.recover_worker()
        self.assertEqual(transport.creates, 1)
        self.assertEqual(self.m.identity['pod_id'], 'new-pod')
        self.assertEqual(self.m._state, 'ready')
        self.assertAlmostEqual(self.m.ledger['spent_usd'], 11.25)

    async def test_stop_of_unknown_create_stays_pending_and_catches_late_pod(self):
        self.m.replacements.transport.mode = 'unknown_absent'
        await self.start()
        transport = self.m.replacements.transport
        await self.m.stop()
        await self.m._stop_task
        self.assertTrue(self.m.replacements.pending())
        self.assertEqual(self.m._state, 'error')
        self.assertFalse(self.m._public()['can_start'])
        pod = new_pod(transport.name)
        self.api.pods[pod['id']] = pod
        await self.m.replacements.recover_worker()
        await self.m.replacements.recover_worker()
        self.assertEqual(self.api.pods['new-pod']['status'], 'EXITED')
        self.assertEqual(self.m._state, 'stopped')
        self.assertEqual(transport.creates, 1)
        self.assertAlmostEqual(self.m.ledger['spent_usd'], 11.25)

    async def test_stop_during_create_cannot_be_erased_by_response(self):
        transport = self.m.replacements.transport
        transport.mode = 'block'
        await self.m.start()
        for _ in range(100):
            if transport.entered.is_set(): break
            await asyncio.sleep(.005)
        self.assertTrue(transport.entered.is_set())
        await self.m.stop()
        self.assertTrue(self.m.replacements.intent()['cancel_requested'])
        transport.release.set()
        await self.m._stop_task
        await self.m.replacements.recover_worker()
        self.assertTrue(self.m.replacements.intent()['cancel_requested'])
        self.assertEqual(self.api.pods['new-pod']['status'], 'EXITED')
        self.assertEqual(transport.creates, 1)
        self.assertEqual(self.m.attached, 0)

    async def test_changed_returned_identity_never_authorizes_stop_of_wrong_pod(self):
        self.api.wrong_identity = True
        await self.start()
        self.assertTrue(self.m.replacements.pending())
        self.assertFalse(any(c[1] == 'new-pod' and c[2] == 'stop' for c in self.api.actions))
        self.assertEqual(self.m.identity['pod_id'], 'old-pod')

    async def test_owned_recipe_mismatch_is_stopped_without_adoption(self):
        self.m.replacements.transport.mode = 'wrong_volume'
        await self.start()
        await self.m.replacements.recover_worker()
        self.assertEqual(self.api.pods['new-pod']['status'], 'EXITED')
        self.assertEqual(self.m.identity['pod_id'], 'old-pod')
        self.assertEqual(self.m._state, 'stopped')

    async def test_unknown_allocation_restart_never_posts_again(self):
        self.m.replacements.transport.mode = 'unknown_absent'
        await self.start()
        deadline = self.m.replacements.intent()['deadline_epoch']
        await self.m.close()
        self.m = self.make_manager()
        await self.m.status()
        await self.m._start_task
        self.assertTrue(self.m.replacements.pending())
        self.assertEqual(self.m.replacements.transport.creates, 0)
        self.assertEqual(self.m.replacements.intent()['deadline_epoch'], deadline)
        self.assertAlmostEqual(self.m.ledger['spent_usd'], 11.25)
        self.assertFalse(self.m._public()['can_start'])

    async def test_never_active_exited_allocation_remains_pending_until_activity(self):
        self.m.replacements.transport.mode = 'fresh_exited'
        await self.start()
        await self.m.stop()
        await self.m._stop_task
        self.assertTrue(self.m.replacements.pending())
        self.assertNotEqual(self.m._state, 'stopped')
        self.api.pods['new-pod']['status'] = 'RUNNING'
        await self.m.replacements.recover_worker()
        await self.m.replacements.recover_worker()
        self.assertFalse(self.m.replacements.pending())
        self.assertEqual(self.m._state, 'stopped')
        self.assertEqual(self.api.pods['new-pod']['status'], 'EXITED')

    async def test_matching_watcher_receipt_resolves_stopped_late_allocation(self):
        self.m.replacements.transport.mode = 'unknown_absent'
        await self.start()
        await self.m.stop()
        await self.m._stop_task
        intent = self.m.replacements.intent()
        pod = new_pod(intent['name'])
        pod['status'] = 'EXITED'
        self.api.pods[pod['id']] = pod
        await self.m.replacements.recover_worker()
        self.assertTrue(self.m.replacements.pending())
        identity = {'pod_id': pod['id'], 'pod_name': pod['name'], 'created_at': pod['createdAt']}
        receipt = {'name': intent['name'], 'created_after': intent['created_after'],
                   'deadline_epoch': intent['deadline_epoch'] + 1,
                   'observed_active_identities': [identity]}
        result = Path(self.m.ledger['allocation_intent']).with_suffix('.result.json')
        controller.private_json(result, receipt)
        await self.m.replacements.recover_worker()
        self.assertTrue(self.m.replacements.pending())
        receipt['deadline_epoch'] = intent['deadline_epoch']
        controller.private_json(result, receipt)
        await self.m.replacements.recover_worker()
        self.assertFalse(self.m.replacements.pending())
        self.assertEqual(self.m._state, 'stopped')
        self.assertFalse(any(action[1:] == ('new-pod', 'stop') for action in self.api.actions))

    async def test_adoption_replays_crash_between_config_and_ledger(self):
        original_write = controller.private_json
        failed = [False]
        def crash_once(path, value):
            if Path(path) == self.m.ledger_path and value.get('identity', {}).get('pod_id') == 'new-pod' and not failed[0]:
                failed[0] = True
                raise asyncio.CancelledError()
            original_write(path, value)
        with patch.object(controller, 'private_json', crash_once):
            await self.start()
        self.assertTrue(failed[0])
        self.assertEqual(json.loads(self.config_path.read_text())['pod_id'], 'new-pod')
        self.assertEqual(json.loads(self.m.ledger_path.read_text())['identity']['pod_id'], 'old-pod')
        await self.m.close()
        self.m = self.make_manager()
        self.assertEqual(self.m.identity['pod_id'], 'new-pod')
        self.assertEqual(self.m.ledger['identity']['pod_id'], 'new-pod')
        self.assertFalse(self.m.replacements.pending())
        self.assertEqual(len(self.m.ledger['previous_pods']), 1)
        self.assertAlmostEqual(self.m.ledger['spent_usd'], 11.25)
        self.assertAlmostEqual(self.m.storage_rate, .05)
        await self.m.status()
        await self.m._start_task
        self.assertEqual(self.m._state, 'ready')
        self.assertEqual(self.m._lease['deadline_epoch'], NOW + 3480)
        self.assertEqual(self.m.replacements.transport.creates, 0)


if __name__ == '__main__':
    unittest.main()
