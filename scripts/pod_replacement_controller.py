"""Optional one-attempt replacement lifecycle, using the already prepaid session."""
from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import runpod_budget as guard
from pod_replacement import ReplacementAPI, ReplacementError, ownership_match, validate_recipe, verify_created
from pod_replacement_watch import identity_of, private_json, replacement_lease, validate_intent


class AllocationPending(RuntimeError):
    """A create may have succeeded; the independent watcher still owns cleanup."""


def finish_adoption(config_path):
    """Replay a private write-ahead identity change after a partial file update."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    state = Path(config['state_dir']).expanduser().resolve()
    transaction = state / 'identity-adoption.json'
    if not transaction.exists():
        return
    journal = json.loads(transaction.read_text())
    if journal.get('phase') != 'prepared':
        return
    new_config, new_ledger = journal['config'], journal['ledger']
    current = {key: config[key] for key in ('pod_id', 'pod_name', 'created_at')}
    new_identity = {key: new_config[key] for key in ('pod_id', 'pod_name', 'created_at')}
    if (current not in (journal['previous_identity'], new_identity)
            or new_ledger['identity'] != new_identity
            or Path(new_config['state_dir']).expanduser().resolve() != state):
        raise ValueError('Interrupted identity adoption does not match this controller')
    lease_path = Path(new_ledger['lease_file'])
    if lease_path.parent.resolve() != state:
        raise ValueError('Interrupted adoption has an invalid lease path')
    lease = guard.validate_lease(json.loads(lease_path.read_text()))
    if any(lease[key] != new_identity[key] for key in new_identity):
        raise ValueError('Interrupted adoption changed its guarded identity')
    old_ledger = json.loads((state / 'budget-ledger.json').read_text())
    if (old_ledger['identity'] not in (journal['previous_identity'], new_identity)
            or old_ledger['spent_usd'] != new_ledger['spent_usd']):
        raise ValueError('Interrupted adoption changed the prepaid budget')
    intent_path = Path(journal['intent_path'])
    adopted_intent = validate_intent(journal['intent'])
    if (intent_path.parent.resolve() != state or adopted_intent['attempt_state'] != 'adopted'
            or adopted_intent.get('returned_identity') != new_identity
            or adopted_intent['deadline_epoch'] != lease['deadline_epoch']):
        raise ValueError('Interrupted adoption changed its allocation intent')
    private_json(config_path, new_config)
    private_json(state / 'budget-ledger.json', new_ledger)
    private_json(intent_path, adopted_intent)
    journal['phase'] = 'committed'
    private_json(transaction, journal)


class ReplacementController:
    def __init__(self, manager):
        self.m = manager
        enabled = manager.config.get('allow_replacement', False)
        if not isinstance(enabled, bool):
            raise ValueError('allow_replacement must be an explicit boolean')
        self.enabled = enabled
        self.recipe = validate_recipe(manager.config.get('replacement')) if enabled else None
        if enabled and self.recipe['max_hourly_rate'] > manager.max_rate:
            raise ValueError('Replacement price ceiling exceeds the session ceiling')
        self.transport = ReplacementAPI(manager.api, preflight_diagnostic=self._record_preflight)
        self.processes = []

    def _record_preflight(self, observation):
        """Keep a bounded private history of the catalog used by this process."""
        path = self.m.state_dir / 'replacement-preflight-diagnostics.json'
        records = []
        if path.exists() and path.stat().st_size <= 512 * 1024:
            try:
                saved = json.loads(path.read_text())
                if isinstance(saved, dict) and isinstance(saved.get('observations'), list):
                    records = saved['observations'][-7:]
            except (OSError, ValueError):
                pass
        observed = self.m.clock()
        records.append({**observation, 'observed_at_epoch': observed,
                        'observed_at': guard.utc(observed), 'process_id': os.getpid()})
        private_json(path, {'schema_version': 1, 'observations': records})

    def intent(self):
        name = self.m.ledger.get('allocation_intent')
        if not name:
            return None
        path = Path(name)
        if path.parent.resolve() != self.m.state_dir:
            raise ValueError('Invalid private allocation intent path')
        return validate_intent(json.loads(path.read_text()))

    def pending(self):
        intent = self.intent()
        return bool(intent and not intent.get('resolved') and intent['attempt_state'] not in {'adopted', 'rejected'})

    def _save_intent(self, intent):
        # A Stop can arrive while create/SSH is awaited. Never let an older
        # response overwrite the cancellation already saved by that request.
        current = self.intent()
        intent['cancel_requested'] = bool(intent.get('cancel_requested') or
            (current and current.get('cancel_requested')) or self.m.ledger.get('stop_requested'))
        for field in ('observed_active_identities', 'accepted_stop_identities'):
            combined = list(current.get(field, [])) if current else []
            for identity in intent.get(field, []):
                if identity not in combined:
                    combined.append(identity)
            if combined:
                intent[field] = combined
        private_json(self.m.ledger['allocation_intent'], intent)
        return intent

    def cancel(self):
        intent = self.intent()
        if intent and not intent.get('resolved'):
            intent['cancel_requested'] = True
            self._save_intent(intent)

    async def _arm(self, path):
        m = self.m
        log_path = path.with_suffix('.log')
        offset = log_path.stat().st_size if log_path.exists() else 0
        with open(log_path, 'ab', buffering=0) as stream:
            os.chmod(log_path, 0o600)
            process = subprocess.Popen([
                sys.executable, str(Path(__file__).with_name('pod_replacement_watch.py')),
                '--intent', str(path), '--api-key-file', str(m.key_file)],
                stdin=subprocess.DEVNULL, stdout=stream, stderr=stream,
                start_new_session=True, close_fds=True)
        self.processes.append(process)
        m.ledger.setdefault('allocation_guard_pids', []).append(process.pid)
        m._save()
        if sys.platform == 'darwin':
            process_sleep = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(process.pid)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True, close_fds=True)
            m._sleep_inhibitors.append(process_sleep)
        for _ in range(50):
            if process.poll() is not None:
                raise ReplacementError('The allocation budget guard could not start.')
            if b'"event": "allocation_guard_armed"' in log_path.read_bytes()[offset:]:
                return
            await asyncio.sleep(.1)
        raise ReplacementError('The allocation budget guard did not confirm startup.')

    async def replace(self):
        """Called once, only after the fixed-host start was definitively rejected."""
        m = self.m
        if not self.enabled or self.pending():
            raise ReplacementError('Replacement allocation is not available.')
        original = copy.deepcopy(m._lease)
        if original['deadline_epoch'] - m.clock() < 180:
            raise ReplacementError('Too little session time remains for another host.')
        old = await m._get_pod()
        if old['status'] != 'EXITED':
            raise ReplacementError('The previous pod must be confirmed paused before replacement.')
        # The volume and region are inherited from the stopped, verified pod.
        required_mounts = {'network': [{'volumeId': self.recipe['volume_id'], 'path': '/workspace'}]}
        if old.get('mounts') != required_mounts or old.get('dataCenterId') != self.recipe['data_center_id']:
            raise ReplacementError('Replacement must reuse the existing pod’s exact network volume and region.')
        await asyncio.to_thread(self.transport.preflight, self.recipe)
        m._check_start()
        token = uuid.uuid4().hex
        intent = {
            'schema_version': 1, 'name': 'cholescope-allocate-' + token,
            'recipe': self.recipe, 'created_after': m.clock(),
            'previous_identity': copy.deepcopy(m.identity), 'session_lease': original,
            'deadline_epoch': original['deadline_epoch'], 'attempt_state': 'prepared',
            'cancel_requested': False,
        }
        path = m.state_dir / ('allocation-' + token + '.json')
        private_json(path, validate_intent(intent))
        m.ledger.update(allocation_intent=str(path), phase='allocating')
        m._save()
        await self._arm(path)
        m._check_start()
        # Record the dispatch before one POST. Recovery never executes a create.
        intent['attempt_state'] = 'dispatched'
        self._save_intent(intent)
        m.ledger['refund_eligible'] = False
        m._save()
        m._state, m._message = 'starting', 'Finding an available GPU host…'
        try:
            pod = await asyncio.to_thread(self.transport.create, self.recipe, intent['name'])
        except guard.RunpodHTTPError as exc:
            if 400 <= exc.status < 500 and exc.status != 408:
                intent['attempt_state'] = 'rejected'
                self._save_intent(intent)
                m.ledger.update(phase='armed', refund_eligible=True)
                m._save()
                raise ReplacementError(exc.public_message or f'RunPod rejected the replacement (HTTP {exc.status}).') from None
            raise AllocationPending('Replacement request is unconfirmed. Its budget guard remains active.') from None
        except Exception:
            raise AllocationPending('Replacement request is unconfirmed. Its budget guard remains active.') from None
        intent['attempt_state'] = 'accepted'
        if ownership_match(pod, intent['name'], intent['created_after']):
            intent['returned_identity'] = identity_of(pod)
            if pod.get('status') in {'PROVISIONING', 'STARTING', 'RUNNING'}:
                intent['observed_active_identities'] = [identity_of(pod)]
        self._save_intent(intent)
        # Re-read the immutable ID from the API before binding or mutating it.
        if 'returned_identity' in intent:
            current = await asyncio.to_thread(m.api.get, intent['returned_identity']['pod_id'])
            if current is not None:
                try:
                    guard.verify_pod(current, intent['returned_identity'])
                    if not ownership_match(current, intent['name'], intent['created_after']):
                        raise ValueError('Replacement identity changed')
                except (ValueError, KeyError, TypeError):
                    self.cancel()
                    raise AllocationPending('Replacement identity is unconfirmed; its guard remains active.') from None
                return await self._consider(intent, [current])
        return await self.reconcile()

    async def _matches(self, intent):
        pods = await asyncio.to_thread(self.transport.list_pods)
        return [pod for pod in pods if ownership_match(pod, intent['name'], intent['created_after'])]

    async def reconcile(self):
        m = self.m
        intent = self.intent()
        if not intent or not self.pending():
            return None
        path = Path(m.ledger['allocation_intent'])
        if intent['attempt_state'] == 'prepared':
            # A crash before durable dispatch cannot have sent a create request.
            intent.update(attempt_state='rejected', resolved=True)
            self._save_intent(intent)
            m.ledger.update(phase='armed', refund_eligible=True)
            m._save()
            await m._mark_stopped()
            return None
        if not any(process.poll() is None for process in self.processes):
            await self._arm(path)
        matches = await self._matches(intent)
        if not matches:
            raise AllocationPending('Replacement request is unconfirmed. Its budget guard remains active.')
        return await self._consider(intent, matches)

    def _merge_watcher_evidence(self, intent):
        path = Path(self.m.ledger['allocation_intent']).with_suffix('.result.json')
        try:
            receipt = json.loads(path.read_text())
        except (OSError, ValueError):
            return
        if not isinstance(receipt, dict):
            return
        if (receipt.get('name') != intent['name'] or receipt.get('created_after') != intent['created_after']
                or receipt.get('deadline_epoch') != intent['deadline_epoch']):
            return
        for field in ('observed_active_identities', 'accepted_stop_identities'):
            entries = receipt.get(field)
            if not isinstance(entries, list):
                continue
            target = intent.setdefault(field, [])
            for identity in entries:
                if not isinstance(identity, dict):
                    continue
                pod = {'id': identity.get('pod_id'), 'name': identity.get('pod_name'),
                       'createdAt': identity.get('created_at')}
                if ownership_match(pod, intent['name'], intent['created_after']) and identity not in target:
                    target.append(identity)
        self._save_intent(intent)

    async def _consider(self, intent, matches):
        m = self.m
        self._merge_watcher_evidence(intent)
        if any(not ownership_match(pod, intent['name'], intent['created_after']) for pod in matches):
            self.cancel()
            raise AllocationPending('Replacement identity is unconfirmed; its guard remains active.')
        mismatch = len(matches) != 1
        if not mismatch:
            try:
                verify_created(matches[0], intent['recipe'], intent['name'], intent['created_after'])
            except ReplacementError:
                mismatch = True
        cancelled = (intent.get('cancel_requested') or m.ledger.get('stop_requested')
                     or m.clock() >= intent['deadline_epoch'] or mismatch)
        if cancelled:
            intent['cancel_requested'] = True
            active = intent.setdefault('observed_active_identities', [])
            for pod in matches:
                if pod.get('status') in {'PROVISIONING', 'STARTING', 'RUNNING'} and identity_of(pod) not in active:
                    active.append(identity_of(pod))
            self._save_intent(intent)
            all_stopped = True
            for pod in matches:
                current = await asyncio.to_thread(m.api.get, pod['id'])
                if current is None:
                    all_stopped = False
                    continue
                guard.verify_pod(current, identity_of(pod))
                if not ownership_match(current, intent['name'], intent['created_after']):
                    raise AllocationPending('Replacement identity is unconfirmed; its guard remains active.')
                proof = (identity_of(current) in intent.get('observed_active_identities', [])
                         or identity_of(current) in intent.get('accepted_stop_identities', []))
                if current.get('status') not in {'EXITED', 'TERMINATED'}:
                    all_stopped = False
                    await asyncio.to_thread(m.api.shutdown, current['id'], 'stop')
                    accepted = intent.setdefault('accepted_stop_identities', [])
                    if identity_of(current) not in accepted:
                        accepted.append(identity_of(current))
                    self._save_intent(intent)
                elif not proof:
                    all_stopped = False
            if not all_stopped:
                raise AllocationPending('Pausing the replacement allocation. Waiting for confirmation.')
            intent['resolved'] = True
            self._save_intent(intent)
            m.ledger.setdefault('previous_allocations', []).extend(identity_of(pod) for pod in matches)
            m.ledger['phase'] = 'armed'
            m._save()
            old = await m._get_pod()
            if old['status'] != 'EXITED':
                raise ReplacementError('The previous pod is no longer paused.')
            await m._mark_stopped()
            return None
        pod = matches[0]
        if pod.get('status') not in {'PROVISIONING', 'STARTING', 'RUNNING'}:
            self.cancel()
            raise AllocationPending('Replacement has not started. Its budget guard remains active.')
        return await self._adopt(intent, pod)

    async def _adopt(self, intent, pod):
        m = self.m
        identity = identity_of(pod)
        lease = replacement_lease(intent['session_lease'], pod)
        if m.identity == identity:
            # Idempotent completion if a prior process committed identity but
            # exited before its in-memory state/intent update completed.
            saved = guard.validate_lease(json.loads(Path(m.ledger['lease_file']).read_text()))
            if saved['deadline_epoch'] != intent['deadline_epoch']:
                raise ReplacementError('Adopted allocation changed its deadline.')
            guard.verify_pod(pod, saved)
            await m._arm_local(Path(m.ledger['lease_file']), identity=identity)
            intent.update(attempt_state='adopted', returned_identity=identity)
            self._save_intent(intent)
            m._lease, m._pod = saved, pod
            return pod
        path = m.state_dir / ('session-replacement-' + pod['id'] + '.json')
        private_json(path, lease)
        # The original name watcher is already enforcing the deadline. Add the
        # exact-ID guard before changing which pod the local UI controls.
        await m._arm_local(path, identity=identity)
        if m._cancel_start.is_set() or m.ledger.get('stop_requested'):
            self.cancel()
            raise AllocationPending('Pausing the replacement allocation. Waiting for confirmation.')
        config = {**m.config, **identity}
        ledger = copy.deepcopy(m.ledger)
        ledger.setdefault('previous_pods', []).append({**m.identity, 'status': 'EXITED', 'at_epoch': m.clock()})
        ledger.update(identity=identity, phase='active', lease_file=str(path), refund_eligible=False)
        # The replacement reuses one network volume; its stopped-storage
        # estimate does not grow for the ephemeral container disk.
        intent.update(attempt_state='adopted', returned_identity=identity)
        active = intent.setdefault('observed_active_identities', [])
        if identity not in active:
            active.append(identity)
        current_intent = self.intent()
        intent['cancel_requested'] = bool(intent.get('cancel_requested') or current_intent.get('cancel_requested'))
        transaction = {'phase': 'prepared', 'previous_identity': m.identity,
                       'config': config, 'ledger': ledger,
                       'intent_path': m.ledger['allocation_intent'], 'intent': intent}
        private_json(m.state_dir / 'identity-adoption.json', transaction)
        private_json(m.config_path, config)
        private_json(m.ledger_path, ledger)
        self._save_intent(intent)
        transaction['phase'] = 'committed'
        private_json(m.state_dir / 'identity-adoption.json', transaction)
        m.config, m.identity, m.ledger, m._lease, m._pod = config, identity, ledger, lease, pod
        return pod

    async def recover_worker(self):
        m = self.m
        try:
            pod = await self.reconcile()
            if pod is not None:
                await m._attach_worker()
        except AllocationPending as exc:
            m._state, m._message = 'error', str(exc)
        except Exception:
            # The independent watcher retains ownership of unknown allocations.
            self.cancel()
            m._state, m._message = 'error', 'Allocation reconciliation needs attention; its guard remains active.'

    async def retire_guards(self):
        """Called only after the current pod and any allocation are resolved stopped."""
        if self.pending():
            return
        m = self.m
        for pid in m.ledger.get('allocation_guard_pids', []):
            try:
                command = await asyncio.to_thread(subprocess.check_output,
                    ['ps', '-p', str(int(pid)), '-o', 'command='], text=True, stderr=subprocess.DEVNULL)
                if (str(Path(__file__).with_name('pod_replacement_watch.py')) in command
                        and str(m.state_dir) in command):
                    os.kill(int(pid), signal.SIGKILL)
            except (OSError, ValueError, subprocess.CalledProcessError):
                pass
        m.ledger['allocation_guard_pids'] = []
        m._save()
