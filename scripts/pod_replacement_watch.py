#!/usr/bin/env python3
"""Independent stop-only watcher for one durable replacement allocation intent.

It is armed before the sole create request, never creates resources itself,
and keeps watching an ambiguous request even after its deadline has passed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time
import uuid

import runpod_budget as guard
from pod_replacement import ReplacementAPI, ownership_match, validate_recipe, verify_created


def private_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, allow_nan=False, indent=2)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_intent(value):
    if not isinstance(value, dict) or value.get('schema_version') != 1:
        raise ValueError('Invalid allocation intent schema')
    validate_recipe(value['recipe'])
    if not isinstance(value['session_lease'], dict):
        raise ValueError('Invalid allocation session lease')
    lease = guard.validate_lease(value['session_lease'])
    if lease['schema_version'] != 2 or lease['action'] != 'stop':
        raise ValueError('Allocation requires a stop-only session lease')
    if value['deadline_epoch'] != lease['deadline_epoch']:
        raise ValueError('Allocation changed the session deadline')
    if not isinstance(value['name'], str) or not value['name'].startswith('cholescope-allocate-'):
        raise ValueError('Allocation needs a unique persisted name')
    uuid.UUID(hex=value['name'].removeprefix('cholescope-allocate-'))
    guard.number(value['created_after'], 'allocation request time')
    if value['attempt_state'] not in {'prepared', 'dispatched', 'accepted', 'adopted', 'rejected'}:
        raise ValueError('Invalid allocation state')
    if not isinstance(value.get('cancel_requested', False), bool):
        raise ValueError('Invalid allocation cancellation')
    return value


def identity_of(pod):
    return {'pod_id': pod['id'], 'pod_name': pod['name'], 'created_at': pod['createdAt']}


def replacement_lease(original, pod):
    """Change only identity/start anchoring while retaining the prepaid deadline."""
    lease = {**original, **identity_of(pod),
             'session_start_epoch': guard.timestamp(pod['createdAt']),
             'deadline_epoch': original['deadline_epoch']}
    return guard.validate_lease(lease)


def _remember_evidence(target, evidence, first):
    if not isinstance(evidence, list):
        return
    for identity in evidence:
        if not isinstance(identity, dict):
            continue
        pod = {'id': identity.get('pod_id'), 'name': identity.get('pod_name'),
               'createdAt': identity.get('created_at')}
        if ownership_match(pod, first['name'], first['created_after']):
            target[(pod['id'], pod['name'], pod['createdAt'])] = identity_of(pod)


def _identity_key(pod):
    return pod['id'], pod['name'], pod['createdAt']


def watch(intent_path, api, *, clock=time.time, monotonic=time.monotonic,
          sleep=time.sleep, stop_requested=lambda: False, log=print, poll=5):
    path = Path(intent_path)
    first = validate_intent(json.loads(path.read_text()))
    deadline = first['deadline_epoch']
    monotonic_deadline = monotonic() + max(0, deadline - clock())
    result_path = path.with_suffix('.result.json')
    observed_active = {}
    accepted_stop = {}
    cancel = False
    last_intent = first
    try:
        previous = json.loads(result_path.read_text())
        if (isinstance(previous, dict) and previous.get('name') == first['name'] and
                previous.get('created_after') == first['created_after'] and
                previous.get('deadline_epoch') == deadline):
            _remember_evidence(observed_active, previous.get('observed_active_identities'), first)
            _remember_evidence(accepted_stop, previous.get('accepted_stop_identities'), first)
            cancel = previous.get('cancel_requested') is True
    except (OSError, ValueError, TypeError):
        pass
    # The first line is the parent's readiness handshake. If it cannot be
    # written, fail before the parent can dispatch its paid create request.
    log(json.dumps({'event': 'allocation_guard_armed', 'name': first['name'],
                    'deadline_epoch': deadline}), flush=True)

    def safe_log(message):
        try:
            log(message, flush=True)
        except (OSError, ValueError):
            pass

    def save_receipt(state, identities, bad):
        try:
            private_json(result_path, {
                'at_epoch': clock(), 'state': state, 'name': first['name'],
                'created_after': first['created_after'], 'identities': identities,
                'recipe_mismatch': bad, 'deadline_epoch': deadline,
                'cancel_requested': cancel,
                'observed_active_identities': list(observed_active.values()),
                'accepted_stop_identities': list(accepted_stop.values()),
            })
        except (OSError, ValueError):
            # Neither the receipt nor its warning may gate paid shutdown.
            safe_log('WARNING: allocation receipt could not be saved; enforcement continues')

    while True:
        try:
            intent = validate_intent(json.loads(path.read_text()))
            for field in ('name', 'recipe', 'created_after', 'session_lease', 'deadline_epoch'):
                if intent[field] != first[field]:
                    raise ValueError('Allocation intent changed after arming')
            last_intent = intent
        except (RuntimeError, ValueError, KeyError, TypeError, OSError):
            # The original immutable identity remains authoritative even if the
            # controller's file disappears or is damaged. Losing that file must
            # never disable the only guard for an in-flight paid allocation.
            cancel = True
            intent = last_intent
            safe_log('WARNING: allocation intent unavailable or changed; stopping its known allocation')
        try:
            cancel = cancel or stop_requested() or intent.get('cancel_requested', False)
            due = cancel or clock() >= deadline or monotonic() >= monotonic_deadline
            _remember_evidence(observed_active, intent.get('observed_active_identities'), first)
            _remember_evidence(accepted_stop, intent.get('accepted_stop_identities'), first)
            matches = [pod for pod in ReplacementAPI(api).list_pods()
                       if ownership_match(pod, first['name'], first['created_after'])]
            identities = [identity_of(pod) for pod in matches]
            bad = len(matches) > 1
            for pod in matches:
                if pod.get('status') in {'PROVISIONING', 'STARTING', 'RUNNING'}:
                    observed_active[_identity_key(pod)] = identity_of(pod)
                try:
                    verify_created(pod, first['recipe'], first['name'], first['created_after'])
                except RuntimeError:
                    bad = True
            should_stop = due or bad or intent['attempt_state'] == 'rejected'
            # A confirmed allocation rejection may safely retire an empty intent.
            # A timeout/5xx/absent listing never proves that no pod will appear.
            done = not matches and intent['attempt_state'] == 'rejected'
            if matches:
                done = all(pod.get('status') in {'EXITED', 'TERMINATED'} and (
                    _identity_key(pod) in observed_active or _identity_key(pod) in accepted_stop)
                           for pod in matches)
            save_receipt('stopped' if done else ('stopping' if should_stop else 'watching'), identities, bad)
            if done:
                safe_log(json.dumps({'event': 'allocation_shutdown_verified', 'name': first['name']}))
                return
            if should_stop:
                for pod in matches:
                    if pod.get('status') in {'EXITED', 'TERMINATED'}:
                        continue
                    # Refresh exact immutable identity before every mutation.
                    try:
                        current = api.get(pod['id'])
                        if current is None:
                            continue
                        guard.verify_pod(current, identity_of(pod))
                        if ownership_match(current, first['name'], first['created_after']):
                            api.shutdown(pod['id'], 'stop')
                            accepted_stop[_identity_key(current)] = identity_of(current)
                    except (RuntimeError, ValueError, KeyError, TypeError, AttributeError, OSError):
                        # One disappearing or inconsistent record must not block
                        # stopping any other owned duplicate from this attempt.
                        safe_log('WARNING: an allocation identity could not be stopped; reconciliation continues')
                # Preserve successful stop evidence before waiting for EXITED,
                # including when this watcher is restarted between polls.
                save_receipt('stopping', identities, bad)
        except (RuntimeError, ValueError, KeyError, TypeError, OSError):
            # Inspection failure cannot erase an unknown paid allocation.
            safe_log('WARNING: allocation reconciliation is pending; no new create is permitted')
        sleep(poll)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intent', required=True, type=Path)
    parser.add_argument('--api-key-file', required=True, type=Path)
    args = parser.parse_args()
    cancelled = [False]
    def cancel(*_):
        cancelled[0] = True
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    watch(args.intent, guard.RunpodAPI(args.api_key_file.read_text().strip()),
          stop_requested=lambda: cancelled[0])


if __name__ == '__main__':
    main()
