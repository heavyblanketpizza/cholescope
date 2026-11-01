import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizePodStatus, podCanConnect, podNotice, podBudgetLabel, podControlState, postPodAction, createPodControls } from './pod-controls.mjs';

test('connects only after model readiness, or on a direct server without pod management', () => {
  for (const state of ['stopped', 'starting', 'loading', 'stopping', 'error']) assert.equal(podCanConnect({ available: true, state }), false);
  assert.equal(podCanConnect(null), false);
  assert.equal(podCanConnect({ available: true, state: 'ready' }), true);
  assert.equal(podCanConnect({ available: false }), true);
});
test('capabilities must be explicit booleans and invalid states are rejected', () => {
  assert.equal(normalizePodStatus({ available: true, state: 'stopped', can_start: 'true' }).can_start, false);
  assert.throws(() => normalizePodStatus({ available: true, state: 'RUNNING' }));
  assert.throws(() => normalizePodStatus(null));
});
test('stopped state gives one quiet next action with accurate cost scope', () => {
  const status = { available: true, state: 'stopped', rate_usd_per_hour: 2.49, session_limit_minutes: 60 };
  assert.equal(podNotice(status), 'Start the pod to begin.');
  assert.equal(podBudgetLabel(status), '$2.49/hr while running · 60 min session cap · Storage charges continue');
  assert.equal(podBudgetLabel({ available: true, state: 'ready', rate_usd_per_hour: NaN }), '');
});
test('start is same-origin POST with explicit intent header and no browser credential material', async () => {
  let received;
  await postPodAction('start', async (url, request) => {
    received = { url, request };
    return { ok: true, json: async () => ({ available: true, state: 'starting' }) };
  });
  assert.equal(received.url, '/api/pod/start');
  assert.equal(received.request.method, 'POST');
  assert.equal(received.request.headers['X-CholeScope-Request'], '1');
  assert.equal(received.request.body, '{}');
  assert.equal(received.request.headers.Authorization, undefined);
});
test('unknown actions never make a request and failed requests surface the provider message', async () => {
  await assert.rejects(postPodAction('terminate', () => { throw new Error('must not run'); }), /Unknown pod action/);
  await assert.rejects(postPodAction('start', async () => ({ ok: false, json: async () => ({ message: 'Budget limit reached.' }) })), /Budget limit reached/);
});

test('error state cannot enable Start and preserves an available Stop action', () => {
  assert.deepEqual(podControlState({ state: 'error', can_start: false, can_stop: false }),
    { startHidden: false, stopHidden: true, startDisabled: true, stopDisabled: true });
  assert.deepEqual(podControlState({ state: 'error', can_start: false, can_stop: true }),
    { startHidden: true, stopHidden: false, startDisabled: true, stopDisabled: false });
  assert.equal(podControlState({ can_stop: true }, true).stopDisabled, true);
});


test('successful polling clears a transient refresh warning but retains a failed action message', async () => {
  const originals = { document: globalThis.document, window: globalThis.window, fetch: globalThis.fetch,
    setInterval: globalThis.setInterval, clearInterval: globalThis.clearInterval };
  const elements = new Map();
  const makeElement = () => ({ dataset: {}, classList: { toggle() {} }, handlers: {},
    addEventListener(name, handler) { this.handlers[name] = handler; } });
  for (const id of ['pod-controls', 'pod-state', 'pod-caption', 'pod-start', 'pod-stop']) elements.set(id, makeElement());
  const status = { available: true, state: 'stopped', can_start: true, can_stop: false, session_limit_minutes: 60 };
  const success = () => ({ ok: true, status: 200, json: async () => status });
  const replies = [success, () => { throw new Error('temporary offline'); }, success,
    () => ({ ok: false, status: 409, json: async () => ({ message: 'Budget limit reached.' }) }), success, success];
  try {
    globalThis.document = { getElementById: (id) => elements.get(id) };
    globalThis.window = { addEventListener() {} };
    globalThis.fetch = async () => replies.shift()();
    globalThis.setInterval = () => 1;
    globalThis.clearInterval = () => {};
    const controls = createPodControls();
    await controls.ready;
    await controls.refresh();
    assert.equal(elements.get('pod-caption').textContent, 'Could not refresh pod status.');
    await controls.refresh();
    assert.equal(elements.get('pod-caption').textContent, '60 min session cap · Storage charges continue');
    elements.get('pod-start').handlers.click();
    await new Promise(setImmediate);
    assert.equal(elements.get('pod-caption').textContent, 'Budget limit reached.');
    await controls.refresh();
    assert.equal(elements.get('pod-caption').textContent, 'Budget limit reached.');
    assert.equal(replies.length, 0);
  } finally { Object.assign(globalThis, originals); }
});


test('a stopped pod surfaces its last startup failure without changing retry capability', () => {
  const status = { available: true, state: 'stopped', last_error: 'GPU capacity is unavailable. Try again later.', can_start: true, can_stop: false };
  assert.equal(podNotice(status), status.last_error);
  assert.equal(podControlState(status).startDisabled, false);
  assert.equal(podControlState({ ...status, can_start: false }).startDisabled, true);
  assert.equal(podNotice({ ...status, last_error: '  ' }), 'Start the pod to begin.');
  assert.equal(podNotice({ ...status, state: 'ready' }), '');
});
