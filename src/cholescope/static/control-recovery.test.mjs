import test from 'node:test';
import assert from 'node:assert/strict';
import { createControlRecovery } from './control-recovery.mjs';

const STORAGE_KEY = 'cholescope-control-token';
const response = (value, status = 200) => ({
  ok: status >= 200 && status < 300, status, json: async () => value,
});
const deferred = () => {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
};
function setup({ token = '', fetcher } = {}) {
  const saved = new Map(token ? [[STORAGE_KEY, token]] : []);
  const button = { handlers: {}, addEventListener(name, handler) { this.handlers[name] = handler; } };
  const calls = [];
  let claims = 0;
  const control = createControlRecovery({
    button,
    storage: () => ({ getItem: (key) => saved.get(key), setItem: (key, value) => saved.set(key, value) }),
    fetcher: (...args) => { calls.push(args); return fetcher(...args); },
    onClaim: () => { claims += 1; },
  });
  return { control, button, saved, calls, get claims() { return claims; } };
}
const render = (control) => control.render({ connected: false, available: true });

test('only an explicit click takes control, with same-origin POST guards and one pending request', async () => {
  const takeover = deferred();
  const state = setup({ fetcher: () => takeover.promise });
  assert.equal(state.calls.length, 0);
  state.button.handlers.click();
  assert.equal(state.calls.length, 1);
  const [url, request] = state.calls[0];
  assert.equal(url, '/api/control/takeover');
  assert.equal(request.method, 'POST');
  assert.equal(request.credentials, 'same-origin');
  assert.equal(request.headers['Content-Type'], 'application/json');
  assert.equal(request.headers['X-CholeScope-Request'], '1');
  assert.equal(request.headers.Authorization, undefined);
  assert.equal(request.body, '{}');
  assert.equal(render(state.control), 'Taking control of the model…');
  assert.equal(state.button.disabled, true);
  assert.equal(state.button.hidden, false);
  assert.equal(state.button.textContent, 'Connecting…');
  state.button.handlers.click();
  await state.control.claim();
  assert.equal(state.calls.length, 1);
  takeover.resolve(response({ control_token: 'new-owner' }));
  await new Promise(setImmediate);
  assert.equal(state.saved.get(STORAGE_KEY), 'new-owner');
  assert.equal(state.control.query, '?control_token=new-owner');
  assert.equal(state.claims, 1);
  assert.equal(render(state.control), '');
  assert.equal(state.button.hidden, true);
  assert.equal(state.button.disabled, false);
});

test('an existing sessionStorage token is reused and safely encoded in status requests', async () => {
  const state = setup({ token: 'tab owner+/=&', fetcher: async () => response({ supported: true, occupied: false }) });
  assert.equal(state.control.query, '?control_token=tab%20owner%2B%2F%3D%26');
  await state.control.refresh();
  assert.equal(state.calls[0][0], '/api/control/status?control_token=tab%20owner%2B%2F%3D%26');
  assert.equal(state.calls[0][1].cache, 'no-store');
  assert.equal(state.calls[0][1].credentials, 'same-origin');
  assert.equal(state.claims, 0);
});

test('status refreshes expose an occupied connection without automatically claiming it', async () => {
  const state = setup({ fetcher: async () => response({ supported: true, occupied: true, requires_claim: true }) });
  await state.control.refresh();
  await state.control.refresh();
  assert.deepEqual(state.calls.map(([url]) => url), ['/api/control/status', '/api/control/status']);
  assert.equal(state.claims, 0);
  assert.equal(state.control.query, '');
  assert.equal(state.saved.size, 0);
  assert.equal(state.control.requiresClaim, true);
  assert.match(render(state.control), /Choose Use this tab to switch here/);
  assert.equal(state.button.hidden, false);
  assert.equal(state.button.textContent, 'Use this tab');
});

test('a revoked token exposes Use this tab even after the other connection has gone away', async () => {
  const state = setup({ token: 'revoked-owner', fetcher: async () => response({ supported: true, occupied: false, requires_claim: true }) });
  await state.control.refresh();
  assert.equal(state.control.requiresClaim, true);
  assert.match(render(state.control), /Use this tab/);
  assert.equal(state.button.hidden, false);
  assert.equal(state.button.disabled, false);
  assert.equal(state.button.textContent, 'Use this tab');
  assert.equal(state.saved.get(STORAGE_KEY), 'revoked-owner');
});

test('a failed takeover preserves the current token and offers a working retry', async () => {
  const replies = [response({ message: 'Unavailable' }, 503), response({ control_token: 'retry-owner' })];
  const state = setup({ token: 'original-owner', fetcher: async () => replies.shift() });
  await state.control.claim();
  assert.equal(state.control.query, '?control_token=original-owner');
  assert.equal(state.saved.get(STORAGE_KEY), 'original-owner');
  assert.equal(state.claims, 0);
  assert.equal(render(state.control), 'Could not take control. Try again.');
  assert.equal(state.button.hidden, false);
  assert.equal(state.button.disabled, false);
  assert.equal(state.button.textContent, 'Use this tab');
  await state.control.claim();
  assert.equal(state.control.query, '?control_token=retry-owner');
  assert.equal(state.saved.get(STORAGE_KEY), 'retry-owner');
  assert.equal(state.claims, 1);
  assert.equal(render(state.control), '');
  assert.equal(state.button.hidden, true);
});

test('standalone status 404 clears stale control ownership and hides the button', async () => {
  const replies = [response({ supported: true, occupied: true, requires_claim: true }), response({}, 404)];
  const state = setup({ fetcher: async () => replies.shift() });
  await state.control.refresh();
  render(state.control);
  assert.equal(state.button.hidden, false);
  await state.control.refresh();
  assert.equal(state.control.requiresClaim, false);
  assert.equal(render(state.control), '');
  assert.equal(state.button.hidden, true);
});

test('standalone status 404 clears a failed takeover retry action', async () => {
  const state = setup({ fetcher: async (url) => url === '/api/control/takeover'
    ? response({ message: 'Unavailable' }, 503) : response({}, 404) });
  await state.control.claim();
  assert.equal(render(state.control), 'Could not take control. Try again.');
  assert.equal(state.button.hidden, false);
  await state.control.refresh();
  assert.equal(render(state.control), '');
  assert.equal(state.button.hidden, true);
  assert.equal(state.control.requiresClaim, false);
});

for (const phase of ['response', 'JSON body']) {
  test(`a stale status ${phase} cannot overwrite a successful new claim`, async () => {
    const stale = deferred();
    const bodyStarted = deferred();
    const oldStatus = { supported: true, occupied: true, requires_claim: true };
    const state = setup({ token: 'old-owner', fetcher: async (url) => {
      if (url === '/api/control/takeover') return response({ control_token: 'new-owner' });
      if (phase === 'response') return stale.promise;
      return { ok: true, status: 200, json: () => { bodyStarted.resolve(); return stale.promise; } };
    } });
    const refresh = state.control.refresh();
    if (phase === 'JSON body') await bodyStarted.promise;
    await state.control.claim();
    stale.resolve(phase === 'response' ? response(oldStatus) : oldStatus);
    await refresh;
    assert.equal(state.control.query, '?control_token=new-owner');
    assert.equal(state.control.requiresClaim, false);
    assert.equal(state.saved.get(STORAGE_KEY), 'new-owner');
    assert.equal(state.claims, 1);
    assert.equal(render(state.control), '');
    assert.equal(state.button.hidden, true);
  });
}
