import test from 'node:test';
import assert from 'node:assert/strict';
import { KeyboardControls } from './controls.mjs';
import { PointerControls } from './pointer-controls.mjs';

class Target {
  constructor() { this.handlers = new Map(); this.hidden = false; }
  addEventListener(name, handler) { this.handlers.set(name, handler); }
  removeEventListener(name) { this.handlers.delete(name); }
  emit(name, data = {}) {
    const event = { key: '', repeat: false, target: { closest: () => null }, preventDefault() {}, ...data };
    this.handlers.get(name)?.(event);
  }
}

function setup() {
  const target = new Target(), visibilityTarget = new Target(), changes = [], arms = [];
  let pauses = 0;
  const controls = new KeyboardControls({ target, visibilityTarget, onChange: (state) => changes.push(state), onArm: (arm) => arms.push(arm), onPause: () => pauses++ });
  controls.setEnabled(true);
  return { target, visibilityTarget, changes, arms, controls, pauses: () => pauses };
}

test('key-up releases a held key even if focus has moved into a form field', () => {
  const s = setup();
  s.target.emit('keydown', { key: 'w' });
  s.target.emit('keyup', { key: 'w', target: { closest: () => ({}) } });
  assert.deepEqual(s.changes.at(-1).keys, []);
});

test('window blur, hidden tab, disable, and destroy release motion', () => {
  for (const reason of ['blur', 'hidden', 'disable', 'destroy']) {
    const s = setup();
    s.target.emit('keydown', { key: 'w' });
    if (reason === 'blur') s.target.emit('blur');
    if (reason === 'hidden') { s.visibilityTarget.hidden = true; s.visibilityTarget.emit('visibilitychange'); }
    if (reason === 'disable') s.controls.setEnabled(false);
    if (reason === 'destroy') s.controls.destroy();
    assert.deepEqual(s.changes.at(-1).keys, [], reason);
  }
});

test('switching arms emits a release for the original arm before enabling the new arm', () => {
  const s = setup();
  s.target.emit('keydown', { key: 'f' });
  s.target.emit('keydown', { key: '2' });
  assert.deepEqual(s.changes.slice(-2), [{ keys: [], selected_arm: 'left' }, { keys: [], selected_arm: 'right' }]);
  assert.deepEqual(s.arms, ['right']);
});

test('auto-repeat cannot duplicate pause commands or held-key changes', () => {
  const s = setup();
  s.target.emit('keydown', { key: 'w' });
  s.target.emit('keydown', { key: 'w', repeat: true });
  assert.equal(s.changes.length, 1);
  s.target.emit('keydown', { key: ' ' });
  s.target.emit('keydown', { key: ' ', repeat: true });
  assert.equal(s.pauses(), 1);
  assert.deepEqual(s.controls.snapshot().keys, []);
});

test('form typing, disconnected inputs, and OS shortcuts cannot become motion commands', () => {
  const s = setup();
  s.target.emit('keydown', { key: 'w', target: { closest: () => ({}) } });
  assert.deepEqual(s.controls.snapshot().keys, []);
  s.controls.setEnabled(false);
  s.target.emit('keydown', { key: 'w' });
  assert.deepEqual(s.controls.snapshot().keys, []);
  s.controls.setEnabled(true);
  s.target.emit('keydown', { key: 'w' });
  s.target.emit('keydown', { key: 's', metaKey: true });
  assert.deepEqual(s.controls.snapshot().keys, []);
});

test('opposing keys remain explicit for the server to resolve, without browser repeat buildup', () => {
  const s = setup();
  for (const key of ['W', 's', 'Shift']) s.target.emit('keydown', { key });
  assert.deepEqual(s.controls.snapshot().keys, ['s', 'shift', 'w']);
  s.target.emit('keyup', { key: 'W' });
  assert.deepEqual(s.controls.snapshot().keys, ['s', 'shift']);
});

class PointerTarget extends Target {
  constructor() { super(); this.captured = new Set(); }
  getBoundingClientRect() { return { left: 10, top: 20, width: 1000, height: 500 }; }
  setPointerCapture(id) { this.captured.add(id); }
  hasPointerCapture(id) { return this.captured.has(id); }
  releasePointerCapture(id) { this.captured.delete(id); }
  focus() {}
  point(name, data = {}) {
    this.emit(name, { pointerId: 1, pointerType: 'mouse', button: 0, buttons: 1, isPrimary: true, clientX: 110, clientY: 120, ...data });
  }
}

function pointerSetup() {
  const target = new PointerTarget(), blurTarget = new Target(), visibilityTarget = new Target();
  const releases = [], feedback = [], arms = [];
  const controls = new PointerControls({ target, blurTarget, visibilityTarget,
    onRelease: (value) => releases.push(value), onFeedback: (value) => feedback.push(value), onSelectArm: (arm) => arms.push(arm) });
  controls.setEnabled(true);
  return { target, blurTarget, visibilityTarget, releases, feedback, arms, controls };
}

function flush(pointer) {
  let message;
  pointer.flush((value) => { message = value; return true; });
  return message;
}

test('drag follows model +X right and +Y down, and successful heartbeats drain once', () => {
  const s = pointerSetup();
  s.target.point('pointerdown');
  s.target.point('pointermove', { clientX: 210, clientY: 170 });
  assert.deepEqual(flush(s.controls), { pointer_active: true, pointer_delta: [.05, .025, 0] });
  assert.deepEqual(flush(s.controls).pointer_delta, [0, 0, 0]);
  assert.equal(s.feedback.at(-1).active, true);
  assert.deepEqual(s.feedback.at(-1).current, [200, 150]);
});

test('quick drag commits its unsent tail once; lost capture and later heartbeats cannot replay it', () => {
  const s = pointerSetup();
  s.target.point('pointerdown');
  s.target.point('pointermove', { clientX: 112 });
  s.target.point('pointerup', { buttons: 0, clientX: 112 });
  s.target.point('lostpointercapture');
  assert.equal(s.releases.length, 1);
  assert.deepEqual(s.releases[0], { pointer_active: false, pointer_commit: true, pointer_delta: [.001, 0, 0], reason: 'pointerup' });
  assert.equal(s.target.captured.size, 0);
  for (let n = 0; n < 2; n++) assert.deepEqual(flush(s.controls), { pointer_active: false, pointer_delta: [0, 0, 0] });
});

test('normal release sends one zero-tail commit after a drained movement', () => {
  const s = pointerSetup();
  s.target.point('pointerdown');
  s.target.point('pointermove', { clientX: 130 });
  flush(s.controls);
  s.target.point('pointerup', { clientX: 130 });
  assert.equal(s.releases[0].pointer_commit, true);
  assert.deepEqual(s.releases[0].pointer_delta, [0, 0, 0]);
});

test('cancel, lost capture, blur, hidden tab, disable, and arm switch discard unflushed motion', () => {
  for (const reason of ['cancel', 'capture-lost', 'blur', 'hidden', 'disabled', 'arm-switch', 'destroy']) {
    const s = pointerSetup();
    s.target.point('pointerdown');
    s.target.point('pointermove', { clientX: 190 });
    if (reason === 'cancel') s.target.point('pointercancel');
    else if (reason === 'capture-lost') s.target.point('lostpointercapture');
    else if (reason === 'blur') s.blurTarget.emit('blur');
    else if (reason === 'hidden') { s.visibilityTarget.hidden = true; s.visibilityTarget.emit('visibilitychange'); }
    else if (reason === 'disabled') s.controls.setEnabled(false);
    else if (reason === 'destroy') s.controls.destroy();
    else s.controls.clear(reason);
    assert.equal(s.releases.length, 1, reason);
    assert.equal(s.releases[0].pointer_commit, false, reason);
    assert.deepEqual(s.releases[0].pointer_delta, [0, 0, 0], reason);
    assert.deepEqual(flush(s.controls).pointer_delta, [0, 0, 0], reason);
    assert.equal(s.target.captured.size, 0, reason);
  }
});

test('failed heartbeat retains intent until success; disconnect cancels retained intent', () => {
  const s = pointerSetup();
  s.target.point('pointerdown');
  s.target.point('pointermove', { clientX: 130 });
  s.controls.flush(() => false);
  assert.deepEqual(flush(s.controls).pointer_delta, [.01, 0, 0]);
  s.target.point('pointermove', { clientX: 150 });
  s.controls.flush(() => false);
  s.controls.setEnabled(false);
  s.controls.setEnabled(true);
  assert.deepEqual(flush(s.controls).pointer_delta, [0, 0, 0]);
});

function totalDelta(messages) {
  return messages.reduce((total, message) => total.map((value, axis) => value + message.pointer_delta[axis]), [0, 0, 0]);
}

function nearVector(actual, expected) {
  actual.forEach((value, axis) => assert.ok(Math.abs(value - expected[axis]) < 1e-10, `${actual} ≈ ${expected}`));
}

test('large movement is split into bounded packets without losing any distance or delaying overflow', () => {
  const s = pointerSetup();
  s.target.point('pointerdown');
  s.target.point('pointermove', { clientX: 1110, clientY: 620 });
  const messages = [];
  assert.equal(s.controls.flush((message) => { messages.push(message); return true; }), true);
  assert.ok(messages.length > 1);
  for (const { pointer_delta } of messages) assert.ok(Math.hypot(...pointer_delta) <= .08 + 1e-10);
  nearVector(totalDelta(messages), [.5, .25, 0]);
  assert.deepEqual(flush(s.controls).pointer_delta, [0, 0, 0]);
});

test('same 600 pixel stroke preserves distance regardless of heartbeat count or release timing', () => {
  for (const ticks of [0, 1, 2, 3, 6, 12]) {
    const s = pointerSetup(), messages = [];
    s.target.point('pointerdown');
    for (let tick = 1; tick <= ticks; tick++) {
      s.target.point('pointermove', { clientX: 110 + 600 * tick / ticks });
      s.controls.flush((message) => { messages.push(message); return true; });
    }
    // With no heartbeat, the whole stroke is reported on release.
    s.target.point('pointerup', { buttons: 0, clientX: 710 });
    messages.push(...s.releases);
    nearVector(totalDelta(messages), [.3, 0, 0]);
    for (const { pointer_delta } of messages) assert.ok(Math.hypot(...pointer_delta) <= .08 + 1e-10);
    assert.deepEqual(flush(s.controls).pointer_delta, [0, 0, 0]);
  }
});

test('equal pixel distances have equal gain on both axes and a reversal returns to origin', () => {
  const s = pointerSetup(), messages = [];
  s.target.point('pointerdown');
  s.target.point('pointermove', { clientX: 310, clientY: 320 });
  s.controls.flush((message) => { messages.push(message); return true; });
  nearVector(totalDelta(messages), [.1, .1, 0]);
  s.target.point('pointerup', { buttons: 0, clientX: 110, clientY: 120 });
  nearVector(totalDelta([...messages, ...s.releases]), [0, 0, 0]);
});

test('a partial batch failure retries only the unsent displacement', () => {
  const s = pointerSetup(), messages = [];
  s.target.point('pointerdown');
  s.target.point('pointermove', { clientX: 710 });
  let attempts = 0;
  assert.equal(s.controls.flush((message) => {
    if (++attempts === 2) return false;
    messages.push(message); return true;
  }), false);
  assert.equal(messages.length, 1);
  s.controls.flush((message) => { messages.push(message); return true; });
  nearVector(totalDelta(messages), [.3, 0, 0]);
  assert.deepEqual(flush(s.controls).pointer_delta, [0, 0, 0]);
});

test('release sends the final coordinate and failed release cannot leak into the next drag', () => {
  const s = pointerSetup();
  s.target.point('pointerdown');
  s.target.point('pointermove', { clientX: 130 });
  const sent = flush(s.controls);
  s.target.point('pointerup', { buttons: 0, clientX: 150 });
  nearVector(totalDelta([sent, ...s.releases]), [.02, 0, 0]);
  s.controls.onRelease = () => false;
  s.target.point('pointerdown');
  s.target.point('pointerup', { buttons: 0, clientX: 710 });
  s.target.point('pointerdown');
  assert.deepEqual(flush(s.controls).pointer_delta, [0, 0, 0]);
});

test('Shift scales pointer translation and wheel depth to one quarter', () => {
  const s = pointerSetup();
  s.target.point('pointerdown');
  s.target.point('pointermove', { clientX: 210, clientY: 170, shiftKey: true });
  s.target.emit('wheel', { deltaY: -100, deltaMode: 0, shiftKey: true });
  assert.deepEqual(flush(s.controls).pointer_delta, [.0125, .00625, .00625]);
});

test('wheel only takes over scrolling during an active drag and normalizes line units', () => {
  const s = pointerSetup();
  let prevented = 0;
  const wheel = () => s.target.emit('wheel', { deltaY: -2, deltaMode: 1, preventDefault() { prevented++; } });
  wheel();
  assert.equal(prevented, 0);
  s.target.point('pointerdown');
  wheel();
  assert.equal(prevented, 1);
  assert.deepEqual(flush(s.controls).pointer_delta, [0, 0, .008]);
  s.target.point('pointerup');
  wheel();
  assert.equal(prevented, 1);
});

test('secondary buttons and another touch cannot initiate or sustain a primary gesture', () => {
  const s = pointerSetup();
  for (const button of [1, 2]) s.target.point('pointerdown', { button });
  s.target.point('pointerdown', { pointerType: 'touch', isPrimary: false });
  assert.equal(s.controls.active, false);
  s.target.point('pointerdown');
  s.target.point('pointermove', { pointerId: 2, clientX: 600 });
  assert.deepEqual(flush(s.controls).pointer_delta, [0, 0, 0]);
  s.target.point('pointermove', { buttons: 2, clientX: 600 });
  assert.equal(s.controls.active, false);
  assert.equal(s.releases[0].pointer_commit, false);
});

test('form targets do not start drags, while a grip selects its arm before capture', () => {
  const s = pointerSetup();
  s.target.point('pointerdown', { target: { closest: (selector) => selector === '[data-drag-arm]' ? null : {} } });
  assert.equal(s.controls.active, false);
  s.target.point('pointerdown', { target: { closest: (selector) => selector === '[data-drag-arm]' ? { dataset: { dragArm: 'right' } } : null } });
  assert.deepEqual(s.arms, ['right']);
  assert.equal(s.controls.active, true);
});

test('OS shortcut modifiers cannot start a drag and cancel an existing gesture without committing', () => {
  for (const modifier of ['ctrlKey', 'metaKey', 'altKey']) {
    const s = pointerSetup();
    s.target.point('pointerdown', { [modifier]: true });
    assert.equal(s.controls.active, false, modifier);
    s.target.point('pointerdown');
    s.target.point('pointermove', { clientX: 130 });
    s.blurTarget.emit('keydown', { [modifier]: true });
    assert.equal(s.controls.active, false, modifier);
    assert.equal(s.releases[0].pointer_commit, false, modifier);
    assert.deepEqual(s.releases[0].pointer_delta, [0, 0, 0], modifier);
  }
});

test('ordinary key events do not drain pending pointer input or disturb a touch gesture', () => {
  const s = pointerSetup();
  s.target.point('pointerdown', { pointerType: 'touch' });
  s.target.point('pointermove', { pointerType: 'touch', buttons: 0, clientX: 130 });
  s.blurTarget.emit('keydown', { key: 'w' });
  assert.equal(s.controls.active, true);
  const keyboardSnapshot = { keys: ['w'], selected_arm: 'left' };
  let sent;
  s.controls.flush((fields) => { sent = { ...keyboardSnapshot, ...fields }; return true; });
  assert.deepEqual(sent, { keys: ['w'], selected_arm: 'left', pointer_active: true, pointer_delta: [.01, 0, 0] });
});
