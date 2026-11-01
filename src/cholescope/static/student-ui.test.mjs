import test from 'node:test';
import assert from 'node:assert/strict';
import { canCompare, chartPoint, finiteXY, sameChallenge, safeMediaURL, trainingHistory } from './student.mjs';

const record = (overrides = {}) => ({ scene_id: 'scene-1', seed: 42, target: [.12, .08], score: { complete: true }, recording: { url: '/clip.mp4' }, ...overrides });

test('comparison requires identical scene, seed, target, and two complete recordings', () => {
  assert.equal(canCompare(record(), record()), true);
  for (const change of [{ scene_id: 'different' }, { seed: 43 }, { target: [.12, -.08] }, { score: { complete: false } }, { recording: null }]) assert.equal(canCompare(record(), record(change)), false);
  assert.equal(canCompare(record(), null), false);
});

test('malformed coordinates cannot match a challenge or enter the command map', () => {
  for (const value of [null, [0], [0, 0, 0], ['0', 0], [Infinity, 0], [NaN, 0]]) {
    assert.equal(finiteXY(value), false);
    assert.equal(sameChallenge(record(), record({ target: value })), false);
    assert.equal(chartPoint(value), null);
  }
});

test('command map follows controller X-right and Y-down with bounded plotting', () => {
  assert.deepEqual(chartPoint([0, 0]), [140, 90]);
  assert.deepEqual(chartPoint([.24, .24]), [252, 157]);
  assert.deepEqual(chartPoint([-.24, -.24]), [28, 23]);
  assert.deepEqual(chartPoint([1, -1]), [252, 23]);
});

test('loss chart accepts only measured finite nonnegative samples', () => {
  const valid = { updates: 50, batch_mse: .012 };
  assert.deepEqual(trainingHistory([valid, { updates: 100, batch_mse: NaN }, { updates: 0, batch_mse: -1 }, { updates: '2', batch_mse: .1 }, null]), [valid]);
  assert.deepEqual(trainingHistory(null), []);
});

test('recording links stay on the serving origin', () => {
  assert.equal(safeMediaURL('/api/runs/one/video.mp4', 'http://127.0.0.1:8792/student'), 'http://127.0.0.1:8792/api/runs/one/video.mp4');
  for (const value of ['https://example.com/clip.mp4', 'javascript:alert(1)', '', null]) assert.throws(() => safeMediaURL(value, 'http://127.0.0.1:8792/student'));
});
