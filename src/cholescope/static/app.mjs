import { createPodControls, podCanConnect, podNotice } from './pod-controls.mjs';
import { KeyboardControls } from './controls.mjs';
import { PointerControls } from './pointer-controls.mjs';
import { createControlRecovery } from './control-recovery.mjs';

const $ = (id) => document.getElementById(id);
const state = {
  backendReady: false, socketReady: false, running: false, paused: false,
  pending: false, sessionStarted: false, sceneId: '', seed: 42, selectedArm: 'left',
  recordings: { a: null, b: null }, recording: null, stoppingRecording: false,
  branch: 'a', comparing: false, hasFrame: false, startedAt: 0, startingRecording: null,
  inputLog: [], socket: null, retry: 1000, message: '',
};
const aVideo = $('reference-video');
const bVideo = $('comparison-video');
let toastTimer;
let pollBusy = false;
let pointer;
let podStatus = null;
let leaving = false;
const recovery = createControlRecovery({ button: $('take-control'), onChange: render, onClaim() {
  const old = state.socket;
  state.socket = null;
  state.socketReady = false;
  old?.close();
  connect();
} });

function toast(message) {
  $('toast').textContent = message;
  $('toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, 6500);
}

function send(message) {
  if (state.socket?.readyState !== WebSocket.OPEN) return false;
  try { state.socket.send(JSON.stringify(message)); return true; }
  catch { return false; }
}

function elapsed() { return Math.max(0, Math.round(performance.now() - state.startedAt)); }
function timeLabel(seconds) {
  if (!Number.isFinite(seconds)) seconds = 0;
  const minutes = Math.floor(seconds / 60);
  return `${String(minutes).padStart(2, '0')}:${(seconds % 60).toFixed(1).padStart(4, '0')}`;
}

function updateKeys(snapshot) {
  for (const keycap of document.querySelectorAll('[data-key]')) {
    keycap.classList.toggle('pressed', snapshot.keys.includes(keycap.dataset.key));
  }
  const active = state.backendReady && state.socketReady && state.running && !state.pending && !state.startingRecording && !state.comparing;
  $('input-label').textContent = active ? (pointer?.active ? `Dragging ${state.selectedArm} arm` : snapshot.keys.length ? 'Sending input' : 'Mouse + keyboard active') : 'Controls inactive';
  $('input-dot').parentElement.classList.toggle('active', active);
}

function submitInput(snapshot) {
  updateKeys(snapshot);
  // Key changes retain pointer activity but never consume or replay its delta.
  return sendInput(snapshot, { pointer_active: Boolean(pointer?.active) });
}

function sendInput(snapshot, pointerFields) {
  const input = { ...snapshot, ...pointerFields };
  if (state.backendReady && state.socketReady && state.running && !state.comparing && send({ type: 'input', ...input })) {
    state.inputLog.push({ t_ms: elapsed(), ...input });
    $('export-inputs').disabled = false;
    return true;
  }
  return false;
}

const keyboard = new KeyboardControls({
  target: window,
  visibilityTarget: document,
  onChange: submitInput,
  onArm: (arm) => {
    pointer?.clear('arm-switch');
    state.selectedArm = arm;
    for (const value of ['left', 'right']) {
      for (const prefix of ['arm', 'drag']) {
        $(`${prefix}-${value}`).classList.toggle('selected', value === arm);
        $(`${prefix}-${value}`).setAttribute('aria-pressed', String(value === arm));
      }
    }
  },
  onPause: () => pauseSession(),
});

pointer = new PointerControls({
  target: $('live-viewport'),
  blurTarget: window,
  visibilityTarget: document,
  onSelectArm: (arm) => keyboard.setArm(arm),
  onRelease: ({ reason, ...fields }) => {
    // The old arm is still selected here when an arm switch cancels a drag.
    return sendInput({ keys: keyboard.snapshot().keys, selected_arm: state.selectedArm }, fields);
  },
  onFeedback: ({ active, start, current, depth }) => {
    $('pointer-feedback').toggleAttribute('hidden', !active);
    $('live-viewport').classList.toggle('pointer-dragging', active);
    updateKeys(keyboard.snapshot());
    if (!active || !start || !current) return;
    const rect = $('live-viewport').getBoundingClientRect();
    $('pointer-feedback').setAttribute('viewBox', `0 0 ${rect.width} ${rect.height}`);
    for (const [name, value] of Object.entries({ x1: start[0], y1: start[1], x2: current[0], y2: current[1] })) $('drag-tether').setAttribute(name, value);
    for (const [id, point] of [['drag-anchor', start], ['drag-ring', current]]) {
      $(id).setAttribute('cx', point[0]);
      $(id).setAttribute('cy', point[1]);
    }
    $('drag-feedback-label').setAttribute('x', Math.max(9, Math.min(rect.width - 126, current[0] + 18)));
    $('drag-feedback-label').setAttribute('y', Math.max(18, Math.min(rect.height - 20, current[1] - 15)));
    $('drag-feedback-label').textContent = depth ? 'Input gesture · depth' : 'Input gesture';
  },
});

function clearInputs(reason) {
  pointer.clear(reason);
  keyboard.clear();
}

function syncPointerAccess() {
  const enabled = state.backendReady && state.socketReady && state.running && state.hasFrame && !state.pending && !state.startingRecording && !state.stoppingRecording && !state.comparing;
  pointer.setEnabled(enabled);
  $('live-viewport').classList.toggle('pointer-enabled', enabled);
  $('pointer-toolbar').hidden = state.comparing;
  $('drag-help').hidden = state.comparing;
  $('drag-left').disabled = !enabled;
  $('drag-right').disabled = !enabled;
  $('drag-help').textContent = enabled
    ? 'Drag to move · Scroll: depth · Shift: precision'
    : state.paused ? 'Resume to drag the tool.' : 'Drag controls activate with the video.';
}

function render() {
  const connected = state.backendReady && state.socketReady;
  const controlNotice = recovery.render({ connected, available: state.backendReady && podCanConnect(podStatus) });
  const active = connected && state.running && !state.pending && !state.startingRecording && !state.comparing;
  keyboard.setEnabled(active);
  syncPointerAccess();
  updateKeys(keyboard.snapshot());
  $('connection').className = `connection ${connected ? 'ready' : 'disconnected'}`;
  // Session state is on the start button and the live video; this word is only the link to the model.
  $('connection-label').textContent = connected ? 'Connected' : 'Offline';
  $('start').disabled = !connected || !state.sceneId || state.pending || Boolean(state.recording) || Boolean(state.startingRecording);
  $('start-label').textContent = state.pending ? 'Preparing…' : state.running ? 'Pause session' : state.paused ? 'Resume session' : 'Start session';
  $('start').querySelector('.play-symbol').textContent = state.running ? 'Ⅱ' : '▶';
  const sessionLocked = state.running || state.paused || state.pending;
  $('scene-select').disabled = sessionLocked || !$('scene-select').value;
  $('seed').disabled = sessionLocked;
  $('random-seed').disabled = sessionLocked;
  $('new-session').hidden = !state.sessionStarted;
  $('new-session').disabled = state.pending || Boolean(state.recording) || Boolean(state.startingRecording);
  $('pause').disabled = !state.running || state.pending;
  $('record-a').disabled = !active || state.stoppingRecording || state.recording === 'b';
  $('record-b').disabled = !active || !state.recordings.a || state.branch !== 'b' || state.stoppingRecording || state.recording === 'a';
  $('reset').disabled = !connected || state.pending || !state.recordings.a || Boolean(state.recording) || Boolean(state.startingRecording);
  $('compare').disabled = !state.recordings.a || !state.recordings.b || Boolean(state.recording) || state.pending;
  for (const branch of ['a', 'b']) {
    $(`record-${branch}`).classList.toggle('recording', state.recording === branch);
    $(`record-${branch}`).lastElementChild.textContent = state.startingRecording === branch ? 'Preparing…' : state.recording === branch ? (state.stoppingRecording ? 'Saving…' : `Finish ${branch.toUpperCase()}`) : `Record ${branch.toUpperCase()}`;
  }
  $('live-viewport').classList.toggle('running', active);
  $('notice').className = `connection-notice ${connected ? 'ready' : 'error'}`;
  $('notice-text').textContent = podNotice(podStatus) || controlNotice || state.message || (connected ? 'Drag the tool. Record A, reset, then record B.' : 'Waiting for the model.');
  if (!state.comparing) {
    // The status line explains idle states; the video itself only labels live ones.
    const label = state.recording ? `Recording ${state.recording.toUpperCase()}` : state.running ? 'Live' : state.paused ? 'Paused' : '';
    const dot = document.createElement('span'); dot.className = 'small-dot';
    $('live-caption').replaceChildren(...(label ? [dot, label] : []));
    $('live-caption').dataset.state = state.recording ? 'recording' : state.running ? 'live' : 'idle';
  }
}

function mediaURL(value) {
  const url = new URL(value, location.href);
  if (url.origin !== location.origin || !['http:', 'https:'].includes(url.protocol)) throw new Error('The server returned an invalid recording URL.');
  return url.href;
}

function applyStatus(message) {
  if (typeof message.backend_ready === 'boolean') state.backendReady = message.backend_ready;
  if (typeof message.message === 'string') state.message = message.message;
  if (['running', 'paused', 'ready', 'idle', 'stopped', 'error'].includes(message.state)) {
    state.running = message.state === 'running';
    state.paused = message.state === 'paused' && state.sessionStarted;
    state.pending = false;
  }
  if (message.state === 'loading' || message.state === 'starting') state.pending = true;
  if (!state.backendReady) {
    state.running = false;
    state.pending = false;
  }
  showMetrics(message);
  render();
}

function showMetrics(message) {
  if (typeof message.fps === 'number' && Number.isFinite(message.fps)) {
    $('fps').replaceChildren(document.createTextNode(message.fps.toFixed(1)), Object.assign(document.createElement('small'), { textContent: ' FPS' }));
  }
  if (typeof message.latency_ms === 'number' && Number.isFinite(message.latency_ms)) {
    $('latency').replaceChildren(document.createTextNode(String(Math.round(message.latency_ms))), Object.assign(document.createElement('small'), { textContent: ' ms' }));
  }
}

function receive(message) {
  switch (message.type) {
    case 'status': applyStatus(message); break;
    case 'metrics': showMetrics(message); break;
    case 'frame': {
      if (state.comparing) break;
      const jpeg = message.jpeg;
      if (typeof jpeg !== 'string' || !jpeg) break;
      $('live-image').src = jpeg.startsWith('data:image/jpeg;base64,') ? jpeg : `data:image/jpeg;base64,${jpeg}`;
      $('live-image').hidden = false;
      $('live-empty').hidden = true;
      const firstFrame = !state.hasFrame;
      state.hasFrame = true;
      if (firstFrame) syncPointerAccess();
      $('live-time').textContent = timeLabel(typeof message.timestamp_ms === 'number' ? message.timestamp_ms / 1000 : typeof message.frame_index === 'number' ? message.frame_index / 10 : elapsed() / 1000);
      break;
    }
    case 'recording_started': {
      if (!['a', 'b'].includes(message.branch)) break;
      state.recording = message.branch;
      state.startingRecording = null;
      state.startedAt = performance.now();
      state.inputLog.push({ type: 'recording_started', branch: message.branch, t_ms: 0, scene_id: state.sceneId, seed: state.seed });
      state.message = `Recording ${message.branch.toUpperCase()} · Drag the tool, then finish.`;
      render();
      break;
    }
    case 'recording': {
      const branch = String(message.branch || '').toLowerCase();
      if (!['a', 'b'].includes(branch) || !message.url) break;
      try {
        state.recordings[branch] = { ...message, url: mediaURL(message.url) };
        const link = $(`download-${branch}`);
        link.href = state.recordings[branch].url;
        link.hidden = false;
        const video = branch === 'a' ? aVideo : bVideo;
        video.src = state.recordings[branch].url;
        video.load();
        if (branch === 'a') {
          $('reference-image').hidden = true;
          $('reference-empty').hidden = true;
          aVideo.hidden = false;
          $('reference-title').textContent = 'Recorded baseline';
          $('reference-kind').textContent = 'AI generated';
          $('reference-kind').classList.add('generated');
        }
        if (message.actions_url) {
          $(`actions-${branch}`).href = mediaURL(message.actions_url);
          $(`actions-${branch}`).hidden = false;
        }
        state.recording = null;
        state.stoppingRecording = false;
        state.message = branch === 'a' ? 'A saved. Reset for B, then try another action.' : 'A and B saved. Ready to compare.';
        render();
      } catch (error) { toast(error.message); }
      break;
    }
    case 'error':
      clearInputs('error');
      state.pending = false;
      state.stoppingRecording = false;
      state.recording = null;
      state.startingRecording = null;
      state.message = message.message || 'The backend could not complete that request.';
      toast(state.message);
      render();
      break;
  }
}

function connect() {
  if (leaving || recovery.requiresClaim || !podCanConnect(podStatus) || [WebSocket.CONNECTING, WebSocket.OPEN].includes(state.socket?.readyState)) return;
  const socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws${recovery.query}`);
  state.socket = socket;
  socket.onopen = () => { if (state.socket !== socket) return; state.socketReady = true; state.retry = 1000; render(); refreshStatus(); };
  socket.onmessage = (event) => {
    if (state.socket !== socket) return;
    try { receive(JSON.parse(event.data)); } catch { toast('Received an unreadable response from the backend.'); }
  };
  socket.onclose = () => {
    if (state.socket !== socket) return;
    state.socketReady = false;
    state.running = false;
    state.paused = false;
    state.pending = false;
    state.recording = null;
    state.startingRecording = null;
    state.stoppingRecording = false;
    pointer.setEnabled(false);
    keyboard.setEnabled(false);
    state.message = podNotice(podStatus) || 'Reconnecting…';
    render();
    void recovery.refresh();
    setTimeout(connect, state.retry);
    state.retry = Math.min(8000, state.retry * 1.6);
  };
  socket.onerror = () => socket.close();
}

async function refreshStatus() {
  if (pollBusy || !podCanConnect(podStatus)) return;
  pollBusy = true;
  try {
    await recovery.refresh();
    const response = await fetch('/api/status', { cache: 'no-store' });
    if (!response.ok) throw new Error('status unavailable');
    applyStatus(await response.json());
  } catch {
    state.backendReady = false;
    state.running = false;
    state.message = podNotice(podStatus) || 'Waiting for the model.';
    render();
  } finally { pollBusy = false; }
}

function setSceneImage() {
  const id = $('scene-select').value;
  state.sceneId = id;
  if (!id) return;
  $('reference-image').src = `/api/scene/${encodeURIComponent(id)}/image`;
  $('reference-image').hidden = false;
  $('reference-empty').hidden = true;
}

async function loadScenes() {
  try {
    const response = await fetch('/api/scenes');
    if (!response.ok) throw new Error('scene list unavailable');
    const data = await response.json();
    const scenes = Array.isArray(data) ? data : data.scenes;
    const select = $('scene-select');
    select.replaceChildren();
    for (const scene of scenes || []) {
      if (typeof scene.id !== 'string') continue;
      const option = document.createElement('option');
      option.value = scene.id;
      option.textContent = scene.name || scene.label || scene.id;
      select.append(option);
    }
    if (!select.children.length) select.append(Object.assign(document.createElement('option'), { value: '', textContent: 'No scenes configured' }));
    else setSceneImage();
    render();
  } catch {
    $('scene-select').replaceChildren(Object.assign(document.createElement('option'), { value: '', textContent: 'Scene library unavailable' }));
  }
}

function resetView() {
  pointer.clear('reset');
  state.hasFrame = false;
  $('live-image').hidden = true;
  $('live-image').removeAttribute('src');
  $('live-empty').hidden = false;
  $('live-time').textContent = '00:00.0';
}

function startSession() {
  if (!state.sceneId || state.pending) return;
  const seed = Number($('seed').value);
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 2147483647) { toast('Choose a whole-number seed between 0 and 2147483647.'); return; }
  exitComparison();
  clearInputs('start');
  state.seed = seed;
  if (!send({ type: 'start', scene_id: state.sceneId, seed })) return;
  state.sessionStarted = true;
  clearRecordings();
  state.pending = true;
  state.paused = false;
  state.startedAt = performance.now();
  state.inputLog = [];
  state.branch = 'a';
  state.message = 'Preparing the first frame…';
  resetView();
  render();
}

function pauseSession() {
  if (!state.running || state.pending) return;
  clearInputs('pause');
  send({ type: 'input', keys: [], selected_arm: state.selectedArm, pointer_active: false });
  if (state.recording && !state.stoppingRecording) {
    state.stoppingRecording = true;
    send({ type: 'record_stop', branch: state.recording });
  }
  if (send({ type: 'pause' })) { state.pending = true; render(); }
}

function toggleRecord(branch) {
  if (!state.running || state.stoppingRecording || state.startingRecording) return;
  if (state.recording === branch) {
    clearInputs('record-stop');
    state.stoppingRecording = true;
    send({ type: 'record_stop', branch });
  } else if (!state.recording) {
    if (!send({ type: 'record_start', branch })) return;
    clearInputs('record-start');
    state.startingRecording = branch;
    state.message = `Preparing branch ${branch.toUpperCase()}…`;
    resetView();
    if (branch === 'a') {
      state.recordings.b = null;
      $('download-b').hidden = true;
      $('actions-b').hidden = true;
      bVideo.removeAttribute('src');
    }
  }
  render();
}

function clearRecordings() {
  state.recordings = { a: null, b: null };
  for (const branch of ['a', 'b']) { $(`download-${branch}`).hidden = true; $(`actions-${branch}`).hidden = true; }
  for (const video of [aVideo, bVideo]) { video.pause(); video.removeAttribute('src'); video.hidden = true; }
  $('reference-title').textContent = 'Starting scene';
  $('reference-kind').textContent = 'Dataset image';
  $('reference-kind').classList.remove('generated');
  $('reference-time').textContent = '00:00.0';
  setSceneImage();
}

function resetForB() {
  if (state.pending || state.recording) return;
  exitComparison();
  clearInputs('reset');
  send({ type: 'input', keys: [], selected_arm: state.selectedArm, pointer_active: false });
  if (!send({ type: 'reset', scene_id: state.sceneId, seed: state.seed })) return;
  state.pending = true;
  state.branch = 'b';
  state.message = `Resetting for B…`;
  state.startedAt = performance.now();
  state.inputLog.push({ type: 'reset', t_ms: 0, scene_id: state.sceneId, seed: state.seed });
  resetView();
  render();
}

function comparisonDuration() {
  const durations = [aVideo.duration, bVideo.duration].filter((value) => Number.isFinite(value) && value > 0);
  return durations.length === 2 ? Math.min(...durations) : 0;
}

function pauseVideos() {
  aVideo.pause();
  bVideo.pause();
  $('play-pause').textContent = '▶';
  $('play-pause').setAttribute('aria-label', 'Play both recordings');
}

async function playVideos() {
  const duration = comparisonDuration();
  if (!duration) { toast('The recordings are still loading. Try playback again in a moment.'); return; }
  if (aVideo.currentTime >= duration - .05) { aVideo.currentTime = 0; bVideo.currentTime = 0; }
  bVideo.currentTime = aVideo.currentTime;
  const results = await Promise.allSettled([aVideo.play(), bVideo.play()]);
  if (results.some((result) => result.status === 'rejected')) {
    pauseVideos(); toast('Your browser could not play one of the recordings. Download the MP4 files to inspect them.'); return;
  }
  $('play-pause').textContent = 'Ⅱ';
  $('play-pause').setAttribute('aria-label', 'Pause both recordings');
}

function enterComparison() {
  if (!state.recordings.a || !state.recordings.b) return;
  if (state.running) pauseSession();
  clearInputs('comparison');
  state.comparing = true;
  aVideo.hidden = false;
  bVideo.hidden = false;
  $('live-image').hidden = true;
  $('live-empty').hidden = true;
  $('reference-image').hidden = true;
  $('reference-empty').hidden = true;
  $('playback-controls').hidden = false;
  $('live-title').textContent = 'Alternative continuation';
  $('live-caption').replaceChildren();
  state.message = 'Synchronized A/B playback.';
  aVideo.currentTime = 0;
  bVideo.currentTime = 0;
  render();
  playVideos();
}

function exitComparison() {
  pauseVideos();
  state.comparing = false;
  bVideo.hidden = true;
  $('live-image').hidden = !state.hasFrame;
  $('live-empty').hidden = state.hasFrame;
  $('playback-controls').hidden = true;
  $('live-title').textContent = 'Model continuation';
  render();
}

function playbackTick() {
  if (state.comparing) {
    const duration = comparisonDuration();
    const current = Math.min(aVideo.currentTime, duration);
    if (duration && aVideo.currentTime >= duration) pauseVideos();
    if (!aVideo.paused && Math.abs(aVideo.currentTime - bVideo.currentTime) > .12) bVideo.currentTime = aVideo.currentTime;
    $('timeline').value = duration ? String(Math.round(current / duration * 1000)) : '0';
    $('playback-time').textContent = `${timeLabel(current)} / ${timeLabel(duration)}`;
    $('reference-time').textContent = timeLabel(current);
    $('live-time').textContent = timeLabel(Math.min(bVideo.currentTime, duration));
  }
  requestAnimationFrame(playbackTick);
}

$('start').addEventListener('click', () => {
  if (state.running) pauseSession();
  else if (state.paused) {
    exitComparison();
    if (send({ type: 'resume' })) { state.pending = true; render(); }
  } else startSession();
  $('start').blur();
});
$('pause').addEventListener('click', () => { pauseSession(); $('pause').blur(); });
$('new-session').addEventListener('click', () => {
  pauseSession();
  state.sessionStarted = false;
  state.paused = false;
  state.branch = 'a';
  exitComparison();
  clearRecordings();
  resetView();
  render();
  $('new-session').blur();
});
$('scene-select').addEventListener('change', setSceneImage);
$('random-seed').addEventListener('click', () => { $('seed').value = String(crypto.getRandomValues(new Uint32Array(1))[0] % 2147483648); });
$('arm-left').addEventListener('click', () => { keyboard.setArm('left'); $('arm-left').blur(); });
$('arm-right').addEventListener('click', () => { keyboard.setArm('right'); $('arm-right').blur(); });
for (const arm of ['left', 'right']) $(`drag-${arm}`).addEventListener('click', () => {
  keyboard.setArm(arm);
  $('live-viewport').focus({ preventScroll: true });
});
for (const branch of ['a', 'b']) $(`record-${branch}`).addEventListener('click', () => { toggleRecord(branch); $(`record-${branch}`).blur(); });
$('reset').addEventListener('click', () => { resetForB(); $('reset').blur(); });
$('compare').addEventListener('click', enterComparison);
$('exit-compare').addEventListener('click', exitComparison);
$('play-pause').addEventListener('click', () => { if (aVideo.paused) playVideos(); else pauseVideos(); });
$('timeline').addEventListener('input', () => { const time = Number($('timeline').value) / 1000 * comparisonDuration(); aVideo.currentTime = time; bVideo.currentTime = time; });
for (const video of [aVideo, bVideo]) video.addEventListener('ended', pauseVideos);
$('reference-image').addEventListener('error', () => {
  $('reference-image').hidden = true;
  if (!state.recordings.a) {
    $('reference-empty').hidden = false;
    $('reference-empty-title').textContent = 'The selected scene image is not available yet.';
  }
});
$('export-inputs').addEventListener('click', () => {
  const data = { format: 'cholescope-control-inputs-v2', description: 'Client mouse, touch, and keyboard intentions. These are not server-confirmed model action tensors.', scene_id: state.sceneId, seed: state.seed, inputs: state.inputLog };
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
  const anchor = document.createElement('a');
  anchor.href = url; anchor.download = `cholescope-controls-${state.seed}.json`;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
window.addEventListener('pagehide', () => {
  leaving = true;
  clearInputs('pagehide');
  send({ type: 'input', keys: [], selected_arm: state.selectedArm, pointer_active: false });
  send({ type: 'pause' });
});

setInterval(() => {
  if (state.backendReady && state.socketReady && state.running && !state.pending && !state.startingRecording && !state.comparing) {
    pointer.flush((fields) => sendInput(keyboard.snapshot(), fields));
  }
}, 100);
setInterval(refreshStatus, 5000);
render();
loadScenes();
createPodControls({ onChange(status) {
  const wasReady = podCanConnect(podStatus);
  podStatus = status;
  if (!podCanConnect(status)) {
    clearInputs('pod-offline');
    state.backendReady = false; state.socketReady = false; state.running = false; state.pending = false;
    state.message = podNotice(status); state.socket?.close();
  } else { if (!wasReady) state.message = ''; connect(); void refreshStatus(); }
  render();
} });
playbackTick();
