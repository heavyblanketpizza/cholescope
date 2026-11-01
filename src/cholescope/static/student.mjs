import { createPodControls, podCanConnect, podNotice } from './pod-controls.mjs';
import { createControlRecovery } from './control-recovery.mjs';
const PHASES = ['before', 'after'];
const MAP_LIMIT = .24;

export function finiteXY(value) {
  return Array.isArray(value) && value.length === 2 && value.every((n) => typeof n === 'number' && Number.isFinite(n));
}

export function sameChallenge(a, b) {
  return Boolean(a && b && a.scene_id === b.scene_id && a.seed === b.seed && finiteXY(a.target) && finiteXY(b.target)
    && a.target.every((value, i) => Math.abs(value - b.target[i]) < 1e-8));
}

export function canCompare(a, b) {
  return sameChallenge(a, b) && a.score?.complete === true && b.score?.complete === true && Boolean(a.recording?.url && b.recording?.url);
}

export function chartPoint(point) {
  if (!finiteXY(point)) return null;
  return [140 + Math.max(-MAP_LIMIT, Math.min(MAP_LIMIT, point[0])) / MAP_LIMIT * 112,
    90 + Math.max(-MAP_LIMIT, Math.min(MAP_LIMIT, point[1])) / MAP_LIMIT * 67];
}

export function trainingHistory(value) {
  return Array.isArray(value) ? value.filter((item) => Number.isFinite(item?.updates) && Number.isFinite(item?.batch_mse) && item.batch_mse >= 0) : [];
}

export function safeMediaURL(value, base) {
  if (typeof value !== 'string' || !value) throw new Error('Missing recording URL.');
  const url = new URL(value, base);
  if (url.origin !== new URL(base).origin || !['http:', 'https:'].includes(url.protocol)) throw new Error('Invalid recording URL.');
  return url.href;
}

function boot() {
  const $ = (id) => document.getElementById(id);
  const state = { socket: null, socketReady: false, backendReady: false, trained: false, training: false,
    pending: false, activePhase: null, sceneId: '', target: [.12, .08], seed: 42,
    recordings: { before: null, after: null }, paths: { before: [[0, 0]], after: [[0, 0]] },
    step: 0, totalSteps: 64, metrics: null, comparing: false, retry: 1000, leaving: false,
    message: 'Connecting to the student lab…', error: false, selectedByUser: false, expectingReset: false };
  const videos = Object.fromEntries(PHASES.map((phase) => [phase, $(`video-${phase}`)]));
  let podStatus = null;
  let pollBusy = false;
  const recovery = createControlRecovery({
    button: $('take-control'),
    onChange: render,
    onClaim() {
      const old = state.socket;
      state.socket = null;
      state.socketReady = false;
      old?.close();
      connect();
    },
  });

  const number = (value, digits = 4) => typeof value === 'number' && Number.isFinite(value)
    ? value !== 0 && Math.abs(value) < 10 ** -digits ? value.toExponential(2) : value.toFixed(digits) : '—';
  const time = (seconds) => {
    const value = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
    return `${String(Math.floor(value / 60)).padStart(2, '0')}:${(value % 60).toFixed(1).padStart(4, '0')}`;
  };
  const challenge = () => ({ target: state.target, scene_id: state.sceneId, seed: state.seed });
  const busy = () => state.pending || state.training || Boolean(state.activePhase);
  const send = (payload) => {
    if (state.socket?.readyState !== WebSocket.OPEN) return false;
    try { state.socket.send(JSON.stringify(payload)); return true; } catch { return false; }
  };
  function notice(message, error = false) { state.message = message; state.error = error; render(); }

  function drawMap(phase, path = state.paths[phase]) {
    const svg = $(`map-${phase}`);
    const goal = chartPoint(state.target);
    const points = path.filter(finiteXY).map(chartPoint);
    const current = points.at(-1) || [140, 90];
    // The map is a distinct coordinate instrument, never an anatomical overlay.
    svg.innerHTML = `<path class="map-grid" d="M28 23H252M28 56.5H252M28 90H252M28 123.5H252M28 157H252M28 23V157M84 23V157M140 23V157M196 23V157M252 23V157" fill="none" stroke-width=".6"/>
      <path class="map-axis" d="M28 90H252M140 23V157" fill="none" stroke-width=".8"/>
      <text x="258" y="94">+X</text><text x="144" y="174">+Y</text><text x="10" y="94">−X</text><text x="144" y="16">−Y</text>
      <circle class="map-origin" cx="140" cy="90" r="2.5"/>
      <circle class="map-goal" cx="${goal[0]}" cy="${goal[1]}" r="7" stroke-width="1"/>
      <path class="map-cross" d="M${goal[0] - 10} ${goal[1]}h20M${goal[0]} ${goal[1] - 10}v20" stroke-width=".8"/>
      <polyline class="map-path" points="${points.map((p) => p.join(',')).join(' ')}" fill="none" stroke-width="1.8" stroke-linejoin="round"/>
      <circle class="map-head" cx="${current[0]}" cy="${current[1]}" r="3.4"/>
      <text x="28" y="174">0 = start</text><text x="190" y="16">goal</text>`;
  }

  function drawLoss(history) {
    const points = trainingHistory(history);
    const svg = $('loss-chart');
    if (!points.length) {
      svg.innerHTML = '<path class="chart-grid" d="M12 94H348M12 53H348M12 12H348" fill="none" stroke-dasharray="3 5"/><text x="180" y="59" text-anchor="middle">Train to reveal measured loss</text>';
      return;
    }
    const max = Math.max(...points.map((p) => p.batch_mse), 1e-8);
    const span = Math.max(1, points.at(-1).updates - points[0].updates);
    const line = points.map((p) => `${12 + (p.updates - points[0].updates) / span * 336},${94 - p.batch_mse / max * 80}`).join(' ');
    svg.innerHTML = `<path class="chart-grid" d="M12 94H348M12 53H348M12 12H348" fill="none" stroke-dasharray="3 5"/><polyline class="chart-line" points="${line}" fill="none" stroke-width="1.7"/><text x="12" y="108">${points[0].updates}</text><text x="348" y="108" text-anchor="end">${points.at(-1).updates} updates</text>`;
    $('loss-initial').textContent = number(points[0].batch_mse, 5);
    $('loss-final').textContent = number(points.at(-1).batch_mse, 5);
  }

  function showMetrics(message) {
    const metrics = message.metrics;
    const history = trainingHistory(message.train_history || metrics?.history);
    if (history.length) drawLoss(history);
    if (!metrics) return;
    state.metrics = metrics;
    if (Number.isFinite(metrics.total_updates)) $('training-steps').textContent = `${metrics.total_updates.toLocaleString()} real updates`;
    const before = metrics.before;
    const after = metrics.after;
    if (!before || !after) return;
    $('evaluation-count').textContent = `${after.goals} held-out goals`;
    $('holdout-distance').textContent = `${number(before.mean_terminal_distance)} → ${number(after.mean_terminal_distance)}`;
    const reached = (item) => `${Math.round(item.goal_reached_fraction * item.goals)}/${item.goals}`;
    $('holdout-success').textContent = `${reached(before)} → ${reached(after)}`;
    $('holdout-neutral').textContent = metrics.neutral ? reached(metrics.neutral) : '—';
    $('holdout-teacher').textContent = metrics.analytic_teacher ? reached(metrics.analytic_teacher) : '—';
    $('evaluation-threshold').textContent = `Reached = final command distance ≤ ${number(after.goal_tolerance, 3)}. Before → after; no clinical success score.`;
    $('training-caption').textContent = `${metrics.algorithm || 'Supervised imitation'}. Training examples are geometric corrections, without medical images.`;
  }

  function render() {
    const locked = busy();
    const canRun = state.socketReady && state.backendReady && state.sceneId && !locked;
    $('connection').className = `connection ${state.socketReady ? 'ready' : 'disconnected'}`;
    $('connection-label').textContent = !state.socketReady ? 'Offline' : state.backendReady ? 'Ready' : 'CPU ready';
    $('watch-before').disabled = !canRun;
    $('watch-after').disabled = !canRun || !state.trained;
    $('train').disabled = !state.socketReady || locked || state.trained;
    $('train').querySelector('strong').textContent = state.training ? 'Training…' : state.trained ? 'Trained' : 'Train';
    $('reset-learning').disabled = !state.socketReady || locked || (!state.trained && !state.recordings.before && !state.recordings.after);
    $('checkpoint-after').hidden = !state.trained;
    $('scene-select').disabled = locked || !$('scene-select').value;
    for (const button of document.querySelectorAll('[data-target]')) button.disabled = locked;
    $('stop').disabled = !state.activePhase && !state.pending;
    const recoveryNotice = recovery.render({ connected: state.socketReady, available: state.backendReady && podCanConnect(podStatus) });
    $('status-message').textContent = podNotice(podStatus) || recoveryNotice || state.message;
    $('status-message').parentElement.classList.toggle('busy', locked);
    $('status-message').parentElement.classList.toggle('error', state.error);
    $('after-kind').textContent = state.trained ? 'Learned weights' : 'Awaiting training';
    for (const phase of PHASES) $(`card-${phase}`).classList.toggle('active', state.activePhase === phase);
    const comparable = canCompare(state.recordings.before, state.recordings.after);
    $('compare').disabled = !comparable || locked;
    $('timeline').disabled = !comparable || locked;
  }

  function pausePlayback() {
    for (const video of Object.values(videos)) video.pause();
    $('compare').textContent = '▶ Play comparison';
  }

  function resetPhase(phase) {
    state.recordings[phase] = null;
    state.paths[phase] = [[0, 0]];
    videos[phase].pause(); videos[phase].removeAttribute('src'); videos[phase].hidden = true;
    for (const item of ['download', 'actions']) $(`${item}-${phase}`).hidden = true;
    $(`distance-${phase}`).textContent = '—';
    $(`result-${phase}`).textContent = 'No trial yet';
    $(`caption-${phase}`).textContent = '';
    $(`time-${phase}`).textContent = '00:00.0';
    const img = $(`image-${phase}`);
    img.src = state.sceneId ? `/api/scene/${encodeURIComponent(state.sceneId)}/image` : '';
    img.hidden = !state.sceneId;
    $(`empty-${phase}`).hidden = Boolean(state.sceneId);
    drawMap(phase);
  }

  function changeChallenge() {
    state.selectedByUser = true;
    pausePlayback(); state.comparing = false;
    for (const phase of PHASES) resetPhase(phase);
    $('timeline').value = '0'; $('playback-time').textContent = '00:00.0 / 00:00.0';
    notice('New challenge selected. Watch both checkpoints with these same starting conditions.');
  }

  function clearLearningEvidence() {
    pausePlayback(); state.comparing = false; state.metrics = null;
    for (const phase of PHASES) resetPhase(phase);
    drawLoss([]);
    for (const id of ['loss-initial', 'loss-final', 'holdout-distance', 'holdout-success', 'holdout-neutral', 'holdout-teacher']) $(id).textContent = '—';
    $('training-steps').textContent = 'No updates yet';
    $('evaluation-count').textContent = 'Not evaluated';
    $('training-caption').textContent = "Loss is how closely the student matches the coordinate teacher's moves.";
    $('evaluation-threshold').textContent = 'No clinical success score is calculated.';
    $('timeline').value = '0'; $('playback-time').textContent = '00:00.0 / 00:00.0';
  }

  function watch(phase) {
    if (busy() || !state.backendReady || !state.sceneId || (phase === 'after' && !state.trained)) return;
    pausePlayback(); state.comparing = false; resetPhase(phase);
    if (!send({ type: 'student_trial', phase, ...challenge() })) return;
    state.activePhase = phase; state.pending = true; state.step = 0;
    $(`caption-${phase}`).textContent = 'Preparing Cosmos visualization';
    notice(`Preparing the ${phase === 'before' ? 'untrained' : 'trained'} student's attempt. The initial scene can take a moment to load.`);
  }

  function stop(reason = 'Trial stopped.') {
    pausePlayback();
    if (!state.activePhase && !state.pending) return;
    send({ type: 'pause' });
    state.pending = false;
    notice(reason);
  }

  async function loadActions(phase, record) {
    if (!record.recording?.actions_url) return;
    try {
      const url = safeMediaURL(record.recording.actions_url, location.href);
      const response = await fetch(url, { cache: 'no-store' });
      if (!response.ok) return;
      const data = await response.json();
      if (state.recordings[phase] !== record) return;
      const rows = Array.isArray(data.actions) ? data.actions : [];
      const path = rows.map((row) => Array.isArray(row) ? row.slice(0, 2) : null).filter(finiteXY);
      if (path.length) { state.paths[phase] = [[0, 0], ...path]; drawMap(phase); }
    } catch { /* The video and explicit score remain available if the action export fails. */ }
  }

  function result(message, quiet = false) {
    const phase = message.phase;
    if (!PHASES.includes(phase) || !sameChallenge(message, challenge())) return;
    const existing = state.recordings[phase];
    if (existing && existing.recording?.run_id === message.recording?.run_id && existing.checkpoint_sha256 === message.checkpoint_sha256) return;
    const record = { ...message, recording: message.recording ? { ...message.recording } : null };
    if (record.recording?.url) {
      try {
        record.recording.url = safeMediaURL(record.recording.url, location.href);
        const video = videos[phase];
        video.src = record.recording.url; video.hidden = false; video.load();
        $(`image-${phase}`).hidden = true; $(`empty-${phase}`).hidden = true;
        $(`download-${phase}`).href = record.recording.url; $(`download-${phase}`).hidden = false;
        if (record.recording.actions_url) { $(`actions-${phase}`).href = safeMediaURL(record.recording.actions_url, location.href); $(`actions-${phase}`).hidden = false; }
      } catch (error) { notice(error.message, true); return; }
    }
    state.recordings[phase] = record;
    const score = record.score || {};
    $(`distance-${phase}`).textContent = number(score.final_distance);
    $(`result-${phase}`).textContent = !score.complete ? 'Stopped early · incomplete trial' : score.reached ? 'Command target reached' : 'Command target missed';
    $(`caption-${phase}`).textContent = score.complete ? 'Recorded · AI generated' : 'Partial recording · AI generated';
    if (Number.isFinite(score.steps)) $(`time-${phase}`).textContent = time(score.steps / 10);
    if (state.activePhase === phase) { state.activePhase = null; state.pending = false; }
    if (!quiet) notice(!score.complete ? 'Trial stopped. Partial attempts are excluded from the synchronized comparison.'
      : phase === 'before' ? state.trained ? 'Untrained attempt saved. Watch the trained student on the same challenge.' : 'First attempt saved. Train the small network, then watch its second attempt.'
        : 'Trained attempt saved. Play the comparison and inspect the command maps and measured distances.');
    void loadActions(phase, record);
    render();
  }

  function receiveStudent(message) {
    const wasTraining = state.training;
    state.trained = Boolean(message.trained);
    state.training = Boolean(message.training);
    if (state.expectingReset && message.state === 'idle' && !state.trained && !state.training && !message.metrics) {
      state.expectingReset = false; state.pending = false; state.activePhase = null;
      clearLearningEvidence();
      state.message = 'Learning reset to the same initial weights. Watch the untrained attempt, then train the student yourself.';
      state.error = false;
    }
    if (typeof message.model_name === 'string') $('model-label').textContent = message.model_name;
    if (Number.isFinite(message.parameter_count)) $('parameter-count').textContent = `${message.parameter_count.toLocaleString()} trainable parameters · Cosmos frozen`;
    showMetrics(message);
    if (message.state === 'running' && PHASES.includes(message.phase)) {
      state.activePhase = message.phase; state.pending = false;
      state.step = Number.isFinite(message.step) ? message.step : 0;
      state.totalSteps = Number.isFinite(message.total_steps) ? message.total_steps : 64;
      if (finiteXY(message.position)) {
        const path = state.paths[message.phase];
        if (!path.at(-1)?.every((v, i) => v === message.position[i])) path.push([...message.position]);
        drawMap(message.phase);
        $(`distance-${message.phase}`).textContent = number(Math.hypot(...message.position.map((v, i) => v - state.target[i])));
        $(`result-${message.phase}`).textContent = `${state.step} / ${state.totalSteps} steps · trial in progress`;
      }
      state.message = `${message.phase === 'before' ? 'Untrained' : 'Trained'} student running · step ${state.step} of ${state.totalSteps}. The map shows commanded motion; the video is Cosmos's prediction.`;
      $(`caption-${message.phase}`).textContent = 'Live · AI generated';
    } else if (['finished', 'cancelled'].includes(message.state)) {
      state.activePhase = null; state.pending = false;
    }
    if (state.training) { state.pending = false; state.message = 'Training the student on synthetic coordinate corrections. Cosmos weights remain frozen.'; }
    else if (wasTraining && state.trained) { state.pending = false; state.message = 'Training complete. The chart and held-out results show measured changes. Watch the trained student next.'; }
    // A server snapshot can still contain the preceding run while a new trial
    // is starting. Never let that old result replace the live attempt.
    for (const item of Object.values(message.results || {})) {
      if (state.activePhase !== item?.phase) result(item, true);
    }
    render();
  }

  function receive(message) {
    switch (message.type) {
      case 'student_state': receiveStudent(message); break;
      case 'student_result': result(message); break;
      case 'status':
        if (typeof message.backend_ready === 'boolean') state.backendReady = message.backend_ready;
        if (!state.backendReady && !busy()) state.message = 'Start the pod to watch attempts.';
        render(); break;
      case 'frame': {
        if (!state.activePhase || typeof message.jpeg !== 'string' || !message.jpeg) break;
        const phase = state.activePhase;
        const img = $(`image-${phase}`);
        img.src = message.jpeg.startsWith('data:image/jpeg;base64,') ? message.jpeg : `data:image/jpeg;base64,${message.jpeg}`;
        img.alt = `Live Cosmos visualization of the ${phase === 'before' ? 'untrained' : 'trained'} student's commands`;
        img.hidden = false; videos[phase].hidden = true; $(`empty-${phase}`).hidden = true;
        $(`time-${phase}`).textContent = time((Number.isFinite(message.frame_index) ? message.frame_index + 1 : state.step) / 10);
        break;
      }
      case 'error':
        state.pending = false; state.activePhase = null; state.training = false; state.expectingReset = false;
        notice(message.message || 'The request could not be completed.', true); break;
    }
  }

  function connect() {
    if (state.leaving || recovery.requiresClaim || !podCanConnect(podStatus) || [WebSocket.CONNECTING, WebSocket.OPEN].includes(state.socket?.readyState)) return;
    const socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws${recovery.query}`);
    state.socket = socket;
    socket.onopen = () => {
      if (state.socket !== socket) return;
      state.socketReady = true; state.retry = 1000;
      notice('Choose an attempt to watch.');
      send({ type: 'student_status' }); void refreshStatus();
    };
    socket.onmessage = (event) => {
      if (state.socket !== socket) return;
      try { receive(JSON.parse(event.data)); } catch { notice('The server sent an unreadable response. Reconnect before starting another attempt.', true); }
    };
    socket.onclose = () => {
      if (state.socket !== socket) return;
      state.socketReady = false; state.pending = false; state.activePhase = null; state.training = false; state.expectingReset = false; pausePlayback();
      notice(podNotice(podStatus) || 'Reconnecting…');
      void recovery.refresh();
      if (!state.leaving) setTimeout(connect, state.retry);
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
      if (!response.ok) throw new Error('Backend status unavailable.');
      const data = await response.json();
      state.backendReady = Boolean(data.backend_ready);
      render();
    } catch { state.backendReady = false; render(); }
    finally { pollBusy = false; }
  }

  async function loadScenes() {
    try {
      const response = await fetch('/api/scenes', { cache: 'no-store' });
      if (!response.ok) throw new Error('Scene list unavailable.');
      const data = await response.json();
      const select = $('scene-select'); select.replaceChildren();
      for (const scene of Array.isArray(data) ? data : data.scenes || []) {
        if (typeof scene.id !== 'string') continue;
        select.append(Object.assign(document.createElement('option'), { value: scene.id, textContent: scene.name || scene.label || scene.id }));
      }
      if (!select.children.length) select.append(Object.assign(document.createElement('option'), { value: '', textContent: 'No scenes configured' }));
      state.sceneId = select.value;
      for (const phase of PHASES) resetPhase(phase);
      render(); send({ type: 'student_status' });
    } catch { $('scene-select').replaceChildren(Object.assign(document.createElement('option'), { value: '', textContent: 'Scene library unavailable' })); }
  }

  function duration() {
    const values = Object.values(videos).map((video) => video.duration);
    return values.every((value) => Number.isFinite(value) && value > 0) ? Math.min(...values) : 0;
  }

  async function playComparison() {
    if (!canCompare(state.recordings.before, state.recordings.after) || busy()) return;
    if (!videos.before.paused) { pausePlayback(); return; }
    const end = duration();
    if (!end) { notice('Recordings are loading. Try playback again in a moment.'); return; }
    state.comparing = true;
    if (videos.before.currentTime >= end - .05) videos.before.currentTime = 0;
    videos.after.currentTime = videos.before.currentTime;
    for (const phase of PHASES) { videos[phase].hidden = false; $(`image-${phase}`).hidden = true; $(`empty-${phase}`).hidden = true; }
    const results = await Promise.allSettled(Object.values(videos).map((video) => video.play()));
    if (results.some((r) => r.status === 'rejected')) { pausePlayback(); notice('Playback is unavailable. Use the video downloads to inspect these attempts.', true); return; }
    $('compare').textContent = 'Ⅱ Pause comparison';
    notice('Synchronized comparison · command coordinates');
  }

  function tick() {
    if (state.comparing) {
      const end = duration();
      const current = Math.min(videos.before.currentTime, end);
      if (end && videos.before.currentTime >= end) pausePlayback();
      if (!videos.before.paused && Math.abs(videos.after.currentTime - videos.before.currentTime) > .12) videos.after.currentTime = videos.before.currentTime;
      $('timeline').value = end ? String(Math.round(current / end * 1000)) : '0';
      $('playback-time').textContent = `${time(current)} / ${time(end)}`;
      for (const phase of PHASES) {
        $(`time-${phase}`).textContent = time(current);
        drawMap(phase, state.paths[phase].slice(0, Math.min(state.paths[phase].length, Math.floor(current * 10) + 2)));
      }
    }
    if (!state.leaving) requestAnimationFrame(tick);
  }

  $('watch-before').addEventListener('click', () => watch('before'));
  $('watch-after').addEventListener('click', () => watch('after'));
  $('train').addEventListener('click', () => {
    if (busy() || state.trained) return;
    pausePlayback(); state.comparing = false;
    if (send({ type: 'student_train' })) { state.training = true; notice('Training from coordinates…'); }
  });
  $('reset-learning').addEventListener('click', () => {
    if (busy()) return;
    if (send({ type: 'student_reset' })) { state.expectingReset = true; state.pending = true; notice('Resetting learning…'); }
  });
  $('stop').addEventListener('click', () => stop());
  $('scene-select').addEventListener('change', () => { state.sceneId = $('scene-select').value; changeChallenge(); });
  for (const button of document.querySelectorAll('[data-target]')) button.addEventListener('click', () => {
    if (busy()) return;
    state.target = button.dataset.target.split(',').map(Number);
    for (const item of document.querySelectorAll('[data-target]')) { item.classList.toggle('selected', item === button); item.setAttribute('aria-pressed', String(item === button)); }
    changeChallenge();
  });
  $('compare').addEventListener('click', playComparison);
  $('timeline').addEventListener('input', () => { const current = Number($('timeline').value) / 1000 * duration(); for (const video of Object.values(videos)) video.currentTime = current; state.comparing = true; });
  for (const video of Object.values(videos)) video.addEventListener('ended', pausePlayback);
  window.addEventListener('blur', () => stop('Trial stopped because the window lost focus. Partial attempts do not count as completed comparisons.'));
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop('Trial stopped while this tab was hidden.'); });
  window.addEventListener('pagehide', () => { state.leaving = true; stop(); state.socket?.close(); });
  for (const phase of PHASES) drawMap(phase);
  drawLoss([]); render();
  createPodControls({ onChange(status) {
    const wasReady = podCanConnect(podStatus);
    podStatus = status;
    if (!podCanConnect(status)) {
      state.backendReady = false; state.socketReady = false; state.pending = false; state.activePhase = null; state.training = false;
      state.message = podNotice(status); state.error = false; state.socket?.close(); pausePlayback();
    } else { if (!wasReady) state.message = 'Loading lesson…'; connect(); void refreshStatus(); }
    render();
  } });
  void loadScenes();
  setInterval(refreshStatus, 10000);
  requestAnimationFrame(tick);
}

if (typeof document !== 'undefined') boot();
