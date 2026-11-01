const STATES = new Set(['stopped', 'starting', 'loading', 'ready', 'stopping', 'error']);

export function normalizePodStatus(value) {
  if (!value || typeof value !== 'object') throw new Error('Pod status unavailable.');
  if (value.available === false) return { available: false, state: 'ready', can_start: false, can_stop: false };
  if (value.available !== true || !STATES.has(value.state)) throw new Error('Pod status unavailable.');
  return { ...value, can_start: value.can_start === true, can_stop: value.can_stop === true };
}

export function podCanConnect(status) {
  return status?.available === false || (status?.available === true && status.state === 'ready');
}

export function podNotice(status) {
  if (!status || status.available === false) return '';
  if (status.state === 'stopped' && typeof status.last_error === 'string' && status.last_error.trim()) return status.last_error;
  return { stopped: 'Start the pod to begin.', starting: 'Starting the GPU…', loading: 'Loading Cosmos…',
    ready: '', stopping: 'Stopping the GPU…', error: status.message || 'Pod unavailable.' }[status.state] || '';
}

export function podControlState(status, pending = false) {
  return { startHidden: status?.can_stop === true, stopHidden: status?.can_stop !== true,
    startDisabled: pending || status?.can_start !== true, stopDisabled: pending || status?.can_stop !== true };
}

export function podBudgetLabel(status) {
  if (!status?.available) return '';
  const parts = [];
  if (Number.isFinite(status.rate_usd_per_hour) && status.rate_usd_per_hour >= 0) parts.push(`$${status.rate_usd_per_hour.toFixed(2)}/hr while running`);
  if (Number.isFinite(status.session_limit_minutes) && status.session_limit_minutes > 0) parts.push(`${status.session_limit_minutes} min session cap`);
  if (status.state === 'stopped') parts.push('Storage charges continue');
  else if (status.remote_guard_enabled === false) parts.push('Keep this Mac online');
  return parts.join(' · ');
}

export async function postPodAction(action, fetcher = fetch) {
  if (!['start', 'stop'].includes(action)) throw new Error('Unknown pod action.');
  const response = await fetcher(`/api/pod/${action}`, { method: 'POST', credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json', 'X-CholeScope-Request': '1' }, body: '{}' });
  let payload;
  try { payload = await response.json(); } catch { throw new Error('The pod request could not be confirmed. Refresh status before retrying.'); }
  if (!response.ok) throw new Error(payload.message || payload.detail || 'The pod request failed.');
  return payload;
}

export function createPodControls({ onChange = () => {} } = {}) {
  const root = document.getElementById('pod-controls');
  const label = document.getElementById('pod-state');
  const caption = document.getElementById('pod-caption');
  const start = document.getElementById('pod-start');
  const stop = document.getElementById('pod-stop');
  let current = null;
  let polling = false;
  let pending = false;
  let leaving = false;
  let error = '';
  let refreshError = '';
  function render() {
    if (!root) return;
    root.hidden = current?.available === false;
    root.dataset.state = current?.state || 'checking';
    label.textContent = { stopped: 'Pod stopped', starting: 'Starting…', loading: 'Loading…', ready: 'Pod running', stopping: 'Stopping…', error: 'Needs attention' }[current?.state] || 'Checking pod…';
    const controls = podControlState(current, pending);
    start.hidden = controls.startHidden;
    stop.hidden = controls.stopHidden;
    start.disabled = controls.startDisabled;
    stop.disabled = controls.stopDisabled;
    start.textContent = pending && current?.state !== 'stopping' ? 'Starting…' : 'Start pod';
    stop.textContent = pending && current?.state === 'stopping' ? 'Stopping…' : 'Stop pod';
    caption.textContent = error || refreshError || (current?.state === 'error' ? current.message || 'Pod unavailable.' : podBudgetLabel(current));
    caption.classList.toggle('error', Boolean(error || refreshError || current?.state === 'error'));
  }
  function update(value) {
    current = normalizePodStatus(value);
    render();
    onChange(current);
  }
  async function refresh() {
    if (polling || leaving) return current;
    polling = true;
    try {
      const response = await fetch('/api/pod/status', { cache: 'no-store', credentials: 'same-origin' });
      if (response.status === 404 || response.status === 405) { refreshError = ''; update({ available: false }); }
      else {
        if (!response.ok) throw new Error('Pod status unavailable.');
        const status = normalizePodStatus(await response.json());
        refreshError = '';
        update(status);
      }
    } catch {
      if (!current) update({ available: true, state: 'error', message: 'Local launcher unavailable. Reopen CholeScope.', can_start: false, can_stop: false });
      else { refreshError = 'Could not refresh pod status.'; render(); }
    } finally { polling = false; }
    return current;
  }
  async function act(action) {
    if (pending || !current?.[`can_${action}`]) return;
    pending = true; error = ''; refreshError = '';
    const previous = current;
    update({ ...current, state: action === 'start' ? 'starting' : 'stopping', can_start: false, can_stop: action === 'stop' });
    try {
      const result = await postPodAction(action);
      if (result.available === true) update(result);
      else await refresh();
    } catch (reason) {
      error = reason.message; update(previous);
      await refresh();
    } finally { pending = false; render(); }
  }
  start?.addEventListener('click', () => void act('start'));
  stop?.addEventListener('click', () => void act('stop'));
  const ready = refresh();
  const timer = setInterval(refresh, 4000);
  window.addEventListener('pagehide', () => { leaving = true; clearInterval(timer); });
  return { ready, refresh, getState: () => current };
}
