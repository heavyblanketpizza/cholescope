const STORAGE_KEY = 'cholescope-control-token';

// sessionStorage keeps ownership with this tab across reloads and mode changes.
export function createControlRecovery({ button, onChange = () => {}, onClaim = () => {},
  fetcher = fetch, storage = () => sessionStorage } = {}) {
  let token = '';
  try { token = storage().getItem(STORAGE_KEY) || ''; } catch { /* Storage may be disabled. */ }
  let status = null;
  let pending = false;
  let error = '';
  let refreshId = 0;
  const query = () => token ? `?control_token=${encodeURIComponent(token)}` : '';

  async function refresh() {
    const id = ++refreshId;
    try {
      const response = await fetcher(`/api/control/status${query()}`, { cache: 'no-store', credentials: 'same-origin' });
      if (id !== refreshId) return;
      if (response.status === 404) { status = null; error = ''; return; }
      if (!response.ok) return;
      const value = await response.json();
      if (id === refreshId && value.supported === true) status = value;
    } catch { /* A connection retry will refresh again. */ }
    finally { if (id === refreshId) onChange(); }
  }

  async function claim() {
    if (pending) return;
    pending = true; error = ''; ++refreshId; onChange();
    try {
      const response = await fetcher('/api/control/takeover', { method: 'POST', credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json', 'X-CholeScope-Request': '1' }, body: '{}' });
      const value = await response.json();
      if (!response.ok || typeof value.control_token !== 'string' || !value.control_token) {
        throw new Error('Could not take control. Try again.');
      }
      ++refreshId;
      token = value.control_token;
      try { storage().setItem(STORAGE_KEY, token); } catch { /* Keep the claim in memory. */ }
      status = { supported: true, occupied: false, requires_claim: false };
      onClaim();
    } catch {
      error = 'Could not take control. Try again.';
    } finally { pending = false; onChange(); }
  }

  button?.addEventListener('click', () => void claim());
  return {
    refresh, claim,
    get query() { return query(); },
    get requiresClaim() { return status?.requires_claim === true; },
    render({ connected, available }) {
      const blocked = status?.supported === true && (status.occupied || status.requires_claim);
      if (button) {
        button.hidden = connected || !available || !(blocked || pending || error);
        button.disabled = pending;
        button.textContent = pending ? 'Connecting…' : 'Use this tab';
      }
      if (connected || !available) return '';
      if (pending) return 'Taking control of the model…';
      return error || (blocked ? 'Another tab has control. Choose Use this tab to switch here.' : '');
    },
  };
}
