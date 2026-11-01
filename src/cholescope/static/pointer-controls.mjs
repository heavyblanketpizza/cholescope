const ZERO = () => [0, 0, 0];

/** Relative input intent only: this class does not locate or track instruments. */
export class PointerControls {
  constructor({ target, blurTarget, visibilityTarget, onFeedback = () => {}, onRelease = () => true, onSelectArm = () => {}, sensitivity = .5, maxDelta = .08 }) {
    if (![sensitivity, maxDelta].every((value) => Number.isFinite(value) && value > 0)) throw new Error('Pointer scales must be finite and positive');
    this.target = target;
    this.blurTarget = blurTarget;
    this.visibilityTarget = visibilityTarget;
    this.onFeedback = onFeedback;
    this.onRelease = onRelease;
    this.onSelectArm = onSelectArm;
    this.sensitivity = sensitivity;
    this.maxDelta = maxDelta;
    this.enabled = false;
    this.active = false;
    this.pointerId = null;
    this.pending = ZERO();
    this.start = null;
    this.current = null;
    this.depth = 0;
    this.listeners = {
      pointerdown: (event) => this.pointerdown(event),
      pointermove: (event) => this.pointermove(event),
      pointerup: (event) => this.endPointer(event, 'pointerup'),
      pointercancel: (event) => this.endPointer(event, 'cancel'),
      lostpointercapture: (event) => this.endPointer(event, 'capture-lost'),
      wheel: (event) => this.wheel(event),
    };
    for (const [name, handler] of Object.entries(this.listeners)) target.addEventListener(name, handler, name === 'wheel' ? { passive: false } : undefined);
    this.onBlur = () => this.clear('blur');
    this.onShortcut = (event) => { if (event.ctrlKey || event.metaKey || event.altKey) this.clear('shortcut'); };
    this.onVisibility = () => { if (visibilityTarget?.hidden) this.clear('hidden'); };
    blurTarget?.addEventListener('blur', this.onBlur);
    blurTarget?.addEventListener('keydown', this.onShortcut);
    visibilityTarget?.addEventListener('visibilitychange', this.onVisibility);
  }

  feedback() {
    this.onFeedback({ active: this.active, start: this.start, current: this.current, depth: this.depth });
  }

  pointerdown(event) {
    if (!this.enabled || this.active || event.isPrimary === false || event.button !== 0) return;
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (![event.clientX, event.clientY].every(Number.isFinite)) return;
    const handle = event.target?.closest?.('[data-drag-arm]');
    if (!handle && event.target?.closest?.('button, input, select, a, [data-pointer-ignore]')) return;
    const rect = this.target.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    if (handle) this.onSelectArm(handle.dataset.dragArm);
    event.preventDefault();
    this.target.focus?.({ preventScroll: true });
    this.pointerId = event.pointerId;
    this.active = true;
    this.pending = ZERO();
    this.depth = 0;
    this.lastClient = [event.clientX, event.clientY];
    // One spatial scale for both axes, fixed for the duration of this drag.
    this.inputScale = this.sensitivity / rect.width;
    this.start = [event.clientX - rect.left, event.clientY - rect.top];
    this.current = [...this.start];
    try { this.target.setPointerCapture(event.pointerId); }
    catch { this.clear('capture-failed'); return; }
    this.feedback();
  }

  pointermove(event) {
    if (!this.active || event.pointerId !== this.pointerId) return;
    if (event.ctrlKey || event.metaKey || event.altKey) { this.clear('shortcut'); return; }
    if (event.pointerType !== 'touch' && (event.buttons & 1) === 0) { this.clear('buttons-released'); return; }
    this.moveTo(event);
  }

  moveTo(event) {
    if (![event.clientX, event.clientY].every(Number.isFinite)) return;
    const rect = this.target.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) { this.clear('viewport-hidden'); return; }
    event.preventDefault();
    const scale = this.inputScale * (event.shiftKey ? .25 : 1);
    // NVIDIA's model frame uses +X right, +Y down, +Z forward.
    this.pending[0] += (event.clientX - this.lastClient[0]) * scale;
    this.pending[1] += (event.clientY - this.lastClient[1]) * scale;
    this.lastClient = [event.clientX, event.clientY];
    this.current = [event.clientX - rect.left, event.clientY - rect.top];
    this.feedback();
  }

  wheel(event) {
    if (!this.enabled || !this.active || !Number.isFinite(event.deltaY)) return;
    if (event.ctrlKey || event.metaKey || event.altKey) { this.clear('shortcut'); return; }
    event.preventDefault();
    const rect = this.target.getBoundingClientRect();
    const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? rect.height : 1;
    const delta = -event.deltaY * unit * .00025 * (event.shiftKey ? .25 : 1);
    this.pending[2] += delta;
    this.depth += delta;
    this.feedback();
  }

  /** Called only by the 100 ms heartbeat; failed sends retain unsent intent. */
  flush(send) {
    return this.sendPending(send);
  }

  sendPending(send, release = false) {
    // Packet size bounds transport, not movement. Send the entire displacement
    // now instead of dropping overflow or replaying it on future heartbeats.
    const count = Math.max(1, Math.ceil(Math.hypot(...this.pending) / this.maxDelta));
    for (let remaining = count; remaining > 0; remaining--) {
      const delta = this.pending.map((value) => value / remaining);
      const fields = release
        ? { pointer_active: false, pointer_commit: true, pointer_delta: delta, reason: 'pointerup' }
        : { pointer_active: this.active, pointer_delta: delta };
      if (!send(fields)) return false;
      this.pending = remaining === 1 ? ZERO() : this.pending.map((value, axis) => value - delta[axis]);
    }
    return true;
  }

  endPointer(event, reason) {
    if (event.pointerId !== this.pointerId) return;
    if (reason === 'pointerup') {
      if (event.ctrlKey || event.metaKey || event.altKey) { this.clear('shortcut'); return; }
      // Browsers may report the final coordinate on up without a preceding move.
      this.moveTo(event);
    }
    this.clear(reason);
  }

  clear(reason = 'cancel') {
    const pointerId = this.pointerId;
    const hadInput = this.active || this.pending.some((value) => value !== 0);
    const commit = reason === 'pointerup' && this.active;
    if (commit) this.sendPending(this.onRelease, true);
    this.active = false;
    this.pointerId = null;
    this.pending = ZERO();
    this.start = null;
    this.current = null;
    this.depth = 0;
    if (pointerId !== null) {
      try { if (this.target.hasPointerCapture?.(pointerId)) this.target.releasePointerCapture(pointerId); }
      catch { /* Capture may already have ended when focus or the tab changed. */ }
    }
    if (hadInput) {
      // Cancellation discards only unsent motion; accepted poses already hold.
      if (!commit) this.onRelease({ pointer_active: false, pointer_commit: false, pointer_delta: ZERO(), reason });
      this.feedback();
    }
  }

  setEnabled(enabled) {
    this.enabled = Boolean(enabled);
    if (!this.enabled) this.clear('disabled');
  }

  destroy() {
    this.clear('destroy');
    for (const [name, handler] of Object.entries(this.listeners)) this.target.removeEventListener(name, handler);
    this.blurTarget?.removeEventListener('blur', this.onBlur);
    this.blurTarget?.removeEventListener('keydown', this.onShortcut);
    this.visibilityTarget?.removeEventListener('visibilitychange', this.onVisibility);
  }
}
