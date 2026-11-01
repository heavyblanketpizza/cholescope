export const MOTION_KEYS = new Set([
  'w', 'a', 's', 'd', 'q', 'e', 'arrowup', 'arrowdown', 'arrowleft',
  'arrowright', 'z', 'x', 'f', 'v', 'shift',
]);

export function isEditable(target) {
  return Boolean(target?.closest?.('input, textarea, select, button, [contenteditable="true"]'));
}

/** A small input boundary: key-up always wins, even after focus changes. */
export class KeyboardControls {
  constructor({ target, visibilityTarget, onChange = () => {}, onArm = () => {}, onPause = () => {} }) {
    this.target = target;
    this.visibilityTarget = visibilityTarget;
    this.onChange = onChange;
    this.onArm = onArm;
    this.onPause = onPause;
    this.keys = new Set();
    this.arm = 'left';
    this.enabled = false;
    this.listeners = {
      keydown: (event) => this.keydown(event),
      keyup: (event) => this.keyup(event),
      blur: () => this.clear(),
    };
    for (const [name, handler] of Object.entries(this.listeners)) target.addEventListener(name, handler);
    this.onVisibility = () => { if (visibilityTarget?.hidden) this.clear(); };
    visibilityTarget?.addEventListener('visibilitychange', this.onVisibility);
  }

  snapshot() { return { keys: [...this.keys].sort(), selected_arm: this.arm }; }
  emit() { this.onChange(this.snapshot()); }

  keydown(event) {
    const key = event.key.toLowerCase();
    if (event.metaKey || event.ctrlKey || event.altKey) { this.clear(); return; }
    if (isEditable(event.target)) return;
    if (key === '1' || key === '2') {
      event.preventDefault();
      if (!event.repeat) this.setArm(key === '1' ? 'left' : 'right');
      return;
    }
    if (!this.enabled) return;
    if (key === ' ') {
      event.preventDefault();
      if (!event.repeat) { this.clear(); this.onPause(); }
      return;
    }
    if (!MOTION_KEYS.has(key)) return;
    event.preventDefault();
    if (!this.keys.has(key)) { this.keys.add(key); this.emit(); }
  }

  keyup(event) {
    const key = event.key.toLowerCase();
    if (this.keys.delete(key)) { event.preventDefault(); this.emit(); }
  }

  clear() {
    const changed = this.keys.size > 0;
    this.keys.clear();
    if (changed) this.emit();
  }

  setArm(arm) {
    if (!['left', 'right'].includes(arm) || arm === this.arm) return;
    this.clear();
    this.arm = arm;
    this.onArm(arm);
    this.emit();
  }

  setEnabled(enabled) {
    this.enabled = Boolean(enabled);
    if (!this.enabled) this.clear();
  }

  destroy() {
    this.clear();
    for (const [name, handler] of Object.entries(this.listeners)) this.target.removeEventListener(name, handler);
    this.visibilityTarget?.removeEventListener('visibilitychange', this.onVisibility);
  }
}
