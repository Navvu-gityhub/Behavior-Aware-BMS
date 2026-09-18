/**
 * Theme state for the BEACON dashboard.
 *
 * Dark is the design's native mode and stays the default: every token in
 * `beacon.css` holds its original dark value, and `[data-theme="light"]`
 * overrides them. An unset or unreadable preference therefore renders exactly
 * the dashboard the design specifies.
 *
 * The preference is a per-viewer convenience, so `localStorage` is the right
 * home for it — but it throws in a private window and returns null with site
 * data cleared, so every access is guarded and the failure path is "dark".
 */

const KEY = 'beacon.theme';
const THEMES = ['dark', 'light'];

function stored() {
  try {
    const v = localStorage.getItem(KEY);
    return THEMES.includes(v) ? v : null;
  } catch {
    return null;
  }
}

function persist(theme) {
  try {
    localStorage.setItem(KEY, theme);
  } catch {
    // A viewer who cannot store a preference still gets a working toggle for
    // this session; it just will not survive a reload.
  }
}

/** Write the theme onto <html>, which is what the CSS selector matches. */
export function applyTheme(theme) {
  const next = THEMES.includes(theme) ? theme : 'dark';
  document.documentElement.dataset.theme = next;
  return next;
}

export function currentTheme() {
  return document.documentElement.dataset.theme === 'light' ? 'light' : 'dark';
}

/**
 * Apply the stored preference.
 *
 * A first-time viewer gets dark, not their OS setting. Dark is the mode this
 * dashboard was designed in - it is what the battery render, the glass
 * surfaces and the accent palette were tuned against - so an unvisited page
 * shows the design as specified. The toggle is one click away in the dock for
 * anyone who wants otherwise, and their choice is what persists.
 *
 * Call before the shell is written so the page never paints dark and then
 * flips.
 */
export function initTheme() {
  return applyTheme(stored() || 'dark');
}

export function toggleTheme() {
  const next = currentTheme() === 'light' ? 'dark' : 'light';
  applyTheme(next);
  persist(next);
  return next;
}

/**
 * Bind the dock's toggle button.
 *
 * Returns a teardown function. The button lives in the shell markup, which is
 * rewritten whenever the payload is re-rendered, so the caller re-binds rather
 * than assuming the node survives.
 */
export function bindThemeToggle(root = document) {
  const btn = root.querySelector('#themeToggle');
  if (!btn) return () => {};

  const sync = () => {
    const theme = currentTheme();
    btn.setAttribute('aria-pressed', String(theme === 'light'));
    btn.title = theme === 'light' ? 'Switch to dark theme' : 'Switch to light theme';
  };

  const onClick = () => {
    toggleTheme();
    sync();
  };

  btn.addEventListener('click', onClick);
  sync();
  return () => btn.removeEventListener('click', onClick);
}
