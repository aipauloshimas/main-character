/* Main Character Studio: the page.

   It talks to the studio server on this computer and to nothing else (no libraries, no network). The
   project file is the one place the state lives: every edit is a merge patch (RFC 7386) that is sent
   to /api/project after 400 ms, and the card is drawn by the server, so what the page shows is what
   the render draws. Claude can change the project file while the page is open (the answer to "Ask
   Claude" does), so the page never trusts its own copy for longer than an edit takes to come back. */
(() => {
'use strict';

const DEBOUNCE_MS = 400;
const PLAN_WIDTH = 1080; // the width the card is drawn at, whatever the clip's shape (config.PLAN_WIDTH): its height is the clip's shape
const NOTE_KEYS = ['tl', 'tr', 'bl', 'br'];
const FRAME_STEP = 3; // the filmstrip shows every third frame
const CARD_KEYS = ['copy', 'positions', 'freeze', 'matte', 'style']; // what a patch has to touch to change the card
const GALLERY_KEYS = ['copy', 'positions', 'freeze', 'matte']; // and the cards of the gallery (not the style: each draws its own)
const FLAG_TEXT = "This note doesn't fit here. Shorten it or drag it somewhere else.";
const HANDLE_NAMES = { tl: 'Top left note', tr: 'Top right note', bl: 'Bottom left note', br: 'Bottom right note', burst: 'Name burst' };
const ROLE_NAMES = { dark: 'Background dark', light: 'Background light', tex: 'Texture', edge: 'Edge', ink: 'Text' };
const TIMING_ROWS = [
  { key: 'hold', label: 'Hold', unit: ' s', step: 0.1, digits: 1, hint: 'How long the card stays on screen.' },
  { key: 'head_speed', label: 'Head speed', unit: 'x', step: 0.05, digits: 2,
    hint: 'How fast the clip plays before the freeze. 1x is normal speed.' },
  { key: 'tail', label: 'Tail', unit: ' s', step: 0.1, digits: 1,
    hint: 'How much of the clip plays after the card. All the way right plays it to the end.' },
  { key: 'release_from', label: 'Release from', unit: ' s', step: 0.05, digits: 2,
    hint: 'Skips ahead after the freeze frame, up to 1 s, before the clip goes on.' },
];
const TEXT_FIELDS = {
  'f-tl': { read: (c) => c.notes.tl, write: (v) => ({ notes: { tl: v } }), limit: 'tl' },
  'f-tr': { read: (c) => c.notes.tr, write: (v) => ({ notes: { tr: v } }), limit: 'tr' },
  'f-bl': { read: (c) => c.notes.bl, write: (v) => ({ notes: { bl: v } }), limit: 'bl' },
  'f-br': { read: (c) => c.notes.br, write: (v) => ({ notes: { br: v } }), limit: 'br' },
  'f-name': { read: (c) => c.name, write: (v) => ({ name: v }), limit: 'name' },
  'f-bubble': { read: (c) => (c.extras || {}).bubble, write: (v) => ({ extras: { bubble: v.trim() === '' ? null : v } }), limit: 'bubble' },
};

/* ---------- small helpers ---------- */

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** A new element: props are attributes (class, text, and on<event> handlers too), children are nodes or text. */
function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(props)) {
    if (value === null || value === undefined || value === false) continue;
    if (name === 'class') node.className = value;
    else if (name === 'text') node.textContent = value;
    else if (name.startsWith('on') && typeof value === 'function') node.addEventListener(name.slice(2), value);
    else node.setAttribute(name, value === true ? '' : String(value));
  }
  for (const child of children.flat()) {
    if (child !== null && child !== undefined && child !== false) node.append(child);
  }
  return node;
}

/* merge-patch:begin (the pieces between these marks run without a page: a test runs them in node) */

const clone = (value) => JSON.parse(JSON.stringify(value));
const isObject = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);

/** The target with a merge patch applied (RFC 7386: a null deletes the key). Neither is changed. */
function applyPatch(target, patch) {
  if (!isObject(patch)) return clone(patch);
  const out = isObject(target) ? clone(target) : {};
  for (const [key, value] of Object.entries(patch)) {
    if (value === null) delete out[key];
    else out[key] = applyPatch(out[key], value);
  }
  return out;
}

/** Two patches, one after the other, as one patch. The later wins, and its nulls stay. */
function joinPatches(first, second) {
  const out = clone(first);
  for (const [key, value] of Object.entries(second)) {
    out[key] = isObject(value) && isObject(out[key]) ? joinPatches(out[key], value) : clone(value);
  }
  return out;
}

/* merge-patch:end */

const characters = (text) => Array.from(text).length;
const assetUrl = (path) => '/' + path.split('/').map(encodeURIComponent).join('/');

/* geometry:begin (the pieces between these marks run without a page: a test runs them in node) */

/* The notes sit on the card, and the card is the whole stage the layout gives, [0, 0, w, h] in px of the plan (1080 wide,
   as tall as the clip's shape makes it). So a box is placed by its own x, y, w and h alone, in shares of the plan, and a
   drag counts plan px. Nothing is taken off for a canvas around the stage, because there is none. */

/** The style of a handle for a box [x, y, w, h] of the plan: shares of the card's width and height, so that it sits on the
    card whatever size the preview is shown at. */
function boxStyle([x, y, w, h], [, , stageW, stageH]) {
  return {
    left: `${(x / stageW) * 100}%`, top: `${(y / stageH) * 100}%`,
    width: `${(w / stageW) * 100}%`, height: `${(h / stageH) * 100}%`,
  };
}

/** Plan px for each px shown: the plan's width over the width of the preview on the page. */
const planPerShown = ([, , stageW], shownWidth) => stageW / shownWidth;

/** A box [x, y, w, h] moved by (dx, dy) plan px, kept on the card: its new x and y. */
function movedBox([x, y, w, h], dx, dy, [, , stageW, stageH]) {
  return [clamp(x + dx, 0, Math.max(0, stageW - w)), clamp(y + dy, 0, Math.max(0, stageH - h))];
}

/** The patch that pins a note: the top left corner of its box, in whole px of the plan. */
const pinPatch = (key, x, y) => ({ positions: { [key]: [Math.round(x), Math.round(y)] } });

/* geometry:end */

/* ---------- the studio server ---------- */

/* api:begin (a test runs these in node, with a stand-in for fetch) */

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

const NO_SERVER = "Can't reach the studio. Is the window that runs it still open?";

async function errorOf(response) {
  let data = null;
  try { data = await response.json(); } catch { /* not JSON */ }
  return new ApiError((data && data.error) || `The studio answered ${response.status}.`, response.status);
}

/** Call the server and return what it answered as JSON. body is a Blob/File (a picture or a sound: sent as it is, with the
    content type the caller gives it), or else an object, which is sent as JSON, and so is {} when a request that is not a GET
    has none: the server takes a route that wants JSON only as application/json, so that a web page on another site cannot
    post to it without the preflight that the server refuses. */
async function api(method, path, body, headers = {}) {
  const init = { method, headers: { ...headers } };
  if (body instanceof Blob) {
    init.body = body;
  } else if (method !== 'GET') {
    init.body = JSON.stringify(body ?? {});
    init.headers['Content-Type'] = 'application/json';
  }
  let response;
  try { response = await fetch(path, init); } catch { throw new ApiError(NO_SERVER, 0); }
  if (!response.ok) throw await errorOf(response);
  return response.json();
}

/* api:end */

/* ---------- state ---------- */

const state = {
  project: null, // as the page has it: the server's last answer with the edits that have not come back yet
  options: null, // the tables the controls are built from (GET /api/options)
  styles: [],
  layout: null, // {stage, elements, flags} of the card, in px of the plan; the stage is the whole card, [0, 0, w, h]
  frames: [], // [first, last] ranges of the frames that can be frozen on (they have a picture and a cutout)
  contrast: null,
  rev: 0, // a new rev is a new card picture
  galleryRev: 0,
  mode: 'still',
  rendering: false,
  previewReady: false, // a preview was made in this visit
  last: null, // {notes} of the last render in this visit
  told: new Set(), // the warnings the person has been told about
};

/* ---------- toasts, errors, the busy screen ---------- */

function toast(message, kind = '') {
  const item = el('div', { class: 'toast ' + kind, text: message });
  $('#toasts').append(item);
  requestAnimationFrame(() => item.classList.add('on'));
  setTimeout(() => {
    item.classList.remove('on');
    setTimeout(() => item.remove(), 300);
  }, 4500);
}

/** What a warning of the server calls a field of the copy, as the person's own label for it. */
const FIELD_NAMES = {
  'copy.notes.tl': 'What you do', 'copy.notes.tr': 'Handle', 'copy.notes.bl': "Your 'but…'", 'copy.notes.br': 'Oddly specific detail',
  'copy.name': 'Name on screen',
  'copy.extras.burst': 'Burst word', 'copy.extras.emphasis': 'Emphasis word', 'copy.extras.bubble': 'Bubble text',
};

/** A warning of the server in the person's words. The ones about a version (versions.C.notes.tl...) say again what the
    ones about the live copy say, because an edit goes to both, so they are left out. */
function wording(warning) {
  if (warning.includes('versions.')) return null;
  let text = warning.replace(/copy\.extras\.stats\[(\d)\]/g, (_, index) => `Stat ${Number(index) + 1}`);
  for (const [path, name] of Object.entries(FIELD_NAMES)) text = text.split(path).join(name);
  return text;
}

/** Warnings from the server, as toasts. A warning is told once while it lasts (the numbers in it do not make it a new one).
    The one about the text contrast is not: the contrast line of the Style tab says it, all the time and in the same place. */
function showWarnings(warnings) {
  const now = new Set();
  for (const warning of (warnings || []).filter((line) => !line.startsWith('Text contrast ')).map(wording)) {
    if (warning === null) continue;
    now.add(warningKey(warning));
    if (!state.told.has(warningKey(warning))) toast(warning);
  }
  state.told = now;
}

const warningKey = (warning) => warning.replace(/\d+(\.\d+)?/g, '#');

function showError(message) {
  $('#error-text').textContent = message;
  $('#error').hidden = false;
}

function clearError() {
  $('#error').hidden = true;
}

function setBusy(text) {
  $('#busy').hidden = !text;
  if (text) $('#busy-text').textContent = text;
}

/* ---------- edits: one patch at a time, after a pause ---------- */

const queue = { patch: null, timer: null, running: Promise.resolve(), cardChanged: false, galleryChanged: false };

/** Change the project: now on the page, and on the server after a pause (or at once). */
function edit(patch, { now = false } = {}) {
  state.project = applyPatch(state.project, patch);
  queue.patch = queue.patch ? joinPatches(queue.patch, patch) : clone(patch);
  const keys = Object.keys(patch);
  queue.cardChanged ||= keys.some((key) => CARD_KEYS.includes(key));
  queue.galleryChanged ||= keys.some((key) => GALLERY_KEYS.includes(key));
  clearTimeout(queue.timer);
  queue.timer = setTimeout(flush, now ? 0 : DEBOUNCE_MS);
  paintLive();
}

/** Send what is waiting. Resolves when the server has answered (so a render can start from what is on the disk). */
function flush() {
  clearTimeout(queue.timer);
  queue.running = queue.running.then(send);
  return queue.running;
}

async function send() {
  if (!queue.patch) return;
  const patch = queue.patch;
  const { cardChanged, galleryChanged } = queue;
  queue.patch = null;
  queue.cardChanged = queue.galleryChanged = false;
  try {
    const answer = await api('POST', '/api/project', patch);
    clearError();
    state.contrast = answer.contrast;
    if (!queue.patch) state.project = answer.project; // (if the person has typed since, their text is newer)
    showWarnings(answer.warnings);
    if (galleryChanged) bumpGallery();
    if (cardChanged) bumpCard();
    paintAll({ keepFocus: true });
  } catch (error) {
    showError(error.message);
    queue.patch = null;
    try { await reload(); } catch { /* the error above stays */ }
  }
}

/** Read the project from the server again (and the contrast of its style), and put it on the page. */
async function reload() {
  const answer = await api('POST', '/api/project', {});
  state.project = answer.project;
  state.contrast = answer.contrast;
  bumpCard();
  bumpGallery();
  paintAll({ keepFocus: false });
  return answer;
}

window.addEventListener('pagehide', () => {
  if (queue.patch) { // an edit that has not been sent yet is not lost when the page is closed
    fetch('/api/project', { method: 'POST', body: JSON.stringify(queue.patch), keepalive: true, headers: { 'Content-Type': 'application/json' } });
  }
});

/* ---------- the card and its drag handles ---------- */

const cardState = { busy: false, again: false, url: null, error: null };

function bumpCard() {
  state.rev += 1;
  refreshCard();
}

function cardWidth() {
  const pixels = $('#preview').clientWidth * (window.devicePixelRatio || 1);
  return clamp(Math.ceil(pixels / 90) * 90, 180, PLAN_WIDTH);
}

/** Draw the card again. One request at a time: an edit that comes while one is drawing is drawn after it. */
function refreshCard() {
  if (cardState.busy) { cardState.again = true; return; }
  runCard();
}

async function runCard() {
  cardState.busy = true;
  cardState.again = false;
  paintNote();
  $('#preview').classList.add('updating-card');
  $('#updating').hidden = false;
  try {
    let response;
    try { response = await fetch(`/api/card.png?w=${cardWidth()}&r=${state.rev}`); } catch { throw new ApiError(NO_SERVER, 0); }
    if (!response.ok) throw await errorOf(response);
    const url = URL.createObjectURL(await response.blob());
    $('#card-img').src = url;
    if (cardState.url) URL.revokeObjectURL(cardState.url);
    cardState.url = url;
    cardState.error = null;
    paintNote();
    await loadLayout();
  } catch (error) {
    cardState.error = error.message;
    paintNote();
  } finally {
    cardState.busy = false;
    if (cardState.again) {
      runCard();
    } else {
      $('#preview').classList.remove('updating-card');
      $('#updating').hidden = true;
    }
  }
}

/** The note over the preview: why the card is not there, or that there is no preview video yet. */
function paintNote() {
  const note = $('#preview-note');
  let text = '';
  if (cardState.error) text = cardState.error;
  else if (!cardState.url) text = 'Drawing the card…';
  else if (state.mode === 'play' && !state.previewReady) text = 'No preview yet. Press Preview video to make one.';
  note.textContent = text;
  note.hidden = !text;
  note.classList.toggle('error-note', Boolean(cardState.error));
}

async function loadLayout() {
  try { state.layout = await api('GET', '/api/layout'); } catch { state.layout = null; }
  renderHandles();
}

/** Put a handle on the card: its box is in px of the plan, and the card is the whole preview. */
function placeHandle(handle, x, y, w, h) {
  Object.assign(handle.style, boxStyle([x, y, w, h], state.layout.stage));
  handle.box = [x, y, w, h];
}

function renderHandles() {
  const layer = $('#handles');
  if ($('.handle.dragging', layer)) return; // a note is being dragged: the drop draws everything again
  const focused = document.activeElement && layer.contains(document.activeElement) ? document.activeElement.dataset.key : null;
  layer.replaceChildren();
  const layout = state.layout;
  $('#flag-note').hidden = !(layout && Object.keys(layout.flags).length);
  if (!layout) return;
  for (const [key, box] of Object.entries(layout.elements)) {
    const flag = layout.flags[key];
    const name = key === 'tr' && state.project.style.hero === 'stats' ? 'Stats panel' : HANDLE_NAMES[key] || key;
    const handle = el('div', {
      class: 'handle' + (flag ? ' flag' : ''), 'data-key': key, tabindex: 0, role: 'button',
      title: flag ? FLAG_TEXT : `${name}. Drag it to move it.`,
      'aria-label': flag ? `${name}. ${FLAG_TEXT}` : `${name}. Drag it, or use the arrow keys, to move it.`,
    });
    placeHandle(handle, ...box);
    handle.addEventListener('pointerdown', (event) => startDrag(event, handle, key));
    handle.addEventListener('keydown', (event) => nudge(event, handle, key));
    layer.append(handle);
    if (key === focused) handle.focus(); // (the arrow keys go on moving the note that was being moved)
  }
}

/** Move a note to a place on the card: positions are the top left corner of its box, in px of the plan. */
function pin(key, x, y, now) {
  edit(pinPatch(key, x, y), { now });
}

function startDrag(event, handle, key) {
  if (event.button !== 0 || !state.layout) return;
  event.preventDefault();
  handle.setPointerCapture(event.pointerId);
  handle.classList.add('dragging');
  const stage = state.layout.stage;
  const scale = planPerShown(stage, $('#preview').getBoundingClientRect().width);
  const box = handle.box;
  const from = { x: event.clientX, y: event.clientY };
  let [x, y] = box;
  let moved = false;
  const move = (e) => {
    if (Math.abs(e.clientX - from.x) + Math.abs(e.clientY - from.y) > 3) moved = true;
    [x, y] = movedBox(box, (e.clientX - from.x) * scale, (e.clientY - from.y) * scale, stage);
    if (moved) placeHandle(handle, x, y, box[2], box[3]);
  };
  const end = () => {
    handle.removeEventListener('pointermove', move);
    handle.removeEventListener('pointerup', end);
    handle.removeEventListener('pointercancel', end);
    handle.classList.remove('dragging');
    if (moved) pin(key, x, y, true); // a drop posts the position and the card is drawn again
  };
  handle.addEventListener('pointermove', move);
  handle.addEventListener('pointerup', end);
  handle.addEventListener('pointercancel', end);
}

function nudge(event, handle, key) {
  const steps = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[event.key];
  if (!steps || !state.layout) return;
  event.preventDefault();
  const [, , w, h] = handle.box;
  const length = event.shiftKey ? 40 : 8;
  const [nx, ny] = movedBox(handle.box, steps[0] * length, steps[1] * length, state.layout.stage);
  placeHandle(handle, nx, ny, w, h);
  pin(key, nx, ny, false);
}

function paintReset() {
  const positions = state.project.positions || {};
  $('#btn-reset').disabled = !Object.values(positions).some((value) => value !== null && value !== undefined);
}

/* ---------- still and play ---------- */

function setMode(mode) {
  state.mode = mode;
  $('#preview').classList.toggle('play', mode === 'play');
  $('#preview-video').hidden = !(mode === 'play' && state.previewReady); // (no video, no player with nothing in it)
  for (const button of $$('#mode button')) {
    const on = button.dataset.mode === mode;
    button.classList.toggle('on', on);
    button.setAttribute('aria-pressed', String(on));
  }
  if (mode !== 'play') $('#preview-video').pause();
  paintNote();
}

/** Let go of a video file, so that the next render can replace it (Windows will not replace a file that is open). A preview
    replaces only preview.mp4, and a render only the files of the results. */
function releaseVideo(video) {
  video.pause();
  video.removeAttribute('src');
  video.load();
}

/* ---------- the freeze frame ---------- */

function frameCount() {
  return state.project.clip.frames || 0;
}

async function loadFrames() {
  try { state.frames = (await api('GET', '/api/frames')).ranges; } catch { state.frames = []; }
}

const hasFrame = (frame) => state.frames.some(([first, last]) => frame >= first && frame <= last);

/** The nearest frame to `frame`, in the direction of `step` (1 or -1), that can be frozen on; null when there is none. */
function nextFrame(frame, step) {
  for (let next = frame + step; next >= 0 && next < frameCount(); next += step) {
    if (hasFrame(next)) return next;
  }
  return null;
}

function buildFilmstrip() {
  const strip = $('#filmstrip');
  strip.replaceChildren();
  const { freeze } = state.project;
  const suggested = new Set((freeze.suggested || []).map((frame) => Math.floor(frame / FRAME_STEP) * FRAME_STEP));
  for (const [first, last] of state.frames) {
    for (let frame = Math.ceil(first / FRAME_STEP) * FRAME_STEP; frame <= last; frame += FRAME_STEP) {
      const picture = el('img', {
        src: `/api/frame/${frame}?w=120`, alt: '', loading: 'lazy', decoding: 'async', draggable: 'false',
      });
      const mark = suggested.has(frame) ? ' (suggested)' : '';
      const button = el('button', {
        type: 'button', class: 'frame' + (suggested.has(frame) ? ' suggested' : ''), 'data-frame': frame,
        title: `Frame ${frame}${mark}`, 'aria-label': `Frame ${frame}${mark}`,
      }, picture);
      button.addEventListener('click', () => setFreeze(frame));
      strip.append(button);
    }
  }
  if (!strip.children.length) strip.append(el('p', { class: 'muted', text: 'No frames to show yet. Run init on the clip again.' }));
  paintFreeze(true);
}

function setFreeze(frame) {
  const next = clamp(Math.round(frame), 0, Math.max(frameCount() - 1, 0));
  if (next === state.project.freeze.frame) return;
  edit({ freeze: { frame: next } });
  paintFreeze(true);
}

function paintFreeze(scroll) {
  const { freeze, clip } = state.project;
  const frame = freeze.frame;
  const seconds = clip.fps ? (frame / clip.fps).toFixed(2) : '?';
  $('#freeze-info').textContent = `Frame ${frame} of ${frameCount()} · ${seconds} s`;
  $('#freeze-info').title = `The card freezes on frame ${frame}, ${seconds} s into the clip. In Still mode, the left and right arrow keys step one frame.`;
  const here = Math.floor(frame / FRAME_STEP) * FRAME_STEP;
  for (const button of $$('#filmstrip .frame')) {
    const on = Number(button.dataset.frame) === here;
    button.classList.toggle('on', on);
    if (on) {
      button.setAttribute('aria-current', 'true');
      if (scroll) {
        const strip = $('#filmstrip');
        strip.scrollLeft = button.offsetLeft - (strip.clientWidth - button.offsetWidth) / 2;
      }
    } else {
      button.removeAttribute('aria-current');
    }
  }
}

document.addEventListener('keydown', (event) => {
  if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
  if (event.ctrlKey || event.metaKey || event.altKey || !state.project) return;
  if (state.mode === 'play') return; // scrubbing the preview must not move the freeze
  if (event.target instanceof Element && event.target.closest('input, textarea, select, [contenteditable], .handle, .tabs, video')) return;
  event.preventDefault();
  const next = nextFrame(state.project.freeze.frame, event.key === 'ArrowRight' ? 1 : -1);
  if (next !== null) setFreeze(next);
});

/* ---------- tabs ---------- */

function selectTab(name) {
  if (!$('#tab-' + name)) name = 'lines';
  for (const button of $$('.tabs [role=tab]')) {
    const on = button.dataset.tab === name;
    button.setAttribute('aria-selected', String(on));
    button.tabIndex = on ? 0 : -1;
  }
  for (const panel of $$('.panel')) panel.hidden = panel.id !== 'tab-' + name;
  $('.panels').scrollTop = 0;
  try { history.replaceState(null, '', '#' + name); } catch { /* not important */ }
  if (name === 'style' && state.project) loadThumbs();
}

function wireTabs() {
  const buttons = $$('.tabs [role=tab]');
  window.addEventListener('hashchange', () => selectTab(location.hash.slice(1) || 'lines'));
  for (const button of buttons) {
    button.addEventListener('click', () => selectTab(button.dataset.tab));
    button.addEventListener('keydown', (event) => {
      const at = buttons.indexOf(button);
      const next = { ArrowLeft: at - 1, ArrowRight: at + 1, Home: 0, End: buttons.length - 1 }[event.key];
      if (next === undefined) return;
      event.preventDefault();
      const target = buttons[(next + buttons.length) % buttons.length];
      selectTab(target.dataset.tab);
      target.focus();
    });
  }
}

/* ---------- Lines ---------- */

/** Change the live copy. */
function editCopy(change) {
  edit({ copy: change });
}

function paintLines({ keepFocus = true } = {}) {
  const { copy, style } = state.project;
  const limits = state.options.soft_limits;
  for (const [id, field] of Object.entries(TEXT_FIELDS)) {
    const input = $('#' + id);
    const value = field.read(copy) ?? '';
    if (!(keepFocus && document.activeElement === input) && input.value !== value) input.value = value;
    const count = $(`[data-count="${id}"]`);
    const length = characters(input.value);
    count.textContent = `${length}/${limits[field.limit]}`;
    count.classList.toggle('over', length > limits[field.limit]);
    input.setAttribute('aria-invalid', String(length > limits[field.limit]));
  }
  const extras = copy.extras || {};
  const burst = style.hero === 'burst';
  const emphasis = style.notes === 'hand';
  const stats = style.hero === 'stats';
  const bubble = Boolean(style.extras && style.extras.bubble);
  $('#field-burst').hidden = !burst;
  $('#field-emphasis').hidden = !(emphasis || (style.extras && style.extras.circle));
  $('#field-stats').hidden = !stats;
  $('#field-bubble').hidden = !bubble;
  $('#extras-fields').hidden = !(burst || emphasis || stats || bubble || (style.extras && style.extras.circle));
  $('#f-burst').placeholder = copy.name ? `${copy.name}!` : '';
  setValue('#f-burst', extras.burst ?? '', keepFocus);
  setValue('#f-emphasis', extras.emphasis ?? '', keepFocus);
  const rows = extras.stats || defaultStats(copy);
  rows.forEach(([label, value], index) => {
    setValue(`#f-stat${index}-label`, label, keepFocus);
    setValue(`#f-stat${index}-value`, String(value), keepFocus);
  });
}

/** The stats the card draws when the copy has none: the first words of three notes. */
function defaultStats(copy) {
  return [['tl', 9], ['bl', 1], ['br', 8]].map(([key, value]) => [(copy.notes[key] || '').slice(0, 14).trimEnd(), value]);
}

function setValue(selector, value, keepFocus) {
  const input = $(selector);
  if (keepFocus && document.activeElement === input) return;
  if (input.value !== value) input.value = value;
}

function wireLines() {
  for (const [id, field] of Object.entries(TEXT_FIELDS)) {
    const input = $('#' + id);
    input.addEventListener('input', () => editCopy(field.write(input.value, state.project.copy)));
    input.addEventListener('blur', () => { flush().then(() => paintLines({ keepFocus: false })); }); // (the server may have cleaned the text)
  }
  for (const [id, key] of [['f-burst', 'burst'], ['f-emphasis', 'emphasis']]) {
    const input = $('#' + id);
    input.addEventListener('input', () => editCopy({ extras: { [key]: input.value.trim() === '' ? null : input.value } }));
  }
  const readStats = () => [0, 1, 2].map((index) => {
    const raw = Number($(`#f-stat${index}-value`).value);
    return [$(`#f-stat${index}-label`).value, clamp(Number.isFinite(raw) ? Math.round(raw) : 0, 0, 10)];
  });
  for (const index of [0, 1, 2]) {
    for (const part of ['label', 'value']) {
      $(`#f-stat${index}-${part}`).addEventListener('input', () => editCopy({ extras: { stats: readStats() } }));
    }
  }
}

/* ---------- Style ---------- */

const thumbUrl = (id) => `/api/card.png?style=${encodeURIComponent(id)}&w=320&r=${state.galleryRev}`;
const thumbs = { stale: true, pass: 0 };

/** The cards of the gallery are out of date (the copy, a note or the frame changed). They are drawn again the
    next time the Style tab is open, and only if it is open: each one costs the server a second or so. */
function bumpGallery() {
  state.galleryRev += 1;
  thumbs.stale = true;
  if (!$('#tab-style').hidden) loadThumbs();
}

/** Draw the thumbnails of the gallery one after the other, each when the card in the preview is not being drawn,
    so that they never slow an edit down. */
async function loadThumbs() {
  if (!thumbs.stale) return;
  thumbs.stale = false;
  const pass = ++thumbs.pass;
  for (const card of $$('#styles-gallery .gcard[data-style]')) {
    while (cardState.busy) await sleep(150);
    if (pass !== thumbs.pass) return; // a newer pass has taken over
    card.classList.remove('broken');
    $('.gwait', card).textContent = 'Drawing…';
    const image = $('img', card);
    await new Promise((resolve) => {
      image.onload = image.onerror = resolve;
      image.src = thumbUrl(card.dataset.style);
    });
  }
}

function buildGallery() {
  const gallery = $('#styles-gallery');
  gallery.replaceChildren();
  for (const entry of state.styles) {
    const image = el('img', { alt: '', decoding: 'async', draggable: 'false' }); // (no src until loadThumbs draws it)
    const wait = el('span', { class: 'gwait', text: 'Drawing…' });
    const card = el('button', { type: 'button', class: 'gcard' + (entry.builtin ? '' : ' mine'), 'data-style': entry.id, 'aria-pressed': 'false' },
      el('span', { class: 'gthumb' }, wait, image),
      el('span', { class: 'gmeta' }, el('b', { text: entry.name }), el('span', { text: entry.builtin ? 'Ready-made' : 'Your style' })));
    image.addEventListener('error', () => { card.classList.add('broken'); wait.textContent = "Couldn't draw this one"; });
    card.addEventListener('click', () => edit({ style: clone(entry.recipe), style_source: entry.id }, { now: true }));
    gallery.append(card);
  }
  const create = el('button', { type: 'button', class: 'gcard create' },
    el('span', { class: 'plus', text: '+' }), el('b', { text: 'Create your own' }), el('span', { text: 'One option per layer, your colors' }));
  create.addEventListener('click', () => {
    const builder = $('#builder');
    builder.scrollIntoView({ behavior: 'smooth', block: 'start' });
    builder.classList.remove('flash');
    void builder.offsetWidth; // (so the flash can play again)
    builder.classList.add('flash');
  });
  gallery.append(create);
}

function buildStyleControls() {
  const { options, labels, palettes, roles } = state.options;
  for (const row of $$('#tab-style .opts[data-layer]')) {
    const layer = row.dataset.layer;
    for (const id of options[layer]) {
      const button = el('button', { type: 'button', class: `opt${layer === 'notes' ? ' f-' + id : ''}`, 'data-v': id, 'aria-pressed': 'false', text: labels[layer][id] });
      button.addEventListener('click', () => tweakStyle({ [layer]: id }, true));
      row.append(button);
    }
  }
  for (const id of options.extras) {
    const button = el('button', { type: 'button', class: 'tog', 'data-v': id, 'aria-pressed': 'false', text: labels.extras[id] });
    button.addEventListener('click', () => {
      const on = !state.project.style.extras[id];
      tweakStyle({ extras: { [id]: on } }, true);
    });
    $('#extras').append(button);
  }
  for (const role of roles) {
    const input = el('input', { type: 'color', id: `sw-${role}`, 'aria-label': ROLE_NAMES[role] });
    input.addEventListener('input', () => tweakStyle({ palette: { [role]: input.value } }, false));
    $('#swatches').append(el('label', { class: 'sw', for: `sw-${role}` }, input, ROLE_NAMES[role]));
  }
  for (const [name, colors] of Object.entries(palettes)) {
    const chip = el('button', { type: 'button', class: 'chip', title: `${name} colors` },
      el('i', {}, ...roles.map((role) => el('b', { style: `background:${colors[role]}` }))), name);
    chip.addEventListener('click', () => tweakStyle({ palette: { ...colors } }, true));
    $('#palettes').append(chip);
  }
}

/** Change the style the person is building. It is theirs from then on: "custom", until they save it under a name. */
function tweakStyle(change, now) {
  const patch = { style: change };
  if (state.project.style_source !== 'custom') {
    patch.style = { ...change, id: 'custom', name: 'Custom' };
    patch.style_source = 'custom';
  }
  edit(patch, { now });
}

function paintStyle({ keepFocus = true } = {}) {
  const { style, style_source: source } = state.project;
  for (const button of $$('#tab-style .opts[data-layer] .opt')) {
    button.setAttribute('aria-pressed', String(style[button.closest('.opts').dataset.layer] === button.dataset.v));
  }
  for (const button of $$('#extras .tog')) button.setAttribute('aria-pressed', String(Boolean(style.extras[button.dataset.v])));
  for (const [role, color] of Object.entries(style.palette)) {
    const input = $(`#sw-${role}`);
    if (input && !(keepFocus && document.activeElement === input) && input.value !== color) input.value = color;
  }
  const known = state.styles.find((entry) => entry.id === source);
  for (const card of $$('#styles-gallery .gcard[data-style]')) card.setAttribute('aria-pressed', String(card.dataset.style === source));
  const current = $('#style-current');
  current.replaceChildren('Current style: ');
  current.append(el('b', { text: known ? known.name : 'Custom' }));
  if (!known) current.append(' · not saved yet. Name it below to keep it.');
  paintContrast();
}

function paintContrast() {
  const box = $('#contrast');
  const value = state.contrast;
  if (typeof value !== 'number') {
    box.className = 'contrast';
    box.textContent = '';
    return;
  }
  const ok = value >= 3;
  const shown = (Math.floor(value * 10) / 10).toFixed(1); // (never 3.0 for a 2.97 that is too low)
  box.className = 'contrast ' + (ok ? 'ok' : 'low');
  box.textContent = ok
    ? `Text contrast ${shown}:1 · easy to read on a phone`
    : `Text contrast ${shown}:1 · hard to read on a phone. Try a darker or lighter text color.`;
}

async function colorsFromImage(file) {
  try {
    const answer = await api('POST', '/api/palette', file, { 'Content-Type': file.type || 'application/octet-stream' });
    clearError();
    tweakStyle({ palette: answer.palette }, true);
    toast('Colors taken from your image');
  } catch (error) {
    showError(error.message);
  }
}

async function saveStyle() {
  const name = $('#style-name').value.trim();
  if (!name) {
    showError('Name your style first.');
    $('#style-name').focus();
    return;
  }
  try {
    await flush();
    const entry = await api('POST', '/api/styles', { name });
    clearError();
    state.styles = await api('GET', '/api/styles');
    buildGallery();
    edit({ style: { id: entry.id, name: entry.name }, style_source: entry.id }, { now: true });
    bumpGallery();
    $('#style-name').value = '';
    toast(`“${entry.name}” saved next to the 4 styles`);
  } catch (error) {
    showError(error.message);
  }
}

/* Describe it to Claude: the request is recorded in the project, the waiter wakes Claude, and Claude answers by writing the style. */

/* ask-button:begin (a test runs this in node, with stand-ins for the page's $ and state) */

const ask = { timer: null, file: null, sending: false };

/** Whether a request of the person's is still waiting for Claude: there is one, and it has no answered_at. */
function askPending() {
  const request = state.project && state.project.style_request;
  return Boolean(request && !request.answered_at);
}

/** The button and the line under it. While a request is pending the button stays enabled and reads "Ask again" (a new
    request replaces the old one), with a Cancel beside it, so that a request that Claude's session never answers is not
    a dead end. The button is disabled only for the moment a request is on its way. */
function paintAsk(text, kind = '') {
  const status = $('#ask-status');
  status.textContent = text || '';
  status.className = 'ask-status ' + kind;
  const pending = askPending();
  $('#btn-ask').textContent = pending ? 'Ask again' : 'Ask Claude';
  $('#btn-ask').disabled = ask.sending;
  $('#btn-ask-cancel').hidden = !pending;
}

/* ask-button:end */

async function askClaude() {
  if (ask.sending) return;
  const text = $('#describe').value.trim();
  if (!text) {
    paintAsk('Write a few words about the style you want first.', 'error');
    $('#describe').focus();
    return;
  }
  ask.sending = true;
  paintAsk(askPending() ? 'Claude is on it…' : 'Sending…');
  try {
    await flush();
    let image = null;
    if (ask.file) {
      const sent = await api('POST', '/api/style/reference', ask.file, { 'X-Filename': encodeURIComponent(ask.file.name) });
      image = sent.path;
    }
    state.project = await api('POST', '/api/style/ask', { text, image });
    clearError();
    ask.sending = false;
    paintAsk('Claude is on it…');
    startAskPolling();
  } catch (error) {
    ask.sending = false;
    paintAsk(error.message, 'error');
  }
}

/** Take the request away: a patch with a null deletes style_request, for the project file and for Claude's waiter. */
function cancelAsk() {
  clearInterval(ask.timer);
  edit({ style_request: null }, { now: true });
  paintAsk('');
}

function startAskPolling() {
  clearInterval(ask.timer);
  ask.timer = setInterval(checkAnswer, 2000);
}

async function checkAnswer() {
  if (queue.patch) return; // the person is in the middle of an edit: look again in 2 seconds
  try {
    const project = await api('GET', '/api/project');
    const request = project.style_request;
    if (!request || request.answered_at) { // answered (or taken away)
      clearInterval(ask.timer);
      try {
        await reload();
      } catch (error) { // (the server does not accept what Claude wrote: it says what is wrong)
        showError(error.message);
        paintAsk('Claude wrote a style the studio cannot use.', 'error');
        return;
      }
      if (request) {
        paintAsk('Claude updated your style', 'done');
        toast('Claude updated your style');
      } else {
        paintAsk('');
      }
    }
  } catch { /* the studio is busy for a moment: look again */ }
}

/* Cutout */

function paintCutout() {
  const { matte } = state.project;
  const edge = Number(matte.edge) || 0;
  if (document.activeElement !== $('#matte-edge')) $('#matte-edge').value = String(edge);
  $('#matte-edge-value').textContent = edge > 0 ? `+${edge}` : String(edge).replace('-', '−');
  $('#matte-detail').value = matte.detail;
}

async function refineCutout(detail) {
  try {
    await flush();
    setBusy('Cutting you out again around the freeze…');
    const answer = await api('POST', '/api/matte/refine', { detail });
    clearError();
    state.project = answer.project;
    bumpCard();
    bumpGallery();
    toast(detail === 'auto' ? 'Cutout detail: automatic' : `Cutout detail: ${detail === 'max' ? 'maximum' : detail}`);
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy('');
    paintAll({ keepFocus: false });
  }
}

function wireStyle() {
  $('#img-palette').addEventListener('change', (event) => {
    const file = event.target.files[0];
    event.target.value = '';
    if (file) colorsFromImage(file);
  });
  $('#describe-image').addEventListener('change', (event) => {
    ask.file = event.target.files[0] || null;
    $('#describe-file').textContent = ask.file ? ask.file.name : '';
  });
  $('#btn-ask').addEventListener('click', askClaude);
  $('#btn-ask-cancel').addEventListener('click', cancelAsk);
  $('#btn-save-style').addEventListener('click', saveStyle);
  $('#style-name').addEventListener('keydown', (event) => { if (event.key === 'Enter') saveStyle(); });
  $('#matte-edge').addEventListener('input', (event) => {
    const value = Number(event.target.value);
    edit({ matte: { edge: value } });
  });
  $('#matte-detail').addEventListener('change', (event) => {
    if (event.target.value !== state.project.matte.detail) refineCutout(event.target.value);
  });
}

/* ---------- Timing ---------- */

function buildTiming() {
  const ranges = state.options.timing.ranges;
  for (const row of TIMING_ROWS) {
    const [low, high] = ranges[row.key];
    const input = el('input', { type: 'range', id: `t-${row.key}`, min: low, max: high, step: row.step, 'aria-label': row.label });
    const value = el('output', { class: 'value', for: input.id });
    input.addEventListener('input', () => {
      // the tail all the way right is the clip's end, kept as the longest tail so it stays the end when the freeze moves
      const atEnd = row.key === 'tail' && Number(input.value) >= Number(input.max);
      value.textContent = atEnd ? TO_THE_END : format(row, Number(input.value));
      edit({ timing: { [row.key]: atEnd ? high : Number(input.value) } });
    });
    $('#timing-sliders').append(el('div', { class: 'trow' }, el('label', { class: 'label', for: input.id, text: row.label }), input, value, el('span', { class: 'hint', text: row.hint })));
  }
  for (const button of $$('#reveal-options .opt')) {
    button.addEventListener('click', () => edit({ timing: { reveal: button.dataset.reveal } }, { now: true }));
  }
  for (const [id, key] of [['tog-boil', 'boil'], ['tog-shutter', 'shutter']]) {
    $('#' + id).addEventListener('click', () => edit({ timing: { [key]: !state.project.timing[key] } }, { now: true }));
  }
  // the clip's own sound is a setting of the audio, not of the timing
  $('#tog-sound').addEventListener('click', () => edit({ audio: { original: !(state.project.audio.original !== false) } }, { now: true }));
}

const format = (row, value) => `${value.toFixed(row.digits)}${row.unit}`;
const TO_THE_END = 'to the end';

/** Seconds of clip left after the freeze frame: the most the tail can play. */
function clipLeft() {
  const { clip, freeze } = state.project;
  return clip.fps ? Math.max(0, clip.duration - freeze.frame / clip.fps) : clip.duration;
}

function paintTiming() {
  const { timing } = state.project;
  for (const row of TIMING_ROWS) {
    const input = $(`#t-${row.key}`);
    if (row.key === 'tail') {
      // the slider ends where the clip does, so its right end means "to the end"
      const left = Math.round(clipLeft() * 10) / 10;
      input.max = String(left);
      const atEnd = timing.tail >= clipLeft() - 1e-6;
      if (document.activeElement !== input) input.value = String(atEnd ? left : timing.tail);
      input.nextElementSibling.textContent = atEnd ? TO_THE_END : format(row, Number(timing.tail));
      continue;
    }
    if (document.activeElement !== input) input.value = String(timing[row.key]);
    input.nextElementSibling.textContent = format(row, Number(timing[row.key]));
  }
  for (const button of $$('#reveal-options .opt')) button.setAttribute('aria-pressed', String(button.dataset.reveal === timing.reveal));
  for (const [id, on] of [['tog-boil', timing.boil], ['tog-shutter', timing.shutter], ['tog-sound', state.project.audio.original !== false]]) {
    $('#' + id).setAttribute('aria-pressed', String(on));
    $('#' + id).textContent = on ? 'On' : 'Off';
  }
  paintFreezeLands();
}

/** Where in the video the freeze lands: the clip plays from its first frame up to the freeze frame at head speed, as the
    timeline does (freeze frame / fps / head speed). The person lines their own sound up with it in their app. */
function paintFreezeLands() {
  const { clip, freeze, timing } = state.project;
  const speed = Number(timing.head_speed) || 1;
  $('#freeze-lands').textContent = clip.fps
    ? `Freeze lands at ${(freeze.frame / clip.fps / speed).toFixed(2)} s`
    : 'Freeze lands at ? s';
}

/* ---------- previews and renders ---------- */

function setRendering(on) {
  state.rendering = on;
  document.body.classList.toggle('rendering', on);
  $('#btn-preview').disabled = on;
  $('#btn-render').disabled = on;
  $('#btn-preview').textContent = on ? 'Rendering…' : 'Preview video';
  $('#btn-render').textContent = on ? 'Rendering…' : 'Render';
}

async function makePreview() {
  if (state.rendering) return;
  setRendering(true);
  try {
    await flush(); // the render reads the file on the disk
    releaseVideo($('#preview-video'));
    const answer = await api('POST', '/api/preview');
    clearError();
    state.previewReady = true;
    const video = $('#preview-video');
    video.src = `${answer.url}?t=${Date.now()}`;
    setMode('play');
    video.play().catch(() => { /* the person presses play */ });
    for (const line of answer.notes || []) if (line) toast(line);
  } catch (error) {
    showError(error.message);
  } finally {
    setRendering(false);
  }
}

async function makeRender() {
  if (state.rendering) return;
  setRendering(true);
  try {
    await flush();
    releaseVideo($('#result-video'));
    state.resultSource = null;
    const answer = await api('POST', '/api/render');
    clearError();
    state.last = { notes: answer.notes || [] };
    await reload();
    paintResults();
    selectTab('results');
    toast('Rendered. Your video is in the Results tab.');
  } catch (error) {
    showError(error.message);
  } finally {
    setRendering(false);
    paintResults();
  }
}

const KINDS = [[/\.mp4$/, 'Video'], [/-cover\.png$/, 'Cover image']];

function paintResults() {
  const { status } = state.project;
  const outputs = (status && status.outputs) || [];
  $('#results-empty').hidden = outputs.length > 0;
  $('#results-body').hidden = outputs.length === 0;
  if (!outputs.length) return;
  const main = outputs.find((path) => /\.mp4$/.test(path));
  const stamp = encodeURIComponent((status && status.rendered_at) || '');
  const source = main ? `${assetUrl(main)}?t=${stamp}` : null;
  const video = $('#result-video');
  if (source && state.resultSource !== source) {
    state.resultSource = source;
    video.src = source;
  }
  video.hidden = !source;
  $('#results-when').textContent = status.rendered_at ? `Rendered ${new Date(status.rendered_at).toLocaleString()}.` : '';
  const lines = state.last ? state.last.notes.filter(Boolean) : [];
  const warning = $('#results-warning');
  warning.textContent = lines.join(' ');
  warning.hidden = lines.length === 0;
  const list = $('#results-files');
  list.replaceChildren();
  for (const path of outputs) {
    const kind = KINDS.find(([pattern]) => pattern.test(path));
    const name = path.split('/').pop();
    list.append(el('li', {}, el('a', { href: assetUrl(path), download: name, text: name }), el('span', { class: 'kind', text: kind ? kind[1] : 'File' })));
  }
}

/* ---------- painting ---------- */

/** While the person types or drags: what depends on the project and is cheap to draw. */
function paintLive() {
  paintLines({ keepFocus: true });
  paintStyle({ keepFocus: true });
  paintTiming();
  paintCutout();
  paintReset();
}

/** The clip's shape, as one number on the page: its width over its height, which the preview, the filmstrip, the gallery and
    the Results video all take, so that each shows the clip whole in its own shape. It is the plan's (project.work): the card
    and the frames are cut at it. */
function paintShape() {
  const { work } = state.project;
  document.documentElement.style.setProperty('--aspect', String(work.width / work.height));
  document.documentElement.classList.add('shaped'); // (the preview box is hidden until the page knows the shape)
}

/** The sentence that init left in the project when it reset the pinned notes (a clip of another shape has another plan):
    shown once, as a toast, and then taken away from the project. */
function showNotice() {
  if (typeof state.project.notice !== 'string' || !state.project.notice.trim()) return;
  toast(state.project.notice);
  edit({ notice: null }, { now: true });
}

function paintAll({ keepFocus = true } = {}) {
  const { clip } = state.project;
  $('#clip-name').textContent = clip.stem || '';
  paintShape();
  paintLines({ keepFocus });
  paintStyle({ keepFocus });
  paintTiming();
  paintCutout();
  paintFreeze(false);
  paintReset();
  paintResults();
  paintContrast();
}

/* ---------- start ---------- */

function wireStatic() {
  wireTabs();
  wireLines();
  wireStyle();
  $('#btn-preview').addEventListener('click', makePreview);
  $('#btn-render').addEventListener('click', makeRender);
  $('#btn-reset').addEventListener('click', () => {
    edit({ positions: Object.fromEntries([...NOTE_KEYS, 'burst'].map((key) => [key, null])) }, { now: true });
  });
  for (const button of $$('#mode button')) button.addEventListener('click', () => setMode(button.dataset.mode));
  $('#error-close').addEventListener('click', clearError);
  let width = cardWidth();
  new ResizeObserver(() => {
    const next = cardWidth();
    if (next !== width && state.project) { width = next; refreshCard(); }
  }).observe($('#preview'));
}

async function boot() {
  wireStatic();
  selectTab(location.hash.slice(1) || 'lines');
  try {
    const [options, start, styles] = await Promise.all([
      api('GET', '/api/options'), api('POST', '/api/project', {}), api('GET', '/api/styles'), loadFrames(),
    ]);
    state.options = options;
    state.project = start.project;
    state.contrast = start.contrast;
    state.styles = styles;
    state.told = new Set((start.warnings || []).map(wording).filter(Boolean).map(warningKey)); // (the counters show these)
  } catch (error) {
    showError(error.status === 0 ? error.message : `The studio could not open this project (${error.message}).`);
    return;
  }
  buildStyleControls();
  buildTiming();
  buildGallery();
  buildFilmstrip();
  paintAll({ keepFocus: false });
  refreshCard();
  showNotice();
  if (!$('#tab-style').hidden) loadThumbs();
  if (askPending()) { // the page was closed while Claude was on it: it goes on waiting, and can ask again or cancel
    $('#describe').value = state.project.style_request.text || '';
    paintAsk('Claude is on it…');
    startAskPolling();
  } else {
    paintAsk('');
  }
}

boot();
})();
