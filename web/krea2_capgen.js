// Krea 2 Anime Caption (capgen): caption preview on the node + tag autocomplete for its `tags` textarea only.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "Krea2AnimeCapgen";
const PREVIEW = "caption_preview";
const DEBOUNCE_MS = 250;        // longer than the usual gap between keystrokes: one request per pause
const RETRY_MS = 1500;
const K = 12;
const CACHE_MAX = 64;           // token -> items of answered requests (this page), least recently used dropped first
const cache = new Map();

function cacheGet(q) {
  const v = cache.get(q);
  if (v !== undefined) { cache.delete(q); cache.set(q, v); }
  return v;
}

function cachePut(q, items) {
  cache.delete(q);
  cache.set(q, items);
  if (cache.size > CACHE_MAX) cache.delete(cache.keys().next().value);
}
const TAGS_LINES = [2, 4];      // tags textarea: minimum / default height in lines
const PREVIEW_LINES = [4, 6];   // caption preview: minimum / default height in lines

// height of n text lines of a textarea (font size from its style; the frontend's default is 10 px)
function linesPx(el, n) {
  const fs = parseFloat(el && getComputedStyle(el).fontSize) || 10;
  return Math.ceil(n * fs * 1.3 + 10);
}

// ---------------------------------------------------------------- caption preview
function addPreview(node) {
  if (node.widgets?.some((w) => w.name === PREVIEW)) return;
  const el = document.createElement("textarea");
  el.readOnly = true;
  el.placeholder = "caption (after a run)";
  el.className = "comfy-multiline-input krea2capgen-preview";
  el.dataset.krea2capgen = "preview";
  const w = node.addDOMWidget(PREVIEW, "krea2capgen_preview", el, {
    serialize: false,
    hideInPanel: true,
    getMinHeight: () => linesPx(el, PREVIEW_LINES[0]),
    getValue: () => el.value,
    setValue: (v) => { el.value = typeof v === "string" ? v : ""; },
  });
  w.serialize = false;
}

function showCaption(node, output) {
  const w = node.widgets?.find((x) => x.name === PREVIEW);
  const t = output?.text;
  if (w) w.value = Array.isArray(t) ? t.filter((x) => x != null).join("\n\n") : (t ?? "");
}

// ---------------------------------------------------------------- autocomplete
const css = `
.krea2capgen-ac{position:fixed;z-index:10000;min-width:260px;max-width:480px;max-height:320px;overflow-y:auto;
 background:var(--comfy-menu-bg,#202020);color:var(--input-text,#ddd);border:1px solid var(--border-color,#4e4e4e);
 border-radius:6px;box-shadow:0 6px 18px rgba(0,0,0,.45);font:12px/1.35 system-ui,sans-serif;padding:3px 0}
.krea2capgen-ac .row{display:flex;gap:8px;align-items:baseline;padding:3px 9px;cursor:pointer;white-space:nowrap}
.krea2capgen-ac.stale .row{opacity:.45;pointer-events:none}
.krea2capgen-ac .row.sel{background:var(--comfy-input-bg,#3a3a3a)}
.krea2capgen-ac .tag{flex:1;overflow:hidden;text-overflow:ellipsis;font-weight:600}
.krea2capgen-ac .cat{font-size:10.5px;opacity:.85}
.krea2capgen-ac .n{font-size:10.5px;opacity:.6;min-width:58px;text-align:right}
.krea2capgen-ac .note{padding:4px 9px;opacity:.75;white-space:normal}
.krea2capgen-ac .cat-general{color:#4fa3ff}.krea2capgen-ac .cat-character{color:#3fbf5a}
.krea2capgen-ac .cat-copyright{color:#c47bff}.krea2capgen-ac .cat-none{color:#999}`;

// suggest_info's form of a token: lower case, "_" = space, one space, "-" glued to the tag
function norm(s) {
  return s.toLowerCase().replace(/_/g, " ").replace(/\s+/g, " ").trim().replace(/^-\s*/, "-");
}

function compact(n) {
  if (!n) return "0";
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + "k";
  return String(n);
}

// pixel position of the caret inside a textarea (mirror div)
function caretXY(ta, pos) {
  const cs = getComputedStyle(ta);
  const div = document.createElement("div");
  for (const p of ["boxSizing", "width", "fontFamily", "fontSize", "fontWeight", "fontStyle", "letterSpacing",
    "lineHeight", "paddingTop", "paddingRight", "paddingBottom", "paddingLeft", "borderTopWidth",
    "borderRightWidth", "borderBottomWidth", "borderLeftWidth", "textTransform", "wordSpacing", "tabSize"]) {
    div.style[p] = cs[p];
  }
  Object.assign(div.style, { position: "absolute", visibility: "hidden", whiteSpace: "pre-wrap",
    overflowWrap: "break-word", top: "0", left: "-9999px" });
  div.textContent = ta.value.slice(0, pos);
  const span = document.createElement("span");
  span.textContent = ta.value.slice(pos) || ".";
  div.appendChild(span);
  document.body.appendChild(div);
  const r = ta.getBoundingClientRect();
  // the canvas zoom scales the textarea (CSS transform); offsetWidth is the unscaled width
  const scale = ta.offsetWidth ? r.width / ta.offsetWidth : 1;
  const lh = parseFloat(cs.lineHeight) || parseFloat(cs.fontSize) * 1.3;
  const x = r.left + (span.offsetLeft - ta.scrollLeft) * scale;
  const y = r.top + (span.offsetTop - ta.scrollTop + lh) * scale;
  div.remove();
  return { x: Math.min(x, r.right), y: Math.min(Math.max(y, r.top), r.bottom) + 2 };
}

// the token = text between the previous comma / line break and the caret
function tokenAt(ta) {
  const pos = ta.selectionStart;
  if (pos !== ta.selectionEnd) return null;
  const v = ta.value;
  const start = Math.max(v.lastIndexOf(",", pos - 1), v.lastIndexOf("\n", pos - 1)) + 1;
  const raw = v.slice(start, pos);
  const lead = raw.match(/^\s*/)[0];
  const q = raw.slice(lead.length);
  return { start, pos, lead, q };
}

class Dropdown {
  constructor() {
    this.el = null; this.ta = null; this.items = []; this.sel = 0; this.timer = null;
    this.q = null;                  // the token the shown items were suggested for
    this.inflight = null;           // {q, ctl}: the request on its way
    this.accepting = false;         // inside accept(): its own input event is not typing
  }
  ensure() {
    if (this.el) return;
    const st = document.createElement("style");
    st.textContent = css;
    document.head.appendChild(st);
    this.el = document.createElement("div");
    this.el.className = "krea2capgen-ac";
    this.el.style.display = "none";
    // keep the focus in the textarea while clicking a row
    this.el.addEventListener("mousedown", (e) => e.preventDefault());
    document.body.appendChild(this.el);
  }
  get open() { return !!this.el && this.el.style.display !== "none"; }
  hide() {
    if (this.el) this.el.style.display = "none";
    this.items = [];
    this.inflight?.ctl.abort();      // its answer is not wanted any more
    this.inflight = null;
    clearTimeout(this.timer);
  }
  // a cached token is shown at once; otherwise one request after the typing pauses
  schedule(ta, delay = DEBOUNCE_MS) {
    clearTimeout(this.timer);
    const tok = tokenAt(ta);
    const hit = tok && cacheGet(tok.q);
    if (hit) { this.show(ta, tok, { status: "ready", items: hit }); return; }
    this.timer = setTimeout(() => this.query(ta), delay);
  }
  async query(ta) {
    const tok = tokenAt(ta);
    if (document.activeElement !== ta || !tok || !tok.q.replace(/^[-\s]+/, "").trim()) { this.hide(); return; }
    const hit = cacheGet(tok.q);
    if (hit) { this.show(ta, tok, { status: "ready", items: hit }); return; }
    if (this.inflight?.q === tok.q) return;          // the same token is already on its way
    this.inflight?.ctl.abort();                      // an older token's answer is not wanted
    const req = this.inflight = { q: tok.q, ctl: new AbortController() };
    let res;
    try {
      const r = await api.fetchApi(`/krea2capgen/suggest?q=${encodeURIComponent(tok.q)}&k=${K}`,
                                   { signal: req.ctl.signal });
      res = await r.json();
    } catch (e) {
      if (req.ctl.signal.aborted || e?.name === "AbortError") return;
      res = { status: "error: " + e, items: [] };
    } finally {
      if (this.inflight === req) this.inflight = null;
    }
    if (res.status === "ready") cachePut(tok.q, res.items);
    this.show(ta, tok, res);
  }
  // show an answer for token tok if the caret is still on that token
  show(ta, tok, res) {
    if (document.activeElement !== ta) return;
    const cur = tokenAt(ta);
    if (!cur || cur.q !== tok.q) return;
    this.ta = ta;
    if (res.status === "loading") {
      this.render([], "capgen is loading (the first use downloads its data, ~450 MB)...", tok);
      this.timer = setTimeout(() => this.query(ta), RETRY_MS);
    } else if (res.status !== "ready") {
      this.render([], "capgen: " + res.status, tok);
    } else if (!res.items.length || (res.items.length === 1 && norm(res.items[0].tag) === norm(tok.q))) {
      this.hide();
    } else {
      this.render(res.items, null, tok);
    }
  }
  render(items, note, tok) {
    this.ensure();
    this.items = items; this.sel = 0; this.q = tok.q;
    this.el.classList.remove("stale");
    this.el.replaceChildren();
    if (note) {
      const d = document.createElement("div");
      d.className = "note"; d.textContent = note;
      this.el.appendChild(d);
    }
    items.forEach((it, i) => {
      const row = document.createElement("div");
      row.className = "row" + (i === 0 ? " sel" : "");
      const tag = document.createElement("span");
      tag.className = "tag"; tag.textContent = it.tag;
      const cat = document.createElement("span");
      cat.className = "cat cat-" + (it.category || "none");
      cat.textContent = it.category || (it.in_training ? "training" : "");
      const tr = document.createElement("span");
      tr.className = "n"; tr.textContent = "train " + compact(it.training_count);
      tr.title = "training images with this tag";
      const po = document.createElement("span");
      po.className = "n"; po.textContent = "posts " + compact(it.post_count);
      po.title = "Danbooru posts with this tag";
      row.append(tag, cat, tr, po);
      row.addEventListener("mouseenter", () => this.select(i));
      row.addEventListener("click", () => { this.select(i); this.accept(); });
      this.el.appendChild(row);
    });
    const p = caretXY(this.ta, tok.pos);
    this.el.style.display = "block";
    const h = this.el.offsetHeight, w = this.el.offsetWidth;
    this.el.style.left = Math.max(4, Math.min(p.x, window.innerWidth - w - 4)) + "px";
    this.el.style.top = (p.y + h > window.innerHeight - 4 ? Math.max(4, p.y - h - 22) : p.y) + "px";
  }
  select(i) {
    if (!this.items.length) return;
    this.sel = (i + this.items.length) % this.items.length;
    [...this.el.querySelectorAll(".row")].forEach((r, j) => r.classList.toggle("sel", j === this.sel));
    this.el.querySelectorAll(".row")[this.sel]?.scrollIntoView({ block: "nearest" });
  }
  // an edit makes the shown items stale: nothing is selectable until the answer for the new token arrives
  invalidate() {
    this.items = []; this.q = null;
    if (this.el) this.el.classList.add("stale");
  }
  fresh() {
    const tok = this.ta && tokenAt(this.ta);
    return !!tok && this.items.length > 0 && tok.q === this.q;
  }
  // replace the token (and the rest of it after the caret) with "tag, "
  accept() {
    const ta = this.ta, it = this.items[this.sel];
    const tok = ta && tokenAt(ta);
    if (!it || !tok || !this.fresh()) { this.hide(); return; }
    const v = ta.value;
    let rest = v.slice(tok.pos);
    rest = rest.slice(rest.match(/^[^,\n]*/)[0].length);
    if (rest.startsWith(",")) rest = rest.slice(1).replace(/^[ \t]+/, "");
    const lead = tok.lead || (tok.start > 0 && v[tok.start - 1] === "," ? " " : "");
    const text = lead + it.tag + ", ", from = tok.start, to = v.length - rest.length;
    const want = v.slice(0, from) + text + rest;
    // insertText is one native edit: Ctrl+Z / Ctrl+Shift+Z undo / redo it (assigning .value would clear the
    // browser's undo history); its input event updates the widget value. Fallback: plain assignment, no undo.
    this.accepting = true;
    try {
      ta.setSelectionRange(from, to);
      if (!document.execCommand("insertText", false, text) || ta.value !== want) {
        ta.value = want;
        ta.dispatchEvent(new Event("input", { bubbles: true }));
      }
    } finally {
      this.accepting = false;
    }
    ta.selectionStart = ta.selectionEnd = from + text.length;
    this.hide();
  }
  onKey(e) {
    if (!this.open || !this.items.length) {
      if (this.open && e.key === "Escape") { e.preventDefault(); e.stopPropagation(); this.hide(); }
      return;
    }
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      this.select(this.sel + (e.key === "ArrowDown" ? 1 : -1));
    } else if ((e.key === "Enter" || e.key === "Tab") && !e.shiftKey && !e.ctrlKey && !e.metaKey && !e.altKey) {
      if (!this.fresh()) { this.hide(); return; }      // stale list: the key does its default
      this.accept();
    } else if (e.key === "Escape") {
      this.hide();
    } else {
      return;
    }
    e.preventDefault();
    e.stopPropagation();
    e.stopImmediatePropagation();
  }
}

const dd = new Dropdown();
const bound = new WeakSet();

function bindTextarea(ta) {
  if (!(ta instanceof HTMLTextAreaElement) || ta.readOnly || bound.has(ta)) return;
  bound.add(ta);
  ta.addEventListener("keydown", (e) => dd.onKey(e), true);
  ta.addEventListener("input", (e) => {
    if (!e.isTrusted || dd.accepting) return;          // our own accept()
    if (dd.ta === ta) dd.invalidate();
    dd.schedule(ta);
  });
  ta.addEventListener("click", () => { if (dd.open) dd.schedule(ta, 0); });
  ta.addEventListener("blur", () => setTimeout(() => { if (document.activeElement !== ta) dd.hide(); }, 120));
}

function tagsElement(node) {
  const w = node.widgets?.find((x) => x.name === "tags");
  return w && (w.element ?? w.inputEl);
}

// Nodes 2.0 (Vue) renders its own textareas: bind the tags textarea of our node when it gets the focus
function graphNode(id) {
  const graphs = [app.canvas?.graph, app.graph].filter(Boolean);
  for (const g of graphs) {
    const n = g.getNodeById?.(id) ?? g.getNodeById?.(Number(id));
    if (n) return n;
  }
  return null;
}

document.addEventListener("focusin", (e) => {
  const ta = e.target;
  if (!(ta instanceof HTMLTextAreaElement) || bound.has(ta) || ta.readOnly) return;
  const host = ta.closest("[data-node-id]");
  const node = host && graphNode(host.dataset.nodeId);
  if (node?.comfyClass !== NODE) return;
  const w = node.widgets?.find((x) => x.name === "tags");
  // the tags widget's own element, or (Vue nodes) the only editable textarea of our node
  if (tagsElement(node) === ta || (w && host.querySelectorAll("textarea:not([readonly])").length === 1)) bindTextarea(ta);
}, true);
// close on a click elsewhere, on queueing (Ctrl/Cmd+Enter keeps the focus in the textarea) and on execution
document.addEventListener("pointerdown", (e) => {
  if (dd.open && !dd.el.contains(e.target) && e.target !== dd.ta) dd.hide();
}, true);
document.addEventListener("keydown", (e) => {
  if (dd.open && e.key === "Enter" && (e.ctrlKey || e.metaKey)) dd.hide();
}, true);
for (const ev of ["promptQueued", "execution_start", "executing"]) api.addEventListener(ev, () => dd.hide());
window.addEventListener("wheel", (e) => { if (dd.open && !dd.el.contains(e.target)) dd.hide(); },
  { passive: true, capture: true });

app.registerExtension({
  name: "Krea2.AnimeCapgen",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      addPreview(this);
      const ta = tagsElement(this);
      const tw = this.widgets?.find((x) => x.name === "tags");
      if (tw?.options) tw.options.getMinHeight = () => linesPx(ta, TAGS_LINES[0]);
      if (ta) bindTextarea(ta);
      // default size of a new node; a loaded workflow's saved size is applied after this (configure)
      const prev = this.widgets?.find((x) => x.name === PREVIEW)?.element;
      const extra = 2 * Math.max(linesPx(ta, TAGS_LINES[1]) - linesPx(ta, TAGS_LINES[0]),
                                 linesPx(prev, PREVIEW_LINES[1]) - linesPx(prev, PREVIEW_LINES[0]));
      const sz = this.computeSize?.();
      if (sz) this.setSize?.([Math.max(this.size?.[0] ?? 0, sz[0], 400), sz[1] + extra]);
      return r;
    };
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (output) {
      onExecuted?.apply(this, arguments);
      showCaption(this, output);
    };
  },
  onNodeOutputsUpdated(outputs) {
    for (const [id, out] of Object.entries(outputs ?? {})) {
      const n = graphNode(id);
      if (n?.comfyClass === NODE) showCaption(n, out);
    }
  },
});
