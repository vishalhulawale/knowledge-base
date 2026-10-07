// Shared docs enhancements for NeuroTrade and the knowledge base. Runs on every page load, including
// instant navigation. Keep byte-identical in both repos: scripts/check_docs_theme_sync.py.

function decoratePriorityBadges(root) {
  root.querySelectorAll(".md-typeset code").forEach((el) => {
    const match = /^P([0-3])$/.exec(el.textContent.trim());
    if (match) el.classList.add("doc-badge", `doc-p${match[1]}`);
  });
}

function renderLearningProgress(root) {
  const box = root.querySelector(".doc-progress");
  if (!box || box.querySelector(".doc-progress-bar")) return;

  const items = [...root.querySelectorAll(".task-list-item")];
  const done = items.filter((li) => li.querySelector("input[type=checkbox]")?.checked);
  done.forEach((li) => li.classList.add("doc-done"));

  const total = items.length;
  const pct = total ? Math.round((done.length / total) * 100) : 0;

  const byPriority = {};
  items.forEach((li) => {
    const p = li.querySelector("code.doc-badge")?.textContent.trim();
    if (!p) return;
    byPriority[p] ??= { done: 0, total: 0 };
    byPriority[p].total += 1;
    if (done.includes(li)) byPriority[p].done += 1;
  });

  const bar = document.createElement("div");
  bar.className = "doc-progress-bar";
  bar.setAttribute("role", "progressbar");
  bar.setAttribute("aria-valuenow", String(pct));
  bar.setAttribute("aria-valuemin", "0");
  bar.setAttribute("aria-valuemax", "100");
  bar.innerHTML = `<span style="width:${pct}%"></span>`;

  const meta = document.createElement("div");
  meta.className = "doc-progress-meta";
  meta.innerHTML =
    `<span>${pct}% complete</span>` +
    Object.keys(byPriority)
      .sort()
      .map((p) => `<span>${p}: ${byPriority[p].done}/${byPriority[p].total}</span>`)
      .join("");

  box.append(bar, meta);
}

// Reading mode: hides the sidebars, tabs and breadcrumbs, and sets the text in one wider,
// larger column for tablets. The choice is stored per device; storage can be unavailable
// (private browsing), in which case the mode still works for the current page view.
const READING_KEY = "docs.readingMode";
const READING_ICON =
  '<svg xmlns="http://www.w3.org/2000/svg" fill="none" stroke="currentColor" stroke-linecap="round" ' +
  'stroke-linejoin="round" stroke-width="2" class="lucide lucide-book-open" viewBox="0 0 24 24">' +
  '<path d="M12 7v14"/><path d="M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 0 1 4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z"/></svg>';

function readingModeStored() {
  try {
    return localStorage.getItem(READING_KEY) === "on";
  } catch {
    return false;
  }
}

function setReadingMode(on) {
  document.documentElement.classList.toggle("doc-reading", on);
  const button = document.querySelector(".doc-reading-toggle");
  if (button) {
    button.setAttribute("aria-pressed", String(on));
    button.title = on ? "Exit reading mode" : "Reading mode";
  }
  try {
    localStorage.setItem(READING_KEY, on ? "on" : "off");
  } catch {
    /* storage unavailable: keep the mode for this page view only */
  }
}

function addReadingToggle() {
  if (document.querySelector(".doc-reading-toggle")) return;
  const anchor = document.querySelector(".md-header__inner [data-md-component=palette]");
  if (!anchor) return;
  const button = document.createElement("button");
  button.type = "button";
  button.className = "md-header__button md-icon doc-reading-toggle";
  button.setAttribute("aria-label", "Reading mode");
  button.innerHTML = READING_ICON;
  button.addEventListener("click", () =>
    setReadingMode(!document.documentElement.classList.contains("doc-reading"))
  );
  anchor.before(button);
  setReadingMode(document.documentElement.classList.contains("doc-reading"));
}

// "Last updated on" in the footer: the deployment time stamped into build-info.js by the
// Docs workflow, shown in IST. A local build has no stamp and says so.
function renderLastUpdated() {
  const host = document.querySelector(".md-copyright") ?? document.querySelector(".md-footer-meta__inner");
  if (!host) return;
  let el = host.querySelector(".doc-last-updated");
  if (!el) {
    el = document.createElement("div");
    el.className = "doc-last-updated";
    host.prepend(el);
  }
  const iso = window.DOCS_BUILD_TIME;
  const date = iso ? new Date(iso) : null;
  if (!date || Number.isNaN(date.getTime())) {
    el.textContent = "Last updated on: local build (not deployed)";
    return;
  }
  const time = document.createElement("time");
  time.dateTime = iso;
  time.title = date.toString();
  time.textContent =
    new Intl.DateTimeFormat("en-IN", {
      timeZone: "Asia/Kolkata",
      day: "numeric",
      month: "short",
      year: "numeric",
      hour: "numeric",
      minute: "2-digit",
      hour12: true
    }).format(date) + " IST";
  el.replaceChildren("Last updated on ", time);
}

// Diagram zoom: every Mermaid diagram gets zoom in/out, reset and full-screen buttons.
// Ctrl/Cmd + scroll (or a trackpad pinch) zooms at the pointer; when zoomed, drag pans and
// two fingers pinch. The theme renders each diagram later, into a closed shadow root on a
// div.mermaid, so the whole element is scaled and moved rather than the SVG inside it.
const ZOOM_MIN = 0.25;
const ZOOM_MAX = 8;
const ZOOM_STEP = 1.25;
const zoomIcon = (paths) =>
  '<svg xmlns="http://www.w3.org/2000/svg" fill="none" stroke="currentColor" stroke-linecap="round" ' +
  `stroke-linejoin="round" stroke-width="2" viewBox="0 0 24 24" aria-hidden="true">${paths}</svg>`;
const ZOOM_ICONS = {
  in: zoomIcon('<circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/><path d="M11 8v6"/><path d="M8 11h6"/>'),
  out: zoomIcon('<circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/><path d="M8 11h6"/>'),
  full: zoomIcon(
    '<path d="M8 3H5a2 2 0 0 0-2 2v3"/><path d="M21 8V5a2 2 0 0 0-2-2h-3"/>' +
      '<path d="M3 16v3a2 2 0 0 0 2 2h3"/><path d="M16 21h3a2 2 0 0 0 2-2v-3"/>'
  ),
  close: zoomIcon('<path d="M18 6 6 18"/><path d="m6 6 12 12"/>')
};

function zoomButton(label, html, onClick) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "doc-zoom__btn";
  button.title = label;
  button.setAttribute("aria-label", label);
  button.innerHTML = html;
  button.addEventListener("click", onClick);
  return button;
}

function enhanceDiagram(diagram) {
  const box = document.createElement("div");
  box.className = "doc-zoom";
  const viewport = document.createElement("div");
  viewport.className = "doc-zoom__viewport";
  diagram.before(box);
  viewport.append(diagram);

  let scale = 1;
  let x = 0;
  let y = 0;
  let placeholder = null;
  const pointers = new Map();
  let pinch = null;

  const isFull = () => box.classList.contains("doc-zoom--full");
  // Keeps the diagram inside its frame on the page; full screen pans freely.
  const clamp = (offset, frame, content) => Math.min(Math.max(offset, Math.min(0, frame - content)), Math.max(0, frame - content));
  const apply = () => {
    if (!isFull()) {
      // Zoomed in on the page, the frame grows with the diagram, up to 75% of the screen.
      const height = diagram.offsetHeight * scale;
      viewport.style.height = scale > 1 ? `${Math.min(height, innerHeight * 0.75)}px` : "";
      x = clamp(x, viewport.clientWidth, diagram.offsetWidth * scale);
      y = clamp(y, viewport.clientHeight, height);
    }
    const moved = scale !== 1 || x !== 0 || y !== 0;
    diagram.style.transform = moved ? `translate(${x}px, ${y}px) scale(${scale})` : "";
    box.classList.toggle("doc-zoom--zoomed", moved);
    reset.textContent = `${Math.round(scale * 100)}%`;
  };
  // Zoom by `factor`, keeping the point (px, py) of the viewport still.
  const zoomAt = (factor, px, py) => {
    const next = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, scale * factor));
    x = px - ((px - x) * next) / scale;
    y = py - ((py - y) * next) / scale;
    scale = next;
    apply();
  };
  const zoomCentre = (factor) => zoomAt(factor, viewport.clientWidth / 2, viewport.clientHeight / 2);
  const resetZoom = () => {
    scale = 1;
    x = 0;
    y = 0;
    apply();
  };
  const onKey = (event) => {
    if (event.key === "Escape") setFull(false);
  };
  const setFull = (on) => {
    if (on === isFull()) return;
    if (on) {
      placeholder = document.createElement("div");
      placeholder.style.height = `${box.offsetHeight}px`;
      box.before(placeholder);
      document.addEventListener("keydown", onKey);
    } else {
      placeholder?.remove();
      placeholder = null;
      document.removeEventListener("keydown", onKey);
    }
    box.classList.toggle("doc-zoom--full", on);
    document.documentElement.classList.toggle("doc-zoom-open", on);
    full.innerHTML = on ? ZOOM_ICONS.close : ZOOM_ICONS.full;
    full.title = on ? "Exit full screen (Esc)" : "Full screen";
    full.setAttribute("aria-label", full.title);
    resetZoom();
    if (on) full.focus();
  };

  const reset = zoomButton("Reset zoom", "100%", resetZoom);
  reset.classList.add("doc-zoom__level");
  const full = zoomButton("Full screen", ZOOM_ICONS.full, () => setFull(!isFull()));
  const bar = document.createElement("div");
  bar.className = "doc-zoom__bar";
  bar.append(
    zoomButton("Zoom out", ZOOM_ICONS.out, () => zoomCentre(1 / ZOOM_STEP)),
    reset,
    zoomButton("Zoom in", ZOOM_ICONS.in, () => zoomCentre(ZOOM_STEP)),
    full
  );
  box.append(bar, viewport);

  const local = (event) => {
    const rect = viewport.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  };
  viewport.addEventListener(
    "wheel",
    (event) => {
      if (!(event.ctrlKey || event.metaKey || isFull())) return;
      event.preventDefault();
      zoomAt(Math.exp(-event.deltaY * 0.002), ...local(event));
    },
    { passive: false }
  );
  viewport.addEventListener("dblclick", resetZoom);
  viewport.addEventListener("pointerdown", (event) => {
    // At 100% outside full screen, leave touches to the page so it still scrolls.
    if (!isFull() && !box.classList.contains("doc-zoom--zoomed")) return;
    viewport.setPointerCapture(event.pointerId);
    pointers.set(event.pointerId, local(event));
  });
  viewport.addEventListener("pointermove", (event) => {
    if (!pointers.has(event.pointerId)) return;
    const [px, py] = local(event);
    const [lx, ly] = pointers.get(event.pointerId);
    pointers.set(event.pointerId, [px, py]);
    if (pointers.size === 1) {
      x += px - lx;
      y += py - ly;
      apply();
      return;
    }
    const [[ax, ay], [bx, by]] = [...pointers.values()];
    const distance = Math.hypot(ax - bx, ay - by);
    const mid = [(ax + bx) / 2, (ay + by) / 2];
    if (pinch) {
      x += mid[0] - pinch.mid[0];
      y += mid[1] - pinch.mid[1];
      zoomAt(distance / pinch.distance, ...mid);
    }
    pinch = { distance, mid };
  });
  const release = (event) => {
    pointers.delete(event.pointerId);
    pinch = null;
  };
  viewport.addEventListener("pointerup", release);
  viewport.addEventListener("pointercancel", release);
}

function enhanceDiagrams(root) {
  root.querySelectorAll(".md-typeset div.mermaid").forEach((diagram) => {
    if (!diagram.parentElement.classList.contains("doc-zoom__viewport")) enhanceDiagram(diagram);
  });
}

// Diagrams render after the page loads, so watch for them as well as running on each page.
let diagramScan = 0;
new MutationObserver(() => {
  cancelAnimationFrame(diagramScan);
  diagramScan = requestAnimationFrame(() => enhanceDiagrams(document));
}).observe(document.body, {
  childList: true,
  subtree: true
});

// Apply the stored mode before the first render to avoid a flash of the sidebars.
document.documentElement.classList.toggle("doc-reading", readingModeStored());

document$.subscribe(() => {
  decoratePriorityBadges(document);
  renderLearningProgress(document);
  addReadingToggle();
  renderLastUpdated();
  enhanceDiagrams(document);
  document.documentElement.classList.remove("doc-zoom-open");
});
