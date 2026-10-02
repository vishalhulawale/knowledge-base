// Knowledge base docs enhancements (same as NeuroTrade docs). Runs on every page load, including instant navigation.

function decoratePriorityBadges(root) {
  root.querySelectorAll(".md-typeset code").forEach((el) => {
    const match = /^P([0-3])$/.exec(el.textContent.trim());
    if (match) el.classList.add("kb-badge", `kb-p${match[1]}`);
  });
}

function renderLearningProgress(root) {
  const box = root.querySelector(".kb-progress");
  if (!box || box.querySelector(".kb-progress-bar")) return;

  const items = [...root.querySelectorAll(".task-list-item")];
  const done = items.filter((li) => li.querySelector("input[type=checkbox]")?.checked);
  done.forEach((li) => li.classList.add("kb-done"));

  const total = items.length;
  const pct = total ? Math.round((done.length / total) * 100) : 0;

  const byPriority = {};
  items.forEach((li) => {
    const p = li.querySelector("code.kb-badge")?.textContent.trim();
    if (!p) return;
    byPriority[p] ??= { done: 0, total: 0 };
    byPriority[p].total += 1;
    if (done.includes(li)) byPriority[p].done += 1;
  });

  const bar = document.createElement("div");
  bar.className = "kb-progress-bar";
  bar.setAttribute("role", "progressbar");
  bar.setAttribute("aria-valuenow", String(pct));
  bar.setAttribute("aria-valuemin", "0");
  bar.setAttribute("aria-valuemax", "100");
  bar.innerHTML = `<span style="width:${pct}%"></span>`;

  const meta = document.createElement("div");
  meta.className = "kb-progress-meta";
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
const READING_KEY = "kb.readingMode";
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
  document.documentElement.classList.toggle("kb-reading", on);
  const button = document.querySelector(".kb-reading-toggle");
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
  if (document.querySelector(".kb-reading-toggle")) return;
  const anchor = document.querySelector(".md-header__inner [data-md-component=palette]");
  if (!anchor) return;
  const button = document.createElement("button");
  button.type = "button";
  button.className = "md-header__button md-icon kb-reading-toggle";
  button.setAttribute("aria-label", "Reading mode");
  button.innerHTML = READING_ICON;
  button.addEventListener("click", () =>
    setReadingMode(!document.documentElement.classList.contains("kb-reading"))
  );
  anchor.before(button);
  setReadingMode(document.documentElement.classList.contains("kb-reading"));
}

// Apply the stored mode before the first render to avoid a flash of the sidebars.
document.documentElement.classList.toggle("kb-reading", readingModeStored());

document$.subscribe(() => {
  decoratePriorityBadges(document);
  renderLearningProgress(document);
  addReadingToggle();
});
