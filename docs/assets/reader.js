/* Reader mode for comfortable reading on tablets. State is per browser (localStorage). */
(function () {
  var root = document.documentElement;
  var KEY = "kb-reader";
  var THEMES = ["light", "sepia", "dark"];
  var state = { on: false, scale: 1, theme: "light" };
  try { Object.assign(state, JSON.parse(localStorage.getItem(KEY) || "{}")); } catch (e) {}
  var siteScheme = null;      // the scheme the site had before reader mode changed it

  function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} }

  function apply() {
    root.classList.toggle("reader", !!state.on);
    root.style.setProperty("--reader-scale", state.scale);
    var body = document.body;
    if (!body) return;
    if (state.on) {
      if (siteScheme === null) siteScheme = body.getAttribute("data-md-color-scheme") || "default";
      root.setAttribute("data-reader-theme", state.theme);
      body.setAttribute("data-md-color-scheme", state.theme === "dark" ? "slate" : "default");
    } else {
      root.removeAttribute("data-reader-theme");
      if (siteScheme !== null) { body.setAttribute("data-md-color-scheme", siteScheme); siteScheme = null; }
    }
    var t = document.querySelector(".reader-bar [data-act=theme]");
    if (t) t.textContent = state.theme.charAt(0).toUpperCase() + state.theme.slice(1);
    progress();
  }

  function progress() {
    var bar = document.querySelector(".reader-progress");
    if (!bar || !state.on) return;
    var max = root.scrollHeight - window.innerHeight;
    bar.style.width = (max > 0 ? Math.min(100, (window.scrollY / max) * 100) : 0) + "%";
  }

  function act(name) {
    if (name === "toggle") state.on = !state.on;
    else if (name === "smaller") state.scale = Math.max(0.85, +(state.scale - 0.1).toFixed(2));
    else if (name === "larger") state.scale = Math.min(1.6, +(state.scale + 0.1).toFixed(2));
    else if (name === "theme") state.theme = THEMES[(THEMES.indexOf(state.theme) + 1) % THEMES.length];
    else if (name === "answers") {
      var all = Array.prototype.slice.call(document.querySelectorAll(".md-content details"));
      var open = all.some(function (d) { return !d.open; });
      all.forEach(function (d) { d.open = open; });
      return;
    } else if (name === "top") { window.scrollTo({ top: 0, behavior: "smooth" }); return; }
    save(); apply();
  }

  function button(label, actName, text) {
    var b = document.createElement("button");
    b.type = "button"; b.setAttribute("aria-label", label); b.title = label;
    b.setAttribute("data-act", actName); b.textContent = text;
    return b;
  }

  function build() {
    if (document.querySelector(".reader-fab")) return;
    var fab = document.createElement("button");
    fab.type = "button"; fab.className = "reader-fab"; fab.title = "Reader mode";
    fab.setAttribute("aria-label", "Enter reader mode"); fab.setAttribute("data-act", "toggle");
    fab.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M21 5c-1.1-.35-2.3-.5-3.5-.5-1.95 0-4.05.4-5.5 1.5-1.45-1.1-3.55-1.5-5.5-1.5S2.45 4.9 1 6v14.65c0 .25.25.5.5.5.1 0 .15-.05.25-.05C3.1 20.45 5.05 20 6.5 20c1.95 0 4.05.4 5.5 1.5 1.35-.85 3.8-1.5 5.5-1.5 1.65 0 3.35.3 4.75 1.05.1.05.15.05.25.05.25 0 .5-.25.5-.5V6c-.6-.45-1.25-.75-2-1zm0 13.5c-1.1-.35-2.3-.5-3.5-.5-1.7 0-4.15.65-5.5 1.5V8c1.35-.85 3.8-1.5 5.5-1.5 1.2 0 2.4.15 3.5.5v11.5z"/></svg>';

    var bar = document.createElement("div");
    bar.className = "reader-bar"; bar.setAttribute("role", "toolbar"); bar.setAttribute("aria-label", "Reader mode controls");
    var sep = function () { var s = document.createElement("span"); s.className = "sep"; return s; };
    bar.appendChild(button("Smaller text", "smaller", "A−"));
    bar.appendChild(button("Larger text", "larger", "A+"));
    bar.appendChild(sep());
    bar.appendChild(button("Change theme: light, sepia, dark", "theme", "Light"));
    bar.appendChild(sep());
    bar.appendChild(button("Show or hide all answers", "answers", "Answers"));
    bar.appendChild(button("Back to top", "top", "Top"));
    bar.appendChild(sep());
    bar.appendChild(button("Exit reader mode", "toggle", "Exit"));

    var prog = document.createElement("div"); prog.className = "reader-progress";
    document.body.appendChild(prog); document.body.appendChild(fab); document.body.appendChild(bar);
  }

  document.addEventListener("click", function (e) {
    var el = e.target.closest && e.target.closest("[data-act]");
    if (el && (el.closest(".reader-bar") || el.classList.contains("reader-fab"))) act(el.getAttribute("data-act"));
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && state.on) act("toggle");
  });

  // Hide the toolbar while scrolling down, show it when scrolling up or at the ends
  var lastY = 0, ticking = false;
  window.addEventListener("scroll", function () {
    if (ticking) return; ticking = true;
    requestAnimationFrame(function () {
      var y = window.scrollY, atEnd = window.innerHeight + y >= root.scrollHeight - 40;
      root.classList.toggle("reader-bar-hidden", y > lastY + 4 && y > 200 && !atEnd);
      if (y < lastY - 4 || atEnd) root.classList.remove("reader-bar-hidden");
      lastY = y; progress(); ticking = false;
    });
  }, { passive: true });

  function init() { build(); apply(); }
  root.classList.toggle("reader", !!state.on);           // avoid a flash of the full layout
  if (window.document$ && window.document$.subscribe) window.document$.subscribe(init);   // instant navigation
  else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
