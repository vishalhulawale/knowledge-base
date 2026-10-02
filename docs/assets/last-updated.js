/* Shows "Last updated on <deployment time>" in the footer. The time comes from assets/build-info.js. */
(function () {
  function render() {
    var host = document.querySelector(".md-copyright") || document.querySelector(".md-footer-meta__inner");
    if (!host) return;
    var el = host.querySelector(".kb-last-updated");
    if (!el) { el = document.createElement("div"); el.className = "kb-last-updated"; host.insertBefore(el, host.firstChild); }
    var iso = window.KB_BUILD_TIME, d = iso ? new Date(iso) : null;
    if (!d || isNaN(d)) { el.textContent = "Last updated on: local build (not deployed)"; return; }
    var text;
    try {
      text = new Intl.DateTimeFormat("en-IN", { timeZone: "Asia/Kolkata", day: "numeric", month: "short", year: "numeric",
        hour: "numeric", minute: "2-digit", hour12: true }).format(d) + " IST";
    } catch (e) { text = d.toUTCString(); }
    el.innerHTML = "";
    el.appendChild(document.createTextNode("Last updated on "));
    var t = document.createElement("time"); t.dateTime = iso; t.title = d.toString(); t.textContent = text;
    el.appendChild(t);
  }
  if (window.document$ && window.document$.subscribe) window.document$.subscribe(render);
  else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", render);
  else render();
})();
