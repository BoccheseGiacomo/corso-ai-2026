/* Corso AI — navigazione slide (nessuna dipendenza) */
(function () {
  "use strict";

  var stage = document.getElementById("stage");
  var slides = Array.prototype.slice.call(stage.querySelectorAll(".slide"));
  var total = slides.length;
  var counter = document.getElementById("counter");
  var bar = document.getElementById("progress-bar");
  var notesPanel = document.getElementById("notes-panel");
  var helpPanel = document.getElementById("help-panel");
  var current = 0;
  var COURSE = document.body.dataset.course || document.title;
  /* ?all nell'indirizzo mostra subito tutti gli elementi (utile per consultare le slide) */
  var SHOW_ALL = /[?&]all(=|&|$)/.test(location.search);

  /* ---------- Piè di pagina e pulsanti copia ---------- */
  slides.forEach(function (s, i) {
    if (i === 0) return;
    var foot = document.createElement("div");
    foot.className = "foot";
    foot.innerHTML = "<span>" + COURSE + "</span><span class=\"fn\">" + (i + 1) + " / " + total + "</span>";
    s.appendChild(foot);
  });

  document.querySelectorAll(".prompt").forEach(function (p) {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "copy-btn";
    btn.title = "Copia";
    btn.setAttribute("aria-label", "Copia il testo");
    btn.innerHTML = '<svg class="ic"><use href="#i-copy"/></svg>';
    btn.addEventListener("click", function (e) {
      e.stopPropagation();
      var text = p.dataset.copy || p.querySelector(".prompt-text").textContent.replace(/\s+/g, " ").trim();
      copyText(text).then(function () {
        btn.classList.add("ok");
        btn.innerHTML = '<svg class="ic"><use href="#i-check"/></svg>';
        setTimeout(function () {
          btn.classList.remove("ok");
          btn.innerHTML = '<svg class="ic"><use href="#i-copy"/></svg>';
        }, 1400);
      });
    });
    p.appendChild(btn);
  });

  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text).catch(function () { fallbackCopy(text); });
    }
    fallbackCopy(text);
    return Promise.resolve();
  }
  function fallbackCopy(text) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); } catch (e) { /* ignora */ }
    document.body.removeChild(ta);
  }

  /* ---------- Scala 1280×720 ---------- */
  function fit() {
    var s = Math.min(window.innerWidth / 1280, window.innerHeight / 720);
    document.documentElement.style.setProperty("--scale", s);
  }
  window.addEventListener("resize", fit);
  fit();

  /* ---------- Frammenti ---------- */
  function frags(i) { return slides[i].querySelectorAll(".f"); }
  function shownFrags(i) { return slides[i].querySelectorAll(".f.on").length; }
  function setFrags(i, n) {
    var list = frags(i);
    for (var k = 0; k < list.length; k++) list[k].classList.toggle("on", k < n);
  }

  /* ---------- Navigazione ---------- */
  function go(i, fragState) {
    i = Math.max(0, Math.min(total - 1, i));
    slides[current].classList.remove("active");
    current = i;
    slides[current].classList.add("active");
    setFrags(current, (fragState === "all" || SHOW_ALL) ? frags(current).length : 0);
    counter.textContent = (current + 1) + " / " + total;
    bar.style.width = ((current + 1) / total * 100) + "%";
    if (location.hash !== "#" + (current + 1)) history.replaceState(null, "", "#" + (current + 1));
    document.title = (current === 0 ? "" : slides[current].dataset.title + " · ") + COURSE;
    renderNotes();
    /* le demo interattive ascoltano questo evento per avviarsi o mettersi in pausa */
    document.dispatchEvent(new CustomEvent("slidechange", { detail: { index: current, slide: slides[current] } }));
  }
  function next() {
    var n = shownFrags(current);
    if (n < frags(current).length) { setFrags(current, n + 1); return; }
    if (current < total - 1) go(current + 1);
  }
  function prev() {
    var n = shownFrags(current);
    if (n > 0) { setFrags(current, n - 1); return; }
    if (current > 0) go(current - 1, "all");
  }

  /* ---------- Note e aiuto ---------- */
  function renderNotes() {
    if (notesPanel.hidden) return;
    var aside = slides[current].querySelector("aside.notes");
    var txt = aside ? aside.textContent.replace(/^\s+|\s+$/g, "").replace(/\n\s+/g, "\n") : "Nessuna nota.";
    notesPanel.textContent = "";
    var b = document.createElement("b");
    b.textContent = "Note · " + (current + 1) + ". " + slides[current].dataset.title + "\n";
    notesPanel.appendChild(b);
    notesPanel.appendChild(document.createTextNode(txt));
  }
  function toggleNotes() { notesPanel.hidden = !notesPanel.hidden; renderNotes(); }
  function toggleHelp() { helpPanel.hidden = !helpPanel.hidden; }

  /* ---------- Panoramica ---------- */
  function toggleOverview(force) {
    var on = typeof force === "boolean" ? force : !document.body.classList.contains("overview");
    document.body.classList.toggle("overview", on);
    if (on) slides[current].scrollIntoView({ block: "center" });
  }
  slides.forEach(function (s, i) {
    s.addEventListener("click", function () {
      if (document.body.classList.contains("overview")) { toggleOverview(false); go(i); }
    });
  });

  /* ---------- Schermo intero ---------- */
  function toggleFullscreen() {
    if (!document.fullscreenElement) {
      (document.documentElement.requestFullscreen || function () {}).call(document.documentElement);
    } else if (document.exitFullscreen) {
      document.exitFullscreen();
    }
  }

  /* ---------- Tastiera ---------- */
  document.addEventListener("keydown", function (e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    var tag = (e.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select") return;
    var overview = document.body.classList.contains("overview");
    switch (e.key) {
      case "ArrowRight": case "ArrowDown": case "PageDown": case " ": case "Enter":
        if (overview) { if (e.key === "Enter") toggleOverview(false); else go(current + 1); }
        else next();
        e.preventDefault(); break;
      case "ArrowLeft": case "ArrowUp": case "PageUp": case "Backspace":
        if (overview) go(current - 1); else prev();
        e.preventDefault(); break;
      case "Home": go(0); e.preventDefault(); break;
      case "End": go(total - 1, "all"); e.preventDefault(); break;
      case "f": case "F": toggleFullscreen(); break;
      case "o": case "O": toggleOverview(); break;
      case "n": case "N": toggleNotes(); break;
      case "?": case "h": case "H": toggleHelp(); break;
      case "Escape":
        if (overview) toggleOverview(false);
        notesPanel.hidden = true; helpPanel.hidden = true; break;
    }
  });

  /* ---------- Download dei materiali ----------
     Il PDF verrebbe aperto dal browser: lo scarichiamo come blob.
     Se non è possibile (es. pagina aperta da file locale), lo apriamo in una nuova scheda. */
  document.querySelectorAll("a.dl").forEach(function (a) {
    a.addEventListener("click", function (e) {
      e.preventDefault();
      e.stopPropagation();
      var href = a.getAttribute("href");
      var name = href.split("/").pop();
      fetch(href).then(function (r) {
        if (!r.ok) throw new Error(r.status);
        return r.blob();
      }).then(function (blob) {
        var url = URL.createObjectURL(blob);
        var tmp = document.createElement("a");
        tmp.href = url;
        tmp.download = name;
        document.body.appendChild(tmp);
        tmp.click();
        document.body.removeChild(tmp);
        setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
      }).catch(function () {
        window.open(href, "_blank");
      });
    });
  });

  /* ---------- Pulsanti ---------- */
  document.getElementById("btn-next").addEventListener("click", next);
  document.getElementById("btn-prev").addEventListener("click", prev);
  document.getElementById("btn-help").addEventListener("click", toggleHelp);

  /* ---------- Tocco ---------- */
  var tx = null, ty = null;
  stage.addEventListener("touchstart", function (e) {
    if (e.target.closest(".demo")) { tx = null; return; } /* niente swipe dentro le demo */
    tx = e.touches[0].clientX; ty = e.touches[0].clientY;
  }, { passive: true });
  stage.addEventListener("touchend", function (e) {
    if (tx === null) return;
    var dx = e.changedTouches[0].clientX - tx;
    var dy = e.changedTouches[0].clientY - ty;
    if (Math.abs(dx) > 50 && Math.abs(dx) > Math.abs(dy)) { if (dx < 0) next(); else prev(); }
    tx = ty = null;
  });

  /* ---------- Mostra i controlli al movimento del mouse ---------- */
  var uiTimer;
  document.addEventListener("mousemove", function () {
    document.body.classList.add("ui");
    clearTimeout(uiTimer);
    uiTimer = setTimeout(function () { document.body.classList.remove("ui"); }, 2200);
  });

  /* ---------- Indirizzo (#n) ---------- */
  function fromHash() {
    var n = parseInt(location.hash.replace("#", ""), 10);
    return isNaN(n) ? 0 : n - 1;
  }
  window.addEventListener("hashchange", function () {
    var i = fromHash();
    if (i !== current) go(i);
  });

  /* ---------- Stampa: mostra tutto ---------- */
  window.addEventListener("beforeprint", function () {
    slides.forEach(function (s, i) { setFrags(i, frags(i).length); });
  });
  window.addEventListener("afterprint", function () {
    slides.forEach(function (s, i) { if (i !== current) setFrags(i, 0); });
  });

  go(fromHash());
})();
