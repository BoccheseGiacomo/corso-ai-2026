/* Riconoscere le cifre: CNN addestrata su MNIST (solo inferenza) con vista interna di filtri e attivazioni */
(function () {
  "use strict";
  var root = document.getElementById("sim");
  if (!window.MNIST_MODEL) { root.innerHTML = '<p class="muted" style="margin:auto">Modello non trovato.</p>'; return; }

  // ---------- pesi ----------
  function dec(t) {
    var bin = atob(t.d), out = new Float32Array(t.n);
    for (var i = 0; i < t.n; i++) { var v = bin.charCodeAt(i); out[i] = (v > 127 ? v - 256 : v) * t.s; }
    return out;
  }
  var W = {};
  for (var k in MNIST_MODEL.w) W[k] = dec(MNIST_MODEL.w[k]);

  function conv(inp, C, S, w, b, O) {
    var out = new Float32Array(O * S * S);
    for (var o = 0; o < O; o++) for (var y = 0; y < S; y++) for (var x = 0; x < S; x++) {
      var s = b[o];
      for (var c = 0; c < C; c++) for (var ky = 0; ky < 5; ky++) {
        var iy = y + ky - 2; if (iy < 0 || iy >= S) continue;
        for (var kx = 0; kx < 5; kx++) { var ix = x + kx - 2; if (ix < 0 || ix >= S) continue; s += inp[(c * S + iy) * S + ix] * w[((o * C + c) * 5 + ky) * 5 + kx]; }
      }
      out[(o * S + y) * S + x] = s > 0 ? s : 0;
    }
    return out;
  }
  function pool(inp, C, S) {
    var h = S / 2, out = new Float32Array(C * h * h);
    for (var c = 0; c < C; c++) for (var y = 0; y < h; y++) for (var x = 0; x < h; x++) {
      var i = (c * S + 2 * y) * S + 2 * x; out[(c * h + y) * h + x] = Math.max(inp[i], inp[i + 1], inp[i + S], inp[i + S + 1]);
    }
    return out;
  }
  function dense(inp, w, b, O, relu) {
    var I = inp.length, out = new Float32Array(O);
    for (var o = 0; o < O; o++) { var s = b[o]; for (var i = 0; i < I; i++) s += w[o * I + i] * inp[i]; out[o] = relu && s < 0 ? 0 : s; }
    return out;
  }
  function forward(x) {
    var a1 = conv(x, 1, 28, W["c1.weight"], W["c1.bias"], 8), p1 = pool(a1, 8, 28);
    var a2 = conv(p1, 8, 14, W["c2.weight"], W["c2.bias"], 16), p2 = pool(a2, 16, 14);
    var f1 = dense(p2, W["f1.weight"], W["f1.bias"], 64, true), lg = dense(f1, W["f2.weight"], W["f2.bias"], 10, false);
    var m = Math.max.apply(null, lg), s = 0, pr = [];
    for (var i = 0; i < 10; i++) { pr[i] = Math.exp(lg[i] - m); s += pr[i]; }
    return { x: x, a1: a1, p1: p1, a2: a2, p2: p2, f1: f1, probs: pr.map(function (v) { return v / s; }) };
  }

  // ---------- colori ----------
  function actColor(t) { // nero → ciano → bianco
    t = Math.max(0, Math.min(1, t));
    if (t < .6) { var u = t / .6; return [10 + 24 * u, 14 + 197 * u, 21 + 217 * u]; }
    var v = (t - .6) / .4; return [34 + 221 * v, 211 + 44 * v, 238 + 17 * v];
  }
  function wColor(t) { // negativo blu, zero scuro, positivo arancio
    t = Math.max(-1, Math.min(1, t));
    var z = [18, 26, 39];
    if (t >= 0) return [z[0] + (251 - z[0]) * t, z[1] + (146 - z[1]) * t, z[2] + (60 - z[2]) * t];
    t = -t; return [z[0] + (59 - z[0]) * t, z[1] + (130 - z[1]) * t, z[2] + (246 - z[2]) * t];
  }
  function paint(cv, data, off, S, scale, colorFn) {
    if (cv.width !== S) { cv.width = S; cv.height = S; }
    var ctx = cv.getContext("2d"), img = ctx.createImageData(S, S);
    for (var i = 0; i < S * S; i++) { var c = colorFn(data[off + i] / scale); img.data[i * 4] = c[0]; img.data[i * 4 + 1] = c[1]; img.data[i * 4 + 2] = c[2]; img.data[i * 4 + 3] = 255; }
    ctx.putImageData(img, 0, 0);
  }
  function maxAbs(a, off, n) { var m = 1e-6; for (var i = 0; i < n; i++) m = Math.max(m, Math.abs(a[off + i])); return m; }

  // ---------- lavagna e pre-elaborazione ----------
  var pad = root.querySelector(".pad"), pc = pad.getContext("2d"), R = 280;
  pad.width = R; pad.height = R;
  var see = root.querySelector(".see"); see.width = 28; see.height = 28;
  var sc = see.getContext("2d"), tmp = document.createElement("canvas"); tmp.width = 28; tmp.height = 28;
  var tc = tmp.getContext("2d"), drawing = false, last = null, ink = false, state = null;

  function clear() { pc.fillStyle = "#000"; pc.fillRect(0, 0, R, R); sc.fillStyle = "#000"; sc.fillRect(0, 0, 28, 28); ink = false; state = null; render(); }
  function pos(e) { var r = pad.getBoundingClientRect(); return [(e.clientX - r.left) / r.width * R, (e.clientY - r.top) / r.height * R]; }
  function stroke(a, b) {
    pc.strokeStyle = "#fff"; pc.lineWidth = 20; pc.lineCap = "round"; pc.lineJoin = "round";
    pc.beginPath(); pc.moveTo(a[0], a[1]); pc.lineTo(b[0], b[1]); pc.stroke(); ink = true; schedule();
  }
  pad.addEventListener("pointerdown", function (e) { drawing = true; last = pos(e); pad.setPointerCapture(e.pointerId); stroke(last, last); e.preventDefault(); });
  pad.addEventListener("pointermove", function (e) { if (drawing) { var q = pos(e); stroke(last, q); last = q; } });
  pad.addEventListener("pointerup", function () { drawing = false; });
  pad.addEventListener("pointercancel", function () { drawing = false; });
  var pending = false;
  function schedule() { if (!pending) { pending = true; requestAnimationFrame(function () { pending = false; classify(); }); } }

  function preprocess() { // ritaglio, cifra in 20×20, centro di massa al centro (come MNIST)
    var d = pc.getImageData(0, 0, R, R).data, x0 = R, y0 = R, x1 = -1, y1 = -1, x, y, i;
    for (y = 0; y < R; y++) for (x = 0; x < R; x++) if (d[(y * R + x) * 4] > 40) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    if (x1 < 0) return null;
    var w = x1 - x0 + 1, h = y1 - y0 + 1, s = 20 / Math.max(w, h), dw = w * s, dh = h * s;
    tc.fillStyle = "#000"; tc.fillRect(0, 0, 28, 28); tc.imageSmoothingQuality = "high";
    tc.drawImage(pad, x0, y0, w, h, (28 - dw) / 2, (28 - dh) / 2, dw, dh);
    var px = tc.getImageData(0, 0, 28, 28).data, m = 0, mx = 0, my = 0;
    for (i = 0; i < 784; i++) { var v = px[i * 4] / 255; m += v; mx += v * (i % 28); my += v * Math.floor(i / 28); }
    sc.fillStyle = "#000"; sc.fillRect(0, 0, 28, 28);
    sc.drawImage(tmp, Math.round(13.5 - mx / m), Math.round(13.5 - my / m));
    px = sc.getImageData(0, 0, 28, 28).data;
    var inp = new Float32Array(784);
    for (i = 0; i < 784; i++) inp[i] = px[i * 4] / 255;
    return inp;
  }
  function classify() { if (!ink) return; var x = preprocess(); state = x ? forward(x) : null; render(); }

  // ---------- vista semplice ----------
  var outBars = root.querySelector(".out-bars"), bigDigit = root.querySelector(".big-digit"), rows = [];
  for (var i = 0; i < 10; i++) {
    var r = document.createElement("div"); r.className = "bar";
    r.innerHTML = '<span class="k">' + i + '</span><span class="t"><span class="f"></span></span><span class="v">–</span>';
    outBars.appendChild(r); rows.push(r);
  }
  function fmt(p) { return p * 100 < 0.1 ? "0%" : (p * 100).toFixed(p > .995 ? 0 : 1) + "%"; }
  function renderSimple() {
    var p = state && state.probs, best = p ? p.indexOf(Math.max.apply(null, p)) : -1;
    rows.forEach(function (r, i) {
      r.classList.toggle("top", i === best);
      r.querySelector(".f").style.width = p ? (p[i] * 100).toFixed(1) + "%" : "0";
      r.querySelector(".v").textContent = p ? fmt(p[i]) : "–";
    });
    bigDigit.textContent = p ? best : "?";
  }

  // ---------- vista interna ----------
  var c1 = root.querySelector(".c1"), c2 = root.querySelector(".c2"), mapIn = root.querySelector(".map-in");
  var neurEl = root.querySelector(".neurons"), outsEl = root.querySelector(".outs"), links = root.querySelector(".links");
  var insp = root.querySelector(".inspector"), c1cells = [], c2cells = [], neurons = [], outs = [], focus = null;
  for (i = 0; i < 8; i++) {
    var cell = document.createElement("div"); cell.className = "cell"; cell.dataset.k = "c1"; cell.dataset.i = i;
    cell.innerHTML = '<canvas class="ker"></canvas><canvas class="act"></canvas>';
    c1.appendChild(cell); c1cells.push(cell);
    paint(cell.querySelector(".ker"), W["c1.weight"], i * 25, 5, maxAbs(W["c1.weight"], 0, 200), wColor);
  }
  for (i = 0; i < 16; i++) {
    var cell2 = document.createElement("div"); cell2.className = "cell"; cell2.dataset.k = "c2"; cell2.dataset.i = i;
    cell2.innerHTML = '<canvas class="act"></canvas>'; c2.appendChild(cell2); c2cells.push(cell2);
  }
  for (i = 0; i < 64; i++) { var n = document.createElement("div"); n.className = "neuron"; n.dataset.k = "fc"; n.dataset.i = i; neurEl.appendChild(n); neurons.push(n); }
  for (i = 0; i < 10; i++) {
    var o = document.createElement("div"); o.className = "o"; o.dataset.k = "out"; o.dataset.i = i;
    o.innerHTML = '<span class="d">' + i + '</span><span class="ob"><i></i></span>'; outsEl.appendChild(o); outs.push(o);
  }
  var EMPTY = { x: new Float32Array(784), a1: new Float32Array(8 * 784), p1: new Float32Array(8 * 196), a2: new Float32Array(16 * 196), p2: new Float32Array(784), f1: new Float32Array(64), probs: null };

  function renderNet() {
    var s = state || EMPTY;
    paint(mapIn, s.x, 0, 28, 1, actColor);
    var m1 = maxAbs(s.a1, 0, s.a1.length), m2 = maxAbs(s.a2, 0, s.a2.length), mf = maxAbs(s.f1, 0, 64);
    c1cells.forEach(function (c, i) { paint(c.querySelector(".act"), s.a1, i * 784, 28, m1, actColor); });
    c2cells.forEach(function (c, i) { paint(c.querySelector(".act"), s.a2, i * 196, 14, m2, actColor); });
    neurons.forEach(function (n, i) { var c = actColor(s.f1[i] / mf); n.style.background = "rgb(" + c.map(Math.round).join(",") + ")"; });
    var p = s.probs, best = p ? p.indexOf(Math.max.apply(null, p)) : -1;
    outs.forEach(function (o, i) { o.classList.toggle("top", i === best); o.querySelector("i").style.width = p ? (p[i] * 100) + "%" : "0"; });
    drawLinks(s, best);
    renderInspector();
  }
  function drawLinks(s, best) {
    var r = links.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    links.width = Math.round(r.width * dpr); links.height = Math.round(r.height * dpr);
    var ctx = links.getContext("2d"), w = links.width, h = links.height, w2 = W["f2.weight"];
    ctx.clearRect(0, 0, w, h);
    var ny = function (i) { var el = neurons[i].getBoundingClientRect(); return (el.top + el.height / 2 - r.top) * dpr; };
    var oy = function (j) { var el = outs[j].getBoundingClientRect(); return (el.top + el.height / 2 - r.top) * dpr; };
    var mx = 1e-6, j, i;
    for (j = 0; j < 10; j++) for (i = 0; i < 64; i++) mx = Math.max(mx, Math.abs(w2[j * 64 + i] * s.f1[i]));
    var fj = focus && focus.k === "out" ? focus.i : (focus && focus.k === "fc" ? -2 : best);
    for (j = 0; j < 10; j++) for (i = 0; i < 64; i++) {
      var c = w2[j * 64 + i] * s.f1[i], t = Math.abs(c) / mx;
      if (focus && focus.k === "fc") { if (i !== focus.i) continue; t = Math.max(t, Math.abs(w2[j * 64 + i]) / maxAbs(w2, 0, 640)); }
      if (t < 0.04) continue;
      var on = fj < 0 || j === fj;
      ctx.strokeStyle = c >= 0 || (focus && focus.k === "fc" && w2[j * 64 + i] >= 0) ? "rgba(34,211,238," + (on ? t : t * .15) + ")" : "rgba(248,113,113," + (on ? t : t * .15) + ")";
      ctx.lineWidth = (on ? 0.5 + 2.5 * t : 0.6) * dpr;
      ctx.beginPath(); ctx.moveTo(0, ny(i)); ctx.bezierCurveTo(w * .5, ny(i), w * .5, oy(j), w, oy(j)); ctx.stroke();
    }
  }

  // ---------- ispettore ----------
  function cvs(S, px) { var c = document.createElement("canvas"); c.style.width = px + "px"; c.style.height = px + "px"; return c; }
  function group(cap, el) { var g = document.createElement("div"); g.className = "insp-group"; g.appendChild(el); var s = document.createElement("span"); s.className = "cap"; s.textContent = cap; g.appendChild(s); return g; }
  function op(t) { var s = document.createElement("span"); s.className = "insp-op"; s.textContent = t; return s; }
  function renderInspector() {
    [c1cells, c2cells, neurons, outs].forEach(function (L) { L.forEach(function (e) { e.classList.remove("hl"); }); });
    insp.innerHTML = "";
    if (!focus) { insp.innerHTML = '<span class="muted hint">passa col mouse su un elemento della rete per vederlo da vicino</span>'; return; }
    var s = state || EMPTY, i = focus.i, c;
    if (focus.k === "c1") {
      c1cells[i].classList.add("hl");
      c = cvs(28, 96); paint(c, s.x, 0, 28, 1, actColor); insp.appendChild(group("ingresso", c)); insp.appendChild(op("✱"));
      c = cvs(5, 96); paint(c, W["c1.weight"], i * 25, 5, maxAbs(W["c1.weight"], i * 25, 25), wColor); insp.appendChild(group("filtro " + (i + 1) + ": il tratto che cerca", c)); insp.appendChild(op("="));
      c = cvs(28, 96); paint(c, s.a1, i * 784, 28, maxAbs(s.a1, i * 784, 784), actColor); insp.appendChild(group("dove lo trova nel disegno", c));
    } else if (focus.k === "c2") {
      c2cells[i].classList.add("hl");
      var many = document.createElement("div"); many.className = "insp-many";
      var wk = W["c2.weight"], mk = maxAbs(wk, i * 200, 200);
      for (var ch = 0; ch < 8; ch++) { c = cvs(5, 34); paint(c, wk, (i * 8 + ch) * 25, 5, mk, wColor); many.appendChild(c); }
      insp.appendChild(group("8 filtri, uno per mappa dello strato 1", many)); insp.appendChild(op("="));
      c = cvs(14, 96); paint(c, s.a2, i * 196, 14, maxAbs(s.a2, i * 196, 196), actColor); insp.appendChild(group("mappa " + (i + 1), c));
    } else if (focus.k === "fc") {
      neurons[i].classList.add("hl");
      var many2 = document.createElement("div"); many2.className = "insp-many"; many2.style.maxWidth = "620px";
      var wf = W["f1.weight"], mf = maxAbs(wf, i * 784, 784);
      for (var m = 0; m < 16; m++) { c = cvs(7, 34); paint(c, wf, i * 784 + m * 49, 7, mf, wColor); many2.appendChild(c); }
      insp.appendChild(group("pesi del neurone " + (i + 1) + " sulle 16 mappe", many2));
      var val = document.createElement("span"); val.className = "big-num"; val.style.fontSize = "30px"; val.textContent = s.f1[i].toFixed(2);
      insp.appendChild(op("→")); insp.appendChild(group("attivazione", val));
    } else if (focus.k === "out") {
      outs[i].classList.add("hl");
      var pv = document.createElement("span"); pv.className = "big-num"; pv.textContent = s.probs ? fmt(s.probs[i]) : "–";
      insp.appendChild(group("probabilità che sia " + i, pv));
      var hint = document.createElement("span"); hint.className = "muted"; hint.style.fontSize = "14px"; hint.style.maxWidth = "380px";
      hint.innerHTML = '<span style="color:var(--cyan)">ciano</span>: neuroni che spingono verso ' + i + ' · <span style="color:var(--red)">rosso</span>: neuroni che spingono contro';
      insp.appendChild(hint);
    } else if (focus.k === "in") {
      c = cvs(28, 96); paint(c, s.x, 0, 28, 1, actColor); insp.appendChild(group("28 × 28 = 784 numeri tra 0 e 1", c));
    }
  }
  root.querySelector(".net").addEventListener("mouseover", function (e) {
    var t = e.target.closest("[data-k]"); if (!t) return;
    var f = { k: t.dataset.k, i: +t.dataset.i || 0 };
    if (focus && focus.k === f.k && focus.i === f.i) return;
    focus = f; renderInspector(); drawLinks(state || EMPTY, state ? state.probs.indexOf(Math.max.apply(null, state.probs)) : -1);
  });
  root.querySelector(".net").addEventListener("mouseleave", function () { focus = null; renderNet(); });

  // ---------- modalità ----------
  var mode = "simple";
  function setMode(m) {
    mode = m; document.body.className = "mode-" + m;
    root.parentNode.querySelectorAll(".seg-b").forEach(function (b) { b.classList.toggle("on", b.dataset.mode === m); });
    render();
  }
  document.querySelectorAll(".seg-b").forEach(function (b) { b.addEventListener("click", function () { setMode(b.dataset.mode); }); });
  function render() { if (mode === "simple") renderSimple(); else renderNet(); }
  window.addEventListener("resize", render);

  root.querySelector(".clear").addEventListener("click", clear);
  setMode(location.hash === "#rete" ? "net" : "simple");
  clear();
})();
