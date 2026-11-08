/* Prezzo della casa: un modello lineare (costo di base + costo al m² × superficie) regolato a mano o con l'addestramento (discesa del gradiente)
   p[0] = costo al m² (normalizzato 0..1), p[1] = costo di base (normalizzato 0..1) */
(function () {
  "use strict";
  var root = document.getElementById("sim");
  var X0 = 30, X1 = 170, Y0 = 0, Y1 = 450;
  var S_MAX = 4, B_MIN = 0, B_MAX = 300;

  var p = [0.15, 0.85], vel = [0, 0], steps = 0, trail = [], pts = [];
  var training = false, dragging = null, hoverX = null;

  function predict(x, q) { q = q || p; return q[0] * S_MAX * x + B_MIN + q[1] * (B_MAX - B_MIN); }
  function rmse(q) {
    var s = 0;
    for (var i = 0; i < pts.length; i++) { var r = predict(pts[i][0], q) - pts[i][1]; s += r * r; }
    return Math.sqrt(s / pts.length);
  }
  function euro(v) { return Math.round(v / 1000) * 1000 < 0 ? "−" + (Math.round(-v / 1000) * 1000).toLocaleString("it-IT") + " €" : (Math.round(v / 1000) * 1000).toLocaleString("it-IT") + " €"; }
  function rnd() { return (Math.random() + Math.random() + Math.random() - 1.5) / 1.5; }
  function newData() {
    var s = 1.4 + Math.random() * 1.4, b = 20 + Math.random() * 90;
    pts = [];
    for (var i = 0; i < 18; i++) { var x = 42 + Math.random() * 118; pts.push([x, Math.max(15, s * x + b + rnd() * 38)]); }
    computeLand(); trail = [p.slice()]; steps = 0;
  }
  function errColor(t) {
    t = Math.max(0, Math.min(1, t));
    var a = [52, 211, 153], b = [251, 191, 36], c = [248, 113, 113];
    var u = t < .5 ? t * 2 : (t - .5) * 2, s = t < .5 ? a : b, e = t < .5 ? b : c;
    return "rgb(" + [0, 1, 2].map(function (i) { return Math.round(s[i] + (e[i] - s[i]) * u); }).join(",") + ")";
  }

  // ---------- grafico ----------
  var svg = root.querySelector(".plot"), W = 800, H = 500, ML = 74, MR = 20, MT = 14, MB = 54;
  function sx(x) { return ML + (x - X0) / (X1 - X0) * (W - ML - MR); }
  function sy(y) { return H - MB - (y - Y0) / (Y1 - Y0) * (H - MT - MB); }
  function clampY(y) { return Math.max(Y0, Math.min(Y1, y)); }
  function drawPlot() {
    var r = svg.getBoundingClientRect(); W = Math.max(300, r.width); H = Math.max(200, r.height);
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    var h = '<defs><filter id="glow"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>' +
      '<clipPath id="cl"><rect x="' + ML + '" y="' + MT + '" width="' + (W - ML - MR) + '" height="' + (H - MT - MB) + '"/></clipPath></defs>';
    for (var gx = 40; gx <= 160; gx += 20)
      h += '<line x1="' + sx(gx) + '" x2="' + sx(gx) + '" y1="' + sy(Y0) + '" y2="' + sy(Y1) + '" stroke="#1A2436"/>' +
        '<text x="' + sx(gx) + '" y="' + (H - MB + 24) + '" text-anchor="middle" font-size="14" fill="#56647C" font-family="Consolas,monospace">' + gx + '</text>';
    for (var gy = 0; gy <= 400; gy += 100)
      h += '<line x1="' + sx(X0) + '" x2="' + sx(X1) + '" y1="' + sy(gy) + '" y2="' + sy(gy) + '" stroke="#1A2436"/>' +
        '<text x="' + (ML - 10) + '" y="' + (sy(gy) + 5) + '" text-anchor="end" font-size="14" fill="#56647C" font-family="Consolas,monospace">' + gy + '</text>';
    h += '<text x="' + ((ML + W - MR) / 2) + '" y="' + (H - 8) + '" text-anchor="middle" font-size="15" font-weight="600" fill="#C9D3E3">Superficie della casa (m²)</text>';
    h += '<text x="16" y="' + ((MT + H - MB) / 2) + '" text-anchor="middle" font-size="15" font-weight="600" fill="#C9D3E3" transform="rotate(-90 16 ' + ((MT + H - MB) / 2) + ')">Prezzo della casa (migliaia di €)</text>';
    h += '<g clip-path="url(#cl)">';
    pts.forEach(function (q) {
      var e = Math.abs(predict(q[0]) - q[1]);
      h += '<line x1="' + sx(q[0]) + '" x2="' + sx(q[0]) + '" y1="' + sy(q[1]) + '" y2="' + sy(predict(q[0])) + '" stroke="' + errColor(e / 120) + '" stroke-width="2.5" opacity=".9"/>';
    });
    h += '<line x1="' + sx(X0) + '" y1="' + sy(predict(X0)) + '" x2="' + sx(X1) + '" y2="' + sy(predict(X1)) + '" stroke="#22D3EE" stroke-width="4" stroke-linecap="round" filter="url(#glow)"/></g>';
    pts.forEach(function (q) { h += '<circle cx="' + sx(q[0]) + '" cy="' + sy(q[1]) + '" r="7" fill="#E7EDF7" stroke="#0A0E15" stroke-width="2.5"/>'; });
    if (hoverX !== null) {
      var y = predict(hoverX), cy = sy(clampY(y)), lab = "casa di " + Math.round(hoverX) + " m² → " + euro(y * 1000);
      h += '<line x1="' + sx(hoverX) + '" x2="' + sx(hoverX) + '" y1="' + sy(Y0) + '" y2="' + cy + '" stroke="#FB923C" stroke-dasharray="4 4"/>' +
        '<circle cx="' + sx(hoverX) + '" cy="' + cy + '" r="8" fill="#FB923C" stroke="#0A0E15" stroke-width="2.5" filter="url(#glow)"/>' +
        '<rect class="tip-bg" y="' + (MT + 6) + '" height="34" rx="8" fill="#172133" stroke="#FB923C"/>' +
        '<text class="tip-tx" y="' + (MT + 28) + '" font-size="15" font-weight="600" fill="#E7EDF7" font-family="Consolas,monospace">' + lab + ' previsti</text>';
    }
    svg.innerHTML = h;
    if (hoverX !== null) placeTip();
  }
  function placeTip() { // il riquadro si adatta alla lunghezza del testo e resta dentro il grafico
    var tx = svg.querySelector(".tip-tx"), bg = svg.querySelector(".tip-bg");
    var tw = tx.getComputedTextLength(), pad = 14, w = tw + 2 * pad, x = sx(hoverX) + 14;
    if (x + w > W - MR) x = sx(hoverX) - 14 - w;   // troppo a destra: va a sinistra del cursore
    x = Math.max(ML, x);
    bg.setAttribute("x", x); bg.setAttribute("width", w);
    tx.setAttribute("x", x + pad);
  }
  svg.addEventListener("pointermove", function (e) {
    var r = svg.getBoundingClientRect(), x = X0 + (e.clientX - r.left - ML) / (W - ML - MR) * (X1 - X0);
    hoverX = (x >= 40 && x <= 165) ? x : null; render();
  });
  svg.addEventListener("pointerleave", function () { hoverX = null; render(); });

  // ---------- manopole ----------
  var A0 = -135, A1 = 135;
  function polar(r, deg) { var a = (deg - 90) * Math.PI / 180; return [60 + r * Math.cos(a), 60 + r * Math.sin(a)]; }
  function arc(r, d0, d1) { var s = polar(r, d0), e = polar(r, d1); return "M" + s[0] + " " + s[1] + "A" + r + " " + r + " 0 " + (d1 - d0 > 180 ? 1 : 0) + " 1 " + e[0] + " " + e[1]; }
  var knobs = Array.prototype.map.call(root.querySelectorAll(".knob"), function (k, ki) {
    var i = ki === 0 ? 1 : 0; // prima manopola = costo di base (p[1]), seconda = costo al m² (p[0])
    var s = k.querySelector("svg");
    s.setAttribute("viewBox", "0 0 120 120");
    s.innerHTML = '<defs><radialGradient id="kg" cx=".4" cy=".3" r=".8"><stop offset="0" stop-color="#2C3A52"/><stop offset="1" stop-color="#121A27"/></radialGradient></defs>' +
      '<path class="k-ring" d="' + arc(54, A0, A1) + '"/><path class="k-arc"/><circle class="k-body" cx="60" cy="60" r="40"/><line class="k-ind" x1="60" y1="60" x2="60" y2="28"/>';
    var o = { el: k, arc: s.querySelector(".k-arc"), ind: s.querySelector(".k-ind"), val: k.querySelector(".val"), i: i }, last = 0;
    function ang(e) { var r = s.getBoundingClientRect(); return Math.atan2(e.clientY - r.top - r.height / 2, e.clientX - r.left - r.width / 2) * 180 / Math.PI; }
    function set(v) { p[i] = Math.max(0, Math.min(1, v)); trail.push(p.slice()); if (trail.length > 600) trail.shift(); render(); }
    k.addEventListener("pointerdown", function (e) { dragging = i; last = ang(e); k.setPointerCapture(e.pointerId); vel = [0, 0]; e.preventDefault(); });
    k.addEventListener("pointermove", function (e) {
      if (dragging !== i) return;
      var a = ang(e), d = a - last; if (d > 180) d -= 360; if (d < -180) d += 360; last = a;
      set(p[i] + d / (A1 - A0));
    });
    k.addEventListener("pointerup", function () { dragging = null; });
    k.addEventListener("pointercancel", function () { dragging = null; });
    k.addEventListener("wheel", function (e) { e.preventDefault(); set(p[i] - Math.sign(e.deltaY) * 0.01); }, { passive: false });
    return o;
  });
  function drawKnobs() {
    knobs.forEach(function (k) {
      var deg = A0 + p[k.i] * (A1 - A0), e = polar(32, deg);
      k.arc.setAttribute("d", p[k.i] > 0.003 ? arc(54, A0, deg) : "");
      k.ind.setAttribute("x2", e[0]); k.ind.setAttribute("y2", e[1]);
      k.el.classList.toggle("auto", training);
    });
    knobs[0].val.textContent = euro((B_MIN + p[1] * (B_MAX - B_MIN)) * 1000);
    knobs[1].val.textContent = (Math.round(p[0] * S_MAX * 1000 / 10) * 10).toLocaleString("it-IT") + " €/m²";
  }

  // ---------- errore e mappa ----------
  var errEl = root.querySelector(".err"), barEl = root.querySelector(".err-bar span"), stepsEl = root.querySelector(".steps");
  var cv = root.querySelector(".land"), cx = cv.getContext("2d"), N = 160, land = null;
  function computeLand() {
    var img = new ImageData(N, N), vals = [], lo = Infinity, hi = 0, j, i, k;
    for (j = 0; j < N; j++) for (i = 0; i < N; i++) { var v = Math.log(rmse([i / (N - 1), 1 - j / (N - 1)]) + 1); vals.push(v); lo = Math.min(lo, v); hi = Math.max(hi, v); }
    for (k = 0; k < vals.length; k++) {
      var t = (vals[k] - lo) / (hi - lo), band = Math.floor(t * 16) % 2 ? 0.8 : 1, u = Math.pow(1 - t, 2.2);
      img.data[k * 4] = (12 + 22 * u) * band + 0; img.data[k * 4 + 1] = (20 + 191 * u) * band; img.data[k * 4 + 2] = (34 + 204 * u) * band; img.data[k * 4 + 3] = 255;
    }
    land = document.createElement("canvas"); land.width = N; land.height = N; land.getContext("2d").putImageData(img, 0, 0);
  }
  function drawLand() {
    var r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1, w = Math.round(r.width * dpr), h = Math.round(r.height * dpr);
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    cx.imageSmoothingEnabled = true; cx.drawImage(land, 0, 0, w, h);
    cx.lineWidth = 2 * dpr; cx.strokeStyle = "rgba(251,146,60,.7)"; cx.beginPath();
    trail.forEach(function (q, i) { var x = q[0] * w, y = (1 - q[1]) * h; i ? cx.lineTo(x, y) : cx.moveTo(x, y); }); cx.stroke();
    cx.beginPath(); cx.arc(p[0] * w, (1 - p[1]) * h, 8 * dpr, 0, 7);
    cx.fillStyle = "#FB923C"; cx.shadowColor = "#FB923C"; cx.shadowBlur = 14 * dpr; cx.fill(); cx.shadowBlur = 0;
    cx.lineWidth = 2.5 * dpr; cx.strokeStyle = "#0A0E15"; cx.stroke();
  }
  function drawErr() {
    var e = rmse(p);
    errEl.textContent = "± " + euro(e * 1000);
    barEl.style.width = Math.min(100, e / 2) + "%"; barEl.style.background = errColor(e / 150);
    stepsEl.textContent = steps ? steps + " passi di addestramento" : "";
  }

  // ---------- addestramento ----------
  function gradStep() {
    var g = [0, 0], n = pts.length;
    for (var i = 0; i < n; i++) { var x = pts[i][0], r = predict(x) - pts[i][1]; g[0] += 2 * r * S_MAX * x / n; g[1] += 2 * r * (B_MAX - B_MIN) / n; }
    var lr = 0.25 / 330000, beta = 0.86;
    for (var k = 0; k < 2; k++) { vel[k] = beta * vel[k] - lr * g[k]; p[k] = Math.max(0, Math.min(1, p[k] + vel[k])); }
    steps++; trail.push(p.slice()); if (trail.length > 600) trail.shift();
  }
  function render() { drawPlot(); drawKnobs(); drawErr(); drawLand(); }
  var raf = null, frame = 0;
  function loop() { raf = null; if (!training) return; if (dragging === null && frame++ % 3 === 0) gradStep(); render(); raf = requestAnimationFrame(loop); }
  var tr = root.querySelector(".train");
  tr.addEventListener("change", function () { training = tr.checked; vel = [0, 0]; if (training && !raf) raf = requestAnimationFrame(loop); render(); });
  root.querySelector(".shuffle").addEventListener("click", function () { p = [Math.random(), Math.random()]; vel = [0, 0]; steps = 0; trail = [p.slice()]; render(); });
  root.querySelector(".data").addEventListener("click", function () { newData(); render(); });
  window.addEventListener("resize", render);

  newData(); render();
})();
