/* La lettera successiva: mini transformer a caratteri (RoPE, uscita condivisa) con visualizzazione dell'attenzione */
(function () {
  "use strict";
  var root = document.getElementById("sim");
  if (!window.CHARGPT_MODEL) { root.innerHTML = '<p class="muted" style="margin:auto">Modello non trovato (modelli/chargpt.js).</p>'; return; }

  var M = CHARGPT_MODEL, CTX = M.meta.ctx, D = M.meta.d, NH = M.meta.heads, NL = M.meta.layers, DH = D / NH, HALF = DH / 2;
  var VOC = M.vocab, V = VOC.length, stoi = {};
  for (var c = 0; c < V; c++) stoi[VOC[c]] = c;

  // ---------- pesi ----------
  function b64(s) { var b = atob(s), u = new Uint8Array(b.length); for (var i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return u; }
  var W = {};
  Object.keys(M.w).forEach(function (k) {
    var t = M.w[k];
    if (t.f) { W[k] = new Float32Array(b64(t.f).buffer); return; }
    var q = new Int8Array(b64(t.q).buffer), s = new Float32Array(b64(t.s).buffer), cols = t.shape[1], out = new Float32Array(q.length);
    for (var i = 0; i < q.length; i++) out[i] = q[i] * s[(i / cols) | 0];
    W[k] = out;
  });

  // ---------- RoPE: tabelle di coseni e seni per posizione ----------
  var COS = new Float32Array(CTX * HALF), SIN = new Float32Array(CTX * HALF);
  for (var pos = 0; pos < CTX; pos++) for (var i = 0; i < HALF; i++) {
    var a = pos / Math.pow(M.meta.rope_base || 10000, 2 * i / DH);
    COS[pos * HALF + i] = Math.cos(a); SIN[pos * HALF + i] = Math.sin(a);
  }
  function ropeInPlace(buf, T, stride, off) { // ruota q o k di ogni testa
    for (var t = 0; t < T; t++) for (var h = 0; h < NH; h++) {
      var b = t * stride + off + h * DH;
      for (var i = 0; i < HALF; i++) {
        var x1 = buf[b + i], x2 = buf[b + i + HALF], co = COS[t * HALF + i], si = SIN[t * HALF + i];
        buf[b + i] = x1 * co - x2 * si; buf[b + i + HALF] = x2 * co + x1 * si;
      }
    }
  }

  // ---------- operazioni ----------
  function linear(x, T, I, w, b, O) {
    var y = new Float32Array(T * O);
    for (var t = 0; t < T; t++) { var xo = t * I; for (var o = 0; o < O; o++) { var s = b ? b[o] : 0, wo = o * I; for (var i = 0; i < I; i++) s += x[xo + i] * w[wo + i]; y[t * O + o] = s; } }
    return y;
  }
  function layernorm(x, T, g, b) {
    var y = new Float32Array(x.length);
    for (var t = 0; t < T; t++) {
      var o = t * D, m = 0, v = 0, i;
      for (i = 0; i < D; i++) m += x[o + i]; m /= D;
      for (i = 0; i < D; i++) { var d = x[o + i] - m; v += d * d; }
      var r = 1 / Math.sqrt(v / D + 1e-5);
      for (i = 0; i < D; i++) y[o + i] = (x[o + i] - m) * r * g[i] + b[i];
    }
    return y;
  }
  function gelu(v) { return 0.5 * v * (1 + Math.tanh(0.7978845608 * (v + 0.044715 * v * v * v))); }

  function forward(ids) {
    var T = ids.length, x = new Float32Array(T * D), att = [], E = W["tok.weight"];
    for (var t = 0; t < T; t++) for (var i = 0; i < D; i++) x[t * D + i] = E[ids[t] * D + i];
    for (var l = 0; l < NL; l++) {
      var P = "blocks." + l + ".";
      var h = layernorm(x, T, W[P + "ln1.weight"], W[P + "ln1.bias"]);
      var qkv = linear(h, T, D, W[P + "qkv.weight"], W[P + "qkv.bias"], 3 * D);
      ropeInPlace(qkv, T, 3 * D, 0); ropeInPlace(qkv, T, 3 * D, D);
      var y = new Float32Array(T * D), A = [], sc = 1 / Math.sqrt(DH);
      for (var hd = 0; hd < NH; hd++) {
        var a = new Float32Array(T * T);
        for (var qi = 0; qi < T; qi++) {
          var mx = -1e9, row = qi * T, ki, dd;
          for (ki = 0; ki <= qi; ki++) {
            var s = 0;
            for (dd = 0; dd < DH; dd++) s += qkv[qi * 3 * D + hd * DH + dd] * qkv[ki * 3 * D + D + hd * DH + dd];
            s *= sc; a[row + ki] = s; if (s > mx) mx = s;
          }
          var sum = 0;
          for (ki = 0; ki <= qi; ki++) { a[row + ki] = Math.exp(a[row + ki] - mx); sum += a[row + ki]; }
          for (ki = 0; ki <= qi; ki++) a[row + ki] /= sum;
          for (dd = 0; dd < DH; dd++) {
            var acc = 0;
            for (ki = 0; ki <= qi; ki++) acc += a[row + ki] * qkv[ki * 3 * D + 2 * D + hd * DH + dd];
            y[qi * D + hd * DH + dd] = acc;
          }
        }
        A.push(a);
      }
      att.push(A);
      var pr = linear(y, T, D, W[P + "proj.weight"], W[P + "proj.bias"], D);
      for (i = 0; i < x.length; i++) x[i] += pr[i];
      h = layernorm(x, T, W[P + "ln2.weight"], W[P + "ln2.bias"]);
      var f = linear(h, T, D, W[P + "fc.weight"], W[P + "fc.bias"], 4 * D);
      for (i = 0; i < f.length; i++) f[i] = gelu(f[i]);
      var f2 = linear(f, T, 4 * D, W[P + "fc2.weight"], W[P + "fc2.bias"], D);
      for (i = 0; i < x.length; i++) x[i] += f2[i];
    }
    var hf = layernorm(x, T, W["lnf.weight"], W["lnf.bias"]);
    return { logits: linear(hf, T, D, E, W["head_bias"], V), att: att, T: T }; // logits per ogni posizione: T × V
  }

  // ---------- stato e interfaccia ----------
  var $ = function (s) { return root.querySelector(s); };
  var input = $(".prompt"), tilesEl = $(".tiles"), barsEl = $(".bars"), ptitle = $(".ptitle");
  var tempIn = $(".temp"), tout = $(".tout"), layerSel = $(".layer"), headSel = $(".head"), btnAuto = $(".auto");
  var START = "era una sera d'autunno e ", SHOW = 320;
  var text = "", res = null, ctxStart = 0, hover = null, newFrom = Infinity, timer = null;

  for (var l = 0; l < NL; l++) layerSel.add(new Option("strato " + (l + 1), l));
  layerSel.value = NL - 1;
  headSel.add(new Option("tutte le teste", -1));
  for (var h = 0; h < NH; h++) headSel.add(new Option("testa " + (h + 1), h));

  function clean(s) {
    s = s.toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "").replace(/[’‘`´]/g, "'");
    return s.split("").filter(function (ch) { return ch in stoi; }).join("");
  }
  function softmax(lg, off, temp) {
    var out = new Float32Array(V), m = -1e9, s = 0, i;
    for (i = 0; i < V; i++) m = Math.max(m, lg[off + i] / temp);
    for (i = 0; i < V; i++) { out[i] = Math.exp(lg[off + i] / temp - m); s += out[i]; }
    for (i = 0; i < V; i++) out[i] /= s;
    return out;
  }
  function run() {
    ctxStart = Math.max(0, text.length - CTX);
    var ctx = text.slice(ctxStart);
    res = ctx.length ? forward(ctx.split("").map(function (ch) { return stoi[ch]; })) : null;
  }
  function sample() {
    var off = (res.T - 1) * V, t = +tempIn.value, i;
    if (t < 0.05) { var b = 0; for (i = 1; i < V; i++) if (res.logits[off + i] > res.logits[off + b]) b = i; return VOC[b]; }
    var p = softmax(res.logits, off, t), r = Math.random(), acc = 0;
    for (i = 0; i < V; i++) { acc += p[i]; if (r < acc) return VOC[i]; }
    return VOC[V - 1];
  }

  // query = posizione che "guarda"; target = lettera che sta prevedendo
  function view() {
    if (hover !== null && hover - 1 >= ctxStart) return { q: hover - 1, target: hover };
    return { q: text.length - 1, target: null };
  }
  function attRow(qAbs) {
    var L = +layerSel.value, H = +headSel.value, T = res.T, qi = qAbs - ctxStart, w = new Float32Array(qi + 1);
    var hs = H < 0 ? Array.from({ length: NH }, function (_, k) { return k; }) : [H];
    hs.forEach(function (hh) { var a = res.att[L][hh]; for (var k = 0; k <= qi; k++) w[k] += a[qi * T + k] / hs.length; });
    return w;
  }

  function draw() {
    var shown = text.slice(-SHOW), start = text.length - shown.length, v = view(), w = res ? attRow(v.q) : null, mx = 0;
    if (w) for (var k = 0; k < w.length; k++) mx = Math.max(mx, w[k]);
    var html = "";
    for (var i = 0; i < shown.length; i++) {
      var abs = start + i, ch = shown[i], cls = "tile", style = "";
      if (ch === " ") cls += " sp";
      if (abs < ctxStart) cls += " out";
      if (abs === v.q) cls += " q";
      if (abs === v.target) cls += " tg";
      if (abs >= newFrom) cls += " new";
      if (w && abs >= ctxStart && abs <= v.q) {
        var x = Math.sqrt(w[abs - ctxStart] / (mx || 1));
        style = ' style="background:rgba(251,146,60,' + (0.04 + 0.96 * x).toFixed(3) + ')' + (x > .65 ? ';color:#1A0D03' : '') + '"';
      }
      html += '<span class="' + cls + '" data-i="' + abs + '"' + style + '>' + (ch === " " ? "·" : ch) + '</span>';
    }
    if (v.target === null) html += '<span class="tile next">?</span>';
    tilesEl.innerHTML = html;
    newFrom = Infinity;
    drawProbs(v);
  }

  var rows = [];
  for (var r = 0; r < 10; r++) {
    var row = document.createElement("div"); row.className = "bar";
    row.innerHTML = '<span class="k"></span><span class="t"><span class="f"></span></span><span class="v"></span>';
    row.addEventListener("click", function () { if (this.dataset.ch && hover === null) append(this.dataset.ch); });
    barsEl.appendChild(row); rows.push(row);
  }
  function drawProbs(v) {
    if (!res) { rows.forEach(function (row) { row.style.visibility = "hidden"; }); ptitle.textContent = ""; return; }
    var p = softmax(res.logits, (v.q - ctxStart) * V, 1), order = [];
    for (var i = 0; i < V; i++) order.push(i);
    order.sort(function (a, b) { return p[b] - p[a]; });
    var chosen = v.target !== null ? text[v.target] : null;
    if (chosen !== null && order.slice(0, 10).indexOf(stoi[chosen]) < 0) order[9] = stoi[chosen]; // la lettera vera sempre visibile
    rows.forEach(function (row, j) {
      var id = order[j], ch = VOC[id];
      row.style.visibility = "";
      row.className = "bar" + (j === 0 ? " top" : "") + (ch === chosen ? " chosen" : "");
      row.dataset.ch = ch;
      row.querySelector(".k").textContent = ch === " " ? "␣" : ch;
      row.querySelector(".f").style.width = (p[id] * 100).toFixed(1) + "%";
      row.querySelector(".v").textContent = (p[id] * 100).toFixed(p[id] < .1 ? 1 : 0) + "%";
    });
    ptitle.innerHTML = chosen !== null ? 'qui ha scritto <b>' + (chosen === " " ? "␣ (spazio)" : chosen) + '</b>, probabilità ' + (p[stoi[chosen]] * 100).toFixed(1).replace(".", ",") + '%' : "␣ = spazio";
  }

  function append(ch) { newFrom = text.length; text += ch; input.value = text.slice(-200); run(); draw(); }
  function step() { if (res) append(sample()); }
  function stopAuto() { clearInterval(timer); timer = null; btnAuto.classList.remove("on"); btnAuto.textContent = "▶ Scrivi da solo"; }
  function startAuto() { btnAuto.classList.add("on"); btnAuto.textContent = "❚❚ Pausa"; timer = setInterval(function () { step(); if (text.length > 3000) stopAuto(); }, 40); }
  function reset(s) { stopAuto(); text = clean(s); input.value = text; hover = null; run(); draw(); }

  $(".step").addEventListener("click", function () { stopAuto(); step(); });
  btnAuto.addEventListener("click", function () { timer ? stopAuto() : startAuto(); });
  $(".reset").addEventListener("click", function () { reset(START); });
  input.addEventListener("input", function () { stopAuto(); var cl = clean(input.value); if (cl !== input.value) input.value = cl; text = cl; run(); draw(); });
  input.addEventListener("keydown", function (e) { if (e.key === "Enter") { e.preventDefault(); step(); } });
  tempIn.addEventListener("input", function () { tout.textContent = +tempIn.value < 0.05 ? "0" : (+tempIn.value).toFixed(1); });
  [layerSel, headSel].forEach(function (s) { s.addEventListener("change", draw); });
  tilesEl.addEventListener("mouseover", function (e) {
    var t = e.target.closest(".tile"); if (!t || !t.dataset.i) return;
    var i = +t.dataset.i; if (hover === i) return; hover = i; draw();
  });
  tilesEl.addEventListener("mouseleave", function () { hover = null; draw(); });

  tempIn.dispatchEvent(new Event("input"));
  reset(START);
})();
