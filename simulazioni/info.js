/* Pannello informazioni: pulsante .info-btn apre #info; si chiude con ×, clic fuori o Esc. Tasto "i" per aprirlo. */
(function () {
  var ov = document.getElementById("info"), btn = document.querySelector(".info-btn");
  if (!ov || !btn) return;
  function open() { ov.classList.add("open"); }
  function close() { ov.classList.remove("open"); }
  btn.addEventListener("click", open);
  ov.addEventListener("click", function (e) { if (e.target === ov || e.target.closest(".info-close")) close(); });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") close();
    var tag = (e.target.tagName || "").toLowerCase();
    if ((e.key === "i" || e.key === "I") && tag !== "input" && tag !== "textarea") { ov.classList.contains("open") ? close() : open(); }
  });
})();
