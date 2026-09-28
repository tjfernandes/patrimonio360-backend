/*
 * Património 360 — «focar» uma peça na visita 3DVista.
 *
 * Instalado em /root/tours/tour-focus-artifact.js no VPS e carregado por cada
 * visita (mnt, mnsr, mnaz, mj) no <head>, ANTES do script inline que trata o
 * comando `navigateToArtifact` (o handler antigo, que abre a janela da peça).
 *
 * Comando (postMessage do widget):
 *   { type: "navigateToArtifact", mode: "focus", overlayId, panoramaKey, inventoryId? }
 *   ou { type: "focusArtifact", overlayId, panoramaKey }
 *
 * O que faz: fecha janelas abertas (e o painel de boas-vindas, se ainda
 * estiver à frente), vai para o panorama com a câmara virada
 * para o hotspot (setPanoramaCameraWithSpot do 3DVista; movimento suave se já
 * estiver no panorama), NÃO abre a janela da peça, e destaca o ícone do
 * hotspot durante uns segundos com uma pulsação de tamanho («breathing»). A
 * imagem do ícone não é tocada: mantém a cor, a animação (os ícones animados
 * do 3DVista são folhas de sprites) e a orientação. O clique do visitante no
 * hotspot continua a abrir a ficha como sempre.
 *
 * Como o listener é registado em fase de captura e chama
 * stopImmediatePropagation(), o handler antigo nunca vê o comando em modo
 * «focus». Sem `mode` (botão «Ir para a sala», deep-links) tudo fica como era.
 *
 * Verificado a 2026-09-16 no player do Museu do Traje: yaw/pitch e
 * width/height dos hotspots alteráveis em tempo real.
 */
(function () {
  "use strict";

  var VERSION = "2026-09-28.1";
  var HIGHLIGHT_MS = 10000;   // duração do destaque
  var PERIOD_MS = 1300;       // período da pulsação
  var SCALE_MAX = 1.65;       // tamanho máximo relativo ao original
  var WAIT_MEDIA_MS = 8000;   // máximo à espera de o panorama ficar ativo
  var INIT_WAIT_MS = 180000;  // máximo à espera de o tour inicializar

  var current = null;   // destaque em curso

  function log() {
    try { console.log.apply(console, ["[p360 focus]"].concat([].slice.call(arguments))); } catch (e) {}
  }

  function getTour() {
    try {
      return window.tour || (window.TDV && window.TDV.PlayerAPI && window.TDV.PlayerAPI.getInstance());
    } catch (e) { return null; }
  }

  function getRootPlayer(tour) {
    if (!tour) return null;
    try { return tour._getRootPlayer ? tour._getRootPlayer() : tour; } catch (e) { return tour; }
  }

  function getValue(obj, key) {
    if (!obj) return undefined;
    if (typeof obj.get === "function") {
      try { var v = obj.get(key); if (v !== undefined) return v; } catch (e) {}
    }
    if (obj[key] !== undefined) return obj[key];
    if (obj.tq && obj.tq[key] !== undefined) return obj.tq[key];
    return undefined;
  }

  function resolveRef(rp, value) {
    if (typeof value !== "string") return value;
    return rp[value.replace(/^this\./, "")] || null;
  }

  function getLabel(obj) {
    var data = getValue(obj, "data") || (obj && obj.tq && obj.tq.data) || {};
    return getValue(obj, "label") || data.label || null;
  }

  // yaw/pitch do hotspot, como o focusOverlayByName do 3DVista
  function overlaySpot(rp, overlay) {
    if (!overlay) return null;
    var cls = getValue(overlay, "class") || "";
    var item = (cls === "VideoPanoramaOverlay" || cls === "QuadVideoPanoramaOverlay")
      ? overlay
      : (getValue(overlay, "items") || [])[0];
    item = resolveRef(rp, item);
    if (!item) return null;
    var icls = getValue(item, "class") || "";
    if (icls === "QuadVideoPanoramaOverlay" || icls === "QuadHotspotPanoramaOverlayImage") {
      var vertices = getValue(item, "vertices") || [];
      var yaw = 0, pitch = 0;
      for (var i = 0; i < vertices.length; i++) {
        var v = resolveRef(rp, vertices[i]);
        var y = getValue(v, "yaw");
        if (y < 0) y += 360;
        yaw += y; pitch += getValue(v, "pitch");
      }
      if (vertices.length) { yaw /= vertices.length; pitch /= vertices.length; }
      if (yaw > 180) yaw -= 360;
      return { item: item, yaw: yaw, pitch: pitch };
    }
    var iy = getValue(item, "yaw"), ip = getValue(item, "pitch");
    if (typeof iy !== "number" || typeof ip !== "number") return null;
    return { item: item, yaw: iy, pitch: ip };
  }

  function closeWindows(rp) {
    var windows = [];
    try { windows = typeof rp.getByClassName === "function" ? rp.getByClassName("Window") || [] : []; } catch (e) {}
    for (var i = 0; i < windows.length; i++) {
      var win = windows[i];
      try {
        if (typeof win.set === "function") win.set("visible", false);
        else if (typeof win.trigger === "function") win.trigger("close");
        else if (typeof win.hide === "function") win.hide();
      } catch (e) {}
    }
    // Fechar a ficha por código não passa pelo resumePlayers da visita, que é
    // quem avisa o assistente, e uma ficha aberta não se distingue pelo
    // `visible` (fica sempre false; visto a 2026-09-17). Como o modo «focus»
    // nunca deixa uma ficha aberta, avisa sempre: sem isto a peça ficava
    // «aberta» no widget.
    try {
      if (window.P360TourEventBridge && window.P360TourEventBridge.handleClosedInfoWindow) {
        window.P360TourEventBridge.handleClosedInfoWindow({ source: "focus_command" });
      }
    } catch (e) {}
    try { if (window.parent && window.parent !== window) window.parent.postMessage({ type: "deselectedArtifacts" }, "*"); } catch (e) {}
  }

  // Painel de boas-vindas (no MNT: Container «Welcome Panel», a 100%, com o
  // texto «Welcome Text»; as outras visitas não têm). Tapa o panorama, por
  // isso quem carrega em «Ver na tour» antes de o fechar não veria nada.
  // Fecha-se como a própria visita faz no clique (setComponentVisibility).
  var WELCOME_NAME = /welcome|bem.?vind|boas.?vind/i;
  function componentName(component) {
    try { var data = getValue(component, "data"); return data && data.name ? String(data.name) : ""; } catch (e) { return ""; }
  }
  function closeWelcome(rp) {
    var containers = [];
    try { containers = typeof rp.getByClassName === "function" ? rp.getByClassName("Container") || [] : []; } catch (e) {}
    for (var i = 0; i < containers.length; i++) {
      var c = containers[i];
      try {
        if (getValue(c, "class") !== "Container" || !getValue(c, "visible")) continue;
        var named = WELCOME_NAME.test(componentName(c));
        var kids = getValue(c, "children") || [];
        for (var k = 0; !named && k < kids.length; k++) named = WELCOME_NAME.test(componentName(kids[k]));
        if (!named) continue;
        if (typeof rp.setComponentVisibility === "function") rp.setComponentVisibility(c, false, 0, null, "hideEffect", false);
        else c.set("visible", false);
        log("painel de boas-vindas fechado:", getValue(c, "id"));
      } catch (e) {}
    }
  }

  function activeMedia(rp) {
    try {
      var list = rp.mainPlayList;
      var items = getValue(list, "items") || [];
      var idx = getValue(list, "selectedIndex");
      var item = resolveRef(rp, items[idx]);
      return { index: idx, media: item ? resolveRef(rp, getValue(item, "media")) : null };
    } catch (e) { return { index: -1, media: null }; }
  }

  // A versão nova do 3DVista (MNT republicado a 2026-09-18) aceita a chamada
  // setPanoramaCameraWithSpot sem erro mas não mexe a câmara. Depois de o
  // panorama ficar ativo, confirma-se a mira e, se ficou longe, aponta-se
  // diretamente no player (yaw/pitch aceitam escrita em qualquer das versões).
  function mainViewer(rp) {
    try { return (rp.getByClassName("PanoramaPlayer") || [])[0] || null; } catch (e) { return null; }
  }

  function angleGap(a, b) {
    var d = Math.abs((a - b) % 360);
    return d > 180 ? 360 - d : d;
  }

  function aimAtSpot(rp, spot) {
    var viewer = mainViewer(rp);
    if (!viewer || !spot) return;
    var yaw = getValue(viewer, "yaw"), pitch = getValue(viewer, "pitch");
    if (typeof yaw === "number" && typeof pitch === "number"
        && angleGap(yaw, spot.yaw) < 3 && angleGap(pitch, spot.pitch) < 3) return;
    try {
      viewer.set("yaw", spot.yaw);
      viewer.set("pitch", spot.pitch);
      log("câmara apontada diretamente (a API do tour não moveu)");
    } catch (e) { log("não consegui apontar a câmara:", e); }
  }

  function mainViewerHfov(rp) {
    try {
      var players = rp.getByClassName("PanoramaPlayer") || [];
      var h = players.length ? getValue(players[0], "hfov") : null;
      return typeof h === "number" && h > 20 && h < 120 ? h : 75;
    } catch (e) { return 75; }
  }

  // ---------- destaque ----------

  function stopHighlight() {
    if (!current) return;
    var c = current; current = null;
    try { if (c.raf) cancelAnimationFrame(c.raf); } catch (e) {}
    try { clearTimeout(c.timer); clearInterval(c.poll); } catch (e) {}
    try {
      var o = c.originals, it = c.item;
      if (o.width !== undefined) it.set("width", o.width);
      if (o.height !== undefined) it.set("height", o.height);
      if (o.hfov !== undefined) it.set("hfov", o.hfov);
      if (o.vfov !== undefined) it.set("vfov", o.vfov);
      if (o.yaw !== undefined) it.set("yaw", o.yaw);
      if (o.pitch !== undefined) it.set("pitch", o.pitch);
    } catch (e) {}
    log("destaque terminado", c.reason || "");
  }

  // Um hotspot «flat» cresce a partir do canto superior esquerdo (medido a
  // 2026-09-16: 60→180 px desloca o centro 60 px para a direita e para
  // baixo). Para a pulsação ficar centrada, o ponto de ancoragem (yaw/pitch)
  // recua metade do acréscimo, convertido de píxeis para graus com a
  // projeção retilínea do viewer: px = f·tan(θ), f = (largura/2)/tan(hfov/2).
  function pxToDeg(rp, px) {
    try {
      var viewer = (rp.getByClassName("PanoramaPlayer") || [])[0];
      var hfov = getValue(viewer, "hfov");
      var width = getValue(viewer, "actualWidth") || window.innerWidth;
      if (!(hfov > 0) || !(width > 0)) return 0;
      var f = (width / 2) / Math.tan(hfov * Math.PI / 360);
      return Math.atan(px / f) * 180 / Math.PI;
    } catch (e) { return 0; }
  }

  function startHighlight(rp, overlay, spot, panoramaIndex) {
    stopHighlight();
    var item = spot.item;
    var originals = {
      width: getValue(item, "width"), height: getValue(item, "height"),
      hfov: getValue(item, "hfov"), vfov: getValue(item, "vfov"),
      yaw: getValue(item, "yaw"), pitch: getValue(item, "pitch"),
    };
    var anchored = typeof originals.yaw === "number" && typeof originals.pitch === "number";
    var cosPitch = anchored ? (Math.cos(originals.pitch * Math.PI / 180) || 1) : 1;
    var usesPixels = typeof originals.width === "number" && typeof originals.height === "number";
    var usesFov = !usesPixels && typeof originals.hfov === "number";
    if (!usesPixels && !usesFov) { log("hotspot sem tamanho alterável; sem pulsação"); }
    var state = { item: item, overlay: overlay, originals: originals, raf: 0, timer: 0, poll: 0, reason: "" };
    current = state;
    var t0 = performance.now();

    function frame() {
      if (current !== state) return;
      var phase = 0.5 - 0.5 * Math.cos(2 * Math.PI * ((performance.now() - t0) / PERIOD_MS));
      var scale = 1 + (SCALE_MAX - 1) * phase;
      try {
        if (usesPixels) {
          var w = originals.width * scale, h = originals.height * scale;
          item.set("width", w); item.set("height", h);
          if (anchored) {
            item.set("yaw", originals.yaw - pxToDeg(rp, (w - originals.width) / 2) / cosPitch);
            item.set("pitch", originals.pitch + pxToDeg(rp, (h - originals.height) / 2));
          }
        }
        else if (usesFov) { item.set("hfov", originals.hfov * scale); if (typeof originals.vfov === "number") item.set("vfov", originals.vfov * scale); }
      } catch (e) {}
      state.raf = requestAnimationFrame(frame);
    }
    state.raf = requestAnimationFrame(frame);
    state.timer = setTimeout(function () { state.reason = "tempo"; stopHighlight(); }, HIGHLIGHT_MS);
    // termina se o visitante mudar de panorama ou abrir uma janela (a ficha)
    state.poll = setInterval(function () {
      if (current !== state) return;
      var now = activeMedia(rp);
      if (now.index !== panoramaIndex) { state.reason = "mudou de panorama"; stopHighlight(); return; }
      try {
        var wins = rp.getByClassName("Window") || [];
        for (var i = 0; i < wins.length; i++) if (getValue(wins[i], "visible") === true) { state.reason = "janela aberta"; stopHighlight(); return; }
      } catch (e) {}
    }, 300);
  }

  // ---------- reparação das ações da visita (sem tocar nos ficheiros) ----------
  //
  // No MNAz o `location` de cada ação foi escrito com aspas curvas
  // («location: “20_Expo_Temp_P3”»): o eval da visita rebenta, o try/catch dela
  // engole o erro e o aviso `openedInfoWindow` nunca sai — o assistente nunca
  // sabe que peça foi aberta. Editar o script_general.js resolvia, mas invalida
  // o hash do projeto e o player passa a mostrar a marca de água do 3DVista
  // (visto a 2026-09-18). Por isso a reparação é feita aqui: lê-se a ação, e
  // quando a visita abre a janela damos nós o aviso, com os números de
  // inventário corrigidos (../inventarios-corrigidos.json, do
  // tools/poi_mapping/validar_inventarios_poi.py).

  var CURVAS = /[“”‘’]/;
  var acoesPorJanela = null;   // window_XXX -> { ids, titulo, local }
  var inventariosCorrigidos = {};
  var ultimoAvisoDaVisita = 0;

  try {
    window.addEventListener("message", function (evento) {
      if (evento.data && evento.data.type === "openedInfoWindow") ultimoAvisoDaVisita = Date.now();
    });
  } catch (e) {}

  function museuAtual() {
    var partes = String(location.pathname || "").split("/").filter(Boolean);
    partes = partes.filter(function (p) { return p.indexOf(".html") === -1; });
    return partes.length ? partes[partes.length - 1] : "";
  }

  function carregarInventarios() {
    try {
      var pedido = new XMLHttpRequest();
      pedido.open("GET", "../inventarios-corrigidos.json", true);
      pedido.onload = function () {
        try { inventariosCorrigidos = (JSON.parse(pedido.responseText) || {})[museuAtual()] || {}; } catch (e) {}
      };
      pedido.send();
    } catch (e) {}
  }

  function lerAcao(texto) {
    if (!texto || texto.indexOf("openedInfoWindow") === -1) return null;
    var janela = /showWindowBase\(this\.(window_[A-Za-z0-9_]+)/.exec(texto);
    var lista = /inventoryIds:\s*\[([^\]]*)\]/.exec(texto);
    if (!janela || !lista) return null;
    var ids = [], re = /\\?["'“‘]([^"'”’\\]+)\\?["'”’]/g, m;
    while ((m = re.exec(lista[1])) !== null) ids.push(m[1]);
    if (!ids.length) return null;
    var titulo = /title:\s*\\?["'“]([^"'”\\]*)/.exec(texto);
    var local = /location:\s*\\?["'“]([^"'”\\]*)/.exec(texto);
    return {
      janela: janela[1], ids: ids,
      titulo: titulo ? titulo[1] : null,
      local: local ? local[1] : null,
      partida: CURVAS.test(texto),
    };
  }

  // As ações vêm do script_general.js (o modelo em memória desta versão não as
  // expõe: 410 panoramas, 0 ações). O ficheiro já está na cache do browser.
  var ACAO_RE = /showWindowBase\(this\.(window_[A-Za-z0-9_]+)[\s\S]{0,600}?openedInfoWindow[\s\S]{0,400}?inventoryIds:\s*\[([^\]]*)\]([\s\S]{0,400}?)\}/g;

  function indexarDoFicheiro(texto) {
    var mapa = {}, partidas = 0, total = 0, m;
    ACAO_RE.lastIndex = 0;
    while ((m = ACAO_RE.exec(texto)) !== null) {
      var ids = [], re = /\\?["'\u201c\u2018]([^"'\u201d\u2019\\]+)\\?["'\u201d\u2019]/g, x;
      while ((x = re.exec(m[2])) !== null) ids.push(x[1]);
      if (!ids.length) continue;
      var resto = m[3] || "";
      var titulo = /title:\s*\\?["'\u201c]([^"'\u201d\\]*)/.exec(resto);
      var local = /location:\s*\\?["'\u201c]([^"'\u201d\\]*)/.exec(resto);
      var partida = /[\u201c\u201d\u2018\u2019]/.test(resto);
      mapa[m[1]] = {
        janela: m[1], ids: ids,
        titulo: titulo ? titulo[1] : null,
        local: local ? local[1] : null,
        partida: partida,
      };
      total++;
      if (partida) partidas++;
    }
    log("ações lidas do script_general.js:", total, "| com eval partido (aspas curvas):", partidas);
    return mapa;
  }

  function indexarAcoes() {
    if (acoesPorJanela) return acoesPorJanela;
    acoesPorJanela = {};   // fica vazio até o ficheiro chegar
    try {
      var pedido = new XMLHttpRequest();
      pedido.open("GET", "script_general.js", true);
      pedido.onload = function () { try { acoesPorJanela = indexarDoFicheiro(pedido.responseText || ""); } catch (e) { log("não consegui ler as ações:", e); } };
      pedido.send();
    } catch (e) { log("não consegui pedir o script_general.js:", e); }
    return acoesPorJanela;
  }

  function avisarFichaAberta(rp, janelaId) {
    var acao = indexarAcoes()[janelaId];
    if (!acao || !acao.partida) return;   // a visita avisa sozinha: não duplicar
    var quando = Date.now();
    setTimeout(function () {
      if (ultimoAvisoDaVisita >= quando) return;   // afinal a visita avisou
      var ids = [];
      for (var i = 0; i < acao.ids.length; i++) {
        var corrigidos = inventariosCorrigidos[acao.ids[i]];
        if (corrigidos && corrigidos.length) ids = ids.concat(corrigidos);
        else ids.push(acao.ids[i]);
      }
      try {
        window.postMessage({ type: "openedInfoWindow", inventoryIds: ids, title: acao.titulo, location: acao.local }, "*");
        log("aviso da ficha aberta dado por nós:", ids.join(", "));
      } catch (e) {}
    }, 250);
  }

  function repararAcoes(rp) {
    if (rp.__p360AcoesReparadas || typeof rp.showWindowBase !== "function") return;
    rp.__p360AcoesReparadas = true;
    indexarAcoes();
    var original = rp.showWindowBase;
    rp.showWindowBase = function (janela) {
      var resultado = original.apply(this, arguments);
      try { avisarFichaAberta(rp, getValue(janela, "id") || (janela && janela.id)); } catch (e) {}
      return resultado;
    };
  }

  carregarInventarios();

  // ---------- navegação ----------

  function focus(data) {
    var tour = getTour();
    var rp = getRootPlayer(tour);
    if (!tour || !rp) { log("tour ainda não disponível"); return false; }
    repararAcoes(rp);
    // Pronto = o que o comando precisa já existe (a lista de panoramas e o
    // panorama pedido). `_isInitialized` pode ficar false muito depois disso.
    var ready = false;
    try { ready = (getValue(rp.mainPlayList, "items") || []).length > 0 && !!rp[data.panoramaKey]; } catch (e) {}
    if (!ready && tour._isInitialized === false) {
      // Sondar em vez de esperar por EVENT_TOUR_INITIALIZED: visto a
      // 2026-09-16 que o evento não chegava (ou já tinha passado) e o comando
      // perdia-se. A visita pode levar mais de um minuto a carregar.
      if (!data.__deferredSince) { log("tour a inicializar; a adiar"); data.__deferredSince = Date.now(); }
      if (Date.now() - data.__deferredSince < INIT_WAIT_MS) { setTimeout(function () { focus(data); }, 300); return true; }
      log("o tour não inicializou a tempo"); return false;
    }
    try { window.P360TourEventBridge && window.P360TourEventBridge.install && window.P360TourEventBridge.install(rp); } catch (e) {}

    var panoramaKey = String(data.panoramaKey || data.panorama_key || "");
    var overlayId = String(data.overlayId || data.overlay_id || "");
    var panorama = rp[panoramaKey];
    if (!panorama) { log("panorama desconhecido:", panoramaKey); return false; }
    var overlay = overlayId ? rp[overlayId] : null;
    var spot = overlay ? overlaySpot(rp, overlay) : null;
    var title = getLabel(panorama);

    stopHighlight();
    closeWindows(rp);
    closeWelcome(rp);

    var item = null, index = -1;
    try {
      item = rp.getPlayListItemByMedia(rp.mainPlayList, panorama);
      index = (getValue(rp.mainPlayList, "items") || []).indexOf(item);
    } catch (e) {}
    var alreadyThere = activeMedia(rp).media === panorama;
    if (spot && item && typeof rp.setPanoramaCameraWithSpot === "function") {
      // Como as ações do próprio tour: primeiro a câmara para o hotspot, depois
      // a seleção do panorama na playlist (a câmara sozinha só faz moveTo se o
      // panorama já estiver ativo; noutro panorama não muda nada).
      rp.setPanoramaCameraWithSpot(rp.mainPlayList, item, spot.yaw, spot.pitch, mainViewerHfov(rp));
      if (!alreadyThere && index >= 0) {
        if (typeof rp.setPlayListSelectedIndex === "function") rp.setPlayListSelectedIndex(rp.mainPlayList, index);
        else rp.mainPlayList.set("selectedIndex", index);
      }
    } else if (typeof rp.setMainMediaByName === "function" && title) {
      rp.setMainMediaByName(title);
    } else {
      log("sem API de navegação"); return false;
    }
    try {
      window.P360TourEventBridge && window.P360TourEventBridge.postPanoramaChanged &&
        window.P360TourEventBridge.postPanoramaChanged({ panoramaId: panoramaKey, title: title, source: "assistant_navigation" });
    } catch (e) {}

    if (!spot) { log("sem hotspot para destacar", overlayId); return true; }
    var started = performance.now();
    var wait = setInterval(function () {
      var now = activeMedia(rp);
      if (now.media === panorama) {
        clearInterval(wait);
        aimAtSpot(rp, spot);
        startHighlight(rp, overlay, spot, now.index);
        try {
          window.P360TourEventBridge && window.P360TourEventBridge.postNavigationCompleted &&
            window.P360TourEventBridge.postNavigationCompleted({ panoramaId: panoramaKey, overlayId: overlayId, title: getLabel(overlay), source: "assistant_navigation" });
        } catch (e) {}
      } else if (performance.now() - started > WAIT_MEDIA_MS) {
        clearInterval(wait); log("o panorama não ficou ativo a tempo");
      }
    }, 200);
    return true;
  }

  window.P360FocusArtifact = { version: VERSION, focus: focus, stopHighlight: stopHighlight };

  window.addEventListener("message", function (event) {
    var data = event && event.data;
    if (!data || typeof data !== "object") return;
    var isFocus = data.type === "focusArtifact" || (data.type === "navigateToArtifact" && data.mode === "focus");
    if (!isFocus) return;
    event.stopImmediatePropagation();  // o handler antigo (abre a ficha) não corre
    log("comando", data);
    focus(data);
  }, true);

  // sem esperar por comando: o visitante pode abrir a ficha pelo hotspot
  (function esperarPlayer(tentativas) {
    var rp = getRootPlayer(getTour());
    if (rp && typeof rp.showWindowBase === "function") { repararAcoes(rp); return; }
    if ((tentativas || 0) < 600) setTimeout(function () { esperarPlayer((tentativas || 0) + 1); }, 500);
  })(0);
})();
