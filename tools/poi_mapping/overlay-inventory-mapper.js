/* Património 360 — overlay inventory mapper (v3)
 *
 * Cola na consola da página do tour (o index.html do 3DVista, não o assistente).
 *
 * Diferenças para a v1:
 *   - não dispara POIs: lê as acções do modelo do player, por isso é instantâneo
 *     e não abre 45 janelas pela frente do utilizador;
 *   - não depende dos postMessage, por isso funciona em tours onde o eval rebenta
 *     (aspas curvas no location, por exemplo);
 *   - aceita aspas normais, curvas e plicas;
 *   - apanha overlays de panoramas e objectos de modelos 3D;
 *   - cai para a leitura do script_general.js quando o runtime não expõe overlays;
 *   - não guarda estado em localStorage: corre sempre do princípio.
 *
 * No fim descarrega panorama-overlays-inventory.json e deixa tudo em
 * window.P360Mapper.results.
 */
(function () {
  "use strict";

  var SMART_QUOTES = /[“”‘’]/;
  var OPEN_QUOTES = '"\'“‘';

  function closingQuote(open) {
    if (open === "“") return "”";
    if (open === "‘") return "’";
    return open;
  }

  function unescapeValue(value) {
    return String(value)
      .replace(/\\+(['"\/])/g, "$1")
      .replace(/\\n/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  /* Lê o conteúdo de uma string a partir da aspa de abertura em `open`.
   * No modelo em memória as aspas são normais ("valor"); no script exportado
   * vêm escapadas (\"valor\"), e aí o terminador é \" e não ". */
  function readQuoted(text, open, escaped) {
    var close = closingQuote(text.charAt(open));
    var i = open + 1;
    var buf = "";
    while (i < text.length) {
      var ch = text.charAt(i);
      if (escaped && ch === "\\" && text.charAt(i + 1) === close) break;
      if (!escaped && ch === close) break;
      if (ch === "\\") {
        buf += ch + text.charAt(i + 1);
        i += 2;
        continue;
      }
      buf += ch;
      i += 1;
    }
    return { value: unescapeValue(buf), end: i + (escaped ? 2 : 1) };
  }

  /* Lê  <campo>: "valor"  tolerando aspas curvas e escapes. */
  function readField(text, name) {
    var re = new RegExp(name + "\\s*:\\s*(\\\\?)([" + OPEN_QUOTES + "])");
    var m = re.exec(text);
    if (!m) return null;
    var open = m.index + m[0].length - 1;
    return readQuoted(text, open, m[1] === "\\").value || null;
  }

  /* Lê a lista  [ "a", "b" ]  com as mesmas tolerâncias. */
  function readStringList(segment) {
    var out = [];
    var i = 0;
    while (i < segment.length) {
      if (OPEN_QUOTES.indexOf(segment.charAt(i)) === -1) {
        i += 1;
        continue;
      }
      var escaped = i > 0 && segment.charAt(i - 1) === "\\";
      var read = readQuoted(segment, i, escaped);
      if (read.value) out.push(read.value);
      i = read.end;
    }
    return out;
  }

  function parseAction(text) {
    if (!text || text.indexOf("openedInfoWindow") === -1) return null;
    var invMatch = /inventoryIds\s*:\s*\[([\s\S]*?)\]/.exec(text);
    if (!invMatch) return null;
    var ids = readStringList(invMatch[1]);
    if (!ids.length) return null;
    return {
      inventoryIds: ids,
      title: readField(text, "title"),
      location: readField(text, "location"),
      smartQuotes: SMART_QUOTES.test(text)
    };
  }

  /* ---------------- leitura do modelo em memória ---------------- */

  function getValue(obj, key) {
    if (!obj) return undefined;
    if (typeof obj.get === "function") {
      try {
        var v = obj.get(key);
        if (v !== undefined) return v;
      } catch (e) {}
    }
    try {
      if (obj[key] !== undefined) return obj[key];
    } catch (e) {}
    if (obj.tq && obj.tq[key] !== undefined) return obj.tq[key];
    return undefined;
  }

  function getId(obj) {
    return getValue(obj, "id") || (obj && obj.id) || (obj && obj.tq && obj.tq.id) || null;
  }

  function actionText(obj) {
    var parts = [];
    var keys = ["click", "select", "start", "open"];
    for (var i = 0; i < keys.length; i++) {
      var v = getValue(obj, keys[i]);
      if (typeof v === "string") parts.push(v);
      if (obj && obj.tq && typeof obj.tq[keys[i]] === "string") parts.push(obj.tq[keys[i]]);
    }
    return parts.join(" ");
  }

  function scanRuntime(rootPlayer) {
    function resolve(value) {
      if (typeof value !== "string") return value;
      return rootPlayer[value.replace(/^this\./, "")] || null;
    }
    function asArray(value) {
      if (!value) return [];
      var list = Array.isArray(value) ? value : [value];
      return list.map(resolve).filter(Boolean);
    }

    var results = [];
    var containers = Object.keys(rootPlayer).filter(function (k) {
      return /^(panorama|model)_/.test(k);
    });

    for (var c = 0; c < containers.length; c++) {
      var key = containers[c];
      var container = rootPlayer[key];
      var children = asArray(getValue(container, "overlays"))
        .concat(asArray(getValue(container, "objects")));

      for (var o = 0; o < children.length; o++) {
        var overlay = children[o];
        var overlayId = getId(overlay);
        if (!overlayId) continue;

        var candidates = [overlay].concat(asArray(getValue(overlay, "areas")));
        for (var a = 0; a < candidates.length; a++) {
          var parsed = parseAction(actionText(candidates[a]));
          if (!parsed) continue;
          results.push({
            overlayId: overlayId,
            panoramaKey: key,
            inventoryIds: parsed.inventoryIds,
            location: parsed.location,
            title: parsed.title,
            _smartQuotes: parsed.smartQuotes
          });
          break;
        }
      }
    }
    return results;
  }

  /* ---------------- leitura do script exportado (reserva) ---------------- */

  function objectSpans(text) {
    var spans = [];
    var stack = [];
    var inStr = false;
    var quote = "";
    var esc = false;
    for (var i = 0; i < text.length; i++) {
      var ch = text.charAt(i);
      if (inStr) {
        if (esc) esc = false;
        else if (ch === "\\") esc = true;
        else if (ch === quote) inStr = false;
        continue;
      }
      if (ch === '"' || ch === "'") {
        inStr = true;
        quote = ch;
      } else if (ch === "{") {
        stack.push(i);
      } else if (ch === "}" && stack.length) {
        spans.push([stack.pop(), i + 1]);
      }
    }
    spans.sort(function (a, b) { return a[0] - b[0]; });
    return spans;
  }

  function scanSource(text) {
    var spans = objectSpans(text);
    var starts = spans.map(function (s) { return s[0]; });

    function innermost(pos) {
      var lo = 0;
      var hi = starts.length - 1;
      var idx = -1;
      while (lo <= hi) {
        var mid = (lo + hi) >> 1;
        if (starts[mid] <= pos) { idx = mid; lo = mid + 1; } else hi = mid - 1;
      }
      while (idx >= 0) {
        if (spans[idx][1] > pos) return spans[idx];
        idx -= 1;
      }
      return null;
    }

    function ownId(objText) {
      var m = /"id"\s*:\s*"([^"]+)"/.exec(objText);
      return m ? m[1] : null;
    }

    function parentOf(childId, prefixes) {
      var needle = '"this.' + childId + '"';
      var from = 0;
      while (true) {
        var at = text.indexOf(needle, from);
        if (at === -1) return null;
        var span = innermost(at);
        if (span) {
          var pid = ownId(text.slice(span[0], span[1]));
          if (pid) {
            for (var p = 0; p < prefixes.length; p++) {
              if (pid.indexOf(prefixes[p]) === 0) return pid;
            }
          }
        }
        from = at + needle.length;
      }
    }

    var results = [];
    var seen = {};
    var from = 0;
    while (true) {
      var at = text.indexOf("openedInfoWindow", from);
      if (at === -1) break;
      from = at + 16;

      var span = innermost(at);
      if (!span) continue;
      var objText = text.slice(span[0], span[1]);
      var areaId = ownId(objText);
      if (!areaId || seen[areaId]) continue;
      seen[areaId] = true;

      var parsed = parseAction(objText);
      if (!parsed) continue;

      var overlayId = /^(overlay_|sprite_)/.test(areaId)
        ? areaId
        : parentOf(areaId, ["overlay_", "sprite_"]);
      if (!overlayId) continue;

      var panoramaKey = parentOf(overlayId, ["panorama_", "model_"]);
      if (!panoramaKey) continue;

      results.push({
        overlayId: overlayId,
        panoramaKey: panoramaKey,
        inventoryIds: parsed.inventoryIds,
        location: parsed.location,
        title: parsed.title,
        _smartQuotes: parsed.smartQuotes
      });
    }
    return results;
  }

  /* ---------------- juntar, avisar, descarregar ---------------- */

  function merge(runtime, source) {
    var out = runtime.slice();
    var known = {};
    runtime.forEach(function (e) { known[e.overlayId] = true; });
    source.forEach(function (e) {
      if (!known[e.overlayId]) {
        known[e.overlayId] = true;
        out.push(e);
      }
    });
    return out;
  }

  function report(entries) {
    var quebrados = entries.filter(function (e) { return e._smartQuotes; });
    var colados = entries.filter(function (e) {
      return e.inventoryIds.some(function (i) {
        return /\d\s*(?:,|;|\be\b|\ba\b)\s*\d/.test(i) || /_\w+_\d+_/.test(i);
      });
    });
    var semLocal = entries.filter(function (e) { return !e.location; });

    console.log("POIs encontrados : " + entries.length);
    console.log("panoramas        : " + new Set(entries.map(function (e) { return e.panoramaKey; })).size);
    console.log("inventários      : " + new Set([].concat.apply([], entries.map(function (e) { return e.inventoryIds; }))).size);

    if (quebrados.length) {
      console.warn("\n" + quebrados.length + " POIs com aspas curvas na acção: o postMessage nunca dispara no tour.");
      console.table(quebrados.map(function (e) {
        return { inventario: e.inventoryIds.join(" | "), titulo: e.title };
      }));
    }
    if (colados.length) {
      console.warn("\n" + colados.length + " POIs com vários inventários numa só string: o backend não os separa todos.");
      console.table(colados.map(function (e) {
        return { inventario: e.inventoryIds.join(" | "), titulo: e.title };
      }));
    }
    if (semLocal.length) {
      console.warn("\n" + semLocal.length + " POIs sem location.");
    }
  }

  function download(entries) {
    var limpo = entries.map(function (e) {
      return {
        overlayId: e.overlayId,
        panoramaKey: e.panoramaKey,
        inventoryIds: e.inventoryIds,
        location: e.location,
        title: e.title
      };
    });
    var blob = new Blob([JSON.stringify(limpo, null, 2)], { type: "application/json" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = "panorama-overlays-inventory.json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    return limpo;
  }

  function waitForTour(timeoutMs) {
    var limite = Date.now() + (timeoutMs || 15000);
    return new Promise(function (resolve) {
      (function check() {
        var tour = window.tour
          || (window.TDV && window.TDV.PlayerAPI && window.TDV.PlayerAPI.getInstance
              && window.TDV.PlayerAPI.getInstance())
          || (window.TDV && window.TDV.PlayerAPI && window.TDV.PlayerAPI.getCurrentPlayer
              && window.TDV.PlayerAPI.getCurrentPlayer());
        if (tour) return resolve(tour);
        if (Date.now() > limite) return resolve(null);
        setTimeout(check, 150);
      })();
    });
  }

  async function run() {
    var tour = await waitForTour();
    if (!tour) {
      console.error("Não encontrei o player. Estás na página do tour?");
      return null;
    }
    var rootPlayer = tour._getRootPlayer ? tour._getRootPlayer() : tour;

    var runtime = scanRuntime(rootPlayer);
    console.log("runtime: " + runtime.length + " POIs");

    var source = [];
    try {
      var nomes = ["script_general.js", "script_mobile.js"];
      for (var i = 0; i < nomes.length && !source.length; i++) {
        var resp = await fetch(nomes[i], { cache: "no-store" });
        if (!resp.ok) continue;
        source = scanSource(await resp.text());
        console.log(nomes[i] + ": " + source.length + " POIs");
      }
    } catch (e) {
      console.warn("Não consegui ler o script exportado:", e);
    }

    var entries = merge(runtime, source);
    report(entries);
    var limpo = download(entries);
    window.P360Mapper.results = limpo;
    console.log("\nJSON descarregado. Também está em window.P360Mapper.results");
    return limpo;
  }

  window.P360Mapper = {
    run: run,
    parseAction: parseAction,
    scanSource: scanSource,
    scanRuntime: scanRuntime,
    results: null
  };

  if (!window.P360_MAPPER_NO_AUTORUN) run();
})();
