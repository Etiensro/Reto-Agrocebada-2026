/* ==========================================================================
   AgroCebada Dashboard — Reto FIRA 2026
   Lógica interactiva de analítica agrícola, modelación geo-residual,
   cartografía GIS, simulador, explicabilidad y viabilidad de crédito.
   Sin dependencias externas obligatorias; datos en datos.js, modelo en modelo.js.
   ========================================================================== */
(function () {
  "use strict";

  const D = window.DATOS;
  const M = window.Modelo;
  const CFG = window.PANEL_CONFIG || { fondo: true, fondoInicial: "satelite", territorio: true, buscador: true };
  const T = CFG.territorio ? window.TERRITORIO : null;

  if (!D || !M) {
    document.body.innerHTML = "<div style='padding:40px;text-align:center;'><h2>Error: No se pudieron cargar los datos (datos.js / modelo.js).</h2></div>";
    return;
  }

  /* ==========================================================================
     UTILIDADES GENERALES Y DOM
     ========================================================================== */
  const NS = "http://www.w3.org/2000/svg";
  const $ = (selector, context = document) => context.querySelector(selector);
  const $$ = (selector, context = document) => [...context.querySelectorAll(selector)];

  function setAttrs(node, attrs) {
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") node.setAttribute("class", v);
      else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    }
    return node;
  }

  function appendChildren(node, children) {
    for (const ch of children.flat(Infinity)) {
      if (ch === null || ch === undefined || ch === false) continue;
      node.append(ch instanceof Node ? ch : document.createTextNode(String(ch)));
    }
    return node;
  }

  const el = (tag, attrs, ...children) => appendChildren(setAttrs(document.createElement(tag), attrs), children);
  const svgEl = (tag, attrs, ...children) => appendChildren(setAttrs(document.createElementNS(NS, tag), attrs), children);
  const empty = (node) => { while (node.firstChild) node.removeChild(node.firstChild); return node; };
  const clamp = (val, min, max) => Math.min(max, Math.max(min, val));

  // Formato numérico en español de México (es-MX)
  const nf = (digits) => new Intl.NumberFormat("es-MX", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  const NF = [0, 1, 2, 3, 4, 5].map(nf);
  const fmtNum = (v, d = 2) => (Number.isFinite(v) ? NF[d].format(v) : "—");
  const fmtSign = (v, d = 2) => (Number.isFinite(v) ? (v > 0 ? "+" : v < 0 ? "−" : "") + NF[d].format(Math.abs(v)) : "—");
  const fmtPct = (v, d = 0) => (Number.isFinite(v) ? NF[d].format(v * 100) + " %" : "—");
  const probTexto = (v, n) => {
    if (!Number.isFinite(v)) return "—";
    const paso = Math.ceil(100 / Math.max(n || 30, 1));
    if (v <= 0) return `< ${paso} %`;
    if (v >= 1) return `> ${100 - paso} %`;
    return fmtPct(v);
  };

  const saveStore = (k, v) => { try { localStorage.setItem(k, v); } catch (e) { } };
  const loadStore = (k) => { try { return localStorage.getItem(k); } catch (e) { return null; } };

  /* ==========================================================================
     PROCESAMIENTO DE DATOS DEL RETO
     ========================================================================== */
  const P = D.parcelas;
  const porId = new Map(P.map((p) => [p.id, p]));
  const ESTADOS = ["Puebla", "Hidalgo", "Tlaxcala"];
  const COLOR_ESTADO = { Puebla: "var(--c-puebla)", Hidalgo: "var(--c-hidalgo)", Tlaxcala: "var(--c-tlaxcala)" };

  const NOMBRE_VAR = {
    log_area: "Superficie del terreno (log ha)",
    evi_2022: "Salud del terreno (EVI primavera 2022)",
    evi_2023: "Salud del terreno (EVI primavera 2023)",
    ndvi_media: "Salud promedio multianual (NDVI 2022–2024)",
  };

  for (const p of P) {
    const xs = p.poligono.map((q) => q[0]);
    const ys = p.poligono.map((q) => q[1]);
    p.cx = xs.reduce((a, b) => a + b, 0) / xs.length;
    p.cy = ys.reduce((a, b) => a + b, 0) / ys.length;
    p.diam = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
    p.produccion = p.area * p.estimado;
    p.error = p.real === null ? null : p.estimado - p.real;
    p.esOculta = p.origen === "ocultas";
  }

  const valoresRend = P.flatMap((p) => [p.estimado, p.real]).filter(Number.isFinite);
  const DOM = [Math.floor(Math.min(...valoresRend) * 2) / 2, Math.ceil(Math.max(...valoresRend) * 2) / 2];

  const MUNICIPIOS = Object.entries(P.reduce((a, p) => ((a[p.municipio] = (a[p.municipio] || 0) + 1), a), {}))
    .filter(([, n]) => n >= 5).sort((a, b) => b[1] - a[1]).map(([m]) => m);

  const conocidas = P.filter((p) => !p.esOculta);
  const ocultas = P.filter((p) => p.esOculta);

  // Estado global reactivo
  const S = {
    vista: "resumen",
    conjunto: "ocultas",
    estado: "todos",
    color: "estimado",
    sel: ocultas.length ? ocultas[0].id : P[0].id,
    umbral: 3.5,
    costoHa: 19250,
    precioTon: 5500,
    tConjunto: "todas",
    tEstado: "todos",
    busca: "",
    filtroRiesgoNivel: "todos",
    ordenTabla: { col: "id", dir: 1 },
    ordenRiesgo: { col: "prob", dir: 1 },
    varAmbientalDispersion: "evi_2022",
    varClimaMensual: "PREC",
    variable: null,
    anio: null,
    mes: null,
  };

  const umbralGuardado = parseFloat(loadStore("agrocebada.umbral"));
  if (Number.isFinite(umbralGuardado)) S.umbral = clamp(umbralGuardado, 1.5, 5.5);

  const residuosDe = (p) => D.residuos_por_zona[p.zona] || D.residuos_por_zona.Todas;
  const probDe = (p, umbral = S.umbral) => M.probabilidad(p.estimado, residuosDe(p), umbral);
  const nivelDe = (pr) => (pr >= 0.8 ? "alta" : pr >= 0.5 ? "media" : "baja");
  const NIVEL_TXT = { alta: "Viable para Crédito (≥80%)", media: "Evaluar con Cuidado (50–80%)", baja: "Riesgo Elevado (<50%)" };
  const NIVEL_TXT_SHORT = { alta: "Viable", media: "Evaluar", baja: "Riesgo" };

  /* ==========================================================================
     PALETAS DE COLOR Y RAMPAS VISUALES
     ========================================================================== */
  const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

  function hexToRgb(c) {
    if (c.startsWith("#")) {
      const h = c.length === 4 ? c.replace(/./g, (m, i) => (i ? m + m : m)) : c;
      return [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
    }
    return (c.match(/[\d.]+/g) || [0, 0, 0]).slice(0, 3).map(Number);
  }

  let SEQ = [], DIV = [];
  function cargarColores() {
    SEQ = ["--s0", "--s1", "--s2", "--s3", "--s4", "--s5", "--s6"].map((n) => hexToRgb(cssVar(n) || "#2D6A4F"));
    DIV = ["--d-neg", "--d-mid", "--d-pos"].map((n) => hexToRgb(cssVar(n) || "#888888"));
  }

  function interp(paradas, t) {
    t = clamp(t, 0, 1);
    const n = paradas.length - 1;
    const i = Math.min(n - 1, Math.floor(t * n));
    const f = t * n - i;
    const c = paradas[i].map((v, k) => Math.round(v + (paradas[i + 1][k] - v) * f));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }

  const colorSeq = (v) => interp(SEQ, (v - DOM[0]) / (DOM[1] - DOM[0]));
  const colorDiv = (v) => interp(DIV, (clamp(v, -1, 1) + 1) / 2);
  const degradadoCss = (paradas) => `linear-gradient(90deg, ${paradas.map((c, i) => `rgb(${c.join(",")}) ${(i / (paradas.length - 1)) * 100}%`).join(", ")})`;

  /* ==========================================================================
     TOOLTIP FLOTANTE
     ========================================================================== */
  const tip = $("#tip");
  function mostrarTip(filas, evt, titulo) {
    empty(tip);
    if (titulo) tip.append(el("b", {}, titulo));
    for (const [a, b] of filas) {
      tip.append(el("div", { class: "tip-row" }, el("span", { class: "suave" }, a), el("span", {}, b)));
    }
    tip.hidden = false;
    const r = tip.getBoundingClientRect();
    let x = evt.clientX + 14, y = evt.clientY + 14;
    if (x + r.width > window.innerWidth - 12) x = evt.clientX - r.width - 14;
    if (y + r.height > window.innerHeight - 12) y = evt.clientY - r.height - 14;
    tip.style.left = Math.max(8, x) + "px";
    tip.style.top = Math.max(8, y) + "px";
  }

  const ocultarTip = () => { if (tip) tip.hidden = true; };

  const filasParcela = (p) => [
    ["Municipio", `${p.municipio}, ${p.estado}`],
    ...(T && S.color === "territorio" ? [[nombreCapaActual(), textoValorTerritorio(p.id)]] : []),
    ["Cosecha estimada", fmtNum(p.estimado) + " t/ha"],
    ...(p.real !== null ? [["Cosecha real medida", fmtNum(p.real) + " t/ha"]] : []),
    ["Superficie", fmtNum(p.area, 1) + " ha"],
    ["Situación", p.esOculta ? "Por estimar (Reto FIRA)" : "Conocida"],
  ];

  /* ==========================================================================
     ICONOS Y PÍLDORAS
     ========================================================================== */
  function iconoRiesgo(nivel) {
    const s = svgEl("svg", { width: 13, height: 13, viewBox: "0 0 14 14", "aria-hidden": "true" });
    const trazo = { fill: "none", stroke: "currentColor", "stroke-width": 1.7, "stroke-linecap": "round", "stroke-linejoin": "round" };
    if (nivel === "alta") {
      s.append(svgEl("circle", { cx: 7, cy: 7, r: 6, ...trazo }), svgEl("path", { d: "M4 7.2l2 2 4-4.4", ...trazo }));
    } else if (nivel === "media") {
      s.append(svgEl("path", { d: "M7 1.6L13 12.2H1z", ...trazo }), svgEl("path", { d: "M7 5.6v3M7 10.3v.1", ...trazo }));
    } else {
      s.append(svgEl("circle", { cx: 7, cy: 7, r: 6, ...trazo }), svgEl("path", { d: "M4.6 4.6l4.8 4.8M9.4 4.6l-4.8 4.8", ...trazo }));
    }
    return s;
  }

  const pildoraRiesgo = (pr, short = false) => {
    const n = nivelDe(pr);
    return el("span", { class: "risk-pill " + n }, iconoRiesgo(n), short ? NIVEL_TXT_SHORT[n] : NIVEL_TXT[n]);
  };

  /* ==========================================================================
     TEMA CLARO / OSCURO / AUTO
     ========================================================================== */
  const MODOS = ["auto", "light", "dark"];
  const MODO_TXT = { auto: "Auto", light: "Claro", dark: "Oscuro" };
  let modoTema = loadStore("agrocebada.tema") || "auto";

  function aplicarTema() {
    const root = document.documentElement;
    if (modoTema === "auto") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", modoTema);
    const tSpan = $("#tema-texto");
    if (tSpan) tSpan.textContent = MODO_TXT[modoTema];
    cargarColores();
    if (mapa.listo) { pintarMapa(); pintarLeyenda(); }
  }

  const btnTema = $("#tema");
  if (btnTema) {
    btnTema.addEventListener("click", () => {
      modoTema = MODOS[(MODOS.indexOf(modoTema) + 1) % MODOS.length];
      saveStore("agrocebada.tema", modoTema);
      aplicarTema();
    });
  }

  /* ==========================================================================
     MAPA INTERACTIVO SVG + MAPLIBRE CARTOGRÁFICO
     ========================================================================== */
  const mapa = { listo: false, vista: { cx: 0, cy: 0, w: 60 }, kmPx: 0.1, arrastre: null };
  const lienzo = $("#lienzo");
  const svgMapa = $("#mapa");
  const capaParcelas = svgEl("g");
  const capaEtiquetas = svgEl("g");
  const halo = svgEl("circle", { class: "halo-seleccion", display: "none" });

  const filtraMapa = (p) => (S.conjunto === "todas" || p.origen === S.conjunto) && (S.estado === "todos" || p.estado === S.estado);

  function valorColor(p) {
    if (S.color === "territorio") return valorTerritorio(p.id);
    return S.color === "estimado" ? p.estimado : S.color === "real" ? p.real : p.error;
  }

  function construirMapa() {
    if (!svgMapa) return;
    svgMapa.append(capaParcelas, capaEtiquetas, halo);

    for (const p of P) {
      const g = svgEl("g", { class: "parcela", "data-id": p.id });
      p._poly = svgEl("polygon", { points: p.poligono.map((q) => `${q[0]},${-q[1]}`).join(" ") });
      p._dot = svgEl("circle", { class: "punto", cx: p.cx, cy: -p.cy });
      g.append(p._poly, p._dot);
      p._g = g;
      capaParcelas.append(g);
    }

    for (const m of MUNICIPIOS) {
      const ps = P.filter((p) => p.municipio === m);
      const t = svgEl("text", { class: "etiqueta-mun", "text-anchor": "middle" }, m);
      t._x = ps.reduce((a, p) => a + p.cx, 0) / ps.length;
      t._y = -Math.max(...ps.map((p) => p.cy));
      capaEtiquetas.append(t);
    }

    // Eventos de puntero e interacción
    svgMapa.addEventListener("pointermove", (e) => {
      if (mapa.arrastre) return;
      const g = e.target.closest && e.target.closest(".parcela");
      const p = g && porId.get(g.dataset.id);
      if (p) mostrarTip(filasParcela(p), e, p.id);
      else ocultarTip();
    });

    svgMapa.addEventListener("pointerleave", ocultarTip);

    svgMapa.addEventListener("pointerdown", (e) => {
      mapa.arrastre = { x: e.clientX, y: e.clientY, cx: mapa.vista.cx, cy: mapa.vista.cy, movio: false, id: e.pointerId };
      ocultarTip();
    });

    window.addEventListener("pointermove", (e) => {
      const a = mapa.arrastre;
      if (!a || e.pointerId !== a.id) return;
      const dx = e.clientX - a.x, dy = e.clientY - a.y;
      if (!a.movio && Math.hypot(dx, dy) < 4) return;
      if (!a.movio) { a.movio = true; if (lienzo) lienzo.classList.add("arrastrando"); }
      mapa.vista.cx = a.cx - dx * mapa.kmPx;
      mapa.vista.cy = a.cy - dy * mapa.kmPx;
      aplicarVista();
    });

    window.addEventListener("pointerup", (e) => {
      const a = mapa.arrastre;
      if (!a || e.pointerId !== a.id) return;
      mapa.arrastre = null;
      if (lienzo) lienzo.classList.remove("arrastrando");
      if (!a.movio) {
        const g = document.elementFromPoint(e.clientX, e.clientY);
        const nodo = g && g.closest && g.closest(".parcela");
        if (nodo && lienzo.contains(nodo)) seleccionar(nodo.dataset.id, { desplazar: true });
      }
    });

    if (lienzo) {
      lienzo.addEventListener("wheel", (e) => {
        e.preventDefault();
        zoom(e.deltaY > 0 ? 1.25 : 0.8, e.clientX, e.clientY);
      }, { passive: false });
    }

    $("#z-mas")?.addEventListener("click", () => zoom(0.7));
    $("#z-menos")?.addEventListener("click", () => zoom(1.4));
    $("#z-todo")?.addEventListener("click", () => ajustarVista(P.filter(filtraMapa)));

    window.addEventListener("resize", () => {
      if (S.vista === "mapa") aplicarVista();
    });

    mapa.listo = true;
  }

  function zoom(factor, cxPantalla, cyPantalla) {
    if (!lienzo) return;
    const r = lienzo.getBoundingClientRect();
    const W = r.width, H = r.height;
    const v = mapa.vista;
    const fx = cxPantalla === undefined ? 0.5 : (cxPantalla - r.left) / W;
    const fy = cyPantalla === undefined ? 0.5 : (cyPantalla - r.top) / H;
    const w2 = clamp(v.w * factor, 0.4, 160);
    const h = (v.w * H) / W, h2 = (w2 * H) / W;
    const wx = v.cx - v.w / 2 + fx * v.w, wy = v.cy - h / 2 + fy * h;
    v.cx = wx - fx * w2 + w2 / 2;
    v.cy = wy - fy * h2 + h2 / 2;
    v.w = w2;
    aplicarVista();
  }

  function ajustarVista(lista) {
    if (!lista || !lista.length) lista = P;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const p of lista) for (const [x, y] of p.poligono) {
      x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, -y); y1 = Math.max(y1, -y);
    }
    const W = (lienzo && lienzo.clientWidth) || 800, H = (lienzo && lienzo.clientHeight) || 540;
    const w = Math.max(x1 - x0, ((y1 - y0) * W) / H, 1.2) * 1.12;
    mapa.vista = { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, w };
    aplicarVista();
  }

  function aplicarVista() {
    if (!lienzo || !svgMapa) return;
    const W = lienzo.clientWidth, H = lienzo.clientHeight;
    if (!W || !H) return;
    const v = mapa.vista;
    const h = (v.w * H) / W;
    svgMapa.setAttribute("viewBox", `${v.cx - v.w / 2} ${v.cy - h / 2} ${v.w} ${h}`);
    mapa.kmPx = v.w / W;
    const kp = mapa.kmPx;

    for (const p of P) {
      const punto = p.diam / kp < 9;
      p._poly.style.display = punto ? "none" : "";
      p._dot.style.display = punto ? "" : "none";
      p._dot.setAttribute("r", 5 * kp);
    }

    for (const t of capaEtiquetas.childNodes) {
      t.setAttribute("x", t._x);
      t.setAttribute("y", t._y - 10 * kp);
      t.setAttribute("font-size", 12.5 * kp);
      t.setAttribute("stroke-width", 4 * kp);
    }

    posicionarHalo();
    sincronizarFondo();

    const opcionesEscala = [0.5, 1, 2, 5, 10, 20, 50];
    const objetivo = 110 * kp;
    const km = opcionesEscala.reduce((a, b) => (Math.abs(b - objetivo) < Math.abs(a - objetivo) ? b : a));
    const escalaNode = $("#escala");
    if (escalaNode) {
      empty(escalaNode).append(el("span", {}, km + " km"), el("div", { class: "regla", style: `width:${km / kp}px` }));
    }
  }

  function posicionarHalo() {
    const p = S.sel && porId.get(S.sel);
    if (!p || !filtraMapa(p)) { halo.setAttribute("display", "none"); return; }
    halo.removeAttribute("display");
    halo.setAttribute("cx", p.cx);
    halo.setAttribute("cy", -p.cy);
    halo.setAttribute("r", Math.max(p.diam / 2 + 6 * mapa.kmPx, 9 * mapa.kmPx));
  }

  function pintarMapa() {
    const orden = [];
    for (const p of P) {
      const visible = filtraMapa(p);
      const v = valorColor(p);
      const conValor = Number.isFinite(v);
      const g = p._g;
      if (!g) continue;
      g.setAttribute("class", "parcela" + (p.esOculta ? " por-estimar" : "") +
        (!visible || !conValor ? " atenuada" : "") + (p.id === S.sel ? " seleccionada" : ""));

      if (visible && conValor) {
        const c = S.color === "error" ? colorDiv(v) : S.color === "territorio" ? colorTerritorio(v) : colorSeq(v);
        p._poly.style.fill = c;
        p._dot.style.fill = c;
      } else {
        p._poly.style.fill = "";
        p._dot.style.fill = "";
      }
      orden.push([visible && conValor ? (p.id === S.sel ? 2 : 1) : 0, p]);
    }
    orden.sort((a, b) => a[0] - b[0]);
    for (const [, p] of orden) capaParcelas.append(p._g);
    posicionarHalo();
  }

  function pintarLeyenda() {
    const cont = $("#leyenda");
    if (!cont) return;
    empty(cont);

    if (S.color === "territorio") {
      leyendaTerritorio(cont);
    } else if (S.color === "error") {
      cont.append(
        el("span", { class: "text-muted", style: "font-size:11px;" }, "Subestima"),
        el("div", { class: "legend-ramp-gradient", style: `background:${degradadoCss(DIV)}` }),
        el("span", { class: "text-muted", style: "font-size:11px;" }, "Sobrestima"),
        el("span", { class: "text-muted", style: "font-size:10px;" }, "(±1,0 t/ha)")
      );
    } else {
      cont.append(
        el("span", { class: "text-muted", style: "font-size:11px;" }, fmtNum(DOM[0], 1)),
        el("div", { class: "legend-ramp-gradient", style: `background:${degradadoCss(SEQ)}` }),
        el("span", { class: "text-muted", style: "font-size:11px;" }, fmtNum(DOM[1], 1) + " t/ha"),
        el("span", { class: "text-muted", style: "font-size:11px;" }, S.color === "real" ? "(Cosecha real)" : "(Cosecha estimada)")
      );
    }
  }

  function pintarChips() {
    const cont = $("#chips-mun");
    if (!cont) return;
    empty(cont).append(el("span", { class: "text-muted", style: "font-size:11px;font-weight:600;" }, "Zoom rápido:"));
    for (const m of MUNICIPIOS) {
      const count = P.filter((p) => p.municipio === m).length;
      cont.append(el("button", {
        type: "button",
        class: "chip-btn",
        onclick: () => ajustarVista(P.filter((p) => p.municipio === m)),
      }, `${m} (${count})`));
    }
  }

  /* ==========================================================================
     SELECCIÓN CENTRALIZADA DE PARCELA
     ========================================================================== */
  function seleccionar(id, opciones = {}) {
    S.sel = id;
    const p = porId.get(id);

    if (p && opciones.enfocar) {
      mapa.vista = { cx: p.cx, cy: -p.cy, w: Math.max(2.4, p.diam * 5) };
    }

    if (p && !filtraMapa(p)) {
      if (S.conjunto !== "todas" && S.conjunto !== p.origen) S.conjunto = "todas";
      if (S.estado !== "todos" && S.estado !== p.estado) S.estado = "todos";
      pintarFiltrosMapa();
    }

    pintarMapa();
    pintarDetalleRapido();

    // Sincronizar selector en la vista de Predicción
    const selPred = $("#sel-parcela-ficha");
    if (selPred && selPred.value !== id) selPred.value = id;
    if (S.vista === "prediccion") pintarFichaPrediccion();

    if (opciones.enfocar) aplicarVista();
    else if (p && mapa.listo) {
      const v = mapa.vista, W = (lienzo && lienzo.clientWidth) || 800, H = (lienzo && lienzo.clientHeight) || 540;
      const h = (v.w * H) / W;
      if (Math.abs(p.cx - v.cx) > v.w / 2 || Math.abs(-p.cy - v.cy) > h / 2) {
        v.cx = p.cx; v.cy = -p.cy; aplicarVista();
      }
    }
  }

  function segmentado(cont, opciones, actual, alElegir) {
    if (!cont) return;
    empty(cont);
    for (const [valor, texto] of opciones) {
      cont.append(el("button", {
        type: "button",
        "aria-pressed": String(valor === actual),
        onclick: () => {
          alElegir(valor);
          segmentado(cont, opciones, valor, alElegir);
        },
      }, texto));
    }
  }

  function pintarFiltrosMapa() {
    segmentado($("#f-conjunto"), [
      ["ocultas", `Por estimar (${ocultas.length})`],
      ["conocidas", `Conocidas (${conocidas.length})`],
      ["todas", `Todas (${P.length})`],
    ], S.conjunto, (v) => { S.conjunto = v; pintarMapa(); });

    segmentado($("#f-estado"), [
      ["todos", "Todos"],
      ...ESTADOS.map((e) => [e, e]),
    ], S.estado, (v) => {
      S.estado = v;
      pintarMapa();
      ajustarVista(P.filter(filtraMapa));
    });

    if (T && $("#f-capa")) {
      pintarControlesTerritorio();
    }
  }

  /* ==========================================================================
     DETALLE RÁPIDO EN EL MAPA (DRAWER LATERAL)
     ========================================================================== */
  function graficaIntervalo(p) {
    const ancho = 300, x0 = 8, x1 = ancho - 8, lo = 1.5, hi = 6.5;
    const X = (v) => x0 + ((clamp(v, lo, hi) - lo) / (hi - lo)) * (x1 - x0);
    const s = svgEl("svg", {
      class: "interval-svg",
      viewBox: `0 0 ${ancho} 64`,
      role: "img",
      "aria-label": `Estimado ${fmtNum(p.estimado)} t/ha, rango al 80 % de ${fmtNum(p.lo80)} a ${fmtNum(p.hi80)}`,
    });

    s.append(svgEl("line", { class: "chart-axis-line", x1: x0, x2: x1, y1: 38, y2: 38 }));
    for (let t = 2; t <= 6; t++) {
      s.append(
        svgEl("line", { class: "chart-axis-line", x1: X(t), x2: X(t), y1: 38, y2: 42 }),
        svgEl("text", { class: "chart-axis-text", x: X(t), y: 54, "text-anchor": "middle" }, t)
      );
    }

    s.append(
      svgEl("rect", { x: X(p.lo90), y: 24, width: Math.max(2, X(p.hi90) - X(p.lo90)), height: 22, fill: "var(--brand-500)", opacity: ".25", rx: 3 }),
      svgEl("rect", { x: X(p.lo80), y: 24, width: Math.max(2, X(p.hi80) - X(p.lo80)), height: 22, fill: "var(--brand-600)", opacity: ".55", rx: 3 }),
      svgEl("line", { x1: X(p.estimado), x2: X(p.estimado), y1: 18, y2: 48, stroke: "var(--text-main)", "stroke-width": 2.5 })
    );

    s.append(
      svgEl("line", { x1: X(S.umbral), x2: X(S.umbral), y1: 8, y2: 38, stroke: "var(--c-hidalgo)", "stroke-dasharray": "3 2", "stroke-width": 1.5 }),
      svgEl("text", { x: X(S.umbral), y: 6, "text-anchor": "middle", fill: "var(--c-hidalgo)", "font-size": "9.5px", "font-weight": "bold" }, "mín. " + fmtNum(S.umbral, 1))
    );

    if (p.real !== null) {
      const x = X(p.real);
      s.append(
        svgEl("path", { d: `M${x} 14 l5 6 -5 6 -5 -6z`, fill: "var(--c-hidalgo)", stroke: "var(--surface-card)", "stroke-width": 1.2 }),
        svgEl("text", { x, y: 11, "text-anchor": "middle", class: "chart-axis-text", fill: "var(--c-hidalgo)", "font-weight": "600" }, "real")
      );
    }
    return s;
  }

  function pintarDetalleRapido() {
    const cont = $("#detalle");
    if (!cont) return;
    empty(cont);

    const p = S.sel && porId.get(S.sel);
    if (!p) {
      cont.append(el("p", { class: "text-muted", style: "padding:16px;text-align:center;" }, "Selecciona una parcela en el mapa para ver sus indicadores."));
      return;
    }

    const prob = probDe(p);

    cont.append(
      el("div", { class: "parcel-drawer-header" },
        el("div", {},
          el("div", { class: "parcel-id-title" }, p.id),
          el("div", { class: "text-secondary", style: "font-size:12px;margin-top:2px;" }, `${p.municipio}, ${p.estado}`)
        ),
        el("span", { class: "badge-status " + (p.esOculta ? "estimar" : "conocida") },
          p.esOculta ? "Por estimar" : "Observada"
        )
      ),

      el("div", { class: "big-stat-callout" },
        el("div", { class: "big-stat-value" }, fmtNum(p.estimado), " ", el("span", { style: "font-size:16px;font-weight:600;" }, "t/ha")),
        el("div", { class: "big-stat-unit" }, p.esOculta ? "Cosecha esperada (Modelo final)" : "Estimación fuera de fold")
      ),

      graficaIntervalo(p),

      el("dl", { class: "data-definition-list" },
        el("dt", {}, "Rango 80% (seguro)"), el("dd", {}, `${fmtNum(p.lo80)} – ${fmtNum(p.hi80)} t/ha`),
        el("dt", {}, "Rango 90% (amplio)"), el("dd", {}, `${fmtNum(p.lo90)} – ${fmtNum(p.hi90)} t/ha`),
        ...(p.real !== null ? [
          el("dt", {}, "Cosecha real medida"), el("dd", {}, `${fmtNum(p.real)} t/ha`),
          el("dt", {}, "Error (est. − real)"), el("dd", {}, `${fmtSign(p.error)} t/ha`),
        ] : []),
        el("dt", {}, "Superficie"), el("dd", {}, `${fmtNum(p.area, 2)} ha`),
        el("dt", {}, "Producción total"), el("dd", {}, `${fmtNum(p.produccion, 1)} t`),
        el("dt", {}, `Prob. superar ${fmtNum(S.umbral, 1)} t/ha`), el("dd", {}, pildoraRiesgo(prob, true))
      ),

      el("div", { style: "margin-top:10px;display:flex;gap:8px;" },
        el("button", {
          class: "btn btn-primary btn-sm",
          type: "button",
          style: "width:100%;",
          onclick: () => {
            activarVista("prediccion");
            seleccionar(p.id);
          },
        }, "Ver Ficha Completa de Predicción →")
      )
    );
  }

  /* ==========================================================================
     SECCIÓN A: RESUMEN GENERAL (DASHBOARD)
     ========================================================================== */
  function pintarResumenGeneral() {
    const kpisCont = $("#resumen-kpis");
    if (!kpisCont) return;
    empty(kpisCont);

    const supTotal = P.reduce((a, p) => a + p.area, 0);
    const prodTotal = P.reduce((a, p) => a + p.produccion, 0);
    const supOcultas = ocultas.reduce((a, p) => a + p.area, 0);
    const prodOcultas = ocultas.reduce((a, p) => a + p.produccion, 0);
    const supConocidas = conocidas.reduce((a, p) => a + p.area, 0);
    const prodConocidas = conocidas.reduce((a, p) => a + (p.real * p.area), 0);

    const rendMedioObs = prodConocidas / (supConocidas || 1);
    const rendMedioEst59 = prodOcultas / (supOcultas || 1);

    const kpi = (label, val, unit, sub, accent = "") => el("div", { class: `kpi-card ${accent}` },
      el("div", { class: "kpi-label" }, label),
      el("div", { class: "kpi-value-row" },
        el("span", { class: "kpi-value" }, val),
        unit ? el("span", { class: "kpi-unit" }, unit) : null
      ),
      sub ? el("div", { class: "kpi-subtext" }, sub) : null
    );

    kpisCont.append(
      kpi("Total de Parcelas", fmtNum(P.length, 0), "parcelas", "100% georreferenciadas en Hidalgo, Puebla y Tlaxcala"),
      kpi("Parcelas por Estimar", fmtNum(ocultas.length, 0), "del reto", `${fmtNum(supOcultas, 0)} ha · ${fmtNum(rendMedioEst59, 2)} t/ha promedio est.`, "accent-amber"),
      kpi("Parcelas Observadas", fmtNum(conocidas.length, 0), "con etiqueta", `${fmtNum(supConocidas, 0)} ha · ${fmtNum(rendMedioObs, 2)} t/ha real ponderado`, "accent-blue"),
      kpi("Superficie Total", fmtNum(supTotal, 0), "ha", `Producción estimada total: ${fmtNum(prodTotal, 0)} toneladas`)
    );

    // Gráfica de Distribución de Rendimiento
    pintarGraficaDistribucionResumen();

    // Tabla de Estados
    pintarTablaEstadosResumen();

    // Lista rápida de las 59 parcelas a calificar
    pintarLista59Resumen();

    // Resumen de calidad del modelo
    pintarCalidadModeloResumen();
  }

  function pintarGraficaDistribucionResumen() {
    const cont = $("#chart-resumen-distribucion");
    if (!cont) return;
    empty(cont);

    const bins = [
      { r: "2.0–2.8", min: 2.0, max: 2.8, obs: 0, est: 0 },
      { r: "2.8–3.4", min: 2.8, max: 3.4, obs: 0, est: 0 },
      { r: "3.4–4.0", min: 3.4, max: 4.0, obs: 0, est: 0 },
      { r: "4.0–4.6", min: 4.0, max: 4.6, obs: 0, est: 0 },
      { r: "4.6–5.2", min: 4.6, max: 5.2, obs: 0, est: 0 },
      { r: "5.2–6.0", min: 5.2, max: 6.0, obs: 0, est: 0 },
    ];

    for (const p of conocidas) {
      const b = bins.find((bin) => p.real >= bin.min && p.real < bin.max) || bins[bins.length - 1];
      b.obs++;
    }
    for (const p of P) {
      const b = bins.find((bin) => p.estimado >= bin.min && p.estimado < bin.max) || bins[bins.length - 1];
      b.est++;
    }

    const ancho = 500, alto = 240, m = { l: 40, r: 20, t: 30, b: 40 };
    const maxVal = Math.max(...bins.flatMap((b) => [b.obs, b.est])) * 1.15;
    const X = (i) => m.l + (i + 0.5) * ((ancho - m.l - m.r) / bins.length);
    const Y = (v) => alto - m.b - (v / maxVal) * (alto - m.t - m.b);

    const s = svgEl("svg", { class: "chart", viewBox: `0 0 ${ancho} ${alto}` });

    // Rejillas
    for (let i = 0; i <= maxVal; i += 15) {
      s.append(
        svgEl("line", { class: "chart-grid-line", x1: m.l, x2: ancho - m.r, y1: Y(i), y2: Y(i) }),
        svgEl("text", { class: "chart-axis-text", x: m.l - 6, y: Y(i) + 4, "text-anchor": "end" }, i)
      );
    }

    const bw = 24;
    bins.forEach((b, i) => {
      const cx = X(i);
      // Barra Observadas
      const yObs = Y(b.obs), hObs = Y(0) - yObs;
      s.append(
        svgEl("rect", { class: "chart-bar-rect", x: cx - bw - 2, y: yObs, width: bw, height: hObs, fill: "var(--brand-700)", rx: 2 }),
        svgEl("text", { class: "chart-axis-text", x: cx - bw / 2 - 2, y: yObs - 4, "text-anchor": "middle" }, b.obs)
      );
      // Barra Estimadas
      const yEst = Y(b.est), hEst = Y(0) - yEst;
      s.append(
        svgEl("rect", { class: "chart-bar-rect", x: cx + 2, y: yEst, width: bw, height: hEst, fill: "var(--brand-400)", rx: 2 }),
        svgEl("text", { class: "chart-axis-text", x: cx + bw / 2 + 2, y: yEst - 4, "text-anchor": "middle" }, b.est)
      );
      // Etiqueta eje X
      s.append(svgEl("text", { class: "chart-axis-text", x: cx, y: alto - m.b + 18, "text-anchor": "middle" }, b.r));
    });

    s.append(
      svgEl("text", { class: "chart-axis-title", x: (m.l + ancho - m.r) / 2, y: alto - 6, "text-anchor": "middle" }, "Rango de rendimiento (t/ha)")
    );

    cont.append(
      el("div", { style: "display:flex;gap:16px;margin-bottom:8px;font-size:12px;color:var(--text-secondary);" },
        el("span", {}, el("i", { style: "display:inline-block;width:10px;height:10px;background:var(--brand-700);margin-right:5px;border-radius:2px;" }), "Rendimiento observado (138 parcelas)"),
        el("span", {}, el("i", { style: "display:inline-block;width:10px;height:10px;background:var(--brand-400);margin-right:5px;border-radius:2px;" }), "Rendimiento estimado (197 parcelas)")
      ),
      s
    );
  }

  function pintarTablaEstadosResumen() {
    const cont = $("#tabla-resumen-estados");
    if (!cont) return;
    empty(cont);

    const filas = ESTADOS.map((e) => {
      const ps = P.filter((p) => p.estado === e);
      const sup = ps.reduce((a, p) => a + p.area, 0);
      const estMed = ps.reduce((a, p) => a + p.estimado * p.area, 0) / (sup || 1);
      const conReal = ps.filter((p) => p.real !== null);
      const realMed = conReal.reduce((a, p) => a + p.real * p.area, 0) / (conReal.reduce((a, p) => a + p.area, 0) || 1);
      const porEstimar = ps.filter((p) => p.esOculta).length;
      return { estado: e, n: ps.length, porEstimar, sup, estMed, realMed, prod: sup * estMed };
    });

    const tabla = el("table", { class: "data-table" },
      el("thead", {}, el("tr", {},
        el("th", {}, "Estado"),
        el("th", { class: "num" }, "Parcelas"),
        el("th", { class: "num" }, "Por Estimar"),
        el("th", { class: "num" }, "Superficie"),
        el("th", { class: "num" }, "Est. Ponderado"),
        el("th", { class: "num" }, "Real Ponderado"),
        el("th", { class: "num" }, "Producción Est.")
      )),
      el("tbody", {}, filas.map((f) => el("tr", {
        onclick: () => {
          S.estado = f.estado;
          activarVista("mapa");
          pintarFiltrosMapa();
          pintarMapa();
          ajustarVista(P.filter(filtraMapa));
        },
      },
        el("td", { style: "font-weight:600;" }, f.estado),
        el("td", { class: "num" }, f.n),
        el("td", { class: "num" }, el("span", { class: "chip-tag" }, f.porEstimar)),
        el("td", { class: "num" }, fmtNum(f.sup, 1) + " ha"),
        el("td", { class: "num", style: "color:var(--brand-700);font-weight:600;" }, fmtNum(f.estMed, 2) + " t/ha"),
        el("td", { class: "num" }, fmtNum(f.realMed, 2) + " t/ha"),
        el("td", { class: "num" }, fmtNum(f.prod, 0) + " t")
      )))
    );
    cont.append(tabla);
  }

  function pintarLista59Resumen() {
    const cont = $("#resumen-lista-59");
    if (!cont) return;
    empty(cont);

    const tabla = el("table", { class: "mini-table" },
      el("thead", {}, el("tr", {},
        el("th", {}, "ID"),
        el("th", {}, "Municipio"),
        el("th", { class: "num" }, "Sup (ha)"),
        el("th", { class: "num" }, "Est. t/ha"),
        el("th", { class: "num" }, "Rango 80%"),
        el("th", {}, "Acción")
      )),
      el("tbody", {}, ocultas.slice(0, 30).map((p) => el("tr", {},
        el("td", { class: "mono", style: "font-weight:600;" }, p.id),
        el("td", {}, p.municipio),
        el("td", { class: "num" }, fmtNum(p.area, 1)),
        el("td", { class: "num", style: "font-weight:600;color:var(--brand-700);" }, fmtNum(p.estimado, 2)),
        el("td", { class: "num" }, `${fmtNum(p.lo80, 2)}–${fmtNum(p.hi80, 2)}`),
        el("td", {}, el("button", {
          class: "btn btn-outline btn-sm",
          style: "padding:2px 7px;font-size:11px;",
          onclick: () => {
            seleccionar(p.id, { enfocar: true });
            activarVista("prediccion");
          },
        }, "Ficha"))
      )))
    );
    cont.append(tabla);
  }

  function pintarCalidadModeloResumen() {
    const cont = $("#resumen-calidad-modelo");
    if (!cont) return;
    empty(cont);

    const m = D.metricas;
    const cv = m.cv138;
    cont.append(
      el("dl", { class: "data-definition-list", style: "font-size:13px;gap:10px 14px;" },
        el("dt", {}, "Modelo seleccionado"), el("dd", {}, m.modelo),
        el("dt", {}, "R² Validación Cruzada (138)"), el("dd", { style: "color:var(--brand-700);font-weight:700;" }, fmtNum(cv.R2, 3)),
        el("dt", {}, "RMSE Validación Cruzada"), el("dd", {}, fmtNum(cv.RMSE_t_ha, 3) + " t/ha"),
        el("dt", {}, "MAE Validación Cruzada"), el("dd", {}, fmtNum(cv.MAE_t_ha, 3) + " t/ha"),
        el("dt", {}, "Predicciones dentro de ±20%"), el("dd", {}, fmtNum(cv.dentro_20_pct, 1) + " %"),
        el("dt", {}, "Mejora sobre predecir la media"), el("dd", { style: "color:var(--brand-700);" }, "+" + fmtNum(m.mejora_vs_media_pct, 1) + " %"),
        el("dt", {}, "Prueba independiente (28)"), el("dd", {}, `R² ${fmtNum(m.prueba.R2, 2)} · RMSE ${fmtNum(m.prueba.RMSE_t_ha, 3)} t/ha`)
      ),
      el("div", { style: "margin-top:14px;display:flex;gap:8px;" },
        el("button", {
          class: "btn btn-outline btn-sm",
          type: "button",
          onclick: () => activarVista("evaluacion"),
        }, "Ver Evaluación Detallada →")
      )
    );
  }

  /* ==========================================================================
     SECCIÓN C: FICHA DE PREDICCIÓN POR PARCELA (DESGLOSE Y EXPORTACIÓN)
     ========================================================================== */
  function iniciarFichaPrediccion() {
    const sel = $("#sel-parcela-ficha");
    if (!sel) return;
    empty(sel);

    const optgroup = (label, items) => el("optgroup", { label }, items.map((p) => el("option", { value: p.id }, `${p.id} · ${p.municipio} (${fmtNum(p.area, 1)} ha)`)));
    sel.append(
      optgroup(`Parcelas a calificar Reto FIRA (${ocultas.length})`, ocultas),
      optgroup(`Parcelas con rendimiento conocido (${conocidas.length})`, conocidas)
    );

    sel.value = S.sel;
    sel.addEventListener("change", (e) => {
      seleccionar(e.target.value);
    });

    const btnImp = $("#btn-imprimir-ficha");
    if (btnImp) btnImp.addEventListener("click", () => window.print());
  }

  function pintarFichaPrediccion() {
    const cont = $("#contenedor-ficha-completa");
    if (!cont) return;
    empty(cont);

    const p = S.sel && porId.get(S.sel);
    if (!p) {
      cont.append(el("p", { class: "text-muted" }, "Selecciona una parcela válida."));
      return;
    }

    // Predicción y Desglose Matemático
    const entrada = {
      x: p.x, y: p.y, area: p.area,
      evi_2022: p.variables.evi_2022,
      evi_2023: p.variables.evi_2023,
      ndvi_media: p.variables.ndvi_media,
    };
    const r = M.predecir(D.modelo, entrada, { excluir: p.esOculta ? undefined : p.id });
    const prob = probDe(p);

    const cardHeader = el("div", { class: "card", style: "margin-bottom:20px;" },
      el("div", { style: "display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-start;gap:16px;border-bottom:1px solid var(--border-subtle);padding-bottom:16px;" },
        el("div", {},
          el("div", { style: "display:flex;align-items:center;gap:10px;" },
            el("h2", { style: "font-size:26px;font-family:var(--font-mono);font-weight:700;" }, p.id),
            el("span", { class: "badge-status " + (p.esOculta ? "estimar" : "conocida") },
              p.esOculta ? "Reto FIRA 2026 · Cosecha por Estimar" : "Parcela de Referencia · Medida en Campo"
            )
          ),
          el("div", { class: "text-secondary", style: "font-size:14px;margin-top:4px;" },
            `${p.municipio}, ${p.estado} · Coordenadas: ${p.lat.toFixed(5)}° N, ${p.lon.toFixed(5)}° W`
          )
        ),
        el("div", { style: "display:flex;gap:10px;" },
          el("button", {
            class: "btn btn-outline btn-sm",
            onclick: () => {
              activarVista("mapa");
              seleccionar(p.id, { enfocar: true });
            },
          }, "Localizar en Mapa"),
          el("button", {
            class: "btn btn-outline btn-sm",
            onclick: () => {
              activarVista("simulador");
              simCargar(p.id);
            },
          }, "Llevar al Simulador")
        )
      ),

      el("div", { class: "grid-4", style: "margin-top:16px;margin-bottom:0;" },
        el("div", { class: "kpi-card" },
          el("div", { class: "kpi-label" }, "Rendimiento Estimado"),
          el("div", { class: "kpi-value-row" },
            el("span", { class: "kpi-value", style: "color:var(--brand-700);" }, fmtNum(p.estimado, 2)),
            el("span", { class: "kpi-unit" }, "t/ha")
          ),
          el("div", { class: "kpi-subtext" }, p.esOculta ? "Oficial concurso FIRA" : "Estimación fuera de fold")
        ),

        el("div", { class: "kpi-card" },
          el("div", { class: "kpi-label" }, "Superficie y Producción"),
          el("div", { class: "kpi-value-row" },
            el("span", { class: "kpi-value" }, fmtNum(p.produccion, 1)),
            el("span", { class: "kpi-unit" }, "toneladas")
          ),
          el("div", { class: "kpi-subtext" }, `En ${fmtNum(p.area, 2)} hectáreas de terreno`)
        ),

        el("div", { class: "kpi-card" },
          el("div", { class: "kpi-label" }, "Rango de Seguridad (80%)"),
          el("div", { class: "kpi-value-row" },
            el("span", { class: "kpi-value", style: "font-size:20px;" }, `${fmtNum(p.lo80, 2)} – ${fmtNum(p.hi80, 2)}`),
            el("span", { class: "kpi-unit" }, "t/ha")
          ),
          el("div", { class: "kpi-subtext" }, `Amplitud: ±${fmtNum((p.hi80 - p.lo80) / 2, 2)} t/ha (${p.zona})`)
        ),

        el("div", { class: "kpi-card" },
          el("div", { class: "kpi-label" }, "Viabilidad Crediticia"),
          el("div", { class: "kpi-value-row" },
            el("span", { class: "kpi-value", style: "font-size:22px;" }, probTexto(prob, residuosDe(p).length))
          ),
          el("div", { class: "kpi-subtext" }, pildoraRiesgo(prob, false))
        )
      )
    );

    // Desglose del Modelo y Vecinos
    const gridDetalle = el("div", { class: "grid-2" },
      // Tarjeta de Desglose Matemático
      el("div", { class: "card" },
        el("div", { class: "card-header" },
          el("h3", { class: "card-title" }, "¿Cómo se Calculó este Resultado?"),
          el("span", { class: "text-muted", style: "font-size:11px;" }, "GeoResidualRegressor")
        ),
        el("p", { class: "text-secondary", style: "font-size:12.5px;margin-bottom:12px;" },
          "El modelo descompone el pronóstico en dos fases científicas: el efecto espacial de la zona y la corrección fenológica residual:"
        ),
        el("table", { class: "data-table" },
          el("tbody", {},
            el("tr", {},
              el("td", { style: "font-weight:600;" }, `1. Efecto geográfico (${D.modelo.n_vecinos} vecinos)`),
              el("td", { class: "num", style: "font-weight:600;" }, fmtNum(r.efecto_zona, 4) + " t/ha")
            ),
            ...r.aportes.map((a) => el("tr", {},
              el("td", { style: "padding-left:24px;color:var(--text-secondary);" },
                `+ Aporte ${NOMBRE_VAR[a.variable] || a.variable}`
              ),
              el("td", { class: "num", style: "color:var(--brand-700);" }, fmtSign(a.aporte, 4) + " t/ha")
            )),
            el("tr", {},
              el("td", { style: "padding-left:24px;color:var(--text-secondary);" }, "Constante (intercepto residual)"),
              el("td", { class: "num" }, fmtSign(D.modelo.intercepto, 4) + " t/ha")
            ),
            el("tr", { style: "background-color:var(--surface-well);border-top:2px solid var(--border-card);" },
              el("td", { style: "font-weight:700;" }, "= Pronóstico Final Estimado"),
              el("td", { class: "num", style: "font-weight:700;color:var(--brand-800);font-size:15px;" }, fmtNum(r.estimado, 4) + " t/ha")
            )
          )
        )
      ),

      // Red de Vecinos Más Cercanos
      el("div", { class: "card" },
        el("div", { class: "card-header" },
          el("h3", { class: "card-title" }, "Parcelas Vecinas Más Influyentes"),
          el("span", { class: "text-muted", style: "font-size:11px;" }, "Ponderación 1/d")
        ),
        el("p", { class: "text-secondary", style: "font-size:12.5px;margin-bottom:12px;" },
          "Parcelas con cosecha medida más próximas que definen el nivel base de fertilidad del suelo:"
        ),
        el("table", { class: "data-table" },
          el("thead", {}, el("tr", {},
            el("th", {}, "Parcela"),
            el("th", { class: "num" }, "Distancia"),
            el("th", { class: "num" }, "Influencia"),
            el("th", { class: "num" }, "Cosecha Real")
          )),
          el("tbody", {}, r.vecinos.slice(0, 6).map((v) => el("tr", {
            onclick: () => seleccionar(v.id),
          },
            el("td", { class: "mono", style: "font-weight:600;color:var(--brand-700);" }, v.id),
            el("td", { class: "num" }, fmtNum(v.d, 2) + " km"),
            el("td", { class: "num" }, fmtPct(v.peso, 1)),
            el("td", { class: "num", style: "font-weight:600;" }, fmtNum(v.real, 2) + " t/ha")
          )))
        )
      )
    );

    cont.append(cardHeader, gridDetalle);
  }

  /* ==========================================================================
     SECCIÓN D: SIMULADOR DE ESCENARIOS
     ========================================================================== */
  const SIM = { base: null };
  const camposSim = {};

  const rangoVar = (k) => {
    const v = P.map((p) => p.variables[k]).filter(Number.isFinite).sort((a, b) => a - b);
    return { min: v[0], max: v[v.length - 1], med: v[v.length >> 1] };
  };
  const RANGOS = { evi22: rangoVar("evi_2022"), evi23: rangoVar("evi_2023"), ndvi: rangoVar("ndvi_media") };

  function iniciarSimulador() {
    camposSim.lon = $("#sim-lon");
    camposSim.lat = $("#sim-lat");
    camposSim.area = $("#sim-area");
    camposSim.evi22 = $("#sim-evi22");
    camposSim.evi23 = $("#sim-evi23");
    camposSim.ndvi = $("#sim-ndvi");

    const sel = $("#sim-parcela");
    if (!sel) return;
    empty(sel);

    const grupo = (rotulo, lista) => el("optgroup", { label: rotulo },
      lista.map((p) => el("option", { value: p.id }, `${p.id} · ${p.municipio} · ${fmtNum(p.area, 1)} ha`)));
    sel.append(grupo(`Por estimar (${ocultas.length})`, ocultas), grupo(`Con rendimiento conocido (${conocidas.length})`, conocidas));

    for (const [k, r] of Object.entries(RANGOS)) {
      const c = camposSim[k];
      if (!c) continue;
      c.min = (Math.floor(r.min * 100) / 100 - 0.02).toFixed(3);
      c.max = (Math.ceil(r.max * 100) / 100 + 0.02).toFixed(3);
      c.step = "0.001";
    }

    sel.addEventListener("change", () => simCargar(sel.value));
    $("#sim-excluir")?.addEventListener("change", simCalcular);
    for (const c of Object.values(camposSim)) c?.addEventListener("input", simCalcular);
    $("#sim-reset")?.addEventListener("click", () => simCargar($("#sim-parcela").value));

    simCargar(ocultas[0].id);
  }

  function simCargar(id) {
    const p = porId.get(id);
    if (!p) return;
    SIM.base = p;
    const sel = $("#sim-parcela");
    if (sel) sel.value = id;

    if (camposSim.lon) camposSim.lon.value = p.lon.toFixed(5);
    if (camposSim.lat) camposSim.lat.value = p.lat.toFixed(5);
    if (camposSim.area) camposSim.area.value = p.area;
    if (camposSim.evi22) camposSim.evi22.value = p.variables.evi_2022;
    if (camposSim.evi23) camposSim.evi23.value = p.variables.evi_2023;
    if (camposSim.ndvi) camposSim.ndvi.value = p.variables.ndvi_media;

    const excl = $("#sim-excluir");
    if (excl) excl.disabled = p.esOculta;

    simCalcular();
  }

  function simCalcular() {
    if (!camposSim.lon) return;
    const lon = parseFloat(camposSim.lon.value), lat = parseFloat(camposSim.lat.value), area = parseFloat(camposSim.area.value);
    const evi22 = parseFloat(camposSim.evi22.value), evi23 = parseFloat(camposSim.evi23.value), ndvi = parseFloat(camposSim.ndvi.value);

    const rotular = (id, v, r) => {
      const nodo = $(id);
      if (nodo) nodo.textContent = `Actual: ${fmtNum(v, 3)} | Mediana regional: ${fmtNum(r.med, 3)} | Rango: ${fmtNum(r.min, 3)} a ${fmtNum(r.max, 3)}`;
    };
    rotular("#sim-evi22-v", evi22, RANGOS.evi22);
    rotular("#sim-evi23-v", evi23, RANGOS.evi23);
    rotular("#sim-ndvi-v", ndvi, RANGOS.ndvi);

    const cont = $("#sim-resultado");
    if (!cont) return;
    empty(cont);

    if (![lon, lat, area].every(Number.isFinite) || area <= 0) {
      cont.append(el("div", { class: "card" }, el("p", { class: "text-muted" }, "Ingresa coordenadas y superficie válidas para calcular la simulación.")));
      return;
    }

    const q = M.proyectar(lon, lat, D.proyeccion);
    const excluir = $("#sim-excluir")?.checked && SIM.base && !SIM.base.esOculta ? SIM.base.id : undefined;
    const r = M.predecir(D.modelo, { x: q.x, y: q.y, area, evi_2022: evi22, evi_2023: evi23, ndvi_media: ndvi }, { excluir });

    const masCercana = P.filter((p) => p.id !== excluir).reduce((a, p) =>
      (Math.hypot(p.x - q.x, p.y - q.y) < Math.hypot(a.x - q.x, a.y - q.y) ? p : a));
    const zona = masCercana.zona;
    const s80 = M.semiamplitud(D.intervalos, zona, 80);
    const s90 = M.semiamplitud(D.intervalos, zona, 90);
    const prob = M.probabilidad(r.estimado, D.residuos_por_zona[zona] || D.residuos_por_zona.Todas, S.umbral);
    const dMin = r.vecinos[0].d;

    const delta = SIM.base ? r.estimado - SIM.base.estimado : 0;

    const resCard = el("div", { class: "card" },
      el("div", { class: "card-header" },
        el("h3", { class: "card-title" }, "Resultado del Escenario Hipotético"),
        el("span", { class: "badge-status simulada" }, "Simulación Activa")
      ),
      el("div", { class: "grid-2", style: "margin-bottom:14px;" },
        el("div", { class: "big-stat-callout" },
          el("div", { class: "big-stat-value", style: "color:var(--brand-700);" }, fmtNum(r.estimado, 2), " ", el("span", { style: "font-size:16px;" }, "t/ha")),
          el("div", { class: "big-stat-unit" }, "Cosecha estimada simulada"),
          SIM.base ? el("div", { style: "font-size:11.5px;color:var(--text-muted);margin-top:4px;" },
            `Variación vs parcela base (${SIM.base.id}): `,
            el("b", { style: `color:${delta >= 0 ? "var(--brand-700)" : "var(--d-pos)"}` }, fmtSign(delta, 2) + " t/ha")
          ) : null
        ),
        el("div", { class: "big-stat-callout" },
          el("div", { class: "big-stat-value" }, fmtNum(r.estimado * area, 1), " ", el("span", { style: "font-size:16px;" }, "t")),
          el("div", { class: "big-stat-unit" }, `Producción total en ${fmtNum(area, 1)} ha`),
          el("div", { style: "font-size:11.5px;color:var(--text-muted);margin-top:4px;" },
            `Probabilidad superar ${fmtNum(S.umbral, 1)} t/ha: `, pildoraRiesgo(prob, true)
          )
        )
      ),

      el("dl", { class: "data-definition-list" },
        el("dt", {}, "Rango 80% (conformal)"), el("dd", {}, `${fmtNum(r.estimado - s80)} a ${fmtNum(r.estimado + s80)} t/ha`),
        el("dt", {}, "Rango 90% (amplio)"), el("dd", {}, `${fmtNum(r.estimado - s90)} a ${fmtNum(r.estimado + s90)} t/ha`),
        el("dt", {}, "Efecto de zona (30 vecinos)"), el("dd", {}, fmtNum(r.efecto_zona, 3) + " t/ha"),
        el("dt", {}, "Corrección fenológica residual"), el("dd", {}, fmtSign(r.correccion, 3) + " t/ha"),
        el("dt", {}, "Distancia al vecino más próximo"), el("dd", {}, fmtNum(dMin, 2) + " km")
      ),

      dMin > 3 ? el("div", { class: "alert-banner warning", style: "margin-top:14px;margin-bottom:0;" },
        el("div", {}, `Este terreno simulado está a ${fmtNum(dMin, 1)} km de la parcela de referencia más cercana. La incertidumbre estadística es mayor al estar fuera del núcleo denso de muestreo.`)
      ) : null
    );

    cont.append(resCard);
  }

  /* ==========================================================================
     SECCIÓN E: ANÁLISIS AMBIENTAL
     ========================================================================== */
  function pintarAnalisisAmbiental() {
    const kpisCont = $("#ambiental-kpis");
    if (!kpisCont) return;
    empty(kpisCont);

    const kpi = (label, val, unit, sub) => el("div", { class: "kpi-card" },
      el("div", { class: "kpi-label" }, label),
      el("div", { class: "kpi-value-row" },
        el("span", { class: "kpi-value" }, val),
        unit ? el("span", { class: "kpi-unit" }, unit) : null
      ),
      sub ? el("div", { class: "kpi-subtext" }, sub) : null
    );

    kpisCont.append(
      kpi("Precipitación Ciclo 2025", "580 – 1,360", "mm", "Extracción CHIRPS acumulada abril–octubre"),
      kpi("Temperatura Media Aprox.", "15.6 – 18.2", "°C", "Extracción CHIRTS-ERA5"),
      kpi("Rango de Elevación", "2,180 – 2,940", "msnm", "INEGI Continuo de Elevaciones (CEM 4.0)"),
      kpi("Índices Satelitales", "3 Sensores", "S2 · Landsat · Planet", "Series temporales filtradas por nubosidad ≤20%")
    );

    // Selector de Dispersión
    const selDisp = $("#sel-var-dispersion");
    if (selDisp && !selDisp.childNodes.length) {
      selDisp.append(
        el("option", { value: "evi_2022" }, "EVI primavera 2022 (Sentinel-2)"),
        el("option", { value: "evi_2023" }, "EVI primavera 2023 (Sentinel-2)"),
        el("option", { value: "ndvi_media" }, "NDVI promedio 2022–2024 (Sentinel-2)"),
        el("option", { value: "area" }, "Superficie de la parcela (ha)")
      );
      selDisp.onchange = () => pintarDispersionAmbiental(selDisp.value);
    }
    pintarDispersionAmbiental(selDisp ? selDisp.value : "evi_2022");

    // Gráfica de Clima Mensual
    const selClima = $("#sel-var-clima-mensual");
    if (selClima) {
      selClima.onchange = () => pintarClimaMensual(selClima.value);
    }
    pintarClimaMensual(selClima ? selClima.value : "PREC");

    // Tabla Comparativa Ambiental
    pintarTablaAmbientalResumen();
  }

  function pintarDispersionAmbiental(varKey) {
    const cont = $("#chart-dispersion-ambiental");
    if (!cont) return;
    empty(cont);

    const puntos = conocidas.map((p) => ({
      id: p.id,
      x: varKey === "area" ? p.area : p.variables[varKey],
      y: p.real,
      estado: p.estado,
    })).filter((pt) => Number.isFinite(pt.x) && Number.isFinite(pt.y));

    const ancho = 520, alto = 340, m = { l: 48, r: 16, t: 20, b: 44 };
    const xs = puntos.map((p) => p.x);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = 1.5, maxY = 6.0;

    const X = (v) => m.l + ((v - minX) / (maxX - minX || 1)) * (ancho - m.l - m.r);
    const Y = (v) => alto - m.b - ((v - minY) / (maxY - minY)) * (alto - m.t - m.b);

    const s = svgEl("svg", { class: "chart", viewBox: `0 0 ${ancho} ${alto}` });

    // Rejilla Y
    for (let t = 2; t <= 6; t++) {
      s.append(
        svgEl("line", { class: "chart-grid-line", x1: m.l, x2: ancho - m.r, y1: Y(t), y2: Y(t) }),
        svgEl("text", { class: "chart-axis-text", x: m.l - 6, y: Y(t) + 4, "text-anchor": "end" }, t)
      );
    }

    // Puntos
    puntos.forEach((p) => {
      const c = svgEl("circle", {
        class: "chart-data-point",
        cx: X(p.x),
        cy: Y(p.y),
        r: 5,
        fill: COLOR_ESTADO[p.estado] || "var(--brand-700)",
      });
      c.addEventListener("pointermove", (e) => mostrarTip([
        ["Parcela", p.id],
        ["Estado", p.estado],
        ["Valor variable", fmtNum(p.x, 3)],
        ["Cosecha real", fmtNum(p.y, 2) + " t/ha"],
      ], e));
      c.addEventListener("pointerleave", ocultarTip);
      s.append(c);
    });

    s.append(
      svgEl("text", { class: "chart-axis-title", x: (m.l + ancho - m.r) / 2, y: alto - 6, "text-anchor": "middle" }, NOMBRE_VAR[varKey] || varKey),
      svgEl("text", { class: "chart-axis-title", transform: `translate(14, ${(m.t + alto - m.b) / 2}) rotate(-90)`, "text-anchor": "middle" }, "Cosecha Real (t/ha)")
    );

    cont.append(
      el("div", { style: "display:flex;gap:14px;margin-bottom:8px;font-size:12px;color:var(--text-secondary);" },
        ...ESTADOS.map((e) => el("span", {}, el("i", { style: `display:inline-block;width:10px;height:10px;background:${COLOR_ESTADO[e]};margin-right:5px;border-radius:50%;` }), e))
      ),
      s
    );
  }

  function pintarClimaMensual(varId) {
    const cont = $("#chart-clima-mensual");
    if (!cont || !T) return;
    empty(cont);

    const variable = T.variables.find((v) => v.id === varId);
    if (!variable || variable.estatica) {
      cont.append(el("p", { class: "text-muted" }, "Variable no temporal."));
      return;
    }

    const anio = 2025;
    const meses = [4, 5, 6, 7, 8, 9, 10];
    const mesesNom = ["Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct"];

    const porEstado = {};
    for (const e of ESTADOS) porEstado[e] = [];

    meses.forEach((m) => {
      const clave = `${varId}|${anio}|${m}`;
      const capa = TERR.capas.get(clave);
      for (const e of ESTADOS) {
        const ps = P.filter((p) => p.estado === e);
        const vals = ps.map((p) => capa && capa.get(p.id) ? capa.get(p.id).valor : null).filter(Number.isFinite);
        const prom = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
        porEstado[e].push(prom);
      }
    });

    const ancho = 520, alto = 340, m = { l: 48, r: 16, t: 20, b: 44 };
    const todosVals = Object.values(porEstado).flat();
    const minV = Math.min(0, ...todosVals), maxV = Math.max(...todosVals) * 1.15;

    const X = (i) => m.l + (i / (meses.length - 1)) * (ancho - m.l - m.r);
    const Y = (v) => alto - m.b - ((v - minV) / (maxV - minV || 1)) * (alto - m.t - m.b);

    const s = svgEl("svg", { class: "chart", viewBox: `0 0 ${ancho} ${alto}` });

    // Rejilla
    const paso = varId === "PREC" ? 50 : 5;
    for (let v = 0; v <= maxV; v += paso) {
      s.append(
        svgEl("line", { class: "chart-grid-line", x1: m.l, x2: ancho - m.r, y1: Y(v), y2: Y(v) }),
        svgEl("text", { class: "chart-axis-text", x: m.l - 6, y: Y(v) + 4, "text-anchor": "end" }, v)
      );
    }

    // Líneas por estado
    for (const e of ESTADOS) {
      const vals = porEstado[e];
      const pts = vals.map((v, i) => `${X(i)},${Y(v)}`).join(" ");
      s.append(svgEl("polyline", {
        fill: "none",
        stroke: COLOR_ESTADO[e],
        "stroke-width": 2.5,
        points: pts,
      }));
      vals.forEach((v, i) => {
        const c = svgEl("circle", { cx: X(i), cy: Y(v), r: 4.5, fill: COLOR_ESTADO[e], stroke: "var(--surface-card)", "stroke-width": 1.5 });
        c.addEventListener("pointermove", (ev) => mostrarTip([
          ["Estado", e],
          ["Mes", mesesNom[i]],
          ["Promedio", fmtNum(v, 1) + " " + variable.unidad],
        ], ev));
        c.addEventListener("pointerleave", ocultarTip);
        s.append(c);
      });
    }

    mesesNom.forEach((nom, i) => {
      s.append(svgEl("text", { class: "chart-axis-text", x: X(i), y: alto - m.b + 18, "text-anchor": "middle" }, nom));
    });

    s.append(
      svgEl("text", { class: "chart-axis-title", x: (m.l + ancho - m.r) / 2, y: alto - 6, "text-anchor": "middle" }, "Mes del ciclo agrícola (2025)"),
      svgEl("text", { class: "chart-axis-title", transform: `translate(14, ${(m.t + alto - m.b) / 2}) rotate(-90)`, "text-anchor": "middle" }, `${variable.nombre} (${variable.unidad})`)
    );

    cont.append(
      el("div", { style: "display:flex;gap:14px;margin-bottom:8px;font-size:12px;color:var(--text-secondary);" },
        ...ESTADOS.map((e) => el("span", {}, el("i", { style: `display:inline-block;width:10px;height:10px;background:${COLOR_ESTADO[e]};margin-right:5px;border-radius:2px;` }), e))
      ),
      s
    );
  }

  function pintarTablaAmbientalResumen() {
    const cont = $("#tabla-resumen-ambiental");
    if (!cont) return;
    empty(cont);

    const stats = [
      { v: "EVI primavera 2022 (Sentinel-2)", min: 0.10, med: 0.28, max: 0.65, u: "adimensional", rol: "Predictor directo en modelo Ridge (+0.062)" },
      { v: "EVI primavera 2023 (Sentinel-2)", min: 0.09, med: 0.26, max: 0.58, u: "adimensional", rol: "Predictor directo en modelo Ridge (+0.032)" },
      { v: "NDVI multianual 2022–2024", min: 0.12, med: 0.35, max: 0.66, u: "adimensional", rol: "Predictor directo en modelo Ridge (+0.030)" },
      { v: "Precipitación acumulada (CHIRPS)", min: 580, med: 840, max: 1361, u: "mm", rol: "Distorsión regional; sin aporte intra-municipio" },
      { v: "Amplitud térmica (CHIRTS)", min: 8.6, med: 11.2, max: 16.8, u: "°C", rol: "Explica R² 0.61 sola; aporte nulo con ubicación" },
      { v: "Elevación topográfica (CEM 4.0)", min: 2180, med: 2540, max: 2940, u: "m", rol: "Contexto territorial estático" },
    ];

    const tabla = el("table", { class: "data-table" },
      el("thead", {}, el("tr", {},
        el("th", {}, "Variable Agroambiental"),
        el("th", { class: "num" }, "Mínimo"),
        el("th", { class: "num" }, "Mediana"),
        el("th", { class: "num" }, "Máximo"),
        el("th", {}, "Unidad"),
        el("th", {}, "Rol en la Modelación Final")
      )),
      el("tbody", {}, stats.map((s) => el("tr", {},
        el("td", { style: "font-weight:600;" }, s.v),
        el("td", { class: "num" }, s.min),
        el("td", { class: "num", style: "font-weight:600;" }, s.med),
        el("td", { class: "num" }, s.max),
        el("td", {}, s.u),
        el("td", { class: "text-secondary" }, s.rol)
      )))
    );
    cont.append(tabla);
  }

  /* ==========================================================================
     SECCIÓN F: EXPLICABILIDAD DEL MODELO
     ========================================================================== */
  function pintarExplicabilidad() {
    const contImp = $("#chart-importancia-variables");
    if (!contImp) return;
    empty(contImp);

    const filas = D.importancia.slice(0, 7);
    const ancho = 520, filaH = 34, m = { l: 160, r: 40, t: 16, b: 36 };
    const alto = m.t + m.b + filas.length * filaH;
    const maxVal = Math.max(...filas.map((f) => f.INCREMENTO_RMSE_MEDIO_T_HA)) * 1.12;

    const X = (v) => m.l + (v / maxVal) * (ancho - m.l - m.r);
    const s = svgEl("svg", { class: "chart", viewBox: `0 0 ${ancho} ${alto}` });

    filas.forEach((f, i) => {
      const y = m.t + i * filaH + 5;
      const bw = Math.max(3, X(f.INCREMENTO_RMSE_MEDIO_T_HA) - m.l);
      const esDom = f.GRUPO_VARIABLES === "Ubicacion";
      s.append(
        svgEl("text", { class: "chart-axis-text", x: m.l - 10, y: y + 16, "text-anchor": "end", fill: "var(--text-main)", "font-weight": esDom ? "700" : "500" }, f.GRUPO_VARIABLES),
        svgEl("rect", { class: "chart-bar-rect", x: m.l, y, width: bw, height: 22, fill: esDom ? "var(--brand-800)" : "var(--brand-500)", rx: 3 }),
        svgEl("text", { class: "chart-axis-text", x: m.l + bw + 8, y: y + 16, "font-weight": "600" }, fmtNum(f.INCREMENTO_RMSE_MEDIO_T_HA, 4) + " t/ha")
      );
    });

    s.append(
      svgEl("line", { class: "chart-axis-line", x1: m.l, x2: m.l, y1: m.t, y2: alto - m.b }),
      svgEl("text", { class: "chart-axis-title", x: (m.l + ancho - m.r) / 2, y: alto - 6, "text-anchor": "middle" }, "Incremento en RMSE al desordenar variables (t/ha)")
    );

    contImp.append(s);

    // Coeficientes Ridge
    pintarCoeficientesRidge();

    // Tabla de índices
    pintarTablaExplicabilidadIndices();
  }

  function pintarCoeficientesRidge() {
    const cont = $("#chart-coeficientes-ridge");
    if (!cont) return;
    empty(cont);

    const mod = D.modelo;
    const coefs = mod.variables.map((v, i) => ({
      nombre: NOMBRE_VAR[v] || v,
      coef: mod.coeficientes[i],
    }));

    const ancho = 520, filaH = 46, m = { l: 300, r: 40, t: 16, b: 36 };
    const alto = m.t + m.b + coefs.length * filaH;
    const maxAbs = Math.max(...coefs.map((c) => Math.abs(c.coef))) * 1.25;

    const zeroX = m.l + (ancho - m.l - m.r) / 2;
    const X = (v) => zeroX + (v / maxAbs) * ((ancho - m.l - m.r) / 2);

    const s = svgEl("svg", { class: "chart", viewBox: `0 0 ${ancho} ${alto}` });

    s.append(svgEl("line", { class: "chart-axis-line", x1: zeroX, x2: zeroX, y1: m.t, y2: alto - m.b, "stroke-width": 1.5 }));

    coefs.forEach((c, i) => {
      const y = m.t + i * filaH + 8;
      const x1 = Math.min(zeroX, X(c.coef));
      const w = Math.abs(X(c.coef) - zeroX);
      const pos = c.coef >= 0;
      s.append(
        svgEl("text", { class: "chart-axis-text", x: m.l - 12, y: y + 16, "text-anchor": "end", fill: "var(--text-main)" }, c.nombre),
        svgEl("rect", { class: "chart-bar-rect", x: x1, y, width: Math.max(2, w), height: 22, fill: pos ? "var(--brand-600)" : "var(--d-pos)", rx: 3 }),
        svgEl("text", { class: "chart-axis-text", x: pos ? zeroX + w + 8 : zeroX - w - 8, y: y + 16, "text-anchor": pos ? "start" : "end", "font-weight": "600" }, fmtSign(c.coef, 4))
      );
    });

    s.append(
      svgEl("text", { class: "chart-axis-title", x: zeroX, y: alto - 6, "text-anchor": "middle" }, "Magnitud y dirección del ajuste residual Ridge")
    );

    cont.append(s);
  }

  function pintarTablaExplicabilidadIndices() {
    const cont = $("#tabla-explicabilidad-indices");
    if (!cont) return;
    empty(cont);

    const ind = D.guia.indices.slice(0, 8);
    const tabla = el("table", { class: "data-table" },
      el("thead", {}, el("tr", {},
        el("th", {}, "Índice / Variable"),
        el("th", { class: "num" }, "R² por sí solo"),
        el("th", { class: "num" }, "RMSE solo"),
        el("th", { class: "num" }, "Aporte s/ Ubicación"),
        el("th", { class: "num" }, "Error Estándar (EE)"),
        el("th", {}, "¿Aporte Claro (>2 EE)?"),
        el("th", {}, "Métrica intra-municipio más asociada")
      )),
      el("tbody", {}, ind.map((f) => el("tr", {},
        el("td", { style: "font-weight:600;" }, f.VARIABLE),
        el("td", { class: "num" }, fmtNum(f.R2_SOLO, 3)),
        el("td", { class: "num" }, fmtNum(f.RMSE_SOLO_T_HA, 3) + " t/ha"),
        el("td", { class: "num", style: "color:var(--brand-700);font-weight:600;" }, fmtSign(f.APORTE_SOBRE_UBICACION_T_HA, 4) + " t/ha"),
        el("td", { class: "num" }, `±${fmtNum(f.APORTE_EE_T_HA, 4)}`),
        el("td", {}, el("span", { class: "chip-tag", style: f.APORTE_CLARO ? "background:var(--risk-high-prob-bg);color:var(--risk-high-prob-text);" : "" },
          f.APORTE_CLARO ? "Sí, significativo" : "No distinguible de cero"
        )),
        el("td", { class: "mono text-secondary" }, f.METRICA_MAS_ASOCIADA_INTRA_MUNICIPIO)
      )))
    );
    cont.append(tabla);
  }

  /* ==========================================================================
     SECCIÓN G: EVALUACIÓN Y MÉTRICAS
     ========================================================================== */
  function pintarEvaluacion() {
    const kpisCont = $("#evaluacion-kpis");
    if (!kpisCont) return;
    empty(kpisCont);

    const m = D.metricas;
    const kpi = (label, val, unit, sub, acc = "") => el("div", { class: `kpi-card ${acc}` },
      el("div", { class: "kpi-label" }, label),
      el("div", { class: "kpi-value-row" },
        el("span", { class: "kpi-value" }, val),
        unit ? el("span", { class: "kpi-unit" }, unit) : null
      ),
      sub ? el("div", { class: "kpi-subtext" }, sub) : null
    );

    kpisCont.append(
      kpi("R² Validación Cruzada", fmtNum(m.cv138.R2, 3), "", "5 folds × 3 rep. sobre 138 parcelas conocidas"),
      kpi("RMSE Validación Cruzada", fmtNum(m.cv138.RMSE_t_ha, 3), "t/ha", `Predecir la media: ${fmtNum(m.cv_media_rmse, 3)} t/ha`),
      kpi("MAE Validación Cruzada", fmtNum(m.cv138.MAE_t_ha, 3), "t/ha", "Error absoluto medio por hectárea"),
      kpi("R² Prueba Independiente", fmtNum(m.prueba.R2, 3), "", `28 parcelas reservadas (RMSE ${fmtNum(m.prueba.RMSE_t_ha, 3)} t/ha)`, "accent-amber")
    );

    // Gráfica de dispersión CV
    dispersionEnContenedor($("#chart-eval-cv"), conocidas.map((p) => ({
      id: p.id, x: p.real, y: p.estimado, estado: p.estado,
    })), "Validación Cruzada 5×3 (138)");

    // Gráfica de dispersión Prueba 28
    const pruebaPts = D.prueba.map((r) => ({
      id: r.ID_POLIGONO, x: r.REAL_T_HA, y: r.PREDICHO_T_HA, estado: porId.get(r.ID_POLIGONO)?.estado || "Puebla",
    }));
    dispersionEnContenedor($("#chart-eval-prueba"), pruebaPts, "Prueba Reservada (28)");

    // Diagnóstico de Valores Atípicos (AGC_015)
    pintarDiagnosticoAtipicos();

    // Tabla Municipios
    pintarTablaMunicipiosEvaluacion();

    // Comparativa de Familias
    pintarTablaComparativaFamilias();
  }

  function dispersionEnContenedor(cont, puntos, titulo) {
    if (!cont) return;
    empty(cont);

    const ancho = 500, alto = 340, m = { l: 48, r: 16, t: 16, b: 44 };
    const rng = [1.5, 6.0];
    const X = (v) => m.l + ((v - rng[0]) / (rng[1] - rng[0])) * (ancho - m.l - m.r);
    const Y = (v) => alto - m.b - ((v - rng[0]) / (rng[1] - rng[0])) * (alto - m.t - m.b);

    const s = svgEl("svg", { class: "chart", viewBox: `0 0 ${ancho} ${alto}` });

    // Líneas de rejilla
    for (let t = 2; t <= 6; t++) {
      s.append(
        svgEl("line", { class: "chart-grid-line", x1: X(t), x2: X(t), y1: m.t, y2: alto - m.b }),
        svgEl("line", { class: "chart-grid-line", x1: m.l, x2: ancho - m.r, y1: Y(t), y2: Y(t) }),
        svgEl("text", { class: "chart-axis-text", x: X(t), y: alto - m.b + 16, "text-anchor": "middle" }, t),
        svgEl("text", { class: "chart-axis-text", x: m.l - 6, y: Y(t) + 4, "text-anchor": "end" }, t)
      );
    }

    // Línea 1:1 diagonal (predicción perfecta)
    s.append(svgEl("line", { class: "chart-diagonal-ref", x1: X(rng[0]), y1: Y(rng[0]), x2: X(rng[1]), y2: Y(rng[1]) }));

    puntos.forEach((p) => {
      const c = svgEl("circle", {
        class: "chart-data-point",
        cx: X(p.x),
        cy: Y(p.y),
        r: 5,
        fill: COLOR_ESTADO[p.estado] || "var(--brand-700)",
      });
      c.addEventListener("pointermove", (e) => mostrarTip([
        ["Parcela", p.id],
        ["Cosecha real", fmtNum(p.x, 2) + " t/ha"],
        ["Cosecha predicha", fmtNum(p.y, 2) + " t/ha"],
        ["Error", fmtSign(p.y - p.x, 2) + " t/ha"],
        ["Estado", p.estado],
      ], e));
      c.addEventListener("pointerleave", ocultarTip);
      s.append(c);
    });

    s.append(
      svgEl("text", { class: "chart-axis-title", x: (m.l + ancho - m.r) / 2, y: alto - 6, "text-anchor": "middle" }, "Cosecha Real Medida (t/ha)"),
      svgEl("text", { class: "chart-axis-title", transform: `translate(14, ${(m.t + alto - m.b) / 2}) rotate(-90)`, "text-anchor": "middle" }, "Cosecha Estimada por Modelo (t/ha)")
    );

    cont.append(
      el("div", { style: "display:flex;justify-content:space-between;margin-bottom:8px;font-size:12px;color:var(--text-secondary);" },
        el("div", { style: "display:flex;gap:12px;" },
          ...ESTADOS.map((e) => el("span", {}, el("i", { style: `display:inline-block;width:9px;height:9px;background:${COLOR_ESTADO[e]};margin-right:4px;border-radius:50%;` }), e))
        ),
        el("span", { class: "text-muted", style: "font-size:11px;" }, "Línea punteada: y = x")
      ),
      s
    );
  }

  function pintarDiagnosticoAtipicos() {
    const cont = $("#evaluacion-diagnostico-atipicos");
    if (!cont) return;
    empty(cont);

    cont.append(
      el("p", {},
        el("b", {}, "Interpretación Estadística Honesta: "),
        "La prueba reservada consta de 28 parcelas. En muestras pequeñas, el R² es altamente vulnerable a valores atípicos. En este caso, la parcela ",
        el("b", { class: "mono" }, "AGC_015"), " (cosecha real de 3.0 t/ha vs estimada de 4.38 t/ha) aporta por sí sola el ",
        el("b", {}, "48 % del error cuadrático total"), " de la prueba reservada. Al excluir únicamente esa parcela, el R² asciende a ",
        el("b", {}, "0.66"), " (consistente con el 0.69 de la validación cruzada)."
      ),
      el("p", { style: "margin-top:10px;" },
        "Dichas discrepancias puntuales corresponden a pérdidas localizadas de cultivo (heladas, plagas o manejo agronómico adverso) que los satélites multiespectrales de resolución moderada no pueden anticipar antes de la cosecha. Ningún modelo evaluado superó este límite sin memorizar los datos."
      )
    );
  }

  function pintarTablaMunicipiosEvaluacion() {
    const cont = $("#tabla-eval-municipios");
    if (!cont) return;
    empty(cont);

    const zonaDe = (p) => (MUNICIPIOS.includes(p.municipio) ? p.municipio : "Otros municipios");
    const grupos = {};
    for (const p of conocidas) (grupos[zonaDe(p)] = grupos[zonaDe(p)] || []).push(p);

    const filas = Object.entries(grupos).map(([z, ps]) => ({
      zona: z,
      n: ps.length,
      realMed: ps.reduce((a, p) => a + p.real, 0) / ps.length,
      rmse: Math.sqrt(ps.reduce((a, p) => a + p.error ** 2, 0) / ps.length),
      mae: ps.reduce((a, p) => a + Math.abs(p.error), 0) / ps.length,
      sesgo: ps.reduce((a, p) => a + p.error, 0) / ps.length,
    })).sort((a, b) => b.n - a.n);

    const tabla = el("table", { class: "data-table" },
      el("thead", {}, el("tr", {},
        el("th", {}, "Municipio"),
        el("th", { class: "num" }, "Parcelas"),
        el("th", { class: "num" }, "Real Medio"),
        el("th", { class: "num" }, "RMSE (t/ha)"),
        el("th", { class: "num" }, "MAE (t/ha)"),
        el("th", { class: "num" }, "Sesgo")
      )),
      el("tbody", {}, filas.map((f) => el("tr", {},
        el("td", { style: "font-weight:600;" }, f.zona),
        el("td", { class: "num" }, f.n),
        el("td", { class: "num" }, fmtNum(f.realMed, 2) + " t/ha"),
        el("td", { class: "num", style: "font-weight:600;color:var(--brand-700);" }, fmtNum(f.rmse, 3)),
        el("td", { class: "num" }, fmtNum(f.mae, 3)),
        el("td", { class: "num" }, fmtSign(f.sesgo, 3))
      )))
    );
    cont.append(tabla);
  }

  function pintarTablaComparativaFamilias() {
    const cont = $("#tabla-comparativa-familias");
    if (!cont) return;
    empty(cont);

    const filas = D.familias;
    const tabla = el("table", { class: "data-table" },
      el("thead", {}, el("tr", {},
        el("th", {}, "Familia Algorítmica"),
        el("th", { class: "num" }, "Modelos Evaluados"),
        el("th", { class: "num" }, "Mejor RMSE CV (t/ha)"),
        el("th", {}, "Veredicto Técnico")
      )),
      el("tbody", {}, filas.map((f) => el("tr", {
        style: f.familia === "GeoResidual" ? "background:var(--risk-high-prob-bg);font-weight:600;" : "",
      },
        el("td", {}, f.familia),
        el("td", { class: "num" }, f.candidatos),
        el("td", { class: "num", style: "font-family:var(--font-mono);color:var(--brand-700);" }, fmtNum(f.mejor_rmse, 4)),
        el("td", {}, f.familia === "GeoResidual" ? "GANADOR: Menor RMSE y mejor captura de autocorrelación espacial" : "Descartado: Mayor error o riesgo de sobreajuste")
      )))
    );
    cont.append(tabla);
  }

  /* ==========================================================================
     SECCIÓN H: RIESGO PRODUCTIVO Y CRÉDITO FIRA
     ========================================================================== */
  function iniciarRiesgoCredito() {
    const sld = $("#umbral");
    const inCosto = $("#calc-costo");
    const inPrecio = $("#calc-precio");

    function recalcularUmbral() {
      const costo = parseFloat(inCosto?.value);
      const precio = parseFloat(inPrecio?.value);
      if (Number.isFinite(costo) && Number.isFinite(precio) && precio > 0) {
        S.umbral = clamp(costo / precio, 1.5, 5.5);
        if (sld) sld.value = S.umbral;
        saveStore("agrocebada.umbral", S.umbral);
        pintarRiesgoCredito();
        pintarDetalleRapido();
      }
    }

    sld?.addEventListener("input", (e) => {
      S.umbral = parseFloat(e.target.value);
      saveStore("agrocebada.umbral", S.umbral);
      pintarRiesgoCredito();
      pintarDetalleRapido();
    });

    inCosto?.addEventListener("input", recalcularUmbral);
    inPrecio?.addEventListener("input", recalcularUmbral);

    $("#filtro-riesgo-nivel")?.addEventListener("change", (e) => {
      S.filtroRiesgoNivel = e.target.value;
      pintarTablaCreditoRiesgo();
    });
  }

  function pintarRiesgoCredito() {
    const uLabel = $("#umbral-v");
    if (uLabel) uLabel.textContent = fmtNum(S.umbral, 2) + " t/ha";

    const grupos = { alta: [], media: [], baja: [] };
    for (const p of ocultas) grupos[nivelDe(probDe(p))].push(p);

    const cont = $("#riesgo-resumen-cards");
    if (!cont) return;
    empty(cont);

    const descripciones = {
      alta: "Probabilidad ≥ 80 % · Excelente candidato para crédito FIRA",
      media: "Probabilidad 50–80 % · Requiere garantías o evaluación técnica",
      baja: "Probabilidad < 50 % · Riesgo elevado de no cubrir el costo",
    };

    const clases = { alta: "accent-blue", media: "accent-amber", baja: "accent-neutral" };

    for (const n of ["alta", "media", "baja"]) {
      const ha = grupos[n].reduce((a, p) => a + p.area, 0);
      const prod = grupos[n].reduce((a, p) => a + p.produccion, 0);
      cont.append(
        el("div", { class: `kpi-card ${clases[n]}` },
          el("div", { class: "kpi-label" }, pildoraRiesgo(n === "alta" ? 0.9 : n === "media" ? 0.6 : 0.3, false)),
          el("div", { class: "kpi-value-row" },
            el("span", { class: "kpi-value" }, grupos[n].length),
            el("span", { class: "kpi-unit" }, "parcelas (" + fmtPct(grupos[n].length / ocultas.length) + ")")
          ),
          el("div", { class: "kpi-subtext" }, `${fmtNum(ha, 0)} ha · ${fmtNum(prod, 0)} t estimadas`),
          el("div", { class: "text-muted", style: "font-size:11px;margin-top:4px;" }, descripciones[n])
        )
      );
    }

    // Gráfica de sensibilidad del portafolio
    pintarSensibilidadRiesgo();

    // Tabla de las 59 parcelas clasificadas
    pintarTablaCreditoRiesgo();
  }

  function pintarSensibilidadRiesgo() {
    const cont = $("#chart-sensibilidad-riesgo");
    if (!cont) return;
    empty(cont);

    const umbrales = [2.0, 2.5, 3.0, 3.2, 3.5, 3.8, 4.0, 4.5, 5.0];
    const datos = umbrales.map((u) => {
      const viables = ocultas.filter((p) => probDe(p, u) >= 0.8).length;
      return { u, pct: (viables / ocultas.length) * 100, n: viables };
    });

    const ancho = 640, alto = 220, m = { l: 48, r: 24, t: 20, b: 36 };
    const xMin = 1.5, xMax = 5.5;
    const datosLinea = [
      { x: 1.5, y: 100 },
      ...datos.map((d) => ({ x: d.u, y: d.pct })),
      { x: 5.5, y: 0 },
    ];
    const X = (u) => m.l + ((u - xMin) / (xMax - xMin)) * (ancho - m.l - m.r);
    const Y = (p) => alto - m.b - (p / 100) * (alto - m.t - m.b);

    const s = svgEl("svg", { class: "chart", viewBox: `0 0 ${ancho} ${alto}` });

    for (let p = 0; p <= 100; p += 25) {
      s.append(
        svgEl("line", { class: "chart-grid-line", x1: m.l, x2: ancho - m.r, y1: Y(p), y2: Y(p) }),
        svgEl("text", { class: "chart-axis-text", x: m.l - 6, y: Y(p) + 4, "text-anchor": "end" }, p + "%")
      );
    }

    const pts = datosLinea.map((d) => `${X(d.x)},${Y(d.y)}`).join(" ");
    s.append(
      svgEl("polyline", { fill: "none", stroke: "var(--brand-700)", "stroke-width": 3, points: pts }),
      // Marcador umbral actual
      svgEl("line", { x1: X(S.umbral), x2: X(S.umbral), y1: m.t, y2: alto - m.b, stroke: "var(--c-hidalgo)", "stroke-dasharray": "4 3", "stroke-width": 1.8 })
    );

    datos.forEach((d) => {
      const c = svgEl("circle", { cx: X(d.u), cy: Y(d.pct), r: 5, fill: "var(--brand-700)", stroke: "var(--surface-card)", "stroke-width": 1.5 });
      c.addEventListener("pointermove", (e) => mostrarTip([
        ["Punto de equilibrio", fmtNum(d.u, 1) + " t/ha"],
        ["Parcelas viables (≥80%)", `${d.n} de 59 (${fmtNum(d.pct, 1)}%)`],
      ], e));
      c.addEventListener("pointerleave", ocultarTip);
      s.append(c);
    });

    umbrales.forEach((u) => {
      s.append(svgEl("text", { class: "chart-axis-text", x: X(u), y: alto - m.b + 16, "text-anchor": "middle" }, u));
    });

    s.append(
      svgEl("text", { class: "chart-axis-title", x: (m.l + ancho - m.r) / 2, y: alto - 6, "text-anchor": "middle" }, "Punto de equilibrio exigido (t/ha)"),
      svgEl("text", { class: "chart-axis-title", transform: `translate(14, ${(m.t + alto - m.b) / 2}) rotate(-90)`, "text-anchor": "middle" }, "% Portafolio Viable")
    );

    cont.append(s);
  }

  function pintarTablaCreditoRiesgo() {
    const cont = $("#tabla-credito-riesgo");
    if (!cont) return;
    empty(cont);

    let lista = ocultas.slice();
    if (S.filtroRiesgoNivel !== "todos") {
      lista = lista.filter((p) => nivelDe(probDe(p)) === S.filtroRiesgoNivel);
    }

    lista.sort((a, b) => probDe(b) - probDe(a));

    const tabla = el("table", { class: "data-table" },
      el("thead", {}, el("tr", {},
        el("th", {}, "Parcela"),
        el("th", {}, "Municipio"),
        el("th", { class: "num" }, "Superficie (ha)"),
        el("th", { class: "num" }, "Cosecha Estimada"),
        el("th", { class: "num" }, "Límite Inferior 80%"),
        el("th", { class: "num" }, "Probabilidad"),
        el("th", {}, "Calificación de Riesgo"),
        el("th", {}, "Acción")
      )),
      el("tbody", {}, lista.map((p) => {
        const pr = probDe(p);
        return el("tr", {},
          el("td", { class: "mono", style: "font-weight:600;" }, p.id),
          el("td", {}, p.municipio),
          el("td", { class: "num" }, fmtNum(p.area, 2)),
          el("td", { class: "num", style: "font-weight:600;color:var(--brand-700);" }, fmtNum(p.estimado, 2) + " t/ha"),
          el("td", { class: "num" }, fmtNum(p.lo80, 2) + " t/ha"),
          el("td", { class: "num", style: "font-weight:700;" }, probTexto(pr, residuosDe(p).length)),
          el("td", {}, pildoraRiesgo(pr, false)),
          el("td", {}, el("button", {
            class: "btn btn-outline btn-sm",
            style: "padding:2px 8px;font-size:11px;",
            onclick: () => {
              seleccionar(p.id, { enfocar: true });
              activarVista("mapa");
            },
          }, "Ver Mapa"))
        );
      }))
    );
    cont.append(tabla);
  }

  /* ==========================================================================
     SECCIÓN I: METODOLOGÍA, PREGUNTAS GUÍA Y GLOSARIO
     ========================================================================== */
  function pintarMetodologia() {
    // Respuestas a las 3 preguntas guía del Reto
    pintarPreguntasGuia();
    pintarFlujoMetodologico();
    pintarFuentesMetodologicas();
    pintarGlosario();
  }

  function pintarPreguntasGuia() {
    const cont = $("#metodologia-preguntas-guia");
    if (!cont) return;
    empty(cont);

    const R = D.guia.resumen;

    cont.append(
      el("div", { class: "grid-3", style: "margin-bottom:0;" },
        // Pregunta 1
        el("div", { class: "card", style: "background:var(--surface-well);" },
          el("h4", { style: "font-size:14px;color:var(--brand-800);margin-bottom:8px;" }, "1. ¿Qué índice es más importante?"),
          el("p", { class: "text-secondary", style: "font-size:12.5px;line-height:1.55;" },
            el("b", {}, "El verdor previo a la siembra (Sentinel-2 EVI y NDVI en abril–mayo 2022–2024). "),
            `Por sí solo, el de mayor poder es ${R.pregunta1_indice_mayor_poder_solo.VARIABLE} (R² ${fmtNum(R.pregunta1_indice_mayor_poder_solo.R2_SOLO, 2)}). Sin embargo, una vez conocida la ubicación, solo EVI y NDVI históricos aportan una mejora estadística superior a dos errores estándar (+0.014 t/ha). Los índices de la temporada 2025 no aportan ganancia significativa sobre la ubicación.`
          )
        ),

        // Pregunta 2
        el("div", { class: "card", style: "background:var(--surface-well);" },
          el("h4", { style: "font-size:14px;color:var(--brand-800);margin-bottom:8px;" }, "2. ¿Qué variable climática tiene mayor peso?"),
          el("p", { class: "text-secondary", style: "font-size:12.5px;line-height:1.55;" },
            el("b", {}, "La amplitud térmica mensual (CHIRTS). "),
            `Usada de manera aislada explica un R² de ${fmtNum(R.pregunta2_variable_climatica_mayor_peso.R2_SOLO, 2)}. No obstante, tras conocer la ubicación geográfica, su aporte sobre el modelo es nulo (${fmtSign(R.pregunta2_variable_climatica_mayor_peso.APORTE_SOBRE_UBICACION_T_HA, 4)} t/ha). Esto obedece a que el raster climático (5 km) distingue regiones macro, pero no diferencias microclimáticas entre parcelas contiguas.`
          )
        ),

        // Pregunta 3
        el("div", { class: "card", style: "background:var(--surface-well);" },
          el("h4", { style: "font-size:14px;color:var(--brand-800);margin-bottom:8px;" }, "3. ¿Qué combinación funciona mejor?"),
          el("p", { class: "text-secondary", style: "font-size:12.5px;line-height:1.55;" },
            el("b", {}, `${R.pregunta3_mejor_combinacion.COMBINACION}. `),
            `Obtiene el menor RMSE (${fmtNum(R.pregunta3_mejor_combinacion.RMSE_MEJOR_T_HA, 3)} t/ha) entre ${D.guia.n_combinaciones} combinaciones posibles. Estadísticamente empata con otras 13 combinaciones que incorporan historial satelital. En ninguna de las combinaciones óptimas aparecen los índices satelitales de 2025, demostrando parsimonia y menor riesgo de sobreajuste.`
          )
        )
      )
    );
  }

  function pintarFlujoMetodologico() {
    const cont = $("#metodologia-flujo");
    if (!cont) return;
    empty(cont);

    cont.append(
      el("ol", { style: "padding-left:18px;margin:0;display:flex;flex-direction:column;gap:8px;" },
        el("li", {}, el("b", {}, "Extracción Zonal Robusta: "), "Procesamiento de 42,551 imágenes satelitales (Sentinel-2, Landsat, Planet) con filtrado estricto de nubes (≤20%). Integración de CHIRPS (precipitación) y CEM 4.0."),
        el("li", {}, el("b", {}, "Ingeniería Fenológica: "), "Construcción de curvas estacionales completas, picos, fechas críticas de inicio/fin y medias históricas 2022–2024 de verdor."),
        el("li", {}, el("b", {}, "Estrategia Anti-Fuga: "), "Partición fija de 110 desarrollo y 28 prueba independiente. Validación cruzada 5×3 con imputación y selección repetidas dentro de cada fold."),
        el("li", {}, el("b", {}, "Algoritmo Geo-Residual: "), "Estimación de línea base zonal por ponderación 1/d sobre 30 vecinos más corrección Ridge L2 regularizada."),
        el("li", {}, el("b", {}, "Inferencia Conformal: "), "Cálculo de intervalos de predicción empíricos (80% y 90%) basados en la distribución real de residuos por municipio.")
      )
    );
  }

  function pintarFuentesMetodologicas() {
    const cont = $("#metodologia-fuentes");
    if (!cont) return;
    empty(cont);

    cont.append(
      el("ul", { style: "padding-left:18px;margin:0;display:flex;flex-direction:column;gap:8px;" },
        el("li", {}, el("b", {}, "FIRA / Reto AgroCebada 2026: "), "Polígonos georreferenciados y rendimientos en campo del ciclo abril–octubre 2025."),
        el("li", {}, el("b", {}, "Copernicus Sentinel-2 & USGS Landsat: "), "Reflectancia en superficie corregida atmosféricamente."),
        el("li", {}, el("b", {}, "PlanetScope NICFI: "), "Imágenes satelitales de alta resolución espacial (Dataset PRO)."),
        el("li", {}, el("b", {}, "CHIRPS & CHIRTS-ERA5 (UC Santa Barbara): "), "Precipitación satelital calibrada con estaciones y temperaturas."),
        el("li", {}, el("b", {}, "INEGI (CEM 4.0): "), "Continuo de Elevaciones Mexicano a 120 m."),
        el("li", {}, el("b", {}, "Transparencia de IA: "), "Desarrollo asistido con modelos LLM documentado con rigor en registro_asistencia_IA.md.")
      )
    );
  }

  function pintarGlosario() {
    const cont = $("#metodologia-glosario");
    if (!cont) return;
    empty(cont);

    const terminos = [
      { t: "Rendimiento (t/ha)", d: "Cantidad de toneladas métricas cosechadas por hectárea de superficie cultivada." },
      { t: "EVI (Enhanced Vegetation Index)", d: "Índice de vegetación mejorado que optimiza la señal en presencia de alta biomasa y desacopla la influencia del fondo del suelo." },
      { t: "NDVI (Normalized Difference Vegetation Index)", d: "Índice normalizado de verdor basado en la absorción en el rojo y reflectancia en el infrarrojo cercano." },
      { t: "RMSE (Root Mean Squared Error)", d: "Raíz del error cuadrático medio; penaliza con mayor severidad las desviaciones grandes entre el pronóstico y el dato real." },
      { t: "MAE (Mean Absolute Error)", d: "Error absoluto medio; magnitud promedio del error en t/ha sin penalización cuadrática." },
      { t: "R² (Coeficiente de Determinación)", d: "Proporción de la varianza del rendimiento explicada por el modelo respecto a una línea base trivial (la media)." },
      { t: "Validación Cruzada (CV)", d: "Técnica de remuestreo donde los datos se dividen en k pliegues para entrenar y evaluar iterativamente sin memorización." },
      { t: "Intervalo Conformal", d: "Técnica estadística no paramétrica que garantiza coberturas empíricas del 80% o 90% a partir de los residuos observados." },
      { t: "Punto de Equilibrio", d: "Rendimiento mínimo en t/ha para que los ingresos de la cosecha igualen exactamente los costos operativos del agricultor." },
    ];

    const grid = el("div", { class: "grid-3", style: "margin-bottom:0;" },
      terminos.map((item) => el("div", { style: "background:var(--surface-subtle);border:1px solid var(--border-subtle);border-radius:var(--radius-md);padding:12px;" },
        el("b", { style: "color:var(--brand-800);font-size:13px;" }, item.t),
        el("p", { class: "text-secondary", style: "font-size:12px;margin-top:4px;" }, item.d)
      ))
    );
    cont.append(grid);
  }

  /* ==========================================================================
     CAPAS DEL TERRITORIO Y MAPLIBRE CARTOGRÁFICO
     ========================================================================== */
  const MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"];
  const TERR = { capas: new Map(), lim: [0, 1] };

  function iniciarTerritorio() {
    if (!T) return;
    for (const [v, anio, mes, i, valor, cob, est] of T.registros) {
      const clave = `${T.variables[v].id}|${anio === null ? "" : anio}|${mes === null ? "" : mes}`;
      let capa = TERR.capas.get(clave);
      if (!capa) TERR.capas.set(clave, (capa = new Map()));
      capa.set(T.ids[i], { valor, cobertura: cob, parcial: est === 1 });
    }
    S.variable = T.variables.find((v) => !v.estatica).id;
    S.anio = T.anios[T.anios.length - 1];
    S.mes = mesesDe(S.variable, S.anio).includes(7) ? 7 : mesesDe(S.variable, S.anio)[0];
  }

  const variableDe = (id) => T && T.variables.find((v) => v.id === id);
  function mesesDe(id, anio) {
    const ms = new Set();
    for (const clave of TERR.capas.keys()) {
      const [v, a, m] = clave.split("|");
      if (v === id && Number(a) === anio && m !== "") ms.add(Number(m));
    }
    return [...ms].sort((a, b) => a - b);
  }

  const claveCapa = (id) => (variableDe(id)?.estatica ? `${id}||` : `${id}|${S.anio}|${S.mes}`);
  const capaTerritorio = (id = S.variable) => TERR.capas.get(claveCapa(id)) || new Map();
  function valorTerritorio(pid, id = S.variable) {
    const r = capaTerritorio(id).get(pid);
    return r && Number.isFinite(r.valor) ? r.valor : null;
  }
  const fechaCapa = (id = S.variable) => (variableDe(id)?.estatica ? "capa estática" : `${MESES[S.mes - 1]} de ${S.anio}`);
  const nombreCapaActual = () => variableDe(S.variable)?.nombre || "Territorio";
  function textoValorTerritorio(pid, id = S.variable) {
    const v = valorTerritorio(pid, id);
    const u = variableDe(id)?.unidad || "";
    return v === null ? "Sin dato" : `${fmtNum(v, u === "m" ? 0 : 1)} ${u}`;
  }
  function actualizarLimitesTerritorio() {
    const vals = [...capaTerritorio().values()].map((r) => r.valor).filter(Number.isFinite);
    TERR.lim = vals.length ? [Math.min(...vals), Math.max(...vals)] : [0, 1];
  }
  const colorTerritorio = (v) => interp(SEQ, TERR.lim[1] === TERR.lim[0] ? 0.5 : (v - TERR.lim[0]) / (TERR.lim[1] - TERR.lim[0]));

  function leyendaTerritorio(cont) {
    const v = variableDe(S.variable);
    if (!v) return;
    const conDato = [...capaTerritorio().values()].filter((r) => Number.isFinite(r.valor)).length;
    cont.append(
      el("span", { class: "text-muted", style: "font-size:11px;" }, fmtNum(TERR.lim[0], 1)),
      el("div", { class: "legend-ramp-gradient", style: `background:${degradadoCss(SEQ)}` }),
      el("span", { class: "text-muted", style: "font-size:11px;" }, `${fmtNum(TERR.lim[1], 1)} ${v.unidad}`),
      el("span", { class: "text-muted", style: "font-size:11px;" }, `${v.nombre} · ${fechaCapa()} (${conDato} parcelas con dato)`)
    );
  }

  function pintarControlesTerritorio() {
    const capa = $("#f-capa");
    if (!capa) return;
    const valorActual = S.color === "territorio" ? "ter:" + S.variable : "est:" + S.color;
    empty(capa);
    capa.append(
      el("optgroup", { label: "Modelo de Rendimiento" },
        el("option", { value: "est:estimado" }, "Rendimiento Estimado (t/ha)"),
        el("option", { value: "est:real" }, "Rendimiento Real Medido (t/ha)"),
        el("option", { value: "est:error" }, "Error de Estimación (t/ha)")
      ),
      el("optgroup", { label: "Territorio y Medio Ambiente" },
        T.variables.map((v) => el("option", { value: "ter:" + v.id }, v.nombre))
      )
    );
    capa.value = valorActual;
    capa.onchange = () => {
      const [tipo, id] = capa.value.split(":");
      if (tipo === "ter") {
        S.color = "territorio";
        S.variable = id;
        actualizarSelectoresFecha();
        actualizarLimitesTerritorio();
      } else {
        S.color = id;
      }
      if (S.color !== "estimado" && S.conjunto === "ocultas") {
        S.conjunto = "todas";
        pintarFiltrosMapa();
      }
      pintarMapa();
      pintarLeyenda();
      pintarDetalleRapido();
    };

    actualizarSelectoresFecha();

    const anio = $("#f-anio"), mes = $("#f-mes");
    if (anio) {
      anio.onchange = () => {
        S.anio = Number(anio.value);
        const ms = mesesDe(S.variable, S.anio);
        if (!ms.includes(S.mes)) S.mes = ms[0];
        actualizarSelectoresFecha();
        actualizarLimitesTerritorio();
        if (S.color === "territorio") { pintarMapa(); pintarLeyenda(); }
        pintarDetalleRapido();
      };
    }
    if (mes) {
      mes.onchange = () => {
        S.mes = Number(mes.value);
        actualizarLimitesTerritorio();
        if (S.color === "territorio") { pintarMapa(); pintarLeyenda(); }
        pintarDetalleRapido();
      };
    }

    const fondo = $("#f-fondo");
    if (fondo) {
      fondo.value = FONDO.tipo;
      fondo.onchange = () => cambiarFondo(fondo.value);
    }

    if (CFG.buscador) iniciarBuscador();
    const csv = $("#descargar-capa");
    if (csv) csv.onclick = descargarCsvCapa;
  }

  function actualizarSelectoresFecha() {
    const estatica = S.color === "territorio" && variableDe(S.variable)?.estatica;
    const gAnio = $("#g-anio"), gMes = $("#g-mes");
    if (gAnio) gAnio.hidden = estatica || S.color !== "territorio";
    if (gMes) gMes.hidden = estatica || S.color !== "territorio";
    const anio = $("#f-anio"), mes = $("#f-mes");
    if (!anio || !mes || !T) return;
    empty(anio); empty(mes);
    for (const a of T.anios) anio.append(el("option", { value: a }, String(a)));
    anio.value = S.anio;
    const idMeses = S.color === "territorio" ? S.variable : T.variables.find((v) => !v.estatica).id;
    for (const m of mesesDe(idMeses, S.anio)) mes.append(el("option", { value: m }, MESES[m - 1]));
    mes.value = S.mes;
  }

  function iniciarBuscador() {
    const lista = $("#ids-mapa"), caja = $("#f-buscar");
    if (!lista || !caja || lista.childNodes.length) return;
    for (const p of P.slice().sort((a, b) => a.id.localeCompare(b.id))) {
      lista.append(el("option", { value: p.id, label: `${p.municipio} (${p.estado})` }));
    }
    const buscar = () => {
      const q = caja.value.trim().toLowerCase();
      if (!q) return;
      const exacta = P.find((p) => p.id.toLowerCase() === q);
      const parecidas = P.filter((p) => p.id.toLowerCase().includes(q) || p.municipio.toLowerCase().includes(q));
      const hallada = exacta || (parecidas.length === 1 ? parecidas[0] : null);
      if (hallada) {
        seleccionar(hallada.id, { enfocar: true });
      } else if (parecidas.length > 1) {
        ajustarVista(parecidas);
      }
    };
    caja.addEventListener("change", buscar);
    caja.addEventListener("keydown", (e) => { if (e.key === "Enter") buscar(); });
  }

  function descargarCsvCapa() {
    const celda = (v) => (v === null || v === undefined ? "" : /[",\r\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : v);
    let nombre, filas;
    if (S.color === "territorio") {
      const v = variableDe(S.variable), capa = capaTerritorio();
      nombre = `agrocebada_${v.id}${v.estatica ? "" : `_${S.anio}_${String(S.mes).padStart(2, "0")}`}`;
      filas = [
        ["ID_POLIGONO", "ESTADO", "MUNICIPIO", "LONGITUD", "LATITUD", "AREA_HA", "VARIABLE", "ANIO", "MES", "VALOR", "UNIDAD"],
        ...P.map((p) => {
          const r = capa.get(p.id) || {};
          return [p.id, p.estado, p.municipio, p.lon, p.lat, p.area, v.id, v.estatica ? "" : S.anio, v.estatica ? "" : S.mes, r.valor, v.unidad];
        })
      ];
    } else {
      nombre = `agrocebada_rendimiento_${S.color}`;
      filas = [
        ["ID_POLIGONO", "ESTADO", "MUNICIPIO", "LONGITUD", "LATITUD", "AREA_HA", "RENDIMIENTO_ESTIMADO_T_HA", "LIMITE_INF_80", "LIMITE_SUP_80", "RENDIMIENTO_REAL_T_HA", "ERROR_T_HA", "SITUACION"],
        ...P.map((p) => [p.id, p.estado, p.municipio, p.lon, p.lat, p.area, p.estimado, p.lo80, p.hi80, p.real, p.error === null ? "" : p.error.toFixed(4), p.esOculta ? "POR_ESTIMAR" : "CONOCIDA"])
      ];
    }
    const texto = "\uFEFF" + filas.map((f) => f.map(celda).join(",")).join("\r\n") + "\r\n";
    const blob = new Blob([texto], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = el("a", { href: url, download: nombre + ".csv" });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  /* ==========================================================================
     FONDO CARTOGRÁFICO MAPLIBRE (SATÉLITE / CALLES / OFFLINE)
     ========================================================================== */
  const FONDO = { mapa: null, listo: false, cargando: false, tipo: CFG.fondo ? CFG.fondoInicial || "satelite" : "ninguno" };
  const CREDITOS = {
    satelite: [["Esri World Imagery", "https://www.arcgis.com/home/item.html?id=10df2279f9684e4a9f6a7f08febac2a9"], ["Maxar, Earthstar"]],
    calles: [["OpenFreeMap", "https://openfreemap.org/"], ["© OpenStreetMap", "https://www.openstreetmap.org/copyright"]],
  };

  function estiloFondo(tipo) {
    if (tipo === "calles") return "https://tiles.openfreemap.org/styles/liberty";
    const esri = "https://server.arcgisonline.com/ArcGIS/rest/services";
    return {
      version: 8,
      sources: {
        imagen: { type: "raster", tileSize: 256, maxzoom: 19, tiles: [`${esri}/World_Imagery/MapServer/tile/{z}/{y}/{x}`] },
        nombres: { type: "raster", tileSize: 256, maxzoom: 19, tiles: [`${esri}/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}`] },
      },
      layers: [{ id: "imagen", type: "raster", source: "imagen" }, { id: "nombres", type: "raster", source: "nombres" }],
    };
  }

  function iniciarFondo() {
    if (!CFG.fondo || !lienzo) return;
    cambiarFondo(FONDO.tipo);
  }

  function estadoFondo(texto) { const e = $("#fondo-estado"); if (e) e.textContent = texto; }

  function cargarMapLibre(alListo, alFallar) {
    if (window.maplibregl) { alListo(); return; }
    if (FONDO.cargando) return;
    FONDO.cargando = true;
    document.head.append(el("link", { rel: "stylesheet", href: "https://unpkg.com/maplibre-gl@5.6.1/dist/maplibre-gl.css" }));
    document.head.append(el("script", {
      src: "https://unpkg.com/maplibre-gl@5.6.1/dist/maplibre-gl.js", async: true,
      onload: () => { FONDO.cargando = false; alListo(); },
      onerror: () => { FONDO.cargando = false; alFallar(); },
    }));
  }

  function cambiarFondo(tipo) {
    FONDO.tipo = tipo;
    const conFondo = tipo !== "ninguno";
    if (lienzo) {
      lienzo.classList.toggle("con-fondo", conFondo);
      lienzo.classList.toggle("fondo-sat", tipo === "satelite");
    }
    const cont = $("#fondo-mapa");
    if (cont) cont.hidden = !conFondo;
    const cred = $("#fondo-creditos");
    if (cred) {
      empty(cred);
      (CREDITOS[tipo] || []).forEach(([texto, href], i) => {
        if (i) cred.append(" · ");
        cred.append(href ? el("a", { href, target: "_blank", rel: "noopener noreferrer" }, texto) : texto);
      });
      cred.hidden = !conFondo;
    }
    if (!conFondo) { estadoFondo("Parcelas · Modo sin fondo cartográfico"); return; }
    estadoFondo("Cargando capa cartográfica…");

    cargarMapLibre(() => {
      try {
        if (!FONDO.mapa) {
          FONDO.mapa = new window.maplibregl.Map({
            container: "fondo-mapa", style: estiloFondo(FONDO.tipo), interactive: false, attributionControl: false,
            renderWorldCopies: false, maxZoom: 22, center: [D.proyeccion.lon0, D.proyeccion.lat0], zoom: 9,
          });
          FONDO.mapa.on("load", () => {
            FONDO.listo = true;
            estadoFondo(FONDO.tipo === "satelite" ? "Satélite con etiquetas" : "Calles y poblados");
            sincronizarFondo();
          });
          FONDO.mapa.on("error", () => estadoFondo("Fondo parcial · Sin conexión completa"));
        } else {
          FONDO.mapa.setStyle(estiloFondo(FONDO.tipo));
          estadoFondo(FONDO.tipo === "satelite" ? "Satélite con etiquetas" : "Calles y poblados");
        }
        sincronizarFondo();
      } catch (e) {
        estadoFondo("Fondo no disponible · Las parcelas continúan visibles");
      }
    }, () => estadoFondo("Modo sin internet · Fondo vectorial continuo"));
  }

  function sincronizarFondo() {
    if (!FONDO.mapa || FONDO.tipo === "ninguno" || !mapa.kmPx) return;
    const v = mapa.vista, py = D.proyeccion, rad = Math.PI / 180;
    const lat = py.lat0 + ((-v.cy) / py.radio_km) / rad;
    const lon = py.lon0 + (v.cx / (py.radio_km * Math.cos(py.lat0 * rad))) / rad;
    const zoomMapa = Math.log2((78271.517 * Math.cos(lat * rad)) / (mapa.kmPx * 1000));
    FONDO.mapa.resize();
    FONDO.mapa.jumpTo({ center: [lon, lat], zoom: clamp(zoomMapa, 0, 22), bearing: 0, pitch: 0 });
  }

  /* ==========================================================================
     NAVEGACIÓN POR SIDEBAR Y ENRUTAMIENTO POR HASH
     ========================================================================== */
  const VISTAS_CONFIG = [
    ["resumen", "Resumen General"],
    ["mapa", "Mapa de Parcelas"],
    ["prediccion", "Predicción por Parcela"],
    ["simulador", "Simulador de Escenarios"],
    ["ambiental", "Análisis Ambiental"],
    ["explicabilidad", "Explicabilidad del Modelo"],
    ["evaluacion", "Evaluación y Métricas"],
    ["credito", "Riesgo y Crédito FIRA"],
    ["metodologia", "Metodología y Fuentes"],
  ];

  function activarVista(id, sinHash = false) {
    if (!VISTAS_CONFIG.some(([v]) => v === id)) id = "resumen";
    S.vista = id;

    // Actualizar sidebar buttons
    $$(".nav-item").forEach((btn) => {
      const match = btn.dataset.vista === id;
      btn.setAttribute("aria-selected", String(match));
    });

    // Actualizar paneles
    VISTAS_CONFIG.forEach(([v]) => {
      const panel = $("#vista-" + v);
      if (panel) panel.hidden = v !== id;
    });

    // Breadcrumb actual
    const bCurrent = $("#breadcrumb-current");
    const vistaItem = VISTAS_CONFIG.find(([v]) => v === id);
    if (bCurrent && vistaItem) bCurrent.textContent = vistaItem[1];

    ocultarTip();
    if (!sinHash) {
      try { history.replaceState(null, "", "#" + id); } catch (e) { }
    }

    // Renderizado reactivo de la vista seleccionada
    if (id === "resumen") pintarResumenGeneral();
    if (id === "mapa") { aplicarVista(); pintarMapa(); pintarDetalleRapido(); }
    if (id === "prediccion") pintarFichaPrediccion();
    if (id === "simulador") simCalcular();
    if (id === "ambiental") pintarAnalisisAmbiental();
    if (id === "explicabilidad") pintarExplicabilidad();
    if (id === "evaluacion") pintarEvaluacion();
    if (id === "credito") pintarRiesgoCredito();
    if (id === "metodologia") pintarMetodologia();

    // En móviles, cerrar sidebar tras clic
    const sidebar = $("#sidebar");
    if (sidebar && window.innerWidth <= 960) sidebar.classList.remove("open");
  }

  /* ==========================================================================
     INICIALIZACIÓN DE LA APLICACIÓN
     ========================================================================== */
  function iniciar() {
    aplicarTema();

    // Eventos de Navegación Sidebar
    $$(".nav-item").forEach((btn) => {
      btn.addEventListener("click", () => activarVista(btn.dataset.vista));
    });

    // Toggle Sidebar en móvil/tablet
    $("#btn-toggle-sidebar")?.addEventListener("click", () => {
      const sb = $("#sidebar");
      if (sb) sb.classList.toggle("open");
    });

    // Botón de mapa desde resumen
    $("#btn-ir-mapa-desde-resumen")?.addEventListener("click", () => activarVista("mapa"));

    // Descarga de predicciones 59 CSV
    $("#btn-descargar-59-csv")?.addEventListener("click", () => {
      const filas = [
        ["ID_POLIGONO", "ESTADO", "MUNICIPIO", "AREA_HA", "RENDIMIENTO_ESTIMADO_T_HA", "LIMITE_INFERIOR_80", "LIMITE_SUPERIOR_80", "LIMITE_INFERIOR_90", "LIMITE_SUPERIOR_90", "ZONA_INTERVALO"],
        ...ocultas.map((p) => [p.id, p.estado, p.municipio, p.area, p.estimado, p.lo80, p.hi80, p.lo90, p.hi90, p.zona])
      ];
      const texto = "\uFEFF" + filas.map((f) => f.join(",")).join("\r\n") + "\r\n";
      const blob = new Blob([texto], { type: "text/csv;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = el("a", { href: url, download: "predicciones_59_agrocebada_fira2026.csv" });
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });

    // Inicializaciones modulares
    construirMapa();
    iniciarTerritorio();
    iniciarFondo();
    if (T) actualizarLimitesTerritorio();
    pintarFiltrosMapa();
    pintarChips();
    iniciarFichaPrediccion();
    iniciarSimulador();
    iniciarRiesgoCredito();
    pintarLeyenda();

    // Cargar vista inicial desde el hash de URL o por defecto 'resumen'
    const hash = (location.hash || "").replace("#", "");
    const vistaInicial = VISTAS_CONFIG.some(([v]) => v === hash) ? hash : "resumen";
    activarVista(vistaInicial, true);

    ajustarVista(ocultas);
    pintarMapa();
    seleccionar(ocultas[0].id);

    window.addEventListener("hashchange", () => {
      const h = (location.hash || "").replace("#", "");
      if (VISTAS_CONFIG.some(([v]) => v === h)) activarVista(h, true);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", iniciar);
  } else {
    iniciar();
  }
})();
