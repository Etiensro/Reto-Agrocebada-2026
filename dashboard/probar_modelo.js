/* Comprueba en Node que modelo.js reproduce las estimaciones guardadas por Python.
 * Uso: node dashboard/probar_modelo.js   (imprime un JSON con el resultado)
 */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const Modelo = require("./modelo.js");

const sandbox = { window: {} };
vm.runInNewContext(fs.readFileSync(path.join(__dirname, "datos.js"), "utf8"), sandbox);
const D = sandbox.window.DATOS;

let maxOcultas = 0, maxConocidasSinSi = 0, maxCeroDist = 0;
const ocultas = D.parcelas.filter((p) => p.origen === "ocultas");
for (const p of ocultas) {
  const r = Modelo.predecir(D.modelo, { x: p.x, y: p.y, area: p.area, ...p.variables });
  maxOcultas = Math.max(maxOcultas, Math.abs(r.estimado - p.estimado));
}
// Una parcela conocida, sin contarse a sí misma, debe dar un valor coherente (no su etiqueta).
const conocidas = D.parcelas.filter((p) => p.origen === "conocidas");
let sumaAbs = 0;
for (const p of conocidas) {
  const r = Modelo.predecir(D.modelo, { x: p.x, y: p.y, area: p.area, ...p.variables }, { excluir: p.id });
  sumaAbs += Math.abs(r.estimado - p.real);
  maxConocidasSinSi = Math.max(maxConocidasSinSi, Math.abs(r.estimado - p.estimado));
  const conSi = Modelo.predecir(D.modelo, { x: p.x, y: p.y, area: p.area, ...p.variables });
  // Con distancia 0 solo cuenta la propia parcela: el vecino devuelve su etiqueta.
  maxCeroDist = Math.max(maxCeroDist, Math.abs(conSi.efecto_zona - p.real));
}
const proy = D.proyeccion;
const q = Modelo.proyectar(ocultas[0].lon, ocultas[0].lat, proy);
const resultado = {
  n_ocultas: ocultas.length,
  max_dif_ocultas: maxOcultas,
  n_conocidas: conocidas.length,
  mae_conocidas_sin_si_misma: sumaAbs / conocidas.length,
  max_dif_conocida_vs_cv: maxConocidasSinSi,
  max_dif_cero_distancia: maxCeroDist,
  proyeccion_dif: Math.max(Math.abs(q.x - ocultas[0].x), Math.abs(q.y - ocultas[0].y)),
  prob_ejemplo: Modelo.probabilidad(3.5, D.residuos_por_zona["Todas"], 3.0),
};
console.log(JSON.stringify(resultado));
