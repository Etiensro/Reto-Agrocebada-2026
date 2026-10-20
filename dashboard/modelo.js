/* Modelo geográfico-residual de rendimiento, portado de predicción.py.
 *
 * rendimiento = promedio ponderado por 1/distancia de los 30 vecinos más cercanos
 *               (coordenadas en km, solo parcelas con rendimiento conocido)
 *             + corrección Ridge sobre superficie y verdor previo a la siembra.
 *
 * Funciona igual en el navegador (window.Modelo) y en Node (require), de modo que
 * tests_dashboard.py comprueba que reproduce las predicciones de Python.
 */
(function (raiz, fabrica) {
  if (typeof module === "object" && module.exports) module.exports = fabrica();
  else raiz.Modelo = fabrica();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  /** Proyección equirectangular local, igual a x_km / y_km de predicción.py. */
  function proyectar(lon, lat, proy) {
    const rad = Math.PI / 180;
    return {
      x: (lon - proy.lon0) * rad * Math.cos(proy.lat0 * rad) * proy.radio_km,
      y: (lat - proy.lat0) * rad * proy.radio_km,
    };
  }

  function mediana(valores) {
    const v = valores.filter(Number.isFinite).sort((a, b) => a - b);
    if (!v.length) return NaN;
    const m = v.length >> 1;
    return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
  }

  /** Vecinos más cercanos con pesos 1/d (si hay distancia 0, solo esos cuentan). */
  function vecinos(modelo, x, y, excluir) {
    const candidatos = [];
    for (const t of modelo.entrenamiento) {
      if (t.id === excluir) continue;
      candidatos.push({ id: t.id, real: t.real, d: Math.hypot(t.x - x, t.y - y) });
    }
    candidatos.sort((a, b) => a.d - b.d);
    const k = Math.min(modelo.n_vecinos, candidatos.length);
    const elegidos = candidatos.slice(0, k);
    const hayCero = elegidos.some((v) => v.d === 0);
    let suma = 0;
    for (const v of elegidos) {
      v.peso = hayCero ? (v.d === 0 ? 1 : 0) : 1 / v.d;
      suma += v.peso;
    }
    let prediccion = 0;
    for (const v of elegidos) {
      v.peso /= suma;
      prediccion += v.peso * v.real;
    }
    return { prediccion, vecinos: elegidos };
  }

  /**
   * entrada: {x, y, area, evi_2022, evi_2023, ndvi_media}; los valores ausentes
   * se imputan con la mediana del entrenamiento, igual que en Python.
   * opciones.excluir: id de una parcela de entrenamiento que no debe contar como vecina.
   */
  function predecir(modelo, entrada, opciones) {
    const excluir = opciones && opciones.excluir;
    const zona = vecinos(modelo, entrada.x, entrada.y, excluir);
    const valores = {
      log_area: Number.isFinite(entrada.area) ? Math.log1p(entrada.area) : NaN,
      evi_2022: entrada.evi_2022,
      evi_2023: entrada.evi_2023,
      ndvi_media: entrada.ndvi_media,
    };
    let correccion = modelo.intercepto;
    const aportes = modelo.variables.map((nombre, i) => {
      let v = valores[nombre];
      const imputado = !Number.isFinite(v);
      if (imputado) v = modelo.medianas[i];
      const aporte = (modelo.coeficientes[i] * (v - modelo.media[i])) / modelo.escala[i];
      correccion += aporte;
      return { variable: nombre, valor: v, imputado, aporte };
    });
    return {
      estimado: zona.prediccion + correccion,
      efecto_zona: zona.prediccion,
      correccion,
      aportes,
      vecinos: zona.vecinos,
    };
  }

  /** Semiamplitud del intervalo para una zona; si no existe, usa la global. */
  function semiamplitud(intervalos, zona, nivel) {
    const fila = intervalos[zona] || intervalos["Todas"];
    return fila[String(nivel)];
  }

  /** Probabilidad empírica de que el rendimiento real alcance el umbral. */
  function probabilidad(estimado, residuos, umbral) {
    if (!residuos || !residuos.length) return NaN;
    let n = 0;
    for (const r of residuos) if (estimado + r >= umbral) n++;
    return n / residuos.length;
  }

  return { proyectar, predecir, vecinos, semiamplitud, probabilidad, mediana };
});
