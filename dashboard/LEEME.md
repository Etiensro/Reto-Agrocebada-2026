# Panel AgroCebada

Panel web para consultar las estimaciones de rendimiento de cebada, ejecutar el modelo y revisar su
desempeño. Es una página estática: no necesita servidor, base de datos ni instalación.

## Abrirlo

**Forma más simple:** abre `dashboard/index.html` con doble clic. Funciona sin internet; con internet
descarga las tipografías de Google Fonts, y sin ella usa las del sistema.

**Con servidor local** (si el navegador bloquea algo):

```bash
python3 -m http.server 8765 --directory dashboard
```

y abre <http://localhost:8765>.

## Dos versiones del mismo panel

| Archivo | Para qué |
| --- | --- |
| `index.html` | El panel con el mapa simple de parcelas. Funciona sin internet. |
| `panel_mapa_satelital.html` | El mismo panel (mismo diseño, mismas secciones) con **fondo satelital**, y además las capas de clima y relieve y el buscador del mapa de `salidas_extractor/mapa_parcelas.html`. Necesita internet para el fondo; sin él, las parcelas y sus datos siguen visibles. |

La versión satelital agrega, solo en el mapa:

- **Capa**: rendimiento estimado, real o error del modelo, y las variables extraídas de los TIFF
  (precipitación, temperatura mínima y máxima, elevación y pendiente), con **año** y **mes**.
- **Fondo del mapa**: satélite con nombres (Esri World Imagery), calles y localidades (OpenFreeMap, el
  mismo fondo del mapa original) o sin fondo.
- **Buscar parcela** por identificador y **descargar el CSV** de la capa visible.
- En la ficha de cada parcela, la sección **Territorio** con el clima del mes elegido, la elevación y la pendiente.

Tu mapa original (`salidas_extractor/mapa_parcelas.html`) no se modifica. Su fondo era de calles y
localidades; la opción satelital es nueva.

## Qué contiene

| Sección | Para qué sirve |
| --- | --- |
| **Mapa** | Las 197 parcelas coloreadas por rendimiento estimado, real o error. Al seleccionar una parcela muestra su estimación, el intervalo, cómo se obtuvo y sus vecinos más influyentes. |
| **Parcelas** | Tabla ordenable y filtrable con todas las estimaciones; se puede copiar como CSV. |
| **Ejecutar el modelo** | Cambia ubicación, superficie y verdor previo a la siembra; el modelo recalcula en el navegador con los mismos parámetros que `predicción.py`. |
| **Riesgo y crédito** | Probabilidad de que cada parcela alcance un rendimiento mínimo que tú defines; niveles alta, media y baja. |
| **Desempeño** | R², RMSE y MAE en validación cruzada y en la prueba reservada, con gráficas real contra estimado. |
| **Preguntas guía** | Respuestas a las tres preguntas guía de las bases, con sus tablas. |
| **Método** | Cómo se construye la estimación, fuentes de datos y límites. |

## Archivos

| Archivo | Contenido |
| --- | --- |
| `index.html`, `estilos.css`, `app.js` | Interfaz. Los estilos y el código son compartidos por las dos versiones. |
| `panel_mapa_satelital.html` | Versión con fondo cartográfico. **Se genera** desde `index.html`. |
| `territorio.js` | Clima y relieve por parcela (28,762 registros de `datos_por_imagen.csv`). **Se genera.** |
| `modelo.js` | Modelo geográfico-residual en JavaScript (vecinos más cercanos + corrección Ridge). |
| `datos.js` | Datos del panel. **Se genera**, no se edita a mano. |
| `construir_datos.py` | Genera `datos.js`, `territorio.js` y `panel_mapa_satelital.html` desde `salidas_prediccion/` y `salidas_extractor/`. |
| `probar_modelo.js` | Comprueba en Node que `modelo.js` reproduce las predicciones de Python. |
| `panel_autonomo.html`, `panel_mapa_satelital_autonomo.html` | Una sola página con todo incorporado (opcional, ver abajo). |

## Actualizarlo después de reentrenar

```bash
.venv/bin/python predicción.py                       # entrena y escribe salidas_prediccion/
.venv/bin/python dashboard/construir_datos.py        # regenera dashboard/datos.js
.venv/bin/python -m unittest tests_dashboard.py      # comprueba que todo coincide
```

El generador solo exporta el modelo geográfico-residual. Si una nueva ejecución elige otro tipo de
modelo, avisa con un error en lugar de mostrar cifras que no corresponden.

## Publicarlo para que el jurado lo consulte

Las bases piden un enlace o el código con instrucciones. Para tener un enlace público:

1. Genera las versiones de un solo archivo: `.venv/bin/python dashboard/construir_datos.py --autonomo`.
2. Súbela a cualquier hosting estático (GitHub Pages, Netlify, Cloudflare Pages) con el nombre `index.html`:
   `panel_autonomo.html` (sin internet) o `panel_mapa_satelital_autonomo.html` (con fondo satelital, 1.1 MB).

No hay datos personales: los identificadores son los del reto (AGC_001 a AGC_197).

## Notas

- Los datos no salen de tu equipo. Lo único que se pide a internet son las tipografías y, en la versión
  satelital, las teselas del fondo (Esri u OpenFreeMap) y la librería MapLibre (unpkg).
- El fondo satelital cita a Esri y Maxar en el mismo mapa, como piden sus condiciones de uso.
- En las parcelas con rendimiento conocido, el estimado mostrado viene de validación cruzada: el modelo
  no vio su propio dato.
- El rendimiento mínimo de la sección de riesgo es un valor de ejemplo editable; ajústalo al costo real.
- La preferencia de tema y el rendimiento mínimo se recuerdan en el navegador (localStorage).
