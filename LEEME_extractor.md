# Extractor AgroCebada

`extraccion.py` lee los TIFF descomprimidos, relaciona sus valores con las parcelas del Shapefile y genera CSV, GeoJSON y un mapa HTML con los datos incorporados. Los datos originales se conservan. El programa extrae clima y topografía; los índices satelitales ya están en los CSV originales y se pueden unir después mediante `ID_POLIGONO`.

## Ejecución

Desde esta carpeta, con el entorno virtual ya preparado:

```bash
.venv/bin/python extraccion.py
```

Para instalar las dependencias en otro equipo con Python 3.10 o posterior:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements_prediccion.txt
.venv/bin/python extraccion.py
```

En Windows, usa `.venv\Scripts\python.exe` y `.venv\Scripts\pip.exe`.

Opciones útiles:

```bash
# Todos los meses de 2025 y las dos capas de topografía.
.venv/bin/python extraccion.py --anio 2025 --salida salidas_2025

# Una sola imagen; la ruta con espacios debe ir entre comillas.
.venv/bin/python extraccion.py --imagen "DATASET_RETO_AGRO_2026/Reto_AgroCebada_CHIRPS_Precipitacion_2022_2025/Precipitacion_mensual/2025/PREC_2025_07.tif" --salida salidas_julio

# Método alternativo: tomar el píxel de un punto interior de cada parcela.
.venv/bin/python extraccion.py --metodo punto --salida salidas_punto

# Solo archivos de datos, sin generar el HTML.
.venv/bin/python extraccion.py --sin-mapa

# Ver todas las opciones.
.venv/bin/python extraccion.py --help
```

También puedes cambiar `--dataset`, `--parcelas`, `--etiquetas` y `--salida`. `--imagen` acepta un TIFF o una carpeta y puede repetirse. El dataset debe estar descomprimido. El Shapefile necesita los archivos `.shp`, `.shx`, `.dbf` y `.prj` juntos; se usa `.cpg` si existe. Para ejecuciones con distintos filtros o métodos, usa diferentes carpetas de salida. Volver a ejecutar sobre la misma carpeta reemplaza los archivos de resultados con el mismo nombre; no elimina archivos de ejecuciones anteriores.

## Archivos generados

En `salidas_extractor/`:

| Archivo | Contenido |
| --- | --- |
| `por_imagen/*.csv` | Un CSV por TIFF/banda, con una fila por parcela y estadísticas de extracción. |
| `datos_por_imagen.csv` | Todos esos registros juntos: parcela × imagen/banda. Incluye fuente, fecha, unidad, CRS, media, extremos, desviación estándar, píxeles y cobertura válida. |
| `clima_mensual.csv` | Una fila por parcela y mes, con precipitación, Tmin, Tmax, sus coberturas y dos indicadores térmicos derivados. |
| `topografia_parcelas.csv` | Una fila por parcela, con elevación y pendiente, extremos, desviación y cobertura. |
| `parcelas_datos.csv` | Una fila por parcela y una columna por capa, útil para unir con datos satelitales o preparar un modelo. |
| `coordenadas_parcelas.csv` | Centroide y punto interior transformados al CRS de cada tipo de raster. Las unidades de coordenadas se indican por fila. |
| `inventario_rasters.csv` | Archivos utilizados, variable, banda, fecha, unidad, CRS, tamaño, transformación, resolución, NoData y parcelas con valor. |
| `parcelas.geojson` | Geometrías en EPSG:4326, con orden longitud/latitud y datos de identificación. Se puede abrir en un SIG. |
| `mapa_parcelas.html` | Mapa interactivo con todas las capas, búsqueda de parcela, leyenda, ficha y descarga del CSV de la capa seleccionada. El fondo de calles y localidades necesita internet; las parcelas y sus datos se pueden consultar sin conexión. |
| `resumen_ejecucion.json` | Conteos, método, sistemas de coordenadas, estados de calidad y notas de la ejecución. |

Los CSV usan coma como separador, punto decimal y UTF-8 con BOM para facilitar su lectura en Excel. Los valores ausentes se guardan como celdas vacías, no como cero. `RENDIMIENTO_T_HA` conserva las etiquetas del concurso: las 59 parcelas de predicción mantienen este campo vacío. El extractor no genera predicciones de rendimiento.

El selector **Fondo del mapa** permite mostrar calles y localidades o dejar solo las parcelas. El fondo se obtiene de [OpenFreeMap](https://openfreemap.org/quick_start/), basado en OpenStreetMap, y se dibuja con MapLibre GL JS. No requiere cuenta ni clave. Las atribuciones se muestran dentro del mapa. El fondo se sincroniza con las coordenadas de las parcelas al buscar, mover o acercar; no interviene en la extracción de valores ni cambia los CSV. Si falta internet o el navegador no admite WebGL, el mapa conserva las parcelas y avisa que el fondo no está disponible.

## Coordenadas y extracción

El programa lee el CRS declarado en `.prj` y en cada TIFF; nunca asume una proyección cuando falta. Normaliza `ID_POLIGON` del SHP a `ID_POLIGONO` y une por identificador, sin depender del orden de filas. Comprueba coincidencia de IDs, área y conjunto con el CSV de etiquetas.

Las parcelas se transforman al CRS de cada raster con `pyproj` y `always_xy=True`: longitud/latitud para coordenadas geográficas y este/norte para proyectadas. En estos datos, clima y parcelas están en EPSG:4326, y el relieve en EPSG:6372. Las geometrías del mapa se entregan en EPSG:4326. El centroide identifica la ubicación; el método puntual usa un punto garantizado dentro de la parcela.

El método predeterminado, `zonal`, intersecta la parcela con cada píxel que toca. Calcula la superficie de cada intersección en una proyección equivalente local Lambert azimutal (LAEA), en metros cuadrados, y obtiene:

`media = suma(valor_del_pixel × superficie_intersectada_valida) / suma(superficies_validas)`

Así evita dar el mismo peso a una intersección diminuta y a otra grande. También obtiene mínimos, máximos y desviación estándar ponderada de los píxeles válidos. Estos estadísticos describen la capa raster, no la incertidumbre de un modelo. Se excluyen NoData, máscaras y valores no finitos; el cero válido se conserva. Se aplican las escalas y offsets declarados por banda.

`cobertura_valida_pct = superficie_con_datos_validos / superficie_geometrica_de_la_parcela × 100`

El estado puede ser `OK`, `COBERTURA_PARCIAL`, `SIN_COBERTURA` o `SIN_DATOS_VALIDOS`. Con el método puntual no se estima cobertura superficial: el campo queda vacío.

## Interpretación

- La lluvia es acumulación mensual en mm: para acumular abril–octubre se suman los meses. El promedio espacial por parcela no suma la lluvia de todos sus píxeles.
- Tmin y Tmax son promedios mensuales de extremos diarios, en °C. No permiten contar directamente días de helada o episodios de calor.
- `temperatura_media_aprox_c = (Tmin + Tmax) / 2`; es una aproximación. `amplitud_termica_promedio_c = Tmax - Tmin`.
- Elevación está en metros y pendiente en grados, no en porcentaje.
- El clima tiene píxeles de 0.05° (aproximadamente 5 km). Diferentes parcelas pueden compartir valores. La topografía tiene píxeles de 120 m. La extracción no añade resolución al dato original.
- En el dataset hay zonas con NoData topográfico. La pendiente de AGC_170 tiene aproximadamente 2.1% de cobertura válida: el valor representa esa pequeña porción y debe revisarse antes de usarlo en un modelo. El mapa muestra las coberturas parciales.
- Las unidades de variables conocidas se toman de los LEEME del reto; para otros TIFF se conserva la unidad declarada por banda o se indica que falta documentación.

## Verificación

```bash
.venv/bin/python -m unittest tests_extraccion.py
```

Las pruebas verifican extracción de parcelas pequeñas, ponderación por superficies desiguales, ceros, datos ausentes, reproyección y valores de referencia del dataset. El resumen de ejecución permite comprobar que se procesaron 146 TIFF y 197 parcelas en una ejecución completa.

Documentación de las bibliotecas: [Rasterio](https://rasterio.readthedocs.io/en/stable/), [pyproj](https://pyproj4.github.io/pyproj/stable/api/transformer.html) y [Shapely](https://shapely.readthedocs.io/en/stable/reference/shapely.intersection.html).
