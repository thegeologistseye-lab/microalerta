# MicroAlerta Scan

Pipeline en Python para detectar, medir y clasificar por forma partículas en fotos de filtros, y para calcular la TDC de MicroAlerta.
**No asigna polímeros por color.** Clasifica la forma (fibra, fragmento o película). Cuando hay un modelo entrenado con etiquetas validadas, también entrega la probabilidad de que cada partícula sea PCM.

## Instalación

```bash
pip install -r requirements.txt
python -m pytest -q          # 34 pruebas con imágenes sintéticas de medidas conocidas
pip install pyserial         # solo para mover la platina con el Arduino
```

## Flujo de trabajo

```
fotos ─► scan ─► particulas.csv + galeria.html ─► (etiquetar) ─► clasificador ─► scan --modelo ─► tdc
```

### 1. Calibrar la escala (µm/píxel)

Se puede elegir una de tres formas:

| Situación | Cómo |
|---|---|
| Fotos con celular por el ocular (las actuales) | `--diam-campo-um`: diámetro del campo visual = número de campo del ocular (FN, en mm) ÷ aumento del objetivo × 1000. Ej.: FN 18 con objetivo 4× → `4500`. El programa detecta el círculo en cada foto, así que el zoom del celular puede variar entre fotos. |
| Portaobjetos micrométrico | `python -m microalerta_scan.cli calibrar --imagen micrometro.jpg --um-entre-marcas 10 --guardar captura.json`. Con la cámara y los motores la escala queda en `captura.json` y `scan` la lee sola; con el celular, ponga el valor en `calibracion.um_por_px` de la ficha o use `--um-por-px` |
| Escáner con cámara fija | lo mismo que con el micrómetro, junto con `--sin-campo-circular` |

Sin calibración, las medidas quedan en píxeles y la TDC no se calcula.

### 2. Procesar una muestra

```bash
python -m microalerta_scan.cli scan --config ejemplos/muestra_ejemplo.json --salida resultados/calle_r1
# o, sin archivo de configuración:
python -m microalerta_scan.cli scan --imagenes fotos/calle/ --salida resultados/calle --diam-campo-um 4500
```

Salidas:
- `resultado.json`: muestra, calibración por foto, conteo, intervalo de confianza del 95 % (Poisson), partículas y avisos. Es el formato para subir a la app. En `conteo` quedan el área escaneada y el área efectiva del filtro (mm²), la fracción analizada y el `limite_minimo_um`: el tamaño más pequeño que se puede medir con ese aumento (por foto, en `campos`).
- `particulas.csv`: más de 30 rasgos por partícula y una columna `etiqueta` vacía para completar.
- `galeria.html`: recortes de todas las partículas, con su número, para etiquetarlas.
- `superpuestas/`: cada foto con los contornos dibujados. Magenta = fibra, verde = fragmento, amarillo = película, gris = descarte según el modelo.
- `recortes/`: una imagen por partícula.

Opciones útiles:
- `--flat filtro_limpio.jpg`: corrección de iluminación plana.
- `--posiciones campos.csv`: con columnas `campo,x_mm,y_mm` (posición de la platina en el centro de cada foto). Entrega las coordenadas de cada partícula sobre el filtro, para buscarla después en el Raman. Con posiciones, los campos pueden solaparse: cada punto del filtro se asigna al campo cuyo centro está más cerca (el área escaneada es la de la unión), y cada partícula se cuenta una sola vez (columna `fuera_de_su_zona`; ver la sección del Rojo de Nilo). Una fibra cortada en un campo se cuenta en el vecino que la ve completa. `resultado.json` indica el `modo_captura`: `mosaico_con_posiciones` o `manual_sin_solape` (fotos a mano: no deben repetirse ni pisarse). Los ejes de la cámara deben coincidir con los de la platina.
- `--parametros mis_param.json`: sobrescribe los umbrales de `config.py`.
- Imágenes multimodales del mismo campo: `campo01_W.jpg` (luz blanca), `campo01_P.jpg` (luz polarizada) y `campo01_F.jpg` (fluorescencia). Se alinean automáticamente. La foto polarizada agrega tres rasgos: brillo medio (`indice_birrefringencia`), puntos más brillantes (`birrefringencia_p95`) y fracción de la partícula que brilla (`frac_brillante_pol`). Ayudan a separar cristales (cuarzo, sal) de fibras plásticas y de material orgánico. La fluorescencia es opcional.

### 3. Etiquetar y entrenar el clasificador PCM / descarte

1. Abra `galeria.html` y anote en `particulas.csv` (columna `etiqueta`) una de estas categorías: `PCM`, `organico`, `mineral` u `otro`.
2. Etiquete al menos unas 50 partículas por clase. Lo ideal es que las PCM estén confirmadas por Raman.
3. Entrene el modelo:
   ```bash
   python -m microalerta_scan.clasificador particulas_etiquetadas.csv --salida modelo_pcm.joblib
   ```
   Se obtiene la matriz de confusión con validación cruzada **por disco** (`muestra_id`: las partículas de un
   mismo disco nunca quedan a la vez en entrenamiento y validación) y la importancia de cada rasgo.
4. Prueba con discos que el modelo no vio (al menos 20 partículas por clase, meta fijada antes):
   ```bash
   python -m microalerta_scan.clasificador prueba_etiquetada.csv --probar modelo_pcm.joblib --meta 0.8 \
          --entrenamiento particulas_etiquetadas.csv
   ```
   Entrega la matriz de confusión, la precisión y la sensibilidad, y rechaza la prueba si algún disco está
   también en el entrenamiento. Con cada campaña nueva, repetir la prueba con sus discos antes de reentrenar.
5. Si la prueba aprueba, vuelva a procesar con `--modelo modelo_pcm.joblib`: desde ese momento el conteo es de
   **PCM**. Si no, se siguen informando **candidatas detectadas por fluorescencia**.

### 4. TDC y calidad del dato

```bash
python -m microalerta_scan.cli tdc --muestras r1/resultado.json r2/resultado.json r3/resultado.json \
                                   --controles c1/resultado.json c2/resultado.json c3/resultado.json
```

`TDC = máx(0, Nm − Nc) / (A × t)`, donde A es el área del captador (la placa expuesta) en m² y t el tiempo en días.

- Con réplicas, cada una guarda su `diferencia_corregida` = (N_m − N̄_c) / (A × t) **sin truncar** (puede ser
  negativa), y `tdc_media` = máx(0, media de esas diferencias). Truncar cada réplica antes de promediar subiría la
  media cerca del fondo. `tdc_de` y `cv_porcentaje` también se calculan sin truncar.
- Si la media de N_m no pasa N̄_c + 3·DE, `bajo_limite_deteccion` = true: el punto se informa como «bajo el límite
  de detección» (`limite_deteccion_tdc` = 3·DE / (A × t)), no como una deposición medida. `de_control_tdc` es la
  DE del fondo en PCM/(m²·día), para informarla junto a la TDC.

- Si el análisis no cubre el filtro completo, el conteo se extrapola con el área de las fotos y `filtro_diametro_mm`. Sin ese diámetro, la TDC queda en `null` y se explica el motivo.
- Criterio de calidad, fórmula exacta (`muestra.evaluar_campana`; queda escrita en `criterio_calidad.formula`
  de la salida de `tdc`, para que la app muestre la misma):

  | Símbolo | Qué es |
  |---|---|
  | N_m, N_c | conteo de cada réplica o control extendido al disco completo: `n_contado ÷ fracción analizada` (`n_estimado_filtro`) |
  | N̄_m, N̄_c | medias de réplicas y de controles |
  | s_c | desviación estándar muestral de los N_c (con n − 1) |
  | σ_P | piso de Poisson: √máx(media de los `n_contado` de los controles, 1) ÷ fracción media analizada en los controles |
  | DE | máx(s_c, σ_P) |

  **Confianza reforzada** si hay ≥ 3 réplicas, ≥ 3 controles y N̄_m > N̄_c + 3·DE. En cualquier otro caso,
  **exploratorio**. Ejemplo: controles en 0 y disco completo → DE = 1 y la media de las réplicas debe pasar de
  3 (3 justas no alcanzan). Con el celular (40 % del disco) → DE = 2,5 en el disco: siguen siendo más de 3
  contadas. Evalúa la calidad del dato, no la identidad química. Es una categoría interna de detección sobre
  el fondo: no es un intervalo de confianza ni una prueba de significación, y no mira cuánto varían las
  réplicas. Por eso la salida de `tdc` trae también `tdc_media`, `tdc_de` y `cv_porcentaje` de las réplicas.

## Rojo de Nilo: prueba, captura en cuadrícula y conteo

Desde la versión 0.2 el método base es el **Rojo de Nilo**: los plásticos brillan con luz azul detrás de un
filtro naranjo. El flujo completo es:

```
captura en cuadrícula (motores) o grilla manual (celular) ─► cuadrícula del disco ─► scan --deteccion fluorescencia ─► tdc
```

### 0. Probar todo sin equipo

```bash
python -m microalerta_scan.cli prueba --salida prueba          # ~1 min; --rapida: disco chico, segundos
```

Simula un disco teñido de 21 mm con partículas conocidas (fragmentos y fibras que brillan, partículas menores
de 20 µm, grumos de tinte, arena y algodón que no brillan), una platina con 80 µm de juego y la cámara de 5 MP.
Captura unas 100 posiciones con solape, arma la cuadrícula, cuenta con el programa y compara con lo que había.
`cuadricula_revisada.jpg` marca en verde las encontradas, en rojo las perdidas y en magenta las contadas de más.

### 1. Platina con motores

- Firmware: `firmware/microalerta_platina/microalerta_platina.ino` (Arduino Nano, dos 28BYJ-48 con ULN2003 en
  D2–D5 y D6–D9, LED azul en D10 y LED blanco en D11 a través de un transistor). Órdenes de texto por el puerto
  serie: `M x y`, `R dx dy`, `Z`, `P`, `L azul blanca`, `V us`.
- Configuración: copie `ejemplos/captura_ejemplo.json` a `captura.json`.
- Calibración, en este orden. Cada comando **escribe solo** sus valores en `captura.json` y deja un registro
  con la fecha en `calibraciones` (las últimas 20); no hay que copiar nada a mano:
  ```bash
  # 1) escala: foto del micrómetro con la misma cámara y resolución que la captura
  python -m microalerta_scan.cli calibrar --imagen micrometro.jpg --um-entre-marcas 10 --guardar captura.json
  # 2) platina: con un disco de papel usado bajo el objetivo y luz blanca (textura que no se repite)
  python -m microalerta_scan.cli calibrar-platina --puerto COM3 --camara 0 --config captura.json
  ```
  `calibrar` guarda `um_por_px` y el tamaño del campo (`campo_ancho_mm`, `campo_alto_mm`). `calibrar-platina`
  mueve cada eje (como máximo 30 % del campo), mide cuánto se corre la imagen y guarda `pasos_mm_x/y`,
  `signo_x/y` y `juego_mm` (sobrepaso = 2 × juego medido, mínimo 0,1 mm). No guarda si la comparación de
  fotos sale débil (calidad < 0,1) y avisa si la cámara está girada más de 1°. `--no-guardar` solo muestra.
  Las rayas del micrómetro no sirven para la platina: se repiten cada 10 µm y engañan a la medición.
- Sentido de los ejes: `signo_x = +1` si al avanzar el motor el centro de la foto avanza sobre el disco hacia
  la derecha de la imagen (en y, hacia abajo). La captura llega a cada campo siempre con el motor avanzando,
  igual que al fijar el cero, así que el juego no corre las posiciones aunque un eje tenga signo −1.

### 2. Capturar

```bash
python -m microalerta_scan.cli capturar --modo motores --puerto COM3 --camara 0 --config captura.json --salida fotos/CALLE-R1
```

Primero pide centrar el disco (órdenes `x 0.5`, `y -1`, `luz azul`, `listo`; la imagen queda en
`vista_previa.jpg`). Después recorre el círculo en filas, siempre de izquierda a derecha y llegando a cada
campo avanzando (si tiene que retroceder, se pasa `juego_mm` y vuelve), y en cada campo saca `campoNNN_F.jpg`
(luz azul) y `campoNNN_W.jpg` (luz blanca). Guarda `campos.csv` con la posición de cada campo sobre el disco
(centro del disco = 0, 0), `captura.json` y la cuadrícula: `cuadricula_F.jpg` y `cuadricula_W.jpg`, el mosaico
del disco con el número de cada campo. Si las luces no están conectadas al Arduino, use `--luces-manual`:
hace una pasada por luz y pide cambiarla entre ellas.

**Modo manual con celular (mientras llega la cámara):**

```bash
python -m microalerta_scan.cli capturar --modo manual --config captura.json --salida fotos/CALLE-R1
# … fotos con el celular por el ocular, con el filtro naranjo delante del lente …
python -m microalerta_scan.cli ordenar --fotos celular/CALLE-R1 --salida fotos/CALLE-R1
```

`capturar --modo manual` arma una grilla de campos circulares que no se tocan (con el objetivo 4× y un campo de
4,5 mm caben 9, que cubren cerca del 40 % del disco), con la lectura del vernier de cada uno, y la deja en
`grilla_manual.png` para imprimir. `ordenar` ordena las fotos del celular por la hora en que se tomaron, les
pone el nombre de cada campo (`--orden por-campo`: azul y blanca en cada campo; `por-pasada`: todas las azules
y después las blancas) y arma la cuadrícula.

### 3. Contar

```bash
python -m microalerta_scan.cli scan --config muestra.json --deteccion fluorescencia \
       --posiciones fotos/CALLE-R1/campos.csv --sin-campo-circular --salida resultados/CALLE-R1
```

(ver `ejemplos/muestra_fluorescencia.json`; con fotos de celular por el ocular, sin `--sin-campo-circular`).
Si no se da escala, `scan` la toma de `captura.json`, que está junto a `campos.csv`. El umbral de fluorescencia
lo toma de `parametros_campana.json` si la ficha lo nombra (`"parametros_campana": "../parametros_campana.json"`)
o si se da `--parametros`. Si la ficha trae además un `fluo_umbral_min` o una escala distintos, gana el archivo
de la puesta a punto y queda un aviso en `resultado.json`, que anota el umbral usado y de dónde salió.

**Umbral de fluorescencia con los blancos (puesta a punto C).** Con las fotos de los 3 blancos de
procedimiento y, si está, la muestra de prueba (7B):

```bash
python -m microalerta_scan.cli umbral --blancos fotos/B1 fotos/B2 fotos/B3 --referencia fotos/PRUEBA \
       --guardar parametros_campana.json
python -m microalerta_scan.cli scan ... --parametros parametros_campana.json
```

Busca en los blancos todas las manchas de ≥ 20 µm que pasan fondo + 6 σ y deja `fluo_umbral_min` sobre la
mancha número n + 1 más brillante (n = número de blancos), por un margen de 1,10, y nunca bajo 12: en promedio,
a lo más 1 mancha falsa por blanco. Con `--referencia` revisa que queden sobre el umbral al menos 90 % de las
partículas de la muestra de prueba; si no, avisa que los blancos brillan demasiado (tinte viejo o mal
enjuagado) y que conviene repetirlos en vez de subir el umbral. El archivo guarda la fecha, los blancos usados,
la regla y las manchas más brillantes. Use el mismo archivo en todas las muestras de la campaña.

- Detecta lo que brilla en la foto `_F`: señal = canales verde y rojo (el azul lo bloquea el filtro), fondo
  estimado con una apertura morfológica grande, umbral = fondo + 6 σ del ruido con un piso `fluo_umbral_min`,
  que se fija con los blancos de procedimiento. El contorno se toma a media altura del brillo de cada partícula,
  para que el halo no las agrande.
- `limite_minimo_um` (20 con papel filtro, provisional hasta la muestra de prueba 7B): lo más corto no se cuenta y
  queda marcado `bajo_limite`. El tamaño es `largo_um`: el Feret máximo en fragmentos y fibras rectas, y el largo
  por la línea media en objetos curvos (fibras). `particulas.csv` guarda también `ancho_um`, `area_um2`,
  `diam_equiv_um` y `feret_max_um`, y `conteo.por_forma` separa fibras, fragmentos y películas.
- `color_fluo` (verde, amarillo, naranjo o rojo) y `razon_rg_fluo`: el Rojo de Nilo cambia de color según la
  polaridad del plástico. Es una orientación, no identifica el polímero.
- Con posiciones, se analiza solo dentro del círculo de filtrado (`filtro_diametro_mm` − `margen_disco_mm`).
- Solapes: la misma partícula vista entera en dos campos se cuenta una vez, en el campo cuyo centro está más
  cerca de su posición promedio. Los objetos más largos que el solape (fibras largas), que ningún campo ve
  enteros, se cuentan una vez por sus trozos (el más grande); su largo queda subestimado y se avisa.
- Los recortes de la galería muestran la foto azul y la blanca lado a lado. Etiquetas sugeridas: `PCM`,
  `organico`, `mineral`, `grumo_tinte`, `otro`.

## Cómo funciona (partículas oscuras, luz blanca)

1. **Campo visual.** Detecta el círculo del ocular con un ajuste que tolera círculos cortados por el borde de la foto.
2. **Canal de trabajo.** Las tinciones saturan un canal y dejan otro casi negro. Se usa el canal no saturado más brillante.
3. **Fondo.** Se ajusta un polinomio robusto de iluminación que corrige el viñeteo sin que los fragmentos grandes lo contaminen. Se trabaja con la razón intensidad/fondo, que no depende del color de la tinción.
4. **Zonas excluidas.** Se descartan el viñeteo fuerte y las zonas oscuras extensas, como el borde del filtro o las sombras. La fracción útil queda registrada.
5. **Segmentación.** Partículas oscuras: núcleo oscuro (semilla) con crecimiento por histéresis, y umbrales adaptados a la textura del papel. Fibras: filtro de crestas de Sato. Las partículas pegadas se separan con watershed conservador.
6. **Medición.** Largo medido por el camino más largo del esqueleto (mide bien las fibras curvas) o por el Feret máximo. Ancho óptico corregido por desenfoque. También: área, circularidad, solidez, rugosidad de borde y contraste y color relativos al papel vecino.
7. **Forma.** Fibra si largo/ancho > 3. Película si es translúcida e irregular (regla heurística que hay que validar). Fragmento en los demás casos.
8. **Duplicados.** Si la misma foto aparece dos veces, se omite y queda registrado en los avisos.

## Límites conocidos

- El **papel** como filtro es un fondo difícil: su textura y sus propias fibras de celulosa generan detecciones que el clasificador tendrá que descartar. Una membrana lisa mejora mucho la detección.
- Las fotos con un canal saturado (rojo muy intenso con Lugol) pierden contraste.
- "Película" no se puede confirmar en 2D, porque no se mide el espesor.
- Sin posiciones, se asume que los campos fotografiados **no se solapan**. Con `campos.csv` (captura con motores) los solapes se resuelven solos.
- En fluorescencia, el umbral depende de la luz, la exposición y el tinte: fíjelo con los blancos de procedimiento y no cambie la exposición durante la campaña.
- La prueba simulada comprueba la captura, la cuadrícula y el conteo, no la química: qué plásticos brillan se mide con el experimento de validación del protocolo.

## Licencia

MIT (ver `LICENSE` en la raíz del repositorio). Es parte de [MicroAlerta](../README.md).
