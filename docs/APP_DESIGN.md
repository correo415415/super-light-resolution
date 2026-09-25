# Diseño de la aplicación: generación de frames en tiempo real tipo *Lossless Scaling*

> **Sólo diseño.  Aquí no hay código todavía.**  El objetivo de este documento es fijar
> la arquitectura, los números que hay que cumplir y las decisiones que hay que tomar
> ANTES de escribir la primera línea, para que cuando el modelo (run 2) esté listo
> sepamos exactamente qué construir y en qué orden.

## 0. Qué hace Lossless Scaling (LS) y qué queremos replicar

Lossless Scaling (LSFG 3.x) es un post-proceso **independiente del juego**: captura lo
que el juego presenta en pantalla, estima el movimiento entre el frame anterior y el
actual con un modelo de flujo óptico propio, genera 1…N frames intermedios y los presenta
él mismo en una ventana superpuesta (o, en Linux con `lsfg-vk`, inyectándose como capa
Vulkan en la cadena de presentación del juego).  No necesita motion vectors del motor
(a diferencia de DLSS-FG / FSR-FG), por eso funciona con cualquier juego, emulador o
vídeo — y por eso es *peor* que ellos en oclusiones y HUD.

Puntos de LS que hay que reproducir sí o sí, porque son los que hacen que sea usable:

| Función LS | Por qué importa |
|---|---|
| Captura DXGI / WGC (Windows) o capa Vulkan (Linux) | Coger frames sin tocar el juego, a resolución completa, sin copia a CPU |
| Modo ×2, ×3, ×4 y **Adaptive** (objetivo de fps fijo) | Adaptive = generar exactamente los frames que faltan para llegar a 60/120/144 Hz aunque el juego varíe |
| **Flow Scale** (resolución a la que se calcula el flujo, p.ej. 50–100 %) | Es el knob principal calidad ↔ coste GPU |
| **Performance mode** (modelo más ligero) | Para GPUs pequeñas / handhelds |
| **Queue Target** y **Max Frame Latency** | Control de la cola de frames → latencia vs estabilidad |
| Desactivar FG si fps < ~10 y recomendación de dejar ~30 % de GPU libre | FG compite por la GPU con el juego; si el juego no deja headroom, el resultado es peor que sin FG |
| Detección de cambios de escena / cortes | Interpolar entre dos frames sin relación da papilla; hay que repetir frame |
| Cap de fps del juego (ellos recomiendan cap externo, p.ej. RTSS) | Frame pacing regular es más importante que fps altos para FG |

Lo que **no** vamos a replicar (al menos al principio): escalado espacial (LS1, FSR,
NIS…), soporte HDR, dual-GPU, DirectX 9 / OpenGL exóticos, cursor.

## 1. Requisitos numéricos (el presupuesto)

Objetivo mínimo viable: **1080p, juego a 60 fps → salida a 120 fps (×2) en una RTX 3060**.

```
Presupuesto por frame generado a 120 Hz de salida:  8.33 ms
  – el juego ya está usando la GPU (p.ej. 60 %) → nos quedan ~3–4 ms de GPU por frame generado
  – captura + copia + presentación                 ~0.5–1 ms
  ⇒ la inferencia del modelo tiene que estar en   ≤ 3 ms @ 1080p  (fp16, TensorRT)
```

Dónde estamos: RIFE completo (run 1) mide **7.4 ms/frame a 448×256 en una T4 con PyTorch
fp16 eager**.  A 1080p son 18× más píxeles.  Con PyTorch eager estaríamos en ~60–100 ms:
**dos órdenes de magnitud fuera del presupuesto**.  Por eso la app no es "ejecutar
inference.py más rápido": necesita una pila de optimización propia.  Palancas, de mayor a
menor impacto:

| Palanca | Ganancia esperada | Coste |
|---|---|---|
| TensorRT fp16 (fusiona convs, elimina overhead de Python/launch) | 3–5× sobre eager | Exportar a ONNX; `grid_sample` es soportado (opset 16+) |
| **Flow Scale**: calcular IFNet a 50 % de resolución y reescalar flujo+máscara ×2 | ~3.5× en la parte IFNet | Pérdida de detalle en bordes finos; es exactamente lo que hace LS |
| Quitar el RefineUNet + Contextnet en modo *performance* | ~30–40 % | Menos nitidez.  RIFE 4.x oficial también lo quitó |
| Modelo más estrecho (`--ifnet_widths 180 110 70`, `--refine_c 8`) | ~1.7× | Hay que reentrenar (fine-tune con `--init_from` no vale si cambian anchuras) |
| INT8 (TensorRT PTQ con calibración) | 1.3–1.8× sobre fp16 en Ampere | Riesgo de artefactos en el flujo; calibrar con frames de juegos |
| Sólo IFBlock de escala 4 y 2 (saltarse escala 1) en performance mode | ~40 % IFNet | Flujo más borroso |
| CUDA Graphs (tamaños fijos) | 10–20 % en modelos pequeños | Trivial en TensorRT |

Objetivo realista tras optimizar: **~3–5 ms a 1080p en 3060** con flow scale 50–75 % y
sin refine, fp16 TRT.  A 1440p habrá que bajar flow scale; 4K sólo en modo performance.
Hay que **medir pronto**: el primer hito de ingeniería es un benchmark, no la app.

### 1.1 Latencia

La interpolación es inherentemente *causal hacia atrás*: para mostrar el frame entre
F(n) y F(n+1) hay que tener F(n+1).  Por tanto **añade al menos 1 frame de juego de
latencia** (16.7 ms a 60 fps) + el tiempo de inferencia + la cola de presentación.  LS
tiene exactamente esta latencia; DLSS-FG también.  No hay forma de eliminarla sin
*extrapolación* (predecir F(n+½) sólo con F(n−1), F(n)), que es otro problema (más
artefactos, pero latencia cero — Intel lo investiga como *ExtraSS*).  Diseño: la v1 hace
interpolación; dejamos la interfaz preparada para probar extrapolación después.

## 2. Arquitectura

```
                 ┌──────────────────────┐
   juego ──────▶ │  Captura              │  DXGI Desktop Duplication / Windows Graphics Capture
  (fullscreen /  │  (texturas GPU,       │  (Windows)   ·   capa Vulkan (Linux, estilo lsfg-vk)
   borderless)   │   sin copia a CPU)    │
                 └─────────┬────────────┘
                           │ F(n) como textura + timestamp de presentación
                           ▼
                 ┌──────────────────────┐
                 │  Frame pacer / cola  │  decide CUÁNTOS frames generar entre F(n-1) y F(n)
                 │  (modo ×N o adaptive)│  y en qué t (RIFE-m ⇒ t arbitrario, p.ej. 1/3, 2/3)
                 └─────────┬────────────┘
                           │ (F(n-1), F(n), t_k, flags: scene_cut, hud_mask)
                           ▼
                 ┌──────────────────────┐
                 │  Motor de inferencia │  TensorRT fp16, tamaños fijos, CUDA graph
                 │  RIFE-m (nuestro)    │  entrada RGB → flujo/máscara a flow-scale → warp a res. completa
                 └─────────┬────────────┘
                           │ frames generados (texturas)
                           ▼
                 ┌──────────────────────┐
                 │  Presentador         │  ventana overlay (DXGI swapchain, flip model) o
                 │  (VRR / vsync /      │  presentación dentro de la capa Vulkan
                 │   pacing uniforme)   │
                 └──────────────────────┘
                 
   UI (config, overlay de stats, perfiles por juego)  ──  proceso aparte, IPC
```

### 2.1 Captura

* **Windows**: dos vías, igual que LS.  *DXGI Desktop Duplication* (baja latencia, captura
  el monitor entero, necesita fullscreen/borderless; falla con algunos overlays) y
  *Windows Graphics Capture* (WGC; captura una ventana concreta, más compatible, ~1 frame
  más de latencia, borde amarillo en versiones antiguas de Windows).  Ambas entregan una
  textura D3D11 en GPU.  Hay que hacer *interop* D3D11 → CUDA (`cudaGraphicsD3D11RegisterResource`)
  para dársela a TensorRT sin pasar por CPU.
* **Linux**: la vía "correcta" es una **capa Vulkan implícita** que intercepta
  `vkQueuePresentKHR`, coge la imagen del swapchain, genera y presenta N imágenes.  Es lo
  que hace lsfg-vk (licencia CC BY-NC-ND → no podemos copiar su código, sí su idea).  Es
  más complejo que la captura de pantalla pero da frame pacing perfecto y latencia mínima.
  Alternativa v0: captura PipeWire/xdg-portal (como OBS) → mucho más lento y con latencia.
* **Timestamps**: cada frame capturado lleva el instante de presentación.  El pacer los
  necesita para saber si el juego va a 58 o a 61 fps.

### 2.2 Frame pacer (el corazón, y lo que la gente nota)

Entradas: cola de frames reales con timestamps; frecuencia de refresco del monitor;
modo (×2, ×3, ×4, adaptive→target Hz).  Salidas: lista de (t_k) a generar por intervalo
y el instante en el que presentar cada uno.

Reglas:

1. **Modo ×N**: entre F(n−1) y F(n) generar N−1 frames en t = k/N.  Con RIFE-m se generan
   directamente (no recursivo); con RIFE clásico sólo ×2/×4.  Ésta es la razón principal
   del run 2.
2. **Adaptive** (target T Hz, juego a f fps variable): por intervalo real Δ = t(n) − t(n−1)
   se generan `round(Δ·T) − 1` frames, en t uniformes.  Ej: 47 fps → 144 Hz ⇒ 2 frames
   unas veces y 3 otras.  RIFE-m lo hace natural; RIFE clásico no puede.
3. **Queue Target** (0/1/2): cuántos frames reales retener antes de presentar.  0 = mínima
   latencia, sensible a jitter; 2 = suave pero +2 frames de lag.  Empezamos en 1.
4. **Cortes de escena**: si `mean|F(n) − F(n−1)|` (a 64×36) > umbral ⇒ no interpolar,
   repetir F(n−1) (misma lógica que `demo_video.py`).  Un modelo grande de flujo también
   "sabe" que el flujo es basura (máscara ≈ 0.5 en todas partes); podemos usar la
   confianza del propio RIFE como segundo criterio.
5. **Desactivación automática**: si el juego cae bajo ~20 fps o el motor de inferencia
   se pasa del presupuesto 3 frames seguidos, dejar pasar frames reales sin generar (LS
   corta bajo 10 fps).  Mejor 40 fps limpios que 80 con tirones.
6. **Cap del juego**: la app no puede capear el juego; documentar que se use RTSS / el
   límite del driver a un divisor exacto del refresco (60 → 120, 72 → 144, 48 → 144).

### 2.3 Motor de inferencia

* Exportación **PyTorch → ONNX → TensorRT** con perfil de tamaños fijos por resolución
  (1080p, 1440p, 4K) y por flow scale.  El `grid_sample` de warplayer exporta con opset ≥16.
  El `F.interpolate` del flujo y el padding a múltiplos de 32 se hacen en el grafo.
* **Dos grafos**: *IFNet a flow-scale* (entrada 2 frames reducidos → flujo + máscara) y
  *fusión a resolución completa* (upsample de flujo ×1/scale, warp de los frames
  originales, blend con máscara, [refine opcional]).  Así el coste del flujo escala con
  `flow_scale²` y sólo el warp final va a resolución completa (barato: es memoria).
* **Precisión**: fp16 en todo; el flujo en píxeles a 1080p llega a ±100 px, cabe en fp16
  (rango 65k, precisión 0.05 px en 100 → suficiente).  INT8 sólo para la parte convolucional
  del IFNet, con calibración sobre frames de juegos.
* **Batching de t**: en modo ×3 los dos frames (t=1/3, 2/3) comparten I0, I1; RIFE-m los
  calcula por separado (el t entra en todos los bloques).  Se pueden lanzar en un batch
  de 2 → mejor ocupación de GPU que dos lanzamientos.
* **Colas y streams**: la inferencia en su propio CUDA stream con prioridad alta; el
  juego usa la GPU a la vez, así que medir *siempre* con el juego corriendo, no aislado.

### 2.4 Presentación

* Windows: ventana overlay *borderless* encima del juego, swapchain **flip model**
  (`DXGI_SWAP_EFFECT_FLIP_DISCARD`), `Present(1, 0)` con vsync o `ALLOW_TEARING` con VRR.
  Max Frame Latency = 1–2 (`SetMaximumFrameLatency`).  El frame real y los generados se
  presentan con espaciado uniforme = 1/(f·N).  Con G-Sync/FreeSync es mucho más fácil:
  presentar cuando esté listo y el monitor sigue.
* Linux (capa Vulkan): reemplazar el `vkQueuePresentKHR` del juego por N presentes con
  `VK_KHR_present_wait`/`present_timing` cuando esté disponible.

### 2.5 HUD, texto y cursor

El problema visible nº 1 de LS: el HUD estático "tiembla" al interpolarse con el fondo
en movimiento, y el texto que scrollea se emborrona.  Opciones (de más simple a mejor):

1. **Nada** (v1).  Documentar.
2. **Máscara estática aprendida en runtime**: píxeles cuya varianza temporal en los
   últimos ~2 s es ≈ 0 (minimapa, barras) se copian directamente de F(n−1) en vez de
   interpolar.  Barato, ayuda mucho en HUD fijo.
3. **Fine-tune del modelo con datos de juego** que incluyan HUD (ver §4): el modelo aprende
   que las regiones sin movimiento no se deforman.
4. El cursor del ratón nunca se interpola (se dibuja encima, como hace LS).

## 3. Tecnología y lenguaje

| Capa | Opción v1 (rápido de construir) | Opción final |
|---|---|---|
| Captura + presentación Windows | **Python + `windows-capture`/`d3dshot` + PyTorch**: sirve para el *prototipo* y para medir, no para el producto (latencia CPU↔GPU) | **C++ (D3D11 + CUDA interop + TensorRT)** o **Rust** (`windows-rs`, `cudarc`, TensorRT vía FFI) |
| Motor | PyTorch fp16 + `torch.compile` | TensorRT engine (.plan) cargado en C++ |
| UI | Argumentos CLI + overlay de texto | Qt / Tauri / egui; perfiles por ejecutable |
| Linux | — | Capa Vulkan en C++ (mirar `VK_LAYER_MESA_overlay` como plantilla de capa, licencia MIT) |

Plan de hitos (sin código aún, sólo orden):

1. **Bench offline**: exportar RIFE-m (run 2) a ONNX/TRT; medir ms a 1080p/1440p con
   flow scale 1.0/0.75/0.5, con/sin refine, fp16/INT8, en la 3060.  Decide todo lo demás.
2. **Prototipo Python en Windows**: captura WGC → PyTorch/TRT (via `torch-tensorrt` o
   `polygraphy`) → ventana overlay.  Objetivo: ver el ×2 funcionando en un vídeo a pantalla
   completa y en un juego ligero.  Medir latencia con cámara a 240 fps o con `PresentMon`.
3. **Frame pacer + adaptive + cortes de escena** en el prototipo.
4. **Reescritura del núcleo en C++/Rust** con D3D11↔CUDA interop y TRT; UI aparte.
5. **Capa Vulkan para Linux**.

## 4. El modelo: qué le falta para juegos (roadmap de entrenamiento)

Vimeo90K es vídeo real: cámara con motion blur, sin HUD, movimientos moderados.  Un
juego tiene bordes perfectamente nítidos, sin blur, HUD estático, movimientos de cámara
de 100+ px/frame a 60 fps, partículas, y cortes al abrir menús.  Plan:

1. **Run 2 (en curso)**: RIFE-m + septuplet + scale-aug + EMA.  Da t arbitrario y mejora
   movimiento grande.  Métrica: `evaluate.py --multi_t` y drop-frame eval en clips de
   `docs/demo_sources.txt` (incluye gameplay).
2. **Dataset de gameplay propio**: grabar con OBS a **120 o 240 fps** (juegos que lo
   permitan; o *rendering* offline a alta tasa), 1080p, sin compresión agresiva
   (`x264 --qp 0` o ProRes).  Cada frame a 120 fps es GT de los frames a 60 → septupletes
   reales con t exacto.  Mezclar 5–10 juegos distintos (3D, 2D pixel-art, racing, texto).
   Fine-tune desde run 2 con `--init_from`, LR bajo, mezclando Vimeo (para no olvidar).
3. **Pérdida perceptual**: hzwer recomienda optimizar LPIPS y no PSNR para calidad visual;
   añadir `λ·LPIPS` en el fine-tune de juegos (no en el benchmark).
4. **Modelo "performance"**: anchuras reducidas + sin refine, entrenado con distilación
   desde el modelo grande (el grande hace de teacher en lugar del bloque privilegiado).
5. **Evaluación en vivo**: la app en modo *benchmark* graba frames reales pares/impares y
   calcula PSNR de los impares reconstruidos (drop-frame eval en tiempo real).  Permite
   comparar versiones del modelo en el mismo juego sin GT externo.

## 5. Riesgos y decisiones abiertas

| Riesgo | Mitigación |
|---|---|
| No llegar a ≤3 ms a 1080p en 3060 | Flow scale 0.5 + sin refine + INT8.  Si aun así no, objetivo 1080p@30→60 (16 ms de presupuesto) que sigue siendo útil |
| `grid_sample` lento / no fusionado en TRT | Implementar warp como plugin CUDA propio (kernel bilineal trivial) |
| Anti-cheat detecta el overlay/capa | Igual que LS: no se inyecta en el proceso del juego en Windows (captura externa), así que es seguro; la capa Vulkan en Linux sí se carga en el proceso → documentar |
| HDR / 10-bit | Fuera de alcance v1; capturar en SDR |
| Licencias | Nuestro código y pesos son propios (entrenados desde cero).  No usar código ni pesos de RIFE oficial (MIT, pero el objetivo es educativo) ni de lsfg-vk (NC-ND) |
| Extrapolación (latencia 0) | Investigar después de la v1; requiere otro modelo (entrada F(n−1), F(n) → F(n+½)) |

## 6. Referencias

* RIFE: Huang et al., *Real-Time Intermediate Flow Estimation for Video Frame Interpolation*, arXiv:2011.06294 (ECCV 2022).
* hzwer, Practical-RIFE issue #124 (consejos: scale-aug, LPIPS, más entrenamiento, septuplet).
* Lossless Scaling (Steam) — guía de ajustes LSFG 3.1: Flow Scale, Performance mode, Queue Target, Max Frame Latency, capturas DXGI/WGC.
* lsfg-vk — https://lsfg-vk.dev (capa Vulkan, CC BY-NC-ND 4.0; sólo como referencia de arquitectura).
* Microsoft Docs: Windows Graphics Capture, DXGI Desktop Duplication, `IDXGISwapChain2::SetMaximumFrameLatency`, `DXGI_PRESENT_ALLOW_TEARING`.
* NVIDIA TensorRT: ONNX `GridSample` (opset 16), INT8 PTQ, CUDA Graphs.
* Intel *ExtraSS* (SIGGRAPH Asia 2023) — extrapolación de frames como alternativa sin latencia.
