# RIFE desde cero — Interpolación de frames de vídeo (VFI) educativa

Implementación propia en PyTorch de **RIFE** (Real-Time Intermediate Flow Estimation,
[arXiv:2011.06294](https://arxiv.org/abs/2011.06294), ECCV 2022), escrita desde la
especificación técnica: sin pesos preentrenados ni código del repositorio oficial.
El objetivo es **aprender**: cada módulo lleva comentarios explicando el *por qué* de cada
decisión, no sólo el *qué*.

```
python inference.py --ckpt best.pth --input in_30fps.mp4 --output out_60fps.mp4
```

---

## Índice

1. [Arquitectura explicada](#1-arquitectura-explicada)
2. [Estructura del proyecto](#2-estructura-del-proyecto)
3. [Instalación](#3-instalación)
4. [Smoke test (valida el pipeline en tu 3060)](#4-smoke-test)
5. [Entrenar](#5-entrenar)
6. [Evaluar](#6-evaluar)
7. [Inferencia en vídeo](#7-inferencia-en-vídeo)
8. [Resultados esperados](#8-resultados-esperados)
9. [Decisiones de diseño y trade-offs](#9-decisiones-de-diseño-y-trade-offs)
10. [RIFE-m: timestep arbitrario y run 2](#10-rife-m-timestep-arbitrario-y-run-2)
11. [Demo con vídeos reales (YouTube / locales)](#11-demo-con-vídeos-reales)
12. [Hacia una app tipo Lossless Scaling](#12-hacia-una-app-tipo-lossless-scaling)
13. [Roadmap](#13-roadmap)

---

## 1. Arquitectura explicada

### El problema

Dados dos frames consecutivos I₀ e I₁, generar el frame intermedio I_t (t = 0.5).
La dificultad: los píxeles se mueven, y no sabemos cuánto ni hacia dónde.

### La idea de RIFE: estimar el flujo *intermedio* directamente

Los métodos anteriores estimaban el flujo óptico F₀→₁ (con PWC-Net, etc.) y luego lo
"desplazaban" a t mediante *forward warping*, lo que genera agujeros y colisiones.  RIFE
entrena una red (**IFNet**) que predice **directamente** F_{t→0} y F_{t→1}: para cada píxel
del frame que queremos crear, "de dónde viene" en I₀ y en I₁.  Con esos flujos basta un
*backward warping* (`grid_sample`), que es denso, diferenciable y sin agujeros.

```
                    ┌──────────────────────────────────────────────────────┐
  I₀ ──┐            │ IFNet (coarse-to-fine)                               │
       ├──► block0 (1/4, c=240) ──► block1 (1/2, c=150) ──► block2 (1/1, c=90) ──► F_{t→0}, F_{t→1}, M
  I₁ ──┘            │     residual ↑              residual ↑                │
                    └──────────────────────────────────────────────────────┘
                                          │
                 w₀ = warp(I₀, F_{t→0})   │   w₁ = warp(I₁, F_{t→1})
                                          ▼
                       Î⁰_t = M ⊙ w₀ + (1−M) ⊙ w₁          ← fusión con máscara de oclusión
                                          │
          Contextnet(I₀, F_{t→0}), Contextnet(I₁, F_{t→1})   ← features piramidales warpeadas
                                          │
                                 RefineUNet ──► Δ ∈ [−1, 1]
                                          ▼
                              Î_t = clamp(Î⁰_t + Δ, 0, 1)
```

#### IFBlock (`model/ifnet.py`)

Cada bloque: 2 convs 3×3 stride 2 (↓4) → 8 convs 3×3 con skip residual → ConvTranspose 4×4
stride 2 → 5 canales (4 de flujo, 1 de logit de máscara) → interpolación bilineal a
resolución completa.  PReLU en todas las capas.

* **Coarse-to-fine con escalas (4, 2, 1)**: el bloque 0 ve la imagen a 1/4 y dentro baja a
  1/16; con 8 convs 3×3 su campo receptivo cubre ~270 px de la imagen original → captura
  movimientos grandes.  Los bloques 1 y 2 refinan a resoluciones mayores.
* **Predicción residual**: cada bloque recibe (I₀, I₁, warp(I₀), warp(I₁), M, flujo previo) y
  predice una *corrección*.  Si el flujo previo es correcto, warp(I₀) ≈ warp(I₁) y el
  residual → 0.  El bloque no tiene que redescubrir el movimiento grueso.
* **Pesos compartidos entre direcciones**: un mismo bloque predice F_{t→0} y F_{t→1} a la vez;
  la simetría temporal la aprende con la augmentación de inversión temporal.

#### Teacher privilegiado (destilación)

Durante el entrenamiento, un cuarto IFBlock idéntico al bloque 2 recibe **además el frame
real I_t**.  Con esa información puede estimar flujos casi perfectos.  El estudiante imita
esos flujos con una pérdida L1 **enmascarada por píxel**: sólo donde el error de
reconstrucción del estudiante supera al del teacher + 0.01.  Si el teacher se equivoca en un
píxel, no forzamos al estudiante a copiarlo.  El teacher se descarta en inferencia (no
tenemos I_t: ¡es lo que queremos generar!).

#### Warping (`model/warplayer.py`)

`out[y, x] = img[y + dy, x + dx]` con interpolación bilineal.  Se construye una malla de
coordenadas normalizada a [−1, 1], se le suma el flujo (escalado por 2/(W−1)) y se llama a
`F.grid_sample`.  Diferenciable respecto a la imagen y al flujo.

#### Contextnet + RefineUNet (`model/refine.py`)

La fusión deja artefactos en oclusiones y bordes.  El **Contextnet** extrae una pirámide de
features (4 niveles) de cada frame y las warpea con el flujo reescalado; el **UNet** recibe
imágenes, warps, máscara, flujo y esos contextos, y predice un residual acotado con `tanh`.
Predecir el residual (no la imagen) hace el entrenamiento estable desde el paso 0.

#### Pérdidas (`model/loss.py`)

```
L = L_rec(Î_t, I_t) + L_rec(Î_teacher, I_t) + 0.01 · L_distill
```

* **LapLoss**: L1 sobre los 5 niveles de la pirámide laplaciana, ponderados 2^k.  Penaliza
  errores estructurales (bordes desplazados) más que una L1 plana.  Se calcula en fp32.
* **L_rec_teacher**: sin ella el teacher no tendría señal de entrenamiento.
* **L_distill**: sobre los 3 bloques del estudiante (supervisión intermedia).

Tamaño: **10.07 M parámetros** en el estudiante (+0.64 M el teacher).

---

## 2. Estructura del proyecto

```
model/
  warplayer.py     backward warping (grid_sample) con malla cacheada
  ifnet.py         IFBlock, IFNet coarse-to-fine, bloque teacher
  refine.py        Contextnet + RefineUNet
  loss.py          LapLoss, distillation_loss, psnr, ssim
  rife.py          RIFE: forward de entrenamiento (con pérdidas) e inference(); RIFE-m (arbitrary_time);
                   load_state_dict_compat (convierte checkpoints RIFE → RIFE-m)
dataset.py         Vimeo90K triplet **y septuplet** (t aleatorio para RIFE-m, scale-aug) y SyntheticTripletDataset
train.py           DDP (torchrun), AMP fp16, warmup+coseno, checkpoint atómico, --resume exacto,
                   --arbitrary_time, --init_from (fine-tune), --ema, --scale_aug
evaluate.py        PSNR/SSIM en test set, baseline, paneles de flujo por bloque; --multi_t (×6 sobre septuplet)
inference.py       vídeo ×2^k (recursivo) o --multi N (RIFE-m, directo), tiling con blending, modo 2 imágenes
demo_video.py      descarga YouTube/recorta, interpola, side-by-side, slow-mo, drop-frame eval sin GT
utils.py           carga de checkpoints (detecta RIFE-m, usa pesos EMA), visualización de flujo (HSV)
tests/test_core.py 17 tests unitarios del núcleo (incl. RIFE-m y EMA)
scripts/make_subset.py         subset de Vimeo90K
scripts/make_fake_septuplet.py septuplet sintético minúsculo para smoke tests en CPU
docs/KAGGLE.md     guía paso a paso para 2×T4 (run 1 y run 2)
docs/kaggle_notebook.ipynb  notebook del run 2 (RIFE-m fine-tune)
docs/demo_sources.txt       qué vídeos probar y por qué
docs/APP_DESIGN.md          diseño de la app de generación de frames en tiempo real (sin código)
docs/results/               curva y evaluación del run 1
```

---

## 3. Instalación

```bash
# PyTorch según tu CUDA: https://pytorch.org/get-started/locally/  (p.ej. cu121 para la 3060)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python tests/test_core.py      # 12 tests, ~5 s en CPU
```

Dataset: descarga `vimeo_triplet.zip` (33 GB) de http://toflow.csail.mit.edu/ y
descomprímelo.  La carpeta `vimeo_triplet/` debe contener `tri_trainlist.txt`,
`tri_testlist.txt` y `sequences/`.  Mirrors en Kaggle: ver `docs/KAGGLE.md`.

---

## 4. Smoke test

Antes de gastar cuota de Kaggle, valida el pipeline completo en la 3060 (~5-10 min):

```bash
# (a) Sin datos: tríos sintéticos con formas en movimiento lineal.
python train.py --synthetic --smoke --amp --out_dir runs/smoke_syn

# (b) Con Vimeo90K (512 tríos, 3 épocas, batch 8):
python train.py --data_root /data/vimeo_triplet --smoke --amp --out_dir runs/smoke_vimeo
```

`--smoke` fija: 3 épocas, 512 tríos de train, 64 de val, warmup 50, batch 8, logs cada 10
pasos.  Al final imprime la **comprobación de cordura**:

```
[val] epoch 2 psnr 27.1 ssim 0.81 (baseline promedio: 25.6 dB)
[SMOKE ✓] modelo 27.10 dB > baseline 25.60 dB
```

El baseline es `PSNR((I₀+I₁)/2, I_t)`, que en el test set de Vimeo90K da **25.6 dB medidos**
(el brief asumía 20-23 dB; esa cifra corresponde a datasets con más movimiento).  Si tras 3
épocas de smoke el modelo está **por debajo**, hay un bug (típicamente: signo del flujo, orden
(x, y) en el warp, normalización de datos, o gt desalineado por una augmentación mal
aplicada).  Referencia real: con el dataset completo la red pasa el baseline dentro de la
primera época (27.8 dB tras 1 603 pasos, ver §8); con 512 tríos y 3 épocas (~190 pasos)
espera 26-28 dB.  Si sólo llegas a 25.6-26 dB probablemente sea normal por tan pocos pasos:
sube a `--max_samples 2000` antes de sospechar un bug.

Con el dataset sintético el baseline es más alto (~25 dB, fondos planos) y el modelo lo
supera tras ~100 pasos.

Extras para depurar en local: `--ifnet_widths 64 48 32 --refine_c 8` (modelo ×10 más
pequeño), `--num_workers 0` (errores del dataset legibles), `--grad_checkpoint`.

---

## 5. Entrenar

### Una GPU (3060, 12 GB)

```bash
python train.py --data_root /data/vimeo_triplet --out_dir runs/rife_local \
    --epochs 50 --batch_size 12 --amp --num_workers 6 --save_every 500
```

Batch 12 con AMP cabe en 12 GB; con `--grad_checkpoint` cabe 16.  El LR se escala
automáticamente por `world_size/4` (con 1 GPU: 7.5e-5 → usa `--no_lr_scale` si quieres 3e-4).
Una época tarda ~20-25 min en la 3060 (en 2×T4 con AMP: 10.1 min medidos).

### Kaggle 2×T4 (DDP)

```bash
torchrun --nproc_per_node=2 train.py --data_root $DATA --out_dir /kaggle/working/checkpoints \
    --epochs 60 --batch_size 16 --amp --num_workers 4 --time_limit 11.2 --save_every 500
# siguiente sesión:
torchrun --nproc_per_node=2 train.py ... --resume /kaggle/input/<version>/checkpoints/last.pth
```

Guía completa (dataset, notebook, persistencia de checkpoints entre sesiones):
**[docs/KAGGLE.md](docs/KAGGLE.md)**.

### Receta (por defecto en `train.py`)

| | |
|---|---|
| Optimizador | AdamW, wd = 1e-3 (decay desacoplado; evita explosión de pesos → NaN) |
| LR | 3e-4 × world_size/4, warmup lineal 2000 pasos, coseno hasta 3e-6 |
| Batch | 16 por GPU, crops 224×224 |
| Augment | crop aleatorio, flip H/V, rotación 90°, inversión temporal |
| Precisión | AMP fp16 (T4 no soporta bf16); LapLoss en fp32 |
| Clip | norma 1.0 |
| Checkpoints | `last.pth` (cada N pasos + fin de época), `best.pth` (mejor PSNR val), `inference.pth` (sin teacher) |
| Opcionales (run 2) | `--arbitrary_time` (RIFE-m), `--init_from ckpt` (fine-tune, adapta RIFE→RIFE-m), `--ema 0.999`, `--scale_aug P SMIN SMAX`, `--max_gap` (septuplet), `--distill_weight` |

`--resume` restaura modelo, optimizador, scaler, step y **salta los batches ya consumidos de
la época en curso** (sampler determinista por época): reanudar es exacto.

---

## 6. Evaluar

```bash
python evaluate.py --ckpt runs/rife/best.pth --data_root /data/vimeo_triplet --amp \
    --vis_dir outputs_vis --n_vis 8 --out_json results.json
```

Imprime PSNR/SSIM promedio sobre los 3 782 tríos de test a 448×256, el baseline de promediar,
las 5 peores muestras y ms/frame.  En `outputs_vis/`:

* `*_panel.png`: I₀ | I₁ | GT | Pred // F_{t→0} | F_{t→1} | máscara | error×4
* `*_blocks.png`: flujo, máscara, fusión y error **tras cada IFBlock** → se ve cómo el
  bloque 0 captura el movimiento grueso y los siguientes afinan bordes.

Los flujos se colorean en HSV (tono = dirección, brillo = magnitud), la convención habitual.

---

## 7. Inferencia en vídeo

```bash
# 30 → 60 fps
python inference.py --ckpt runs/rife/inference.pth --input in.mp4 --output out.mp4 --fp16
# ×4 (recursivo: entre A y B genera M, luego entre A-M y M-B)
python inference.py --ckpt ... --input in.mp4 --output out.mp4 --exp 2 --fp16
# 4K: tiling con solape + escalas mayores para ver movimientos grandes
python inference.py --ckpt ... --input in4k.mp4 --output out.mp4 --tile 1024 --tile_overlap 128 --scales 8 4 2 --fp16
# dos imágenes
python inference.py --ckpt ... --img0 a.png --img1 b.png --output mid.png
# RIFE-m: ×3 directo (24 → 72 fps), cualquier factor, sin recursión (ver §10)
python inference.py --ckpt runs/rifem/inference.pth --input in.mp4 --output out.mp4 --multi 3 --fp16
```

Velocidad esperada en la 3060 con `--fp16`: ~25-30 fps de entrada a 720p, ~10-12 a 1080p
(sin tiling).  OpenCV no conserva el audio; para reinyectarlo:
`ffmpeg -i out.mp4 -i in.mp4 -c copy -map 0:v -map 1:a out_audio.mp4`.

---

## 8. Resultados

### Run 1 — Kaggle 2×T4, 60 épocas, una sola sesión (~10 h)

Configuración exacta: `torchrun --nproc_per_node=2 train.py --epochs 60 --batch_size 16 --amp
--num_workers 4` (LR efectivo 1.5e-4 = 3e-4 × 2/4, warmup 2000, coseno → 1.5e-6).
**10.1 min/época**, 96 180 pasos, sin NaN ni reinicios.  Datos: `docs/results/`.

| Vimeo90K test (448×256) | PSNR | SSIM |
|---|---|---|
| Baseline (I₀+I₁)/2 | 25.6 dB | 0.77 |
| Época 1 | 27.8 dB | 0.819 |
| Época 2 | 30.9 dB | 0.900 |
| Época 10 | 33.3 dB | 0.945 |
| Época 30 | 34.1 dB | 0.955 |
| **Época 60 (best.pth), 3 782 tríos** | **34.33 dB** | **0.957** |
| Misma red, 500 primeros tríos (`eval.json`) | 34.40 dB | 0.958 |
| RIFE paper, 300 épocas, 4 GPUs | 35.6 dB | 0.980 |

Curva de validación completa en `docs/results/val_curve_run1.csv`.  Observaciones:

* **La curva es logarítmica**: +3 dB en la época 2, +5.5 dB en la 10, y sólo +1 dB más en las
  50 restantes.  El último tramo (ep 50→60, LR < 1e-5) aporta 0.02 dB: el schedule coseno ya
  estaba "apagado".  Para seguir mejorando hay que entrenar con un plan más largo desde el
  inicio (`--epochs 150`), no reanudar este.
* **Distribución por muestra** (500 tríos): p5 = 27.0, mediana = 34.0, p95 = 42.7 dB.  43/500
  muestras por debajo de 28 dB, todas con movimiento grande u oclusiones fuertes
  (peor: `00003/0115` a 22.8 dB).  Ahí está el margen de mejora.
* **La cascada funciona como debe** (`docs/results/blocks_00001_0402.png`): la fusión mejora
  bloque a bloque (37.3 → 38.1 → 40.2 dB en esa muestra) y la máscara se activa sólo en
  bordes de oclusión.
* Inferencia: **7.4 ms/frame** a 448×256 en T4 con fp16 (≈135 fps).
* Baseline real de promediar: **25.6 dB** en Vimeo90K, no 20-23 como asumía el brief (esa
  cifra corresponde a datasets con más movimiento, p.ej. UCF101/SNU-FILM hard).

### Diferencia con el paper (−1.3 dB) y cómo cerrarla

| Causa probable | Impacto estimado | Acción |
|---|---|---|
| 60 vs 300 épocas | ~0.6-0.8 dB | `--epochs 150-200` en 2-3 sesiones con `--resume` |
| LR efectivo 1.5e-4 (2 GPUs) vs 3e-4 (4 GPUs) con batch total 32 vs 64 | ~0.2 dB | `--no_lr_scale` (LR 3e-4 con batch 32 es estable en RIFE) |
| UNet c=16 vs implementación oficial (c=32 aprox.) | ~0.1-0.2 dB | `--refine_c 32` |
| Sin fine-tuning final a LR bajo con crops mayores | ~0.1 dB | última sesión con `--crop_size 256` |

Plan sugerido para el run 2 (≈30 h, 3 sesiones): `--epochs 180 --no_lr_scale --refine_c 32
--time_limit 11.2`, reanudando entre sesiones.  Objetivo: **≥ 35.0 dB**.

---

## 9. Decisiones de diseño y trade-offs

| Decisión | Alternativa | Por qué la elegida |
|---|---|---|
| **LapLoss** | L1 + pérdida FFT / Charbonnier | Sin hiperparámetros, estable, es la del paper.  L1+FFT afila texturas pero puede introducir ringing y es más sensible a pesos. |
| **UNet c=16** (~1.5 M params) | c=32-64 | El grueso de la calidad viene del flujo; el UNet grande da +0.1-0.2 dB a costa de ×2 tiempo.  `--refine_c` lo cambia. |
| **`tanh` en el residual** | `clamp` | tanh acota suavemente y mantiene gradiente; clamp lo anula fuera de [−1, 1]. |
| **AdamW wd=1e-3** | Adam + L2 | Decay desacoplado actúa igual en todos los pesos; con Adam clásico la L2 se diluye por √v en pesos con gradientes grandes y el flujo puede explotar. |
| **fp16 + GradScaler** | bf16 | T4 (Turing) no tiene bf16.  La 3060 sí; puedes cambiarlo en `train.py` si sólo entrenas en local. |
| **Backward warping** | Forward (splatting) | Denso, sin agujeros, trivial con `grid_sample`. |
| **padding_mode="border"** | "zeros" | Bordes negros en los márgenes cuando el flujo apunta fuera → artefactos visibles. |
| **Scheduler por paso** | por época | Reanudar sólo necesita el contador de pasos. |
| **Sampler reanudable** | perder/repetir la época parcial | Kaggle corta sesiones a mitad de época; sin esto pierdes hasta 15 min de cómputo por sesión. |
| **Tiling con rampa lineal** | tiles sin solape | Evita costuras.  Limita el movimiento máximo a ~tile/4 → tiles grandes para 4K. |

---

## 10. RIFE-m: timestep arbitrario y run 2

RIFE clásico sólo sabe generar el frame en **t = 0.5**.  Para ×4 hay que recurrir
(A→M→B, luego A→M₁→M, M→M₂→B) y para ×3 no hay forma.  **RIFE-m** (sección 4.3 del paper)
añade a la entrada de cada IFBlock un canal constante con el valor de *t*:

```
IFBlock0:   cat(I0, I1, t)                              6+1 = 7 canales
IFBlock1/2: cat(I0, I1, t, warp0, warp1, mask_logit)   17+1 = 18
teacher:    cat(..., gt)                                20+1 = 21
```

Entrenarlo requiere GT en t ≠ 0.5 → **Vimeo90K septuplet** (7 frames): se muestrean tres
índices ordenados (i₀, i_t, i₁) y t = (i_t−i₀)/(i₁−i₀); la inversión temporal implica
t → 1−t.  Peso de distilación 0.005 (0.01 en RIFE).  Coste: RIFE-m es ~0.1 dB peor que RIFE
en t = 0.5, a cambio de cualquier t sin acumular error.

**Fine-tune en vez de reentrenar.**  `RIFE.load_state_dict_compat` copia un checkpoint
RIFE a un modelo RIFE-m insertando el canal *t* con peso **cero**: el modelo convertido es
numéricamente idéntico al original para cualquier t (test `test_convert_rife_to_rifem_is_identity`)
y sólo tiene que aprender a *usar* t.  Así el run 2 arranca desde los 34.3 dB del run 1.

```bash
# smoke en CPU con septuplet sintético
python scripts/make_fake_septuplet.py runs/fake_sep
python train.py --data_root runs/fake_sep --ifnet_widths 16 12 8 --refine_c 4 --crop_size 64 --batch_size 4 \
    --epochs 1 --num_workers 0 --out_dir runs/t_rife
python train.py --data_root runs/fake_sep --ifnet_widths 16 12 8 --refine_c 4 --crop_size 64 --batch_size 4 \
    --arbitrary_time --init_from runs/t_rife/best.pth --ema 0.99 --scale_aug 0.5 0.5 1.5 --epochs 2 --num_workers 0 \
    --out_dir runs/t_rifem
# ¿usa el canal t?  ×6 sobre im1→im7: PSNR por t
python evaluate.py --ckpt runs/t_rifem/best.pth --data_root runs/fake_sep --multi_t
```

Receta del run 2 (Kaggle, `docs/kaggle_notebook.ipynb`, detalles en `docs/KAGGLE.md` §9):
septuplet + `--arbitrary_time --init_from best.pth(run1) --ema 0.999 --scale_aug 0.5 0.5 1.5 --lr 1e-4 --no_lr_scale`,
90 épocas en 2 sesiones.  Además de RIFE-m incorpora dos consejos del autor de RIFE
(hzwer, Practical-RIFE #124): *scale augmentation* para movimiento grande y más tiempo de
entrenamiento; y EMA de pesos (se valida y exporta con los pesos EMA).

---

## 11. Demo con vídeos reales

```bash
pip install yt-dlp   # y ffmpeg en el PATH
python demo_video.py --ckpt runs/rife/inference.pth --source "https://youtu.be/XXXX" --start 00:01:00 --duration 8 \
    --height 720 --fp16 --out_dir outputs/demo1
python demo_video.py --ckpt ... --sources docs/demo_sources.txt --out_dir outputs/batch   # varios clips
python demo_video.py --ckpt runs/rifem/inference.pth --source pelicula_24fps.mp4 --multi 3 --out_dir outputs/x3
```

Por clip genera `input.mp4`, `interp_xN.mp4`, `sidebyside.mp4` (original repitiendo frames |
interpolado), `slowmo.mp4` y **`dropframe_eval.json`**: se tiran los frames impares, se
reconstruyen desde los pares y se mide PSNR/SSIM contra los reales (métrica objetiva sin GT
externo, pesimista porque el movimiento es el doble).  Detecta cortes de escena y duplica el
frame en vez de interpolar.  `docs/demo_sources.txt` explica qué tipo de vídeos probar
(fácil/difícil: naturaleza, deportes, animación 2D, gameplay con HUD, texto en scroll, cámara
lenta real como GT).

---

## 12. Hacia una app tipo Lossless Scaling

El objetivo final es una aplicación que capture lo que un juego presenta, genere frames
intermedios y los presente en tiempo real (×2/×3/adaptativo), sin tocar el juego.
`docs/APP_DESIGN.md` recoge el diseño completo **sin código**: presupuesto (≤ 3 ms por
frame a 1080p en una 3060 ⇒ TensorRT fp16/INT8 + *flow scale* + modo performance),
arquitectura (captura DXGI/WGC o capa Vulkan, *frame pacer* con modo adaptativo y cola
configurable, motor TRT, presentación flip-model/VRR), manejo de HUD y cortes, hitos de
ingeniería y roadmap del modelo (dataset de gameplay a 120/240 fps, LPIPS, modelo lite
destilado).  RIFE-m es el prerequisito: el pacer necesita generar el frame en el *t* exacto.

---

## 13. Roadmap

- [x] Núcleo: warp, IFNet coarse-to-fine, teacher, Contextnet+UNet, LapLoss, destilación
- [x] Dataset Vimeo90K + sintético, augmentación
- [x] train.py: DDP, AMP, warmup+coseno, checkpoint atómico, resume exacto, time limit
- [x] evaluate.py con visualización por bloque; inference.py con tiling y ×2^k
- [x] Guía Kaggle + notebook
- [x] Run 1 en Kaggle: 60 épocas / 1 sesión → **34.33 dB / 0.957** (curva en `docs/results/`)
- [x] **RIFE-m**: canal t en los 4 IFBlocks, dataset septuplet, conversión RIFE→RIFE-m, `--multi N`, eval `--multi_t`
- [x] Fine-tune (`--init_from`), EMA de pesos, scale augmentation
- [x] `demo_video.py` (YouTube/local, drop-frame eval) + lista de vídeos a probar
- [x] `docs/APP_DESIGN.md`: diseño de la app de frame generation en tiempo real
- [ ] **Run 2 en Kaggle**: RIFE-m fine-tune sobre septuplet (notebook listo) → objetivo: t=0.5 ≥ run 1 y PSNR plano en `--multi_t`
- [ ] Probar run 1 vs run 2 con `demo_video.py` en los clips de `docs/demo_sources.txt`
- [ ] Verificar velocidad de inferencia en la 3060 (720p/1080p) con `inference.pth`
- [ ] Export a ONNX / TensorRT + benchmark a 1080p con flow scale (hito 1 de APP_DESIGN)
- [ ] Dataset de gameplay propio (OBS 120/240 fps) y fine-tune con LPIPS
- [ ] Modelo "lite" destilado para tiempo real
- [ ] Evaluación en UCF101 / Middlebury / SNU-FILM (bancos estándar de VFI)
- [ ] Experimentos: L1+FFT vs LapLoss, UNet c=32, sin teacher (medir cuánto aporta la destilación)
