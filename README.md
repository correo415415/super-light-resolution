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
10. [Roadmap](#10-roadmap)

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
  rife.py          RIFE: forward de entrenamiento (con pérdidas) e inference()
dataset.py         Vimeo90KTriplet (+augment) y SyntheticTripletDataset (smoke sin datos)
train.py           DDP (torchrun), AMP fp16, warmup+coseno, checkpoint atómico, --resume exacto
evaluate.py        PSNR/SSIM en test set, baseline, paneles de flujo por bloque
inference.py       vídeo ×2^k (recursivo), tiling con blending, modo 2 imágenes
utils.py           carga de checkpoints, visualización de flujo (HSV)
tests/test_core.py 12 tests unitarios del núcleo
scripts/make_subset.py  subset de Vimeo90K
docs/KAGGLE.md     guía paso a paso para 2×T4
docs/kaggle_notebook.ipynb
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
[val] epoch 2 psnr 27.8 ssim 0.88 (baseline promedio: 22.1 dB)
[SMOKE ✓] modelo 27.80 dB > baseline 22.10 dB
```

El baseline es `PSNR((I₀+I₁)/2, I_t)`, que en Vimeo90K da **~20-23 dB**.  Si tras 3 épocas
de smoke el modelo está **por debajo**, hay un bug (típicamente: signo del flujo, orden
(x, y) en el warp, normalización de datos, o gt desalineado por una augmentación mal
aplicada).  Con 512 tríos y 3 épocas el modelo debería estar en ~24-28 dB.

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
Una época tarda ~20-25 min en la 3060.

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
```

Velocidad esperada en la 3060 con `--fp16`: ~25-30 fps de entrada a 720p, ~10-12 a 1080p
(sin tiling).  OpenCV no conserva el audio; para reinyectarlo:
`ffmpeg -i out.mp4 -i in.mp4 -c copy -map 0:v -map 1:a out_audio.mp4`.

---

## 8. Resultados esperados

Vimeo90K test (448×256, PSNR promediado por imagen):

| Configuración | PSNR | SSIM |
|---|---|---|
| Baseline (I₀+I₁)/2 | ~20-23 dB | ~0.65 |
| Smoke test (512 tríos, 3 épocas) | 24-28 dB | ~0.85 |
| ~10 épocas completas (1 sesión Kaggle) | ~32-33 dB | ~0.96 |
| ~60 épocas (3-4 sesiones) | ~34-34.5 dB | ~0.975 |
| 300 épocas (paper, 4 GPUs) | **35.6 dB** | 0.980 |

La curva es muy pronunciada al principio (la mayor parte del PSNR se gana en las primeras
10 épocas) y luego lenta; el último dB cuesta ~200 épocas.  Con 30 h/semana, **60-100 épocas
en 2-4 sesiones** es un objetivo realista que debería darte 34-35 dB.

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

## 10. Roadmap

- [x] Núcleo: warp, IFNet coarse-to-fine, teacher, Contextnet+UNet, LapLoss, destilación
- [x] Dataset Vimeo90K + sintético, augmentación
- [x] train.py: DDP, AMP, warmup+coseno, checkpoint atómico, resume exacto, time limit
- [x] evaluate.py con visualización por bloque; inference.py con tiling y ×2^k
- [x] Guía Kaggle + notebook
- [ ] Verificar cifras del smoke test y velocidad en la 3060 con Vimeo90K real
- [ ] Primer entrenamiento en Kaggle (10 épocas) y publicar curva PSNR/época
- [ ] **RIFE-m**: timestep t arbitrario como canal de entrada → ×4/×8 sin recursión
- [ ] Evaluación en UCF101 / Middlebury / SNU-FILM (bancos estándar de VFI)
- [ ] Export a ONNX / TensorRT para inferencia en tiempo real
- [ ] Modelo "lite" (c = 120/75/45) para 1080p en tiempo real en la 3060
- [ ] Experimentos: L1+FFT vs LapLoss, UNet c=32, sin teacher (medir cuánto aporta la destilación)
