# Entrenar RIFE en Kaggle (2×T4) paso a paso

Kaggle da ~30 h/semana de GPU, sesiones de máximo 12 h y **borra el disco de trabajo al
terminar la sesión** salvo lo que quede en `/kaggle/working` al hacer *Save Version*.
Todo el flujo está diseñado alrededor de esas tres restricciones.

## 0. Resumen del flujo

```
 sesión 1: dataset (input) + código (input)  → entrena 11 h → guarda last.pth en /kaggle/working
                                                                    │  "Save Version"
 sesión 2: dataset + código + OUTPUT de la sesión 1 (input) → --resume → 11 h → guarda ...
 sesión N: ...
```

## 1. El dataset Vimeo90K

### Opción A (recomendada): usar un mirror ya existente en Kaggle

`vimeo_triplet.zip` oficial ocupa 33 GB; subirlo tú mismo es posible (el límite por dataset
es 100 GB) pero lento.  Hay mirrors públicos.  En el momento de escribir esto, buscando
`vimeo` en *Datasets* aparecen, entre otros:

| Dataset Kaggle | Tamaño | Comentario |
|---|---|---|
| `chenshu123/vimeo-triplet` | 34.7 GB | Tamaño idéntico al `vimeo_triplet` oficial descomprimido → probablemente el dataset completo |
| `ekalabyaghosh/vimeo-90k-video-dataset` | 34.7 GB | Idem |
| `mineras/vimeo90k-small` | 3.5 GB | Subset; útil para pruebas rápidas |

> Los nombres/dueños pueden cambiar.  Búscalos en https://www.kaggle.com/datasets?search=vimeo
> y comprueba que contengan `tri_trainlist.txt`, `tri_testlist.txt` y `sequences/`.

Añade el dataset al notebook con **Add Input → Datasets** y localiza la ruta:

```python
!find /kaggle/input -maxdepth 3 -name "tri_trainlist.txt"
# → p.ej. /kaggle/input/vimeo-triplet/vimeo_triplet/tri_trainlist.txt
DATA_ROOT = "/kaggle/input/vimeo-triplet/vimeo_triplet"   # la carpeta que CONTIENE tri_trainlist.txt
```

### Opción B: subir el dataset tú mismo

1. Descarga `vimeo_triplet.zip` de http://toflow.csail.mit.edu/ (33 GB).
2. Descomprímelo en local y súbelo con la CLI (mucho más fiable que el navegador):
   ```bash
   pip install kaggle           # y coloca ~/.kaggle/kaggle.json (Account → Create API token)
   mkdir vimeo_upload && cd vimeo_upload
   # mete dentro la carpeta vimeo_triplet/ (o el zip sin descomprimir: Kaggle lo extrae)
   kaggle datasets init -p .    # crea dataset-metadata.json; edita "title" e "id"
   kaggle datasets create -p . --dir-mode zip
   ```
3. Kaggle tarda ~1 h en procesarlo.  Después úsalo como en la opción A.

### Opción C: subset para pruebas

Si sólo quieres validar el pipeline en Kaggle antes de gastar cuota, crea un dataset con
las primeras 2 000 líneas de `tri_trainlist.txt` y sus carpetas (script: `scripts/make_subset.py`).

## 2. Subir el código

Dos alternativas:

* **Clonar desde GitHub** dentro del notebook (requiere *Internet: ON* en Settings):
  ```python
  !git clone https://github.com/correo415415/super-light-resolution.git /kaggle/working/rife
  ```
* **Subirlo como dataset** (funciona sin internet): comprime el repo sin `.git` ni `runs/`
  y súbelo como dataset `rife-code`.  Luego `!cp -r /kaggle/input/rife-code /kaggle/working/rife`.

## 3. Configurar el notebook

En **Settings** (panel derecho):

* Accelerator → **GPU T4 x2**
* Persistence → *Variables and Files* no sirve para GPU; usaremos Save Version.
* Internet → ON si clonas de GitHub.

## 4. Primera sesión de entrenamiento

Usa el notebook `docs/kaggle_notebook.ipynb` (o copia estas celdas):

```python
# --- Celda 1: setup ---
import os, glob
DATA_ROOT = glob.glob("/kaggle/input/*/vimeo_triplet")[0]  # ajusta si tu mirror tiene otra estructura
assert os.path.exists(f"{DATA_ROOT}/tri_trainlist.txt"), DATA_ROOT
!git clone -q https://github.com/correo415415/super-light-resolution.git /kaggle/working/rife
%cd /kaggle/working/rife
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

```python
# --- Celda 2: (opcional) smoke test de 3 minutos, comprueba que todo va ---
!torchrun --nproc_per_node=2 train.py --data_root {DATA_ROOT} --smoke --amp \
    --out_dir /kaggle/working/smoke --num_workers 2
```

```python
# --- Celda 3: entrenamiento real ---
OUT = "/kaggle/working/checkpoints"
!torchrun --nproc_per_node=2 train.py \
    --data_root {DATA_ROOT} --out_dir {OUT} \
    --epochs 60 --batch_size 16 --amp --num_workers 4 \
    --time_limit 11.2 --save_every 500
```

Notas sobre los flags:

* `--time_limit 11.2` → el script para limpiamente a las 11 h 12 min y guarda `last.pth`.
  Deja margen para que el notebook termine y Kaggle empaquete el output antes del límite
  de 12 h.  **Sin esto pierdes la sesión entera si Kaggle la mata.**
* `--epochs`: fija el TOTAL de épocas del plan (determina el schedule de LR).  Con 2×T4,
  batch 16×2 y AMP, una época de Vimeo (1 603 pasos) tarda **10.1 min medidos** → ≈ 65 épocas
  por sesión de 11 h.  El run 1 (60 épocas, 34.33 dB) cupo en una sola sesión.  El paper usa
  300 épocas; con tu cuota (30 h/semana ≈ 2.5 sesiones) un plan realista es **150-180 épocas
  en 3 sesiones** (ver resultados en el README).
  No cambies `--epochs` entre sesiones o el schedule de LR se desalinea.
* `--save_every 500` → `last.pth` se refresca cada ~4 min.
* `--num_workers 4`: Kaggle da 4 vCPU; con 2 procesos DDP son 4 workers por proceso = 8
  hilos.  Si ves `DataLoader` como cuello de botella (GPU util < 80 %), baja a 2.
* Si aparece OOM: añade `--grad_checkpoint` (ahorra ~35 % de VRAM, +30 % tiempo) o
  `--batch_size 12`.

Cuando termine: **Save Version → Save & Run All (Commit)**… ¡NO!  Eso re-ejecuta todo.
Usa **Save Version → Quick Save** con *"Save output"*: guarda el estado actual de
`/kaggle/working` como output de la versión.

> Alternativa robusta: ejecuta el notebook en modo **"Save & Run All"** desde el principio
> (batch mode, se ejecuta en segundo plano hasta 12 h aunque cierres el navegador).  El
> `--time_limit` garantiza que acaba a tiempo y el output se guarda automáticamente.  Es la
> forma recomendada de gastar la cuota sin tener la pestaña abierta.

## 5. Sesiones siguientes: reanudar

1. Abre el notebook → **Add Input → Your Work → Notebooks** → elige la versión anterior de
   este mismo notebook.  Su output aparece en `/kaggle/input/<slug-del-notebook>/checkpoints/last.pth`.
2. Cambia la celda 3 por:

```python
import glob
RESUME = glob.glob("/kaggle/input/*/checkpoints/last.pth")[0]
print("reanudando desde", RESUME)
!torchrun --nproc_per_node=2 train.py \
    --data_root {DATA_ROOT} --out_dir {OUT} --resume {RESUME} \
    --epochs 60 --batch_size 16 --amp --num_workers 4 \
    --time_limit 11.2 --save_every 500
```

`--resume` restaura modelo, AdamW, GradScaler, `step`, `epoch` y `best_psnr`, y **salta los
batches ya consumidos de la época en curso** (el sampler es determinista por época), así que
no se pierde ni se repite ningún dato.  El output de la nueva sesión vuelve a `/kaggle/working/checkpoints`.

3. Repite hasta completar las épocas.  Cada versión guardada ocupa ~120 MB (last + best).

## 6. Evaluar y descargar

```python
!python evaluate.py --ckpt /kaggle/working/checkpoints/best.pth --data_root {DATA_ROOT} \
    --amp --vis_dir /kaggle/working/vis --n_vis 8 --out_json /kaggle/working/eval.json
```

Descarga `inference.pth` (≈40 MB, sin teacher ni optimizador) desde la pestaña *Output* para
usarlo en tu 3060 con `inference.py`.

## 7. Monitorizar

* Los logs de `train.py` salen por stdout cada `--log_every` pasos con loss, PSNR del batch,
  LR y norma del gradiente.
* TensorBoard escribe en `{OUT}/tb`.  Para verlo en Kaggle:
  ```python
  %load_ext tensorboard
  %tensorboard --logdir /kaggle/working/checkpoints/tb
  ```
* Señales de alarma: `gnorm` > 50 sostenido o `loss` creciendo → baja LR; PSNR de
  validación < baseline (`psnr_avg_baseline` en el log) tras la 1.ª época → bug.

## 8. Problemas frecuentes

| Síntoma | Causa / solución |
|---|---|
| `NCCL error` al arrancar | Falta `--nproc_per_node=2` o el accelerator no es T4×2.  Comprueba `nvidia-smi`. |
| Se queda en `[train] world_size=2` sin avanzar | DataLoader lento con dataset en `/kaggle/input` (disco de red).  Baja `--num_workers` o copia un subset a `/kaggle/working` (¡ojo al límite de 20 GB!). |
| `CUDA out of memory` | `--grad_checkpoint`, `--batch_size 12`, o `--refine_c 8`. |
| La sesión murió y no hay output | No usaste `--time_limit`, o el margen era pequeño.  Usa 11.0 para ir seguro. |
| Loss NaN | Revisa `--weight_decay 1e-3` (AdamW).  Si persiste, `--lr 2e-4` y `--grad_clip 0.5`. |

---

## 9. Run 2: RIFE-m fine-tune sobre Vimeo **septuplet**

El run 1 (60 épocas, triplet) dio **34.33 dB / 0.957 SSIM** y la curva estaba saturada
(≈ +0.02 dB/época al final).  Seguir con la misma receta no compensa.  El run 2 cambia
tres cosas a la vez, cada una con motivación clara:

| Cambio | Flag | Por qué |
|---|---|---|
| **RIFE-m** (timestep arbitrario) | `--arbitrary_time` | Necesario para ×3, ×5… directos y para la app tipo Lossless Scaling (generar el frame en el t exacto que toca, no siempre 0.5).  Añade 1 canal de entrada por IFBlock (7/18/18/21). |
| **Septuplet** en vez de triplet | dataset con `sep_trainlist.txt` | RIFE-m necesita GT en t≠0.5: con 7 frames se muestrean (i0, it, i1) y t=(it−i0)/(i1−i0).  Además el septuplet **incluye** el triplet (gap 2, centro) y añade movimiento hasta 3× mayor (gap 6). |
| **Fine-tune desde el run 1** | `--init_from best.pth` | La conversión RIFE→RIFE-m pone el canal t a **cero** → el modelo arranca numéricamente idéntico al run 1 (34.3 dB) y sólo tiene que aprender a *usar* t.  Entrenar RIFE-m desde cero costaría otras 2–3 sesiones para llegar al mismo sitio. |
| **EMA de pesos** | `--ema 0.999` | Media móvil de los pesos; se valida y exporta con ella.  Típicamente +0.1–0.3 dB y menos ruido época a época.  Se guarda en el checkpoint, reanudable. |
| **Scale augmentation** | `--scale_aug 0.5 0.5 1.5` | Recomendación de hzwer (autor de RIFE) para cerrar el gap entre entrenamiento (448×256, movimiento pequeño) e inferencia (1080p, movimiento grande). |
| Distill weight | `--distill_weight` (auto 0.005) | El paper usa 0.005 para RIFE-m (0.01 para RIFE). |
| LR más bajo | `--lr 1e-4 --warmup_steps 500 --no_lr_scale` | Es un fine-tune: no queremos destruir lo aprendido en las primeras iteraciones. |

### 9.1 Dataset septuplet en Kaggle

`vimeo_septuplet.zip` oficial son 82 GB (http://toflow.csail.mit.edu/).  Mirrors públicos en
Kaggle en el momento de escribir esto (búscalos en https://www.kaggle.com/datasets?search=vimeo+septuplet
y comprueba que contengan `sep_trainlist.txt`, `sep_testlist.txt` y `sequences/*/*/im1..im7.png`):

| Dataset Kaggle | Tamaño | Comentario |
|---|---|---|
| `maiimaii/vimeo-septuplet` (título "dataset_mai") | 87.9 GB, 642k ficheros | Carpeta `vimeo_septuplet/` — tamaño compatible con el dataset completo (91 701 secuencias) |
| `wangsally/vimeo-90k-7` | ? | Otro mirror del septuplet |

El notebook localiza la raíz con `glob('/kaggle/input/**/sep_trainlist.txt')`.  En el mirror de
`maiimaii` la ruta real es `/kaggle/input/datasets/maiimaii/vimeo-septuplet/vimeo_septuplet (1)/vimeo_septuplet`
— **con un espacio y paréntesis** — por eso en las celdas shell `DATA_ROOT` va siempre entre comillas
(`--data_root "{DATA_ROOT}"`); sin ellas `torchrun` recibe la ruta partida en dos argumentos.  Ojo: 88 GB en
`/kaggle/input` (disco de red) → la primera época puede ir más lenta por la caché; `--num_workers 4`
y batch 16 por GPU fueron suficientes en el run 1 (10 min/época sobre 51k tríos).  El septuplet
tiene 64 612 secuencias de train → **≈13–15 min/época**, ≈45 épocas por sesión de 11.2 h.

### 9.2 Plan de sesiones

```
sesión 1: --init_from best.pth(run 1) --arbitrary_time --ema 0.999 --scale_aug 0.5 0.5 1.5 --epochs 90 --time_limit 11.2
          → ~45 épocas.  Guarda versión.
sesión 2: --resume checkpoints_run2/last.pth (mismo --epochs 90)  → completa el coseno.
(sesión 3, opcional): --resume con --epochs 135 ⇒ OJO: cambiar --epochs alarga el coseno y sube el LR de golpe;
          si vas a hacerlo, decide el total ANTES de la sesión 1.)
```

El notebook `docs/kaggle_notebook.ipynb` ya está configurado así.  La celda 1 **detecta sola** el modo:

* Si entre los inputs hay `**/checkpoints_run2/last.pth` (versión anterior de este notebook) → reanuda.
* Si no, busca `**/checkpoints/best.pth` (output del notebook del run 1) → fine-tune.
* Si no encuentra nada, lista los `.pth` visibles y para con un mensaje claro.

Los outputs de un notebook añadido como input se montan en
`/kaggle/input/notebooks/<usuario>/<slug-del-notebook>/<lo que había en /kaggle/working>`; el run 1
escribió en `/kaggle/working/checkpoints/`, así que su `best.pth` está un nivel por debajo del slug
(por eso el glob es recursivo `**`).  Comprueba en la salida de la celda 1 ("Inputs montados") que
aparecen tanto el dataset como el notebook del run 1; si falta el segundo: *Add Input → Your Work →
Notebooks → versión del run 1*.

### 9.3 Qué mirar en los resultados

1. **Validación t=0.5 (test septuplet, im3→im5)**: debe empezar ≈ al nivel del run 1 y no bajar.
   El paper reporta RIFE-m ≈0.1 dB por debajo de RIFE en t=0.5; EMA y scale-aug deberían compensar.
   Los números no son directamente comparables con el 34.33 del test triplet (otro conjunto).
2. **`evaluate.py --multi_t`** (im1→im7, t=1/6…5/6): un RIFE clásico convertido se hunde en
   t=1/6 y 5/6 (predice siempre el centro).  RIFE-m fine-tuneado debe dar PSNR parecido en los 5 t.
   Ésta es la prueba de que el canal t se ha aprendido.
3. **Drop-frame eval en vídeos reales** (`demo_video.py`, sin GT): PSNR de reconstruir frames
   impares a partir de los pares.  Compara run 1 vs run 2 en los mismos clips de `docs/demo_sources.txt`.

### 9.4 Limitaciones de `--init_from`

* `--refine_c` y `--ifnet_widths` **deben coincidir** con el checkpoint origen; si no, esos
  tensores se omiten (quedan aleatorios) y el fine-tune pierde sentido.  El log imprime
  `N tensores copiados, 4 adaptados RIFE→RIFE-m, M omitidos` — M debe ser 0 (o sólo teacher si
  partes de `inference.pth`).
* `--init_from` se ignora si hay `--resume` (el resume ya trae los pesos).
* Se prefieren los pesos EMA del checkpoint origen si los tiene.
