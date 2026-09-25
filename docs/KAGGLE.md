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
  batch 16×2 y AMP, una época de Vimeo (~1 600 pasos) tarda ≈ 12-15 min → ≈ 45 épocas por
  sesión de 11 h.  El paper usa 300 épocas; con tu cuota (30 h/semana ≈ 2.5 sesiones) un
  plan realista es **60-100 épocas en 2-4 sesiones** (ver resultados esperados en el README).
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
