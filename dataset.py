"""
Dataset Vimeo90K (triplet y septuplet).
========================================

Estructura esperada en disco (la del zip oficial `vimeo_triplet.zip`):

    vimeo_triplet/
        tri_trainlist.txt      ← una línea por trío: "00001/0001"
        tri_testlist.txt
        sequences/
            00001/
                0001/
                    im1.png    ← I_0
                    im2.png    ← I_t (ground truth, t=0.5)
                    im3.png    ← I_1
                0002/ ...

Cada frame es 448×256 RGB.  ~51k tríos de train, ~3.8k de test.

Augmentación (sólo en train)
----------------------------
- Crop aleatorio 224×224.  224 = 7·32 → múltiplo de 32 sin padding.
  Además de regularizar, un crop más pequeño permite batch 16 en una T4.
- Flip horizontal y vertical (p=0.5 cada uno).  El flujo es equivariante
  a flips, así que el gt sigue siendo válido.
- Inversión temporal (p=0.5): intercambiar I_0 e I_1.  El frame intermedio
  es el mismo, y así la red aprende que las dos direcciones son simétricas
  (recuerda: los IFBlocks comparten pesos para ambas direcciones).
- Rotación 90° (p=0.5): opcional; Vimeo es apaisado, rotar ayuda a que
  los movimientos verticales estén tan representados como los horizontales.

Todas las augmentaciones se aplican con numpy ANTES de convertir a tensor,
que es más rápido que hacerlas en torch en el DataLoader (CPU).

Smoke test
----------
`max_samples` recorta la lista para validar el pipeline rápidamente.  Si no
tienes Vimeo90K a mano, `SyntheticTripletDataset` genera tríos con formas
que se mueven linealmente: el frame intermedio es exactamente la posición
media, así que la red DEBE ser capaz de aprenderlo (>30 dB en pocos cientos
de pasos).  Si no lo hace, hay bug.
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _read_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    """HWC uint8 → CHW float32 en [0,1]."""
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)


class Vimeo90KTriplet(Dataset):
    """Vimeo90K triplet o septuplet, con timestep.

    Triplet  (`vimeo_triplet`,  tri_trainlist.txt,  im1..im3): t = 0.5 siempre.
    Septuplet (`vimeo_septuplet`, sep_trainlist.txt, im1..im7): en train se
    eligen 3 índices aleatorios ordenados (i0 < it < i1) de los 7 frames y
        t = (it - i0) / (i1 - i0)          ∈ {1/6, 1/5, ..., 5/6, 1/2 ...}
    Es la receta del RIFE-m oficial.  Con `sep_mode="triplet"` se usan siempre
    frames consecutivos con el central (im3, im4, im5 → t=0.5): así el
    septuplet sirve también para entrenar RIFE clásico con 91k secuencias en
    vez de 51k (el autor recomienda "sólo septuplet, incluye al triplet").

    Scale augmentation (`scale_aug`)
    --------------------------------
    Consejo nº1 del autor de RIFE para cerrar el gap train/inferencia: los
    crops de 224 de Vimeo (448×256) contienen movimientos de hasta ~20 px,
    pero en 1080p real el movimiento puede ser de 100+ px.  Con probabilidad
    p reescalamos los 3 frames por un factor s∈[min,max] ANTES del crop:
    s<1 comprime el movimiento (más contexto por crop), s>1 lo amplía.  El
    crop final sigue siendo 224×224, así que el batch apila igual.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        crop_size: int | tuple[int, int] | None = 224,
        augment: bool | None = None,
        max_samples: int | None = None,
        rotate: bool = True,
        arbitrary_time: bool = False,
        scale_aug: tuple[float, float, float] = (0.0, 1.0, 1.0),
        max_gap: int = 6,
        list_dir: str | Path | None = None,
    ):
        """
        Args:
            root:        carpeta `vimeo_triplet` o `vimeo_septuplet` (se detecta).
            split:       "train" o "test".
            crop_size:   tamaño del crop aleatorio (train).  None → imagen entera.
            augment:     por defecto True en train, False en test.
            max_samples: recorta la lista (smoke test / debug).
            rotate:      añadir rotación 90° a las augmentaciones.
            arbitrary_time: (septuplet) muestrear t aleatorio; si False, t=0.5
                         con frames consecutivos alrededor del centro.
            scale_aug:   (prob, s_min, s_max) de reescalado previo al crop.
            max_gap:     (septuplet) separación máxima i1-i0 (2..6).  Gaps
                         grandes = movimientos grandes; 6 usa todo el rango.
        """
        self.root = Path(root)
        self.split = split
        self.augment = (split == "train") if augment is None else augment
        self.rotate = rotate
        self.arbitrary_time = arbitrary_time
        self.scale_aug = scale_aug
        self.max_gap = max(2, min(6, max_gap))
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        self.crop_size = crop_size

        # Detección triplet / septuplet por el nombre de las listas.
        if (self.root / "sep_trainlist.txt").exists():
            self.n_frames = 7
            prefix = "sep"
        elif (self.root / "tri_trainlist.txt").exists():
            self.n_frames = 3
            prefix = "tri"
        else:
            raise FileNotFoundError(
                f"No encuentro tri_trainlist.txt ni sep_trainlist.txt en {self.root}.  "
                "¿Apunta --data_root a la carpeta 'vimeo_triplet' / 'vimeo_septuplet'?"
            )
        # list_dir: carpeta alternativa con las listas (p.ej. listas "limpias" de
        # scripts/check_dataset.py cuando el dataset está en un disco de sólo lectura).
        list_root = Path(list_dir) if list_dir else self.root
        list_file = list_root / f"{prefix}_{'trainlist' if split == 'train' else 'testlist'}.txt"
        with open(list_file) as f:
            self.samples = [line.strip() for line in f if line.strip()]
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def _pick_indices(self) -> tuple[int, int, int, float]:
        """Devuelve (i0, it, i1, t) con índices 1-based de imK.png."""
        if self.n_frames == 3:
            return 1, 2, 3, 0.5
        if self.augment and self.arbitrary_time:
            # Tres índices distintos ordenados, con separación <= max_gap.
            while True:
                i0, it, i1 = sorted(random.sample(range(1, 8), 3))
                if i1 - i0 <= self.max_gap:
                    break
            return i0, it, i1, (it - i0) / (i1 - i0)
        if self.augment:
            # RIFE clásico sobre septuplet: cualquier trío consecutivo (t=0.5).
            i0 = random.randint(1, 5)
            return i0, i0 + 1, i0 + 2, 0.5
        return 3, 4, 5, 0.5  # test determinista: centro de la secuencia

    def _load(self, idx: int, i0: int, it: int, i1: int):
        d = self.root / "sequences" / self.samples[idx]
        return _read_rgb(d / f"im{i0}.png"), _read_rgb(d / f"im{it}.png"), _read_rgb(d / f"im{i1}.png")

    def _augment(self, img0, gt, img1, t: float):
        # Scale augmentation (antes del crop).
        p, smin, smax = self.scale_aug
        if p > 0 and random.random() < p and self.crop_size is not None:
            s = random.uniform(smin, smax)
            h, w = img0.shape[:2]
            ch, cw = self.crop_size
            # No bajar por debajo del tamaño del crop.
            s = max(s, ch / h, cw / w)
            nh, nw = max(ch, int(round(h * s))), max(cw, int(round(w * s)))
            interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
            img0, gt, img1 = (cv2.resize(im, (nw, nh), interpolation=interp) for im in (img0, gt, img1))
        # Crop aleatorio.
        if self.crop_size is not None:
            h, w = img0.shape[:2]
            ch, cw = self.crop_size
            y = random.randint(0, h - ch)
            x = random.randint(0, w - cw)
            img0, gt, img1 = (im[y : y + ch, x : x + cw] for im in (img0, gt, img1))
        # Flips.  Se aplican a los 3 frames a la vez (¡si no, el gt deja de serlo!).
        if random.random() < 0.5:
            img0, gt, img1 = (im[:, ::-1] for im in (img0, gt, img1))
        if random.random() < 0.5:
            img0, gt, img1 = (im[::-1, :] for im in (img0, gt, img1))
        # Rotación 90° (sólo si el crop es cuadrado, para que el batch apile).
        if self.rotate and self.crop_size is not None and self.crop_size[0] == self.crop_size[1]:
            if random.random() < 0.5:
                img0, gt, img1 = (np.rot90(im) for im in (img0, gt, img1))
        # Inversión temporal: intercambiar I_0 e I_1 implica t → 1 - t.
        if random.random() < 0.5:
            img0, img1 = img1, img0
            t = 1.0 - t
        return img0, gt, img1, t

    # Los mirrors públicos de Vimeo90K a veces traen PNGs truncados o secuencias
    # incompletas (p.ej. `libpng error: IDAT: CRC error` + FileNotFoundError en
    # 00023/0424 del mirror septuplet de Kaggle).  Un solo fichero malo NO
    # debe tirar un entrenamiento de 11 h: en train sustituimos la muestra por
    # otra aleatoria (y avisamos), en test la saltamos de forma determinista.
    MAX_RETRIES = 8

    def _load_safe(self, idx: int, i0: int, it: int, i1: int):
        try:
            imgs = self._load(idx, i0, it, i1)
        except (FileNotFoundError, cv2.error) as e:
            return None, e
        if any(im is None or im.shape[:2] != imgs[0].shape[:2] for im in imgs):
            return None, ValueError("frame vacío o tamaños distintos")
        return imgs, None

    def __getitem__(self, idx: int) -> dict:
        i0, it, i1, t = self._pick_indices()
        imgs, err = self._load_safe(idx, i0, it, i1)
        tries = 0
        while imgs is None and tries < self.MAX_RETRIES:
            warnings.warn(f"[dataset] muestra {self.samples[idx]} ilegible ({err}); se sustituye")
            tries += 1
            # train: otra aleatoria; test: la siguiente (determinista, reproducible)
            idx = random.randrange(len(self.samples)) if self.augment else (idx + 1) % len(self.samples)
            i0, it, i1, t = self._pick_indices()
            imgs, err = self._load_safe(idx, i0, it, i1)
        if imgs is None:
            raise RuntimeError(f"{self.MAX_RETRIES} muestras ilegibles seguidas; ¿data_root correcto?  último error: {err}")
        img0, gt, img1 = imgs
        if self.augment:
            img0, gt, img1, t = self._augment(img0, gt, img1, t)
        elif self.crop_size is not None:
            # En validación con crop, crop CENTRAL determinista.
            h, w = img0.shape[:2]
            ch, cw = self.crop_size
            y, x = (h - ch) // 2, (w - cw) // 2
            img0, gt, img1 = (im[y : y + ch, x : x + cw] for im in (img0, gt, img1))
        return {
            "img0": _to_tensor(img0),
            "gt": _to_tensor(gt),
            "img1": _to_tensor(img1),
            "timestep": torch.tensor(t, dtype=torch.float32),
            "name": self.samples[idx],
        }


class SyntheticTripletDataset(Dataset):
    """Tríos sintéticos para smoke tests sin datos reales.

    Genera un fondo con gradiente + ruido suave y N rectángulos/círculos de
    color que se trasladan linealmente entre I_0 e I_1.  El frame intermedio
    tiene cada forma exactamente a mitad de camino.  Movimiento máximo
    configurable (por defecto ±24 px, similar a Vimeo).
    """

    def __init__(self, n: int = 2000, size: int = 224, n_shapes: int = 4, max_motion: int = 24, seed: int = 0):
        # Limitamos el movimiento para imágenes pequeñas (tests en CPU con size=64).
        max_motion = min(max_motion, max(2, size // 8))
        self.n, self.size, self.n_shapes, self.max_motion, self.seed = n, size, n_shapes, max_motion, seed

    def __len__(self) -> int:
        return self.n

    def _render(self, rng: np.random.Generator, shapes: list[dict], t: float) -> np.ndarray:
        s = self.size
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32) / s
        bg = np.stack([0.3 + 0.4 * xx, 0.3 + 0.4 * yy, 0.5 + 0.2 * np.sin(6 * xx) * np.cos(6 * yy)], -1)
        img = (bg * 255).astype(np.uint8).copy()
        for sh in shapes:
            cx = int(round(sh["x"] + t * sh["dx"]))
            cy = int(round(sh["y"] + t * sh["dy"]))
            color = tuple(int(c) for c in sh["color"])
            if sh["kind"] == "rect":
                cv2.rectangle(img, (cx - sh["r"], cy - sh["r"]), (cx + sh["r"], cy + sh["r"]), color, -1)
            else:
                cv2.circle(img, (cx, cy), sh["r"], color, -1)
        return img

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        s, m = self.size, self.max_motion
        shapes = [
            dict(
                kind=rng.choice(["rect", "circle"]),
                x=rng.integers(m + 10, s - m - 10),
                y=rng.integers(m + 10, s - m - 10),
                dx=rng.integers(-m, m + 1),
                dy=rng.integers(-m, m + 1),
                r=int(rng.integers(max(3, s // 28), max(4, s // 8))),
                color=rng.integers(0, 256, 3),
            )
            for _ in range(self.n_shapes)
        ]
        img0 = self._render(rng, shapes, 0.0)
        gt = self._render(rng, shapes, 0.5)
        img1 = self._render(rng, shapes, 1.0)
        return {"img0": _to_tensor(img0), "gt": _to_tensor(gt), "img1": _to_tensor(img1),
                "timestep": torch.tensor(0.5), "name": f"synthetic/{idx:06d}"}


def build_dataset(args, split: str) -> Dataset:
    """Fábrica usada por train.py / evaluate.py según los flags."""
    if getattr(args, "synthetic", False):
        if split == "train":
            n = args.max_samples or 2000
        else:
            n = getattr(args, "max_val_samples", None) or 200
        return SyntheticTripletDataset(n=n, size=args.crop_size, seed=0 if split == "train" else 1)
    return Vimeo90KTriplet(
        args.data_root,
        split=split,
        crop_size=args.crop_size if split == "train" else None,
        max_samples=args.max_samples if split == "train" else getattr(args, "max_val_samples", None),
        arbitrary_time=getattr(args, "arbitrary_time", False),
        scale_aug=tuple(getattr(args, "scale_aug", (0.0, 1.0, 1.0))),
        max_gap=getattr(args, "max_gap", 6),
        list_dir=getattr(args, "list_dir", None),
    )
