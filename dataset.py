"""
Dataset Vimeo90K triplet.
=========================

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
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        crop_size: int | tuple[int, int] | None = 224,
        augment: bool | None = None,
        max_samples: int | None = None,
        rotate: bool = True,
    ):
        """
        Args:
            root:        carpeta `vimeo_triplet`.
            split:       "train" o "test".
            crop_size:   tamaño del crop aleatorio (train).  None → imagen entera.
            augment:     por defecto True en train, False en test.
            max_samples: recorta la lista (smoke test / debug).
            rotate:      añadir rotación 90° a las augmentaciones.
        """
        self.root = Path(root)
        self.split = split
        self.augment = (split == "train") if augment is None else augment
        self.rotate = rotate
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        self.crop_size = crop_size

        list_file = self.root / ("tri_trainlist.txt" if split == "train" else "tri_testlist.txt")
        if not list_file.exists():
            raise FileNotFoundError(
                f"No encuentro {list_file}.  ¿Apunta --data_root a la carpeta 'vimeo_triplet'?"
            )
        with open(list_file) as f:
            self.samples = [line.strip() for line in f if line.strip()]
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_triplet(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = self.root / "sequences" / self.samples[idx]
        return _read_rgb(d / "im1.png"), _read_rgb(d / "im2.png"), _read_rgb(d / "im3.png")

    def _augment(self, img0, gt, img1):
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
        # Inversión temporal.
        if random.random() < 0.5:
            img0, img1 = img1, img0
        return img0, gt, img1

    def __getitem__(self, idx: int) -> dict:
        img0, gt, img1 = self._load_triplet(idx)
        if self.augment:
            img0, gt, img1 = self._augment(img0, gt, img1)
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
        return {"img0": _to_tensor(img0), "gt": _to_tensor(gt), "img1": _to_tensor(img1), "name": f"synthetic/{idx:06d}"}


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
    )
