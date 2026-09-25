"""
Utilidades compartidas por evaluate.py e inference.py.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from model import RIFE


def load_model(ckpt_path: str | Path, device: torch.device, refine_c: int | None = None) -> RIFE:
    """Carga un checkpoint de train.py (last.pth / best.pth / inference.pth).

    El checkpoint guarda `args`, de donde leemos la configuración del modelo
    para no tener que pasarla a mano.  Si el checkpoint es de inferencia (sin
    teacher), cargamos con strict=False y el teacher queda con pesos
    aleatorios (no se usa en inferencia).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    widths = tuple(args.get("ifnet_widths", (240, 150, 90)))
    rc = refine_c or args.get("refine_c", 16)
    model = RIFE(ifnet_widths=widths, refine_c=rc)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    missing = [k for k in missing if not k.startswith("ifnet.teacher")]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint incompatible.  faltan={missing[:5]} sobran={unexpected[:5]}")
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Visualización de flujo
# ---------------------------------------------------------------------------
def flow_to_color(flow: np.ndarray, max_mag: float | None = None) -> np.ndarray:
    """Flujo (H, W, 2) → imagen RGB uint8 (codificación HSV estándar).

    Tono = dirección del movimiento, valor = magnitud.  Es la convención de
    Middlebury/Sintel que usa casi toda la literatura de flujo óptico.
    """
    dx, dy = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(dx.astype(np.float32), dy.astype(np.float32), angleInDegrees=True)
    if max_mag is None:
        max_mag = max(float(np.percentile(mag, 99)), 1e-3)
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)  # OpenCV usa H en [0, 180)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) en [0,1] → (H, W, 3) uint8 RGB."""
    return (t.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def _label(img: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def make_debug_grid(img0, img1, gt, pred, aux: dict) -> np.ndarray:
    """Panel 2×4 para depurar una muestra (tensores sin batch):

        I_0      | I_1      | GT        | Pred
        F_{t→0}  | F_{t→1}  | Máscara M | |Pred-GT|×4
    """
    h, w = gt.shape[-2:]
    flow = aux["flow"].detach().float().cpu().numpy()
    f0 = flow_to_color(flow[0:2].transpose(1, 2, 0))
    f1 = flow_to_color(flow[2:4].transpose(1, 2, 0))
    mask = (aux["mask"][0].detach().float().cpu().numpy() * 255).astype(np.uint8)
    mask = np.repeat(mask[..., None], 3, axis=-1)
    err = (pred.float() - gt.float()).abs().mean(0, keepdim=True).repeat(3, 1, 1) * 4
    row0 = np.concatenate([tensor_to_uint8(x) for x in (img0, img1, gt, pred)], axis=1)
    row1 = np.concatenate([f0, f1, mask, tensor_to_uint8(err)], axis=1)
    grid = np.ascontiguousarray(np.concatenate([row0, row1], axis=0))
    for i, lab in enumerate(["I0", "I1", "GT", "Pred", "flow t->0", "flow t->1", "mask", "|err|x4"]):
        _label(grid, lab, (i % 4) * w + 5, (i // 4) * h + 18)
    return grid
