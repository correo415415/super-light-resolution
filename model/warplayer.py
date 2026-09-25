"""
Backward warping diferenciable.
=================================

Idea central de RIFE (y de casi todo VFI basado en flujo):

    Î_t(x) = I_0( x + F_{t→0}(x) )

Es decir: para cada píxel `x` del frame intermedio que queremos generar,
miramos "hacia atrás" al frame fuente en la posición desplazada por el flujo
y muestreamos ese color.  A esto se le llama *backward warping*.

¿Por qué backward y no forward?
-------------------------------
- Forward warping ("empujar" cada píxel de I_0 a su destino) deja agujeros
  (píxeles del destino a los que no llega nadie) y colisiones (varios píxeles
  caen en el mismo sitio).  Requiere splatting y es difícil de hacer
  diferenciable de manera estable.
- Backward warping garantiza que TODOS los píxeles del destino reciben un
  valor, porque cada uno "tira" de la fuente.  Se implementa trivialmente con
  `F.grid_sample`, que es bilineal y diferenciable tanto respecto a la imagen
  como respecto a las coordenadas (y por tanto respecto al flujo).

Convenciones
------------
- El flujo tiene forma (B, 2, H, W).  Canal 0 = desplazamiento en x
  (columnas), canal 1 = desplazamiento en y (filas), medidos en PÍXELES.
- `grid_sample` espera coordenadas normalizadas en [-1, 1], donde -1 es el
  borde izquierdo/superior y +1 el borde derecho/inferior.  Por eso
  construimos una malla base normalizada y sumamos el flujo escalado:
      flow_x_norm = flow_x * 2 / (W - 1)
  (con `align_corners=True`, el píxel 0 mapea a -1 y el W-1 a +1, así que
  la distancia entre píxeles adyacentes es 2/(W-1)).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Cache de mallas base por (device, dtype, H, W).  Construir la malla con
# `meshgrid` en cada forward es barato pero no gratis; en entrenamiento se
# llama decenas de veces por iteración (3 bloques × 2 direcciones × teacher...).
_GRID_CACHE: dict[tuple, torch.Tensor] = {}


def _base_grid(B: int, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Devuelve la malla de coordenadas normalizadas (B, H, W, 2) con
    orden (x, y), que es lo que espera `grid_sample`."""
    key = (str(device), dtype, H, W)
    grid = _GRID_CACHE.get(key)
    if grid is None:
        # linspace(-1, 1, W): W puntos equiespaciados entre los dos bordes.
        xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        # indexing="ij" → grid_y varía por filas, grid_x por columnas.
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1)  # (H, W, 2), orden (x, y)
        _GRID_CACHE[key] = grid
    # expand no copia memoria: la misma malla sirve para todo el batch.
    return grid.unsqueeze(0).expand(B, -1, -1, -1)


def warp(img: torch.Tensor, flow: torch.Tensor, padding_mode: str = "border") -> torch.Tensor:
    """Backward-warp de `img` según `flow`.

    Args:
        img:  (B, C, H, W) imagen (o cualquier mapa de features) fuente.
        flow: (B, 2, H, W) flujo en píxeles.  `flow[:, 0]` = dx, `flow[:, 1]` = dy.
              Semántica: out[y, x] = img[y + dy, x + dx].
        padding_mode: qué hacer cuando el flujo apunta fuera de la imagen.
              "border" replica el borde, que suele dar mejor resultado visual
              que rellenar con ceros ("zeros") en los márgenes del frame.

    Returns:
        (B, C, H, W) imagen warpeada.
    """
    B, _, H, W = img.shape
    assert flow.shape == (B, 2, H, W), f"flow {tuple(flow.shape)} no casa con img {tuple(img.shape)}"

    # El flujo llega en píxeles; lo convertimos al espacio normalizado de grid_sample.
    # Nota: si W == 1 la división daría inf; no ocurre en la práctica pero lo
    # protegemos con max(...,1) para robustez.
    scale_x = 2.0 / max(W - 1, 1)
    scale_y = 2.0 / max(H - 1, 1)
    # (B, 2, H, W) → (B, H, W, 2) para sumarlo a la malla.
    flow_norm = torch.stack((flow[:, 0] * scale_x, flow[:, 1] * scale_y), dim=-1)

    grid = _base_grid(B, H, W, img.device, flow_norm.dtype) + flow_norm

    # grid_sample necesita que grid e img tengan el mismo dtype (importante
    # bajo AMP, donde img puede ser fp16 y el flujo fp32 o viceversa).
    return F.grid_sample(
        img,
        grid.to(img.dtype),
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )
