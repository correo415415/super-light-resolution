"""
Funciones de pérdida.
=====================

1. LapLoss (pérdida piramidal laplaciana)
-----------------------------------------
Una L1 directa sobre píxeles penaliza igual un error de bajo nivel (color
ligeramente distinto en una zona plana) que uno estructural (un borde
desplazado un píxel).  El ojo humano —y el PSNR/SSIM— castigan más lo
segundo.

La pirámide laplaciana descompone la imagen en bandas de frecuencia:
    L_k = G_k - upsample(G_{k+1})      (G = pirámide gaussiana)
Cada nivel L_k contiene los detalles de una escala concreta.  Sumar la L1
sobre todos los niveles equivale a una L1 "consciente de la escala": los
errores en bordes finos aparecen en los niveles altos, los errores de forma
en los bajos, y todos cuentan.  El último nivel (el residuo gaussiano más
grueso) captura el color global.

Trade-off LapLoss vs L1 + FFT (usada en algunas variantes VFI):
  - L1+FFT (pérdida en el dominio de Fourier) afila resultados y ayuda con
    texturas, pero es más sensible a hiperparámetros y puede introducir
    ringing.
  - LapLoss es la elección del paper original de RIFE, es estable y
    prácticamente no tiene hiperparámetros (nº de niveles).
  Usamos LapLoss por defecto con 5 niveles.

2. Pérdida de destilación privilegiada
--------------------------------------
Queremos que el flujo del estudiante se parezca al del teacher, pero SÓLO
donde el teacher lo hace mejor.  Si el teacher se equivoca en un píxel,
forzar al estudiante a imitarlo sería contraproducente.  Por eso:

    mask = 1[ |merged_student - gt| > |merged_teacher - gt| + margen ]
    L_distill = mean( |flow_student - flow_teacher| · mask )

El margen (0.01) evita que píxeles donde ambos son casi iguales generen
gradiente ruidoso.  El teacher se trata como constante (`detach`): el
gradiente sólo fluye al estudiante.

Se aplica sobre la salida de los 3 bloques del estudiante.  Esto es una
supervisión intermedia que acelera la convergencia de los bloques gruesos.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gauss_kernel(channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Kernel gaussiano 5×5 binomial, replicado para conv depthwise."""
    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device, dtype=dtype)
    k = torch.outer(k, k)
    k = k / k.sum()
    return k.expand(channels, 1, 5, 5).contiguous()


def _gauss_blur(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    # Padding "reflect" evita que los bordes se oscurezcan al filtrar.
    x = F.pad(x, (2, 2, 2, 2), mode="reflect")
    return F.conv2d(x, kernel, groups=x.shape[1])


def laplacian_pyramid(img: torch.Tensor, kernel: torch.Tensor, levels: int) -> list[torch.Tensor]:
    """Devuelve [L_0, ..., L_{levels-2}, G_{levels-1}]."""
    pyr = []
    current = img
    for _ in range(levels - 1):
        blurred = _gauss_blur(current, kernel)
        down = blurred[:, :, ::2, ::2]  # decimación ×2
        # Volvemos a subir para restar: lo que se pierde es el detalle de este nivel.
        up = F.interpolate(down, size=current.shape[-2:], mode="bilinear", align_corners=False)
        pyr.append(current - up)
        current = down
    pyr.append(current)
    return pyr


class LapLoss(nn.Module):
    def __init__(self, levels: int = 5):
        super().__init__()
        self.levels = levels
        self._kernel_cache: dict[tuple, torch.Tensor] = {}

    def _kernel(self, x: torch.Tensor) -> torch.Tensor:
        key = (x.shape[1], str(x.device), x.dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _gauss_kernel(x.shape[1], x.device, x.dtype)
        return self._kernel_cache[key]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Calculamos en fp32 aunque estemos bajo AMP: las diferencias entre
        # niveles son pequeñas y en fp16 perderíamos precisión.
        pred = pred.float()
        target = target.float()
        kernel = self._kernel(pred)
        pyr_p = laplacian_pyramid(pred, kernel, self.levels)
        pyr_t = laplacian_pyramid(target, kernel, self.levels)
        # Pesamos cada nivel por 2^k: los niveles gruesos tienen menos
        # píxeles, así compensamos para que su contribución no se diluya.
        return sum((2**k) * F.l1_loss(p, t) for k, (p, t) in enumerate(zip(pyr_p, pyr_t)))


def distillation_loss(
    flows_student: list[torch.Tensor],
    merged_student: list[torch.Tensor],
    flow_teacher: torch.Tensor,
    merged_teacher: torch.Tensor,
    gt: torch.Tensor,
    margin: float = 0.01,
) -> torch.Tensor:
    """Pérdida de destilación enmascarada por píxel (ver docstring del módulo).

    Args:
        flows_student:  lista de (B, 4, H, W), flujo tras cada bloque del estudiante.
        merged_student: lista de (B, 3, H, W), fusión tras cada bloque.
        flow_teacher:   (B, 4, H, W).
        merged_teacher: (B, 3, H, W).
        gt:             (B, 3, H, W).
    """
    flow_teacher = flow_teacher.detach()
    # Error por píxel del teacher (promediado en canales de color).
    err_teacher = (merged_teacher.detach() - gt).abs().mean(dim=1, keepdim=True)

    loss = gt.new_zeros(())
    for flow_s, merged_s in zip(flows_student, merged_student):
        err_student = (merged_s.detach() - gt).abs().mean(dim=1, keepdim=True)
        # 1 donde el estudiante lo hace claramente peor que el teacher.
        mask = (err_student > err_teacher + margin).to(flow_s.dtype)
        loss = loss + ((flow_s - flow_teacher).abs() * mask).mean()
    return loss


# ---------------------------------------------------------------------------
# Métricas (útiles para validación durante el entrenamiento y evaluate.py)
# ---------------------------------------------------------------------------
def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """PSNR por imagen del batch → (B,)."""
    mse = ((pred.float() - target.float()) ** 2).flatten(1).mean(dim=1)
    return 10.0 * torch.log10(max_val**2 / mse.clamp_min(1e-10))


def _ssim_window(size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    w = torch.outer(g, g)
    return w.expand(channels, 1, size, size).contiguous()


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """SSIM (Wang et al. 2004) por imagen del batch → (B,).  Imágenes en [0,1].

    Implementación estándar con ventana gaussiana 11×11, σ=1.5, sin padding
    (modo 'valid'), que es lo que usan las evaluaciones de VFI habituales.
    """
    pred = pred.float()
    target = target.float()
    C = pred.shape[1]
    w = _ssim_window(window_size, sigma, C, pred.device, pred.dtype)
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    mu_p = F.conv2d(pred, w, groups=C)
    mu_t = F.conv2d(target, w, groups=C)
    sigma_pp = F.conv2d(pred * pred, w, groups=C) - mu_p**2
    sigma_tt = F.conv2d(target * target, w, groups=C) - mu_t**2
    sigma_pt = F.conv2d(pred * target, w, groups=C) - mu_p * mu_t

    ssim_map = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / (
        (mu_p**2 + mu_t**2 + C1) * (sigma_pp + sigma_tt + C2)
    )
    return ssim_map.flatten(1).mean(dim=1)
