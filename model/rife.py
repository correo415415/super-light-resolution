"""
RIFE: modelo completo.
======================

Une las piezas:

    (I_0, I_1) ─► IFNet ─► flujos F_{t→0}, F_{t→1}, máscara M
                            │
                            ▼
                warp(I_0, F_{t→0}), warp(I_1, F_{t→1})
                            │
                            ▼
              fusión  Î_t^0 = M·w0 + (1-M)·w1
                            │
        Contextnet(I_0,F_{t→0}), Contextnet(I_1,F_{t→1})
                            │
                            ▼
              UNet ─► residual ∈ [-1,1]
                            │
                            ▼
              Î_t = clamp(Î_t^0 + residual, 0, 1)

Este archivo contiene:
  - `RIFE.forward`: forward de entrenamiento con teacher; devuelve un dict
    con todas las salidas intermedias y las pérdidas ya calculadas.
  - `RIFE.inference`: forward limpio (sin teacher, sin gt) con padding
    automático para que H, W sean múltiplos de 32.

Nota sobre t ≠ 0.5 (RIFE vs RIFE-m)
-----------------------------------
Con `arbitrary_time=False` (RIFE, paper ECCV) sólo se interpola t=0.5 y ×4
se hace por recursión.  Con `arbitrary_time=True` (RIFE-m) el modelo recibe t
como canal extra y genera cualquier instante: permite ×3, 24→60 fps o cámara
lenta continua eligiendo cada t libremente.  Matiz importante: en t=0.5 un
RIFE puro suele ser ligeramente mejor (toda su capacidad va a un instante);
la ganancia de RIFE-m está en los t no centrales, no en el central.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ifnet import IFNet
from .loss import LapLoss, distillation_loss
from .refine import Contextnet, RefineUNet


class RIFE(nn.Module):
    def __init__(
        self,
        ifnet_widths: tuple[int, int, int] = (240, 150, 90),
        refine_c: int = 16,
        distill_weight: float = 0.01,
        distill_margin: float = 0.01,
        lap_levels: int = 5,
        use_checkpoint: bool = False,
        arbitrary_time: bool = False,
    ):
        super().__init__()
        self.arbitrary_time = arbitrary_time
        self.ifnet = IFNet(ifnet_widths, use_checkpoint=use_checkpoint, arbitrary_time=arbitrary_time)
        self.contextnet = Contextnet(refine_c)
        self.unet = RefineUNet(refine_c)
        self.lap = LapLoss(lap_levels)
        self.distill_weight = distill_weight
        self.distill_margin = distill_margin

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------
    @staticmethod
    def pad_to_multiple(x: torch.Tensor, multiple: int = 32) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        """Padding (reflejado) hasta que H y W sean múltiplos de `multiple`.

        ¿Por qué 32?  El IFBlock0 trabaja a escala 1/4 y dentro baja ×4 más
        → 1/16.  El Contextnet/UNet bajan hasta 1/16 también.  Para que las
        divisiones sean exactas y las skip connections casen, usamos 32
        (margen de seguridad si alguien pasa escalas (8,4,2)).
        Devuelve también el padding aplicado para recortar la salida.
        """
        _, _, h, w = x.shape
        ph = (multiple - h % multiple) % multiple
        pw = (multiple - w % multiple) % multiple
        pad = (0, pw, 0, ph)  # (izq, der, arriba, abajo)
        if ph or pw:
            # 'replicate' funciona con cualquier tamaño de padding; 'reflect'
            # fallaría si el padding supera el tamaño de la imagen.
            x = F.pad(x, pad, mode="replicate")
        return x, pad

    def refine(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        warped0: torch.Tensor,
        warped1: torch.Tensor,
        mask: torch.Tensor,
        flow: torch.Tensor,
        merged: torch.Tensor,
    ) -> torch.Tensor:
        """Aplica Contextnet + UNet y devuelve la imagen final en [0,1]."""
        ctx0 = self.contextnet(img0, flow[:, 0:2])
        ctx1 = self.contextnet(img1, flow[:, 2:4])
        residual = self.unet(img0, img1, warped0, warped1, mask, flow, ctx0, ctx1)
        # tanh acota el residual a [-1, 1] de forma suave (mejor gradiente
        # que un clamp duro, que anula el gradiente fuera del rango).
        residual = torch.tanh(residual)
        return torch.clamp(merged + residual, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------
    def forward(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        gt: torch.Tensor,
        scales: tuple[float, float, float] = (4.0, 2.0, 1.0),
        timestep=0.5,
    ) -> dict:
        """Forward de entrenamiento.  `timestep`: float o tensor (B,) con el t
        de cada muestra (sólo relevante en RIFE-m).

        Espera imágenes ya con H, W múltiplos de 32 (los crops de 224×224 lo
        cumplen: 224 = 7·32).

        Returns dict con:
            pred:          Î_t final (B,3,H,W)
            merged:        lista de fusiones por bloque (sin refinar)
            merged_teacher, flows, flow_teacher, masks, mask_teacher
            loss, loss_rec, loss_rec_teacher, loss_distill
        """
        out = self.ifnet(img0, img1, gt=gt, scales=scales, timestep=timestep)
        flow = out["flows"][-1]
        mask = out["masks"][-1]
        warped0, warped1 = out["warped"]
        merged = out["merged"][-1]

        pred = self.refine(img0, img1, warped0, warped1, mask, flow, merged)

        # ---- Pérdidas ----
        # L_rec: la salida final refinada debe parecerse al gt.
        loss_rec = self.lap(pred, gt)
        # L_rec_teacher: el teacher también aprende a reconstruir (si no,
        # no tendría ninguna señal de entrenamiento: nadie le supervisa el flujo).
        loss_rec_teacher = self.lap(out["merged_teacher"], gt)
        # L_distill: los flujos del estudiante imitan al teacher donde éste es mejor.
        loss_distill = distillation_loss(
            out["flows"], out["merged"], out["flow_teacher"], out["merged_teacher"], gt, self.distill_margin
        )
        loss = loss_rec + loss_rec_teacher + self.distill_weight * loss_distill

        out.update(
            pred=pred,
            loss=loss,
            loss_rec=loss_rec,
            loss_rec_teacher=loss_rec_teacher,
            loss_distill=loss_distill,
        )
        return out

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------
    @torch.no_grad()
    def inference(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        scales: tuple[float, float, float] = (4.0, 2.0, 1.0),
        return_aux: bool = False,
        timestep: float = 0.5,
    ):
        """Interpola el frame en el instante `timestep` entre img0 e img1.

        Args:
            img0, img1: (B,3,H,W) en [0,1], cualquier H, W.
            scales:     (4,2,1) por defecto.  Para movimientos muy grandes o
                        vídeo 4K, (8,4,2) suele funcionar mejor.
            return_aux: si True devuelve también flujo, máscara y fusión sin
                        refinar (para visualización/debugging).
            timestep:   t ∈ (0,1).  Sólo tiene efecto en RIFE-m; en RIFE
                        clásico sólo se admite 0.5.
        """
        if not self.arbitrary_time and abs(float(timestep) - 0.5) > 1e-6:
            raise ValueError("este checkpoint es RIFE (t fijo = 0.5); para t arbitrario usa un modelo RIFE-m")
        _, _, h, w = img0.shape
        multiple = int(32 * max(scales) / 4)  # 32 para (4,2,1); 64 para (8,4,2)
        img0_p, pad = self.pad_to_multiple(img0, multiple)
        img1_p, _ = self.pad_to_multiple(img1, multiple)

        out = self.ifnet(img0_p, img1_p, gt=None, scales=scales, timestep=timestep)
        flow = out["flows"][-1]
        mask = out["masks"][-1]
        warped0, warped1 = out["warped"]
        merged = out["merged"][-1]
        pred = self.refine(img0_p, img1_p, warped0, warped1, mask, flow, merged)

        pred = pred[:, :, :h, :w]
        if not return_aux:
            return pred
        return pred, {
            "flow": flow[:, :, :h, :w],
            "mask": mask[:, :, :h, :w],
            "merged": merged[:, :, :h, :w],
            "warped0": warped0[:, :, :h, :w],
            "warped1": warped1[:, :, :h, :w],
        }

    # ------------------------------------------------------------------
    def student_parameters(self):
        """Parámetros que se usan en inferencia (sin el teacher).  Útil para
        contar tamaño real del modelo desplegado."""
        for name, p in self.named_parameters():
            if not name.startswith("ifnet.teacher"):
                yield p

    def export_inference_state_dict(self) -> dict:
        """state_dict sin el teacher, para checkpoints de despliegue."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("ifnet.teacher")}

    def load_state_dict_compat(self, state: dict) -> dict:
        """Carga un checkpoint aunque cambie el número de canales de entrada de
        los IFBlocks (RIFE → RIFE-m) o falte el teacher (inference.pth).

        Para la primera conv de cada IFBlock (`encoder.0.0.weight`, forma
        (C_out, C_in, 3, 3)):
          - C_in igual → copia directa;
          - el checkpoint tiene C_in-1 (RIFE → RIFE-m) → se copian los canales
            existentes y el canal t se inserta con peso CERO en su posición
            (índice 6, justo después de I_0, I_1).  Con peso cero el modelo
            convertido es numéricamente idéntico al original para cualquier t;
            el fine-tune sólo tiene que aprender a USAR t.
        Devuelve {copied, adapted, skipped} para el log.
        """
        own = self.state_dict()
        new_state, adapted, skipped = {}, [], []
        for k, v_own in own.items():
            if k not in state:
                skipped.append(k)
                continue
            v = state[k]
            if v.shape == v_own.shape:
                new_state[k] = v
            elif (
                k.endswith("encoder.0.0.weight") and v.dim() == 4
                and v.shape[0] == v_own.shape[0] and v.shape[2:] == v_own.shape[2:]
                and v_own.shape[1] == v.shape[1] + 1
            ):
                w = torch.zeros_like(v_own)
                w[:, :6] = v[:, :6]   # I_0, I_1
                w[:, 7:] = v[:, 6:]   # warps, máscara, flujo, gt: desplazados 1
                new_state[k] = w      # canal 6 (t) queda a cero
                adapted.append(k)
            else:
                skipped.append(k)
        self.load_state_dict(new_state, strict=False)
        return {"copied": len(new_state) - len(adapted), "adapted": adapted, "skipped": skipped}
