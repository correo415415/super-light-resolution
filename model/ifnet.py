"""
IFNet: estimación de flujo intermedio coarse-to-fine.
======================================================

La contribución central de RIFE es estimar DIRECTAMENTE los flujos
intermedios F_{t→0} y F_{t→1} (del frame que queremos generar hacia los dos
frames de entrada), en lugar de estimar F_{0→1} con una red de flujo óptico
clásica y luego "invertir/desplazar" ese flujo hacia t (que es lo que hacían
métodos anteriores como DAIN o SuperSloMo y que introduce agujeros y
artefactos).

Cómo lo hace: una cascada de bloques (IFBlock) que trabaja de grueso a fino.

    bloque0 (escala 1/4): mira las dos imágenes muy reducidas y da una primera
                          estimación grosera del flujo (cubre movimientos grandes).
    bloque1 (escala 1/2): recibe las imágenes, las imágenes YA warpeadas con el
                          flujo anterior, la máscara previa y el flujo previo, y
                          predice una CORRECCIÓN (residual).
    bloque2 (escala 1/1): idem, a resolución completa, para refinar detalles.

¿Por qué residual?  Si cada bloque tuviera que predecir el flujo completo,
el bloque fino tendría que "redescubrir" los movimientos grandes con un
campo receptivo pequeño.  Prediciendo sólo la diferencia respecto a la
estimación previa, cada bloque se concentra en lo que el anterior no pudo
resolver.  Las imágenes warpeadas son la pista clave: si el flujo anterior es
correcto, warp(I_0) ≈ warp(I_1) y el residual debería ser ~0.

Salida de cada bloque: 5 canales
    [0:2] → F_{t→0}   (dx, dy) en píxeles
    [2:4] → F_{t→1}
    [4:5] → logit de la máscara de fusión M (sigmoid se aplica al final).

Sobre "compartir pesos en ambas direcciones": un mismo bloque predice
conjuntamente los dos flujos (hacia 0 y hacia 1) con los mismos pesos.  No
hay una red para cada dirección; la simetría temporal se aprende gracias a
la augmentación por inversión temporal (intercambiar I_0 e I_1).

Bloque teacher (destilación privilegiada)
----------------------------------------
Durante el entrenamiento añadimos un cuarto IFBlock, idéntico al bloque2,
que además recibe el frame ground-truth I_t.  Con esa información
"privilegiada" puede estimar flujos mucho mejores.  El estudiante (los 3
bloques normales) se entrena para imitar esos flujos.  En inferencia el
teacher se descarta: no tenemos I_t (¡es lo que queremos generar!).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .warplayer import warp


def conv_prelu(in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, padding: int = 1) -> nn.Sequential:
    """Conv 2D + PReLU.

    PReLU (Parametric ReLU) tiene una pendiente APRENDIBLE para la parte
    negativa (una por canal).  En regresión de flujo esto ayuda: los
    residuales de flujo son simétricos alrededor de 0 y una ReLU dura
    mataría la mitad de la señal en las capas intermedias.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=True),
        nn.PReLU(out_ch),
    )


class IFBlock(nn.Module):
    """Un bloque de estimación de flujo intermedio.

    Estructura (según la especificación):
        entrada (a escala 1/s)
          → 2 × [conv 3×3 stride 2 + PReLU]        (↓4 en resolución)
          → 8 × [conv 3×3 + PReLU]  + skip residual (cuerpo)
          → ConvTranspose2d 4×4 stride 2            (↑2)  → 5 canales
          → interpolación bilineal ×(2·s)           (vuelve a resolución completa)

    Args:
        in_channels: canales TOTALES de la entrada al encoder.  Incluye los 4
                     del flujo previo, que se concatenan dentro del forward
                     (después de reescalarlos a la resolución del bloque).
        c:           anchura del bloque (240 / 150 / 90 en RIFE).
        use_checkpoint: si True, el cuerpo de 8 convs se ejecuta con
                     gradient checkpointing (recalcula activaciones en el
                     backward para ahorrar VRAM a cambio de ~30% más cómputo).
    """

    def __init__(self, in_channels: int, c: int, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # Downsample ×4.  Usamos c//2 canales en la primera conv para no
        # gastar cómputo a la resolución más alta del bloque.
        self.encoder = nn.Sequential(
            conv_prelu(in_channels, c // 2, 3, 2, 1),
            conv_prelu(c // 2, c, 3, 2, 1),
        )
        # Cuerpo residual: 8 convs 3×3 a resolución 1/(4s).  A esa resolución
        # el campo receptivo efectivo de 8 convs 3×3 es 17 píxeles → 68·s
        # píxeles en la imagen original.  Con s=4 el bloque0 "ve" ~270 px,
        # suficiente para movimientos grandes.
        self.body = nn.Sequential(*[conv_prelu(c, c, 3, 1, 1) for _ in range(8)])
        # Cabeza: ConvTranspose2d 4×4 stride 2 pad 1 duplica la resolución
        # exactamente (H_out = (H-1)*2 - 2 + 4 = 2H).  Sin activación: el flujo
        # y el logit de la máscara son valores libres.
        self.head = nn.ConvTranspose2d(c, 5, kernel_size=4, stride=2, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        flow: torch.Tensor | None,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:     (B, in_channels, H, W) entrada a RESOLUCIÓN COMPLETA.
            flow:  (B, 4, H, W) flujo previo a resolución completa, o None
                   para el primer bloque.
            scale: factor de reducción al que trabaja este bloque (4, 2, 1).

        Returns:
            flow_delta: (B, 4, H, W) residual de flujo, en píxeles de la
                        resolución completa.
            mask_delta: (B, 1, H, W) residual del logit de la máscara.
        """
        if scale != 1:
            x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            # El flujo mide desplazamientos en píxeles: al reducir la imagen
            # por `scale`, los desplazamientos también se reducen por `scale`.
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) / scale
            x = torch.cat((x, flow), dim=1)

        feat = self.encoder(x)
        if self.use_checkpoint and self.training and feat.requires_grad:
            body_out = checkpoint(self.body, feat, use_reentrant=False)
        else:
            body_out = self.body(feat)
        feat = body_out + feat  # skip residual global del cuerpo

        out = self.head(feat)  # (B, 5, H/(2s), W/(2s))

        # Volvemos a resolución completa.  El factor es 2·scale porque el
        # encoder bajó ×4, la cabeza subió ×2 (neto ×1/2) y además la entrada
        # estaba a 1/scale.
        out = F.interpolate(out, scale_factor=2.0 * scale, mode="bilinear", align_corners=False)

        # El flujo se predijo en píxeles de la resolución H/(2s); al
        # reescalarlo a resolución completa hay que multiplicarlo por 2s.
        flow_delta = out[:, :4] * (2.0 * scale)
        mask_delta = out[:, 4:5]
        return flow_delta, mask_delta


class IFNet(nn.Module):
    """Cascada de 3 IFBlocks (estudiante) + 1 IFBlock teacher opcional.

    Canales de entrada (RIFE, t fijo = 0.5):
        bloque0:   I_0, I_1                                        → 3+3      = 6
        bloque1/2: I_0, I_1, warp(I_0), warp(I_1), M, flujo previo → 6+6+1+4  = 17
        teacher:   lo mismo + I_t (gt)                             → 17+3     = 20

    RIFE-m (`arbitrary_time=True`): tiempo arbitrario t ∈ (0, 1)
    ---------------------------------------------------------------
    Se añade UN canal extra a la entrada de cada bloque: un mapa constante
    con el valor t.  Con eso la red sabe "a qué distancia" de I_0 e I_1 está
    el frame que debe generar y puede escalar el flujo en consecuencia
    (F_{t→0} ≈ -t·F_{0→1}, F_{t→1} ≈ (1-t)·F_{0→1} para movimiento lineal).
    Es lo mismo que hace el IFNet_m oficial (7 / 18 / 18 / 21 canales).

    Compatibilidad con checkpoints RIFE (t=0.5): los pesos del canal t se
    inicializan a CERO, así que un modelo RIFE-m convertido produce
    exactamente lo mismo que el RIFE original para cualquier t.  Fine-tunear
    desde el run 1 aprende el uso de t en pocas épocas en vez de re-aprender
    todo desde cero.  Ver `RIFE.load_state_dict_compat`.
    """

    def __init__(
        self,
        widths: tuple[int, int, int] = (240, 150, 90),
        use_checkpoint: bool = False,
        arbitrary_time: bool = False,
    ):
        super().__init__()
        self.arbitrary_time = arbitrary_time
        extra = 1 if arbitrary_time else 0
        c0, c1, c2 = widths
        self.blocks = nn.ModuleList(
            [
                IFBlock(6 + extra, c0, use_checkpoint),
                IFBlock(17 + extra, c1, use_checkpoint),
                IFBlock(17 + extra, c2, use_checkpoint),
            ]
        )
        # Teacher: idéntico en forma al bloque2 pero con 3 canales extra (I_t).
        self.teacher = IFBlock(20 + extra, c2, use_checkpoint)

    def _time_map(self, img0: torch.Tensor, timestep) -> torch.Tensor | None:
        """Mapa constante (B, 1, H, W) con el valor t (None si RIFE clásico).
        `timestep`: float (mismo t para el batch) o tensor (B,) con un t por
        muestra (necesario en entrenamiento RIFE-m)."""
        if not self.arbitrary_time:
            return None
        B, _, H, W = img0.shape
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(float(timestep), device=img0.device, dtype=img0.dtype)
        timestep = timestep.to(img0.device, img0.dtype).reshape(-1, 1, 1, 1)
        return timestep.expand(B, 1, H, W)

    # ------------------------------------------------------------------
    @staticmethod
    def _fuse(warped0: torch.Tensor, warped1: torch.Tensor, mask_logit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fusión Î_t = M ⊙ warp(I_0) + (1-M) ⊙ warp(I_1), con M = sigmoid(logit).

        La máscara aprende a resolver oclusiones: donde un objeto sólo es
        visible en I_0 (porque en I_1 está tapado), M→1 y viceversa.
        """
        mask = torch.sigmoid(mask_logit)
        return warped0 * mask + warped1 * (1.0 - mask), mask

    def forward(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        gt: torch.Tensor | None = None,
        scales: tuple[float, float, float] = (4.0, 2.0, 1.0),
        timestep=0.5,
    ) -> dict:
        """
        Args:
            img0, img1: (B, 3, H, W) en [0, 1].  H y W deben ser múltiplos de
                        16·max(scales)/4 (=16 para las escalas por defecto);
                        el wrapper `RIFE` se encarga del padding.
            gt:         (B, 3, H, W) frame intermedio real.  Si se pasa (y
                        estamos en modo train), se ejecuta el teacher.
            scales:     escalas de los 3 bloques.  Para vídeo 4K se puede usar
                        (8, 4, 2) y así los bloques ven movimientos mayores.
            timestep:   t del frame a generar (float o tensor (B,)).  Sólo se
                        usa si `arbitrary_time=True`; RIFE clásico lo ignora.

        Returns dict con:
            flows:   lista de 3 tensores (B, 4, H, W), flujo acumulado tras cada bloque.
            masks:   lista de 3 tensores (B, 1, H, W), máscara (ya con sigmoid).
            merged:  lista de 3 tensores (B, 3, H, W), fusión tras cada bloque.
            warped:  (warp(I_0), warp(I_1)) con el flujo final del estudiante.
            flow_teacher / mask_teacher / merged_teacher: sólo si hay gt.
        """
        flows, mask_logits, merged = [], [], []
        flow = mask_logit = None
        warped0, warped1 = img0, img1  # antes de tener flujo, "warp" = identidad
        tmap = self._time_map(img0, timestep)
        t_in = (tmap,) if tmap is not None else ()  # canal t (sólo RIFE-m)

        for block, scale in zip(self.blocks, scales):
            if flow is None:
                # Bloque 0: sólo ve las imágenes originales (+ t en RIFE-m).
                flow, mask_logit = block(torch.cat((img0, img1, *t_in), dim=1), None, scale)
            else:
                # Bloques 1 y 2: ven además los warps y la máscara previos y
                # predicen un residual que se SUMA a la estimación anterior.
                x = torch.cat((img0, img1, *t_in, warped0, warped1, mask_logit), dim=1)
                flow_d, mask_d = block(x, flow, scale)
                flow = flow + flow_d
                mask_logit = mask_logit + mask_d

            warped0 = warp(img0, flow[:, 0:2])
            warped1 = warp(img1, flow[:, 2:4])
            flows.append(flow)
            mask_logits.append(mask_logit)
            merged.append((warped0, warped1))

        out: dict = {"flows": flows}

        # ---------------- Teacher (sólo entrenamiento) ----------------
        if gt is not None:
            x_tea = torch.cat((img0, img1, *t_in, warped0, warped1, mask_logit, gt), dim=1)
            flow_d, mask_d = self.teacher(x_tea, flow, scale=1.0)
            flow_tea = flow + flow_d
            mask_logit_tea = mask_logit + mask_d
            w0_tea = warp(img0, flow_tea[:, 0:2])
            w1_tea = warp(img1, flow_tea[:, 2:4])
            merged_tea, mask_tea = self._fuse(w0_tea, w1_tea, mask_logit_tea)
            out.update(flow_teacher=flow_tea, mask_teacher=mask_tea, merged_teacher=merged_tea)

        # Aplicamos sigmoid y fusionamos en cada nivel (útil para visualizar
        # cómo mejora la predicción bloque a bloque).
        masks, merged_imgs = [], []
        for (w0, w1), ml in zip(merged, mask_logits):
            m_img, m = self._fuse(w0, w1, ml)
            merged_imgs.append(m_img)
            masks.append(m)

        out.update(masks=masks, merged=merged_imgs, warped=(warped0, warped1))
        return out
