"""
Refinamiento: Contextnet + UNet.
=================================

La fusión Î_t = M·warp(I_0) + (1-M)·warp(I_1) funciona bien donde el flujo es
correcto, pero deja artefactos en:
  - oclusiones (zonas visibles en un solo frame),
  - bordes de objetos (el warping bilineal emborrona),
  - zonas donde el flujo es simplemente erróneo.

Para corregirlos, un pequeño UNet predice un RESIDUAL que se suma a la fusión.
Predecir el residual (y no la imagen completa) es clave: la red parte de una
estimación ya decente y sólo tiene que aprender correcciones, lo que hace que
el entrenamiento sea estable desde el principio.

Contextnet: "features warpeadas"
--------------------------------
Además de las imágenes warpeadas en RGB, le damos al UNet features
convolucionales de I_0 e I_1 extraídas a varias escalas y WARPEADAS con el
flujo (reescalado a cada nivel).  Esto aporta información semántica alineada
al frame intermedio: el UNet sabe "qué había" en cada posición de la imagen
fuente, no sólo su color.  Es la misma idea que en "Context-aware synthesis"
(Niklaus & Liu, CVPR 2018) pero con un extractor entrenado desde cero.

Trade-off: tamaño del UNet
--------------------------
- UNet grande (c≈64, 4 niveles): +0.1-0.2 dB PSNR pero duplica el tiempo
  de inferencia y la VRAM.
- UNet pequeño (c=16 base, 4 niveles, como hacemos aquí): casi todo el
  beneficio (el grueso de la calidad viene del flujo) a coste marginal.
Elegimos el pequeño por defecto; `c` es un parámetro y puedes subirlo.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .warplayer import warp


def conv_prelu(in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride, padding),
        nn.PReLU(out_ch),
    )


def deconv_prelu(in_ch: int, out_ch: int) -> nn.Sequential:
    """ConvTranspose 4×4 stride 2 pad 1 → duplica H y W exactamente."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
        nn.PReLU(out_ch),
    )


class Contextnet(nn.Module):
    """Pirámide de 4 niveles de features, cada uno warpeado con el flujo.

    Nivel k trabaja a resolución 1/2^k (k=1..4) con c·2^(k-1) canales.
    Devuelve la lista [f1, f2, f3, f4] ya warpeadas.
    """

    def __init__(self, c: int = 16):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Sequential(conv_prelu(3, c, 3, 2, 1), conv_prelu(c, c)),
                nn.Sequential(conv_prelu(c, 2 * c, 3, 2, 1), conv_prelu(2 * c, 2 * c)),
                nn.Sequential(conv_prelu(2 * c, 4 * c, 3, 2, 1), conv_prelu(4 * c, 4 * c)),
                nn.Sequential(conv_prelu(4 * c, 8 * c, 3, 2, 1), conv_prelu(8 * c, 8 * c)),
            ]
        )

    def forward(self, img: torch.Tensor, flow: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            img:  (B, 3, H, W)
            flow: (B, 2, H, W) flujo hacia ESTA imagen (F_{t→0} para I_0).
        """
        feats = []
        x = img
        for stage in self.stages:
            x = stage(x)  # baja ×2
            # Reducimos el flujo a la resolución del nivel y lo dividimos por 2
            # (desplazamientos en píxeles del nivel).  Como vamos nivel a
            # nivel, cada vez es ×0.5 respecto al anterior.
            flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
            feats.append(warp(x, flow))
        return feats


class RefineUNet(nn.Module):
    """UNet ligero que predice el residual de la imagen fusionada.

    Entrada (a resolución completa), 17 canales:
        I_0, I_1 (6) + warp(I_0), warp(I_1) (6) + M (1) + flujo (4)
    y en cada nivel del encoder concatenamos los contextos de ambas imágenes.

    Salida: (B, 3, H, W) residual sin activación; el wrapper aplica
    tanh (o clamp) para dejarlo en [-1, 1].
    """

    def __init__(self, c: int = 16):
        super().__init__()
        # Encoder.  Cada nivel recibe la salida anterior + contextos de I_0 e I_1
        # (2·c_k canales extra en el nivel k).
        self.down0 = nn.Sequential(conv_prelu(17, 2 * c, 3, 2, 1), conv_prelu(2 * c, 2 * c))
        self.down1 = nn.Sequential(conv_prelu(4 * c, 4 * c, 3, 2, 1), conv_prelu(4 * c, 4 * c))
        self.down2 = nn.Sequential(conv_prelu(8 * c, 8 * c, 3, 2, 1), conv_prelu(8 * c, 8 * c))
        self.down3 = nn.Sequential(conv_prelu(16 * c, 16 * c, 3, 2, 1), conv_prelu(16 * c, 16 * c))
        # Decoder con skip connections (concatenación).
        self.up0 = deconv_prelu(32 * c, 8 * c)
        self.up1 = deconv_prelu(16 * c, 4 * c)
        self.up2 = deconv_prelu(8 * c, 2 * c)
        self.up3 = deconv_prelu(4 * c, c)
        self.out = nn.Conv2d(c, 3, 3, 1, 1)

    def forward(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        warped0: torch.Tensor,
        warped1: torch.Tensor,
        mask: torch.Tensor,
        flow: torch.Tensor,
        ctx0: list[torch.Tensor],
        ctx1: list[torch.Tensor],
    ) -> torch.Tensor:
        x = torch.cat((img0, img1, warped0, warped1, mask, flow), dim=1)  # 17 ch
        s0 = self.down0(x)                                         # 1/2,  2c
        s1 = self.down1(torch.cat((s0, ctx0[0], ctx1[0]), dim=1))  # 1/4,  4c
        s2 = self.down2(torch.cat((s1, ctx0[1], ctx1[1]), dim=1))  # 1/8,  8c
        s3 = self.down3(torch.cat((s2, ctx0[2], ctx1[2]), dim=1))  # 1/16, 16c
        x = self.up0(torch.cat((s3, ctx0[3], ctx1[3]), dim=1))     # 1/8,  8c
        x = self.up1(torch.cat((x, s2), dim=1))                    # 1/4,  4c
        x = self.up2(torch.cat((x, s1), dim=1))                    # 1/2,  2c
        x = self.up3(torch.cat((x, s0), dim=1))                    # 1/1,  c
        return self.out(x)
