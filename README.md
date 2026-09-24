# RIFE desde cero — Interpolación de frames de vídeo (VFI) educativa

Implementación propia en PyTorch de la arquitectura **RIFE** (Real-Time Intermediate
Flow Estimation, [arXiv:2011.06294](https://arxiv.org/abs/2011.06294), ECCV 2022),
escrita desde la especificación técnica, sin pesos preentrenados ni código del repositorio
oficial.  El objetivo es aprender: cada módulo lleva comentarios explicando el *por qué*.

> Estado: **en construcción**.  Ver el roadmap al final.

## Estructura

```
model/
  warplayer.py   backward warping diferenciable (grid_sample)
  ifnet.py       IFBlock + IFNet coarse-to-fine + bloque teacher
  refine.py      Contextnet + UNet de refinamiento
  loss.py        LapLoss, pérdida de destilación enmascarada, PSNR/SSIM
  rife.py        modelo completo: forward de entrenamiento e inferencia
tests/
  test_core.py   tests unitarios del núcleo
```

## Tests

```bash
pip install -r requirements.txt
python tests/test_core.py        # o: python -m pytest tests/ -v
```
