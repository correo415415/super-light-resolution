"""
Interpolación de vídeo ×2^k con RIFE.
=====================================

Uso:
    python inference.py --ckpt runs/rife/best.pth --input in.mp4 --output out.mp4            # 30→60 fps
    python inference.py --ckpt ... --input in.mp4 --output out.mp4 --exp 2                    # ×4
    python inference.py --ckpt ... --input in.mp4 --output out.mp4 --tile 512 --tile_overlap 64  # vídeo 4K
    python inference.py --ckpt ... --img0 a.png --img1 b.png --output mid.png                 # 2 imágenes

Cómo funciona ×2^k
------------------
El modelo sólo interpola t=0.5.  Para ×4 hacemos recursión: entre (A, B)
generamos M, y luego entre (A, M) y (M, B).  Con `--exp k` obtenemos 2^k−1
frames intermedios.  Cada nivel de recursión acumula un poco de error, así
que para ×8 o más, RIFE-m (con timestep arbitrario) sería preferible.

Tiling
------
Para resoluciones grandes (4K) la VRAM no alcanza (el IFBlock0 con c=240 a
resolución 1/4 sigue siendo grande).  Partimos cada par de frames en tiles
con solapamiento (overlap), interpolamos cada tile y recomponemos con un
"blending" lineal en la zona de solape para no ver costuras.

Trade-off: un tile pequeño limita el movimiento máximo que la red puede
resolver (un objeto que salta más de ~tile/4 px cae fuera del tile).  Para
4K conviene tile ≥ 1024 y, además, `--scales 8 4 2` para que la red vea
movimientos mayores.

Pipeline de vídeo
-----------------
Leemos frame a frame con OpenCV (BGR uint8 → RGB float [0,1]), interpolamos
en GPU y escribimos con VideoWriter.  El audio no se conserva (OpenCV no lo
maneja); para reinyectarlo:
    ffmpeg -i out.mp4 -i in.mp4 -c copy -map 0:v -map 1:a out_audio.mp4
Con `--fp16` la inferencia va ~2× más rápida en GPUs con tensor cores.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from utils import load_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--input", type=str, help="vídeo de entrada")
    p.add_argument("--img0", type=str, help="(modo imágenes) primer frame")
    p.add_argument("--img1", type=str, help="(modo imágenes) segundo frame")
    p.add_argument("--output", required=True)
    p.add_argument("--exp", type=int, default=1, help="factor 2^exp (1 → ×2)")
    p.add_argument("--scales", type=float, nargs=3, default=(4, 2, 1))
    p.add_argument("--tile", type=int, default=0, help="tamaño de tile (0 = sin tiling)")
    p.add_argument("--tile_overlap", type=int, default=64)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--fps", type=float, default=None, help="fps de salida (por defecto fps_in × 2^exp)")
    p.add_argument("--max_frames", type=int, default=None, help="procesar sólo los primeros N frames")
    p.add_argument("--codec", type=str, default="mp4v")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Inferencia de un par (con o sin tiling)
# ---------------------------------------------------------------------------
class Interpolator:
    def __init__(self, model, scales, tile: int, overlap: int, fp16: bool, device):
        self.model, self.scales, self.tile, self.overlap, self.fp16, self.device = model, tuple(scales), tile, overlap, fp16, device

    @torch.no_grad()
    def _run(self, img0: torch.Tensor, img1: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16 and self.device.type == "cuda"):
            return self.model.inference(img0, img1, scales=self.scales).float()

    @torch.no_grad()
    def middle(self, img0: torch.Tensor, img1: torch.Tensor) -> torch.Tensor:
        """Frame en t=0.5.  (1,3,H,W) → (1,3,H,W)."""
        if self.tile <= 0:
            return self._run(img0, img1)
        return self._tiled(img0, img1)

    def _tiled(self, img0: torch.Tensor, img1: torch.Tensor) -> torch.Tensor:
        _, _, H, W = img0.shape
        t, o = self.tile, self.overlap
        stride = t - o
        out = torch.zeros_like(img0)
        weight = torch.zeros(1, 1, H, W, device=img0.device)
        # Rampa lineal en los bordes del tile para fundir los solapes.
        ramp = torch.ones(t, device=img0.device)
        if o > 0:
            r = torch.linspace(0, 1, o, device=img0.device)
            ramp[:o] = r
            ramp[-o:] = torch.minimum(ramp[-o:], r.flip(0))
        w2d = (ramp[:, None] * ramp[None, :])[None, None]

        ys = list(range(0, max(H - t, 0) + 1, stride))
        xs = list(range(0, max(W - t, 0) + 1, stride))
        if ys[-1] + t < H:
            ys.append(H - t)
        if xs[-1] + t < W:
            xs.append(W - t)
        for y in ys:
            for x in xs:
                y0, x0 = max(0, y), max(0, x)
                y1, x1 = min(H, y0 + t), min(W, x0 + t)
                pred = self._run(img0[:, :, y0:y1, x0:x1], img1[:, :, y0:y1, x0:x1])
                wt = w2d[:, :, : y1 - y0, : x1 - x0]
                out[:, :, y0:y1, x0:x1] += pred * wt
                weight[:, :, y0:y1, x0:x1] += wt
        return out / weight.clamp_min(1e-6)

    @torch.no_grad()
    def between(self, img0: torch.Tensor, img1: torch.Tensor, exp: int) -> list[torch.Tensor]:
        """Devuelve los 2^exp − 1 frames intermedios en orden temporal (recursivo)."""
        if exp == 0:
            return []
        mid = self.middle(img0, img1)
        if exp == 1:
            return [mid]
        return self.between(img0, mid, exp - 1) + [mid] + self.between(mid, img1, exp - 1)


# ---------------------------------------------------------------------------
# E/S
# ---------------------------------------------------------------------------
def bgr_to_tensor(frame: np.ndarray, device) -> torch.Tensor:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).to(device).permute(2, 0, 1)[None].float().div_(255.0)


def tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
    rgb = (t[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def run_images(args, interp: Interpolator, device):
    img0 = bgr_to_tensor(cv2.imread(args.img0), device)
    img1 = bgr_to_tensor(cv2.imread(args.img1), device)
    mids = interp.between(img0, img1, args.exp)
    out = Path(args.output)
    if len(mids) == 1:
        cv2.imwrite(str(out), tensor_to_bgr(mids[0]))
        print(f"guardado {out}")
    else:
        out.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(mids):
            cv2.imwrite(str(out / f"mid_{i:03d}.png"), tensor_to_bgr(m))
        print(f"guardados {len(mids)} frames en {out}/")


def run_video(args, interp: Interpolator, device):
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(args.input)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    factor = 2**args.exp
    fps_out = args.fps or fps_in * factor
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*args.codec), fps_out, (W, H))
    print(f"[inference] {W}x{H} @ {fps_in:.2f}fps → {fps_out:.2f}fps  ({n_total} frames, ×{factor}, tile={args.tile or 'off'})")

    ok, prev = cap.read()
    if not ok:
        raise RuntimeError("vídeo vacío")
    prev_t = bgr_to_tensor(prev, device)
    writer.write(prev)
    n_in, n_out = 1, 1
    t0 = time.time()
    while True:
        ok, cur = cap.read()
        if not ok or (args.max_frames and n_in >= args.max_frames):
            break
        cur_t = bgr_to_tensor(cur, device)
        for mid in interp.between(prev_t, cur_t, args.exp):
            writer.write(tensor_to_bgr(mid))
            n_out += 1
        writer.write(cur)
        n_out += 1
        n_in += 1
        prev_t = cur_t
        if n_in % 50 == 0:
            el = time.time() - t0
            print(f"  {n_in}/{n_total} frames  {n_in / el:.1f} fps_in/s  ETA {(n_total - n_in) / max(n_in / el, 1e-6):.0f}s")
    cap.release()
    writer.release()
    print(f"[inference] listo: {n_in} → {n_out} frames en {time.time() - t0:.1f}s → {args.output}")
    print("  (sin audio; para copiarlo: ffmpeg -i out.mp4 -i in.mp4 -c copy -map 0:v -map 1:a out_audio.mp4)")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)
    interp = Interpolator(model, args.scales, args.tile, args.tile_overlap, args.fp16, device)
    if args.img0 and args.img1:
        run_images(args, interp, device)
    elif args.input:
        run_video(args, interp, device)
    else:
        raise SystemExit("indica --input (vídeo) o --img0/--img1 (imágenes)")


if __name__ == "__main__":
    main()
