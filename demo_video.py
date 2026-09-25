"""
Demo de interpolación sobre vídeos reales (YouTube o locales) con evaluación
sin ground-truth y comparativa visual.
==============================================================================

Uso básico (una URL de YouTube o un fichero local):

    python demo_video.py --ckpt runs/rife/inference.pth --source "https://youtu.be/XXXX" \
        --start 00:01:00 --duration 10 --height 720 --out_dir outputs/demo1

    python demo_video.py --ckpt ... --source mi_video.mp4 --exp 2 --out_dir outputs/demo2

Varias fuentes a la vez (batch), desde un fichero de texto (una por línea,
"url_o_ruta [start] [duration]"):

    python demo_video.py --ckpt ... --sources docs/demo_sources.txt --out_dir outputs/bench

Qué produce, para cada fuente, en `out_dir/<nombre>/`:

    input.mp4          clip original recortado (H = --height)
    interp_x2.mp4      resultado interpolado (×2^exp fps)
    sidebyside.mp4     original (izq) | interpolado (der), a la fps del interpolado,
                       el original repite frames → se ve la diferencia de fluidez
    slowmo.mp4         el interpolado reproducido a la fps del original (cámara lenta)
    dropframe_eval.json  métricas SIN ground-truth (ver abajo)
    worst_XX.png       paneles de los peores frames del drop-frame test

Evaluación sin ground-truth: "drop-frame test"
----------------------------------------------
No tenemos el frame intermedio real de un vídeo a 30 fps... pero SÍ lo
tenemos si tomamos el vídeo a 30 fps, nos quedamos con los frames pares
(15 fps efectivos) y pedimos al modelo que reconstruya los impares.  El PSNR
entre lo reconstruido y el frame impar real mide cómo se comporta el modelo
con un movimiento DOBLE del que verá en uso normal (por eso es una cota
pesimista) — pero permite comparar checkpoints y vídeos entre sí de forma
objetiva, y detectar los peores casos (escenas con cortes, movimiento
enorme, transparencias...).

Descarga de YouTube: usa `yt-dlp` (pip install yt-dlp).  Sólo úsalo con
contenido cuyo uso esté permitido (tus propios vídeos, Creative Commons,
o uso personal de evaluación).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from inference import Interpolator, bgr_to_tensor, tensor_to_bgr
from model import psnr, ssim
from utils import load_model, make_debug_grid


# ---------------------------------------------------------------------------
# Adquisición del clip
# ---------------------------------------------------------------------------
def _is_url(s: str) -> bool:
    return bool(re.match(r"^https?://", s))


def _slug(s: str) -> str:
    s = re.sub(r"^https?://(www\.)?", "", s)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return s[:60] or "clip"


def download_youtube(url: str, dst: Path, height: int) -> Path:
    """Descarga el mejor MP4 con altura <= height (sin audio, más rápido)."""
    if shutil.which("yt-dlp") is None:
        sys.exit("yt-dlp no está instalado: pip install yt-dlp")
    dst.mkdir(parents=True, exist_ok=True)
    out = dst / "source.%(ext)s"
    fmt = f"bestvideo[height<={height}][ext=mp4]/bestvideo[height<={height}]/best[height<={height}]"
    cmd = ["yt-dlp", "-f", fmt, "--no-playlist", "-o", str(out), url]
    print("[demo] descargando:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    files = sorted(dst.glob("source.*"))
    if not files:
        sys.exit("yt-dlp no produjo ningún fichero")
    return files[0]


def cut_clip(src: Path, dst: Path, start: str | None, duration: float | None, height: int) -> Path:
    """Recorta y reescala con ffmpeg a un MP4 con fps constante (necesario:
    los vídeos de YouTube pueden ser VFR y OpenCV se lía)."""
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg no está instalado")
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(src)]
    if duration:
        cmd += ["-t", str(duration)]
    # -2 = par más cercano manteniendo aspecto; -an sin audio; CFR forzado.
    cmd += ["-vf", f"scale=-2:{height}", "-an", "-vsync", "cfr", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(dst)]
    subprocess.run(cmd, check=True)
    return dst


def read_frames(path: Path, max_frames: int | None = None) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok or (max_frames and len(frames) >= max_frames):
            break
        frames.append(f)
    cap.release()
    return frames, fps


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    """Escribe con OpenCV (mp4v) y re-encodea a H.264 con ffmpeg si está
    disponible, para que se reproduzca en cualquier navegador."""
    h, w = frames[0].shape[:2]
    tmp = path.with_suffix(".tmp.mp4")
    wr = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        wr.write(f)
    wr.release()
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], check=True)
        tmp.unlink()
    else:
        tmp.rename(path)


# ---------------------------------------------------------------------------
# Detección de cortes de escena
# ---------------------------------------------------------------------------
def is_scene_cut(a: np.ndarray, b: np.ndarray, thresh: float = 0.35) -> bool:
    """Heurística barata: diferencia media de histogramas en escala reducida.
    Interpolar a través de un corte produce un "morph" horrible; lo correcto
    es DUPLICAR el frame.  Lossless Scaling y FlowFrames hacen lo mismo."""
    sa = cv2.resize(a, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    sb = cv2.resize(b, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return float(np.abs(sa - sb).mean()) > thresh


# ---------------------------------------------------------------------------
# Núcleo de la demo
# ---------------------------------------------------------------------------
def interpolate_frames(frames, interp: Interpolator, exp: int, device, scene_thresh: float, verbose=True):
    """Devuelve la lista de frames interpolados (BGR) y estadísticas."""
    out = [frames[0]]
    n_cuts = 0
    t0 = time.time()
    prev_t = bgr_to_tensor(frames[0], device)
    for i in range(1, len(frames)):
        cur = frames[i]
        cur_t = bgr_to_tensor(cur, device)
        if is_scene_cut(frames[i - 1], cur, scene_thresh):
            n_cuts += 1
            out.extend([frames[i - 1]] * (2**exp - 1))  # duplicar en vez de interpolar
        else:
            out.extend(tensor_to_bgr(m) for m in interp.between(prev_t, cur_t, exp))
        out.append(cur)
        prev_t = cur_t
        if verbose and i % 60 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(frames)-1} pares  {i/el:.1f} pares/s")
    el = time.time() - t0
    return out, {"pairs": len(frames) - 1, "scene_cuts": n_cuts, "seconds": el, "pairs_per_s": (len(frames) - 1) / max(el, 1e-6)}


@torch.no_grad()
def dropframe_eval(frames, interp: Interpolator, device, out_dir: Path, n_worst: int = 3, scene_thresh: float = 0.35):
    """Reconstruye los frames impares a partir de los pares y mide PSNR/SSIM."""
    results = []
    for i in range(1, len(frames) - 1, 2):
        # Saltamos tríos con corte de escena (en cualquiera de los dos huecos):
        # medirían el detector de cortes, no la interpolación.
        if is_scene_cut(frames[i - 1], frames[i], scene_thresh) or is_scene_cut(frames[i], frames[i + 1], scene_thresh):
            continue
        a = bgr_to_tensor(frames[i - 1], device)
        b = bgr_to_tensor(frames[i + 1], device)
        gt = bgr_to_tensor(frames[i], device)
        pred = interp.middle(a, b)
        base = (a + b) / 2
        results.append({
            "frame": i,
            "psnr": psnr(pred, gt).item(),
            "ssim": ssim(pred, gt).item(),
            "psnr_baseline": psnr(base, gt).item(),
        })
    if not results:
        return {"n": 0}
    ps = np.array([r["psnr"] for r in results])
    summary = {
        "n": len(results),
        "psnr_mean": float(ps.mean()),
        "psnr_p5": float(np.percentile(ps, 5)),
        "psnr_median": float(np.median(ps)),
        "ssim_mean": float(np.mean([r["ssim"] for r in results])),
        "psnr_baseline_mean": float(np.mean([r["psnr_baseline"] for r in results])),
        "worst": sorted(results, key=lambda r: r["psnr"])[:n_worst],
    }
    # Paneles de los peores casos (flujo, máscara, error) para entender el fallo.
    for k, r in enumerate(summary["worst"]):
        i = r["frame"]
        a = bgr_to_tensor(frames[i - 1], device)
        b = bgr_to_tensor(frames[i + 1], device)
        gt = bgr_to_tensor(frames[i], device)
        pred, aux = interp.model.inference(a, b, scales=interp.scales, return_aux=True)
        grid = make_debug_grid(a[0], b[0], gt[0], pred[0].float(), {kk: v[0] for kk, v in aux.items()})
        cv2.imwrite(str(out_dir / f"worst_{k:02d}_frame{i}_{r['psnr']:.1f}dB.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    return summary


def side_by_side(orig: list[np.ndarray], interp_frames: list[np.ndarray], factor: int) -> list[np.ndarray]:
    """Original (repitiendo cada frame `factor` veces) | interpolado."""
    out = []
    h, w = orig[0].shape[:2]
    bar = np.zeros((h, 4, 3), np.uint8)
    for j, f in enumerate(interp_frames):
        o = orig[min(j // factor, len(orig) - 1)]
        row = np.concatenate([o, bar, f], axis=1)
        cv2.putText(row, "original", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(row, "original", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(row, f"RIFE x{factor}", (w + 14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(row, f"RIFE x{factor}", (w + 14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        out.append(row)
    return out


def run_one(source: str, start, duration, args, interp: Interpolator, device) -> dict:
    name = _slug(source)
    out_dir = Path(args.out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n===== {source} → {out_dir}")

    src = download_youtube(source, out_dir / "raw", args.height) if _is_url(source) else Path(source)
    clip = cut_clip(src, out_dir / "input.mp4", start, duration, args.height)
    frames, fps = read_frames(clip, args.max_frames)
    if len(frames) < 3:
        print("  clip demasiado corto; saltando")
        return {"source": source, "error": "clip corto"}
    h, w = frames[0].shape[:2]
    print(f"  {len(frames)} frames @ {fps:.2f} fps, {w}x{h}")

    # 1) Interpolación ×2^exp
    interp_frames, stats = interpolate_frames(frames, interp, args.exp, device, args.scene_thresh)
    factor = 2**args.exp
    write_video(out_dir / f"interp_x{factor}.mp4", interp_frames, fps * factor)
    write_video(out_dir / "slowmo.mp4", interp_frames, fps)
    write_video(out_dir / "sidebyside.mp4", side_by_side(frames, interp_frames, factor), fps * factor)
    print(f"  interpolado: {stats['pairs_per_s']:.1f} pares/s, {stats['scene_cuts']} cortes de escena detectados")

    # 2) Drop-frame eval
    ev = dropframe_eval(frames, interp, device, out_dir, scene_thresh=args.scene_thresh) if not args.no_eval else {}
    if ev.get("n"):
        print(f"  drop-frame: PSNR {ev['psnr_mean']:.2f} dB (p5 {ev['psnr_p5']:.2f}, baseline {ev['psnr_baseline_mean']:.2f})  SSIM {ev['ssim_mean']:.4f}")
    result = {"source": source, "name": name, "fps_in": fps, "size": [w, h], "frames": len(frames), **stats, "dropframe": ev}
    with open(out_dir / "dropframe_eval.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def parse_sources(args) -> list[tuple[str, str | None, float | None]]:
    if args.sources:
        items = []
        for line in Path(args.sources).read_text().splitlines():
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.split()
            items.append((parts[0], parts[1] if len(parts) > 1 else args.start, float(parts[2]) if len(parts) > 2 else args.duration))
        return items
    if not args.source:
        sys.exit("indica --source o --sources")
    return [(args.source, args.start, args.duration)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--source", type=str, help="URL de YouTube o ruta local")
    p.add_argument("--sources", type=str, help="fichero con varias fuentes (una por línea)")
    p.add_argument("--start", type=str, default=None, help="inicio del recorte (hh:mm:ss)")
    p.add_argument("--duration", type=float, default=8.0, help="segundos del clip")
    p.add_argument("--height", type=int, default=720, help="altura de trabajo (360/540/720/1080)")
    p.add_argument("--exp", type=int, default=1, help="×2^exp")
    p.add_argument("--scales", type=float, nargs=3, default=(4, 2, 1))
    p.add_argument("--tile", type=int, default=0)
    p.add_argument("--tile_overlap", type=int, default=64)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--scene_thresh", type=float, default=0.35)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--no_eval", action="store_true")
    p.add_argument("--out_dir", type=str, default="outputs/demo")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)
    interp = Interpolator(model, args.scales, args.tile, args.tile_overlap, args.fp16, device)

    results = [run_one(src, st, du, args, interp, device) for src, st, du in parse_sources(args)]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n===== RESUMEN =====")
    print(f"{'clip':40s} {'fps':>5s} {'pares/s':>8s} {'PSNR':>6s} {'p5':>6s} {'base':>6s} {'SSIM':>6s}")
    for r in results:
        ev = r.get("dropframe", {})
        if ev.get("n"):
            print(f"{r['name'][:40]:40s} {r['fps_in']:5.1f} {r['pairs_per_s']:8.1f} {ev['psnr_mean']:6.2f} {ev['psnr_p5']:6.2f} {ev['psnr_baseline_mean']:6.2f} {ev['ssim_mean']:6.3f}")
        else:
            print(f"{r.get('name', r['source'])[:40]:40s}  {r.get('error', '')}")


if __name__ == "__main__":
    main()
