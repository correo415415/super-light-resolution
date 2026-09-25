"""
Evaluación en el test set de Vimeo90K (PSNR / SSIM) + visualización de flujos.
===============================================================================

Uso:
    python evaluate.py --ckpt runs/rife/best.pth --data_root /path/vimeo_triplet
    python evaluate.py --ckpt runs/rife/best.pth --data_root ... --vis_dir outputs_vis --n_vis 16
    python evaluate.py --baseline_only --data_root ...      # sólo el baseline de promediar

Qué mide
--------
* PSNR y SSIM del frame interpolado vs im2.png, promediados sobre los 3782
  tríos de `tri_testlist.txt`, a resolución completa 448×256 (sin crop).
* Baseline "promediar frames": (I_0 + I_1)/2.  En Vimeo90K da ~20-23 dB.
  Cualquier modelo entrenado debe superarlo con claridad; RIFE completo
  llega a ~35 dB.
* Con `--vis_dir` guarda, para las primeras `n_vis` muestras:
    - `*_panel.png`: imágenes, flujos finales, máscara y error.
    - `*_blocks.png`: flujo/máscara/fusión tras CADA IFBlock (coarse → fine),
      para ver qué aporta cada nivel de la cascada.

Nota sobre PSNR: lo calculamos por imagen y promediamos los dB (convención
de los papers de VFI), no el PSNR del MSE promedio.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SyntheticTripletDataset, Vimeo90KTriplet
from model import psnr, ssim
from utils import _label, flow_to_color, load_model, make_debug_grid, tensor_to_uint8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--scales", type=float, nargs=3, default=(4, 2, 1))
    p.add_argument("--vis_dir", type=str, default=None)
    p.add_argument("--n_vis", type=int, default=8)
    p.add_argument("--baseline_only", action="store_true")
    p.add_argument("--out_json", type=str, default=None)
    p.add_argument(
        "--multi_t", action="store_true",
        help="(septuplet + RIFE-m) evaluar ×6: im1→im7 prediciendo im2..im6 con t=k/6; PSNR por t",
    )
    args = p.parse_args()
    if not args.synthetic and not args.data_root:
        p.error("indica --data_root o --synthetic")
    if not args.baseline_only and not args.ckpt:
        p.error("indica --ckpt (o --baseline_only)")
    return args


@torch.no_grad()
def visualize_block_flows(model, img0, img1, gt, out_path: Path, scales) -> None:
    """Panel con F_{t→0}, máscara, fusión y error tras cada IFBlock."""
    img0_p, _ = model.pad_to_multiple(img0, 32)
    img1_p, _ = model.pad_to_multiple(img1, 32)
    out = model.ifnet(img0_p, img1_p, gt=None, scales=tuple(scales))
    h, w = gt.shape[-2:]
    rows = []
    for k, (flow, mask, merged) in enumerate(zip(out["flows"], out["masks"], out["merged"])):
        merged = merged[:, :, :h, :w].float()
        fc = flow_to_color(flow[0, 0:2, :h, :w].float().cpu().numpy().transpose(1, 2, 0))
        m = np.repeat((mask[0, 0, :h, :w].float().cpu().numpy() * 255).astype(np.uint8)[..., None], 3, -1)
        mg = tensor_to_uint8(merged[0])
        err = tensor_to_uint8((merged[0] - gt[0]).abs().mean(0, keepdim=True).repeat(3, 1, 1) * 4)
        row = np.ascontiguousarray(np.concatenate([fc, m, mg, err], axis=1))
        _label(row, f"block{k} (1/{int(scales[k])})  flow | mask | merged | err   PSNR {psnr(merged, gt).item():.2f} dB", 5, 18)
        rows.append(row)
    cv2.imwrite(str(out_path), cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))


@torch.no_grad()
def eval_multi_t(model, ds: Vimeo90KTriplet, device, amp: bool, scales, batch_size: int) -> dict:
    """Evaluación multi-frame (×6) sobre septuplet: I0=im1, I1=im7, GT=im2..im6
    con t = 1/6..5/6.  Devuelve PSNR/SSIM por t y la media.

    Sirve para comprobar que RIFE-m realmente usa el canal t: un RIFE
    convertido sin fine-tune dará el mismo frame (t=0.5) para todos los t y
    su PSNR se hundirá en t=1/6 y 5/6.  También es la métrica de las
    tablas "×6 / multi-frame" de la literatura (gap de 6 frames = movimiento
    3× mayor que en el triplet, así que los números son más bajos).
    """
    if ds.n_frames != 7:
        raise SystemExit("--multi_t requiere el dataset septuplet")
    ts = [k / 6 for k in range(1, 6)]
    sums = {k: [0.0, 0.0] for k in range(1, 6)}
    sum_base = 0.0
    n = 0
    names = ds.samples
    for b in range(0, len(names), batch_size):
        idxs = range(b, min(len(names), b + batch_size))
        frames = [[_to_tensor(ds._load(i, 1, k, 7)[1]) for k in range(1, 8)] for i in idxs]
        stack = torch.stack([torch.stack(f) for f in frames]).to(device)  # (B,7,3,H,W)
        img0, img1 = stack[:, 0], stack[:, 6]
        for k, t in zip(range(1, 6), ts):
            gt = stack[:, k]
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                pred = model.inference(img0, img1, scales=tuple(scales), timestep=t).float()
            sums[k][0] += psnr(pred, gt).sum().item()
            sums[k][1] += ssim(pred, gt).sum().item()
        sum_base += psnr((img0 + img1) / 2, stack[:, 3]).sum().item()
        n += img0.shape[0]
    per_t = {f"t={k}/6": {"psnr": sums[k][0] / n, "ssim": sums[k][1] / n} for k in range(1, 6)}
    mean_psnr = sum(v["psnr"] for v in per_t.values()) / 5
    return {"n": n, "per_t": per_t, "mean_psnr": mean_psnr, "baseline_avg_psnr_t05": sum_base / n}


def _to_tensor(img_rgb) -> torch.Tensor:
    return torch.from_numpy(img_rgb.copy()).permute(2, 0, 1).float() / 255.0


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.amp and device.type == "cuda"

    if args.multi_t:
        ds = Vimeo90KTriplet(args.data_root, split="test", crop_size=None, augment=False, max_samples=args.max_samples)
        model = load_model(args.ckpt, device)
        if not model.arbitrary_time:
            print("[eval] AVISO: el checkpoint es RIFE clásico; se evaluará con t=0.5 fijo (los extremos saldrán mal).")
        res = eval_multi_t(model, ds, device, amp, args.scales, args.batch_size)
        print("=" * 64)
        print(f"[eval ×6] {res['n']} septuplets  (baseline promedio en t=0.5: {res['baseline_avg_psnr_t05']:.2f} dB)")
        for k, v in res["per_t"].items():
            print(f"  {k}: PSNR {v['psnr']:.2f} dB  SSIM {v['ssim']:.4f}")
        print(f"  media : PSNR {res['mean_psnr']:.2f} dB")
        if args.out_json:
            with open(args.out_json, "w") as f:
                json.dump({**res, "ckpt": args.ckpt}, f, indent=2)
            print(f"Guardado {args.out_json}")
        return

    if args.synthetic:
        ds = SyntheticTripletDataset(n=args.max_samples or 200, size=224, seed=1)
    else:
        ds = Vimeo90KTriplet(args.data_root, split="test", crop_size=None, augment=False, max_samples=args.max_samples)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"[eval] {len(ds)} tríos, device={device}")

    model = None if args.baseline_only else load_model(args.ckpt, device)
    vis_dir = Path(args.vis_dir) if args.vis_dir else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    sum_psnr = sum_ssim = sum_psnr_base = sum_ssim_base = 0.0
    n = n_vis_done = 0
    per_sample = []
    infer_time = 0.0
    t0 = time.time()

    for batch in loader:
        img0, img1, gt = (batch[k].to(device) for k in ("img0", "img1", "gt"))
        base = (img0 + img1) / 2
        sum_psnr_base += psnr(base, gt).sum().item()
        sum_ssim_base += ssim(base, gt).sum().item()

        if model is not None:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_inf = time.time()
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                pred, aux = model.inference(img0, img1, scales=tuple(args.scales), return_aux=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_time += time.time() - t_inf
            pred = pred.float()
            p, s = psnr(pred, gt), ssim(pred, gt)
            sum_psnr += p.sum().item()
            sum_ssim += s.sum().item()
            for i, name in enumerate(batch["name"]):
                per_sample.append({"name": name, "psnr": p[i].item(), "ssim": s[i].item()})

            if vis_dir and n_vis_done < args.n_vis:
                for i in range(img0.shape[0]):
                    if n_vis_done >= args.n_vis:
                        break
                    tag = f"{n_vis_done:03d}_" + batch["name"][i].replace("/", "_")
                    grid = make_debug_grid(img0[i], img1[i], gt[i], pred[i], {k: v[i] for k, v in aux.items()})
                    cv2.imwrite(str(vis_dir / f"{tag}_panel.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
                    visualize_block_flows(model, img0[i : i + 1], img1[i : i + 1], gt[i : i + 1],
                                          vis_dir / f"{tag}_blocks.png", args.scales)
                    n_vis_done += 1
        n += img0.shape[0]

    results = {"n": n, "baseline_avg_psnr": sum_psnr_base / n, "baseline_avg_ssim": sum_ssim_base / n}
    if model is not None:
        results.update(psnr=sum_psnr / n, ssim=sum_ssim / n, ms_per_frame=1000 * infer_time / n, ckpt=args.ckpt)

    print("=" * 64)
    print(f"Baseline (I0+I1)/2 : PSNR {results['baseline_avg_psnr']:.2f} dB  SSIM {results['baseline_avg_ssim']:.4f}")
    if model is not None:
        print(f"Modelo             : PSNR {results['psnr']:.2f} dB  SSIM {results['ssim']:.4f}  ({results['ms_per_frame']:.1f} ms/frame)")
        if results["psnr"] < results["baseline_avg_psnr"]:
            print("⚠  El modelo está por debajo del baseline → bug o entrenamiento insuficiente.")
        worst = sorted(per_sample, key=lambda d: d["psnr"])[:5]
        print("Peores 5 muestras  :", ", ".join(f"{d['name']} ({d['psnr']:.1f})" for d in worst))
    if vis_dir:
        print(f"Visualizaciones en {vis_dir}")
    print(f"Tiempo total {time.time() - t0:.1f}s")
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({**results, "per_sample": per_sample}, f, indent=2)
        print(f"Guardado {args.out_json}")


if __name__ == "__main__":
    main()
