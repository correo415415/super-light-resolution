"""
Entrenamiento de RIFE.
======================

Uso (una GPU, p.ej. 3060 local, smoke test sintético):
    python train.py --synthetic --smoke --out_dir runs/smoke

Uso (una GPU, Vimeo90K subset):
    python train.py --data_root /path/vimeo_triplet --smoke --out_dir runs/smoke_vimeo

Uso (Kaggle 2×T4, DDP):
    torchrun --nproc_per_node=2 train.py --data_root /kaggle/input/vimeo90k/vimeo_triplet \
        --out_dir /kaggle/working/checkpoints --epochs 30 --batch_size 16 --amp

Reanudar:
    torchrun --nproc_per_node=2 train.py ... --resume /kaggle/input/mi-checkpoint/last.pth

Decisiones de diseño
--------------------
* **DDP (DistributedDataParallel)** en vez de DataParallel: cada proceso
  tiene su GPU, su copia del modelo y su shard de datos; sólo se sincronizan
  gradientes (all-reduce).  Es ~2× más eficiente que DataParallel en 2 GPUs
  y es el estándar.  `torchrun` lanza los procesos y define las variables
  de entorno RANK / LOCAL_RANK / WORLD_SIZE.
* **AMP (fp16 autocast + GradScaler)**: las T4 tienen tensor cores fp16 →
  ~2× velocidad y mitad de VRAM.  La LapLoss se calcula en fp32 internamente
  (ver loss.py).  El GradScaler evita underflow de gradientes pequeños.
  Nota: usamos fp16 y no bf16 porque las T4 (Turing) no soportan bf16.
* **Scheduler por PASO (no por época)**: warmup lineal 2000 pasos + coseno
  hasta lr_min.  Se implementa como función de `step` para que reanudar sea
  trivial (sólo guardamos el contador de pasos).
* **Checkpoint completo** = modelo + optimizador + scaler + step + epoch +
  args.  Guardamos `last.pth` cada N pasos y al final de cada época, y
  `best.pth` cuando mejora el PSNR de validación.  Con `--time_limit` el
  script se detiene limpiamente antes de que Kaggle mate la sesión.
* **Gradient clipping** (norma 1.0): red de seguridad contra picos de
  gradiente en los primeros pasos, cuando el flujo es basura.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from dataset import build_dataset
from model import RIFE, psnr, ssim


# ---------------------------------------------------------------------------
# Argumentos
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrenamiento RIFE desde cero")
    # Datos
    p.add_argument("--data_root", type=str, default=None, help="carpeta vimeo_triplet")
    p.add_argument("--synthetic", action="store_true", help="usar dataset sintético (sin datos reales)")
    p.add_argument("--crop_size", type=int, default=224)
    p.add_argument("--max_samples", type=int, default=None, help="limitar tríos de train (debug)")
    p.add_argument("--max_val_samples", type=int, default=None, help="limitar tríos de validación")
    p.add_argument("--num_workers", type=int, default=4)
    # Modelo
    p.add_argument("--refine_c", type=int, default=16, help="anchura base del Contextnet/UNet")
    p.add_argument(
        "--ifnet_widths", type=int, nargs=3, default=(240, 150, 90), metavar=("C0", "C1", "C2"),
        help="anchuras de los 3 IFBlocks (reducir sólo para depurar en CPU)",
    )
    p.add_argument("--grad_checkpoint", action="store_true", help="gradient checkpointing en IFBlocks")
    # Optimización
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=16, help="por GPU")
    p.add_argument("--lr", type=float, default=3e-4, help="LR base para world_size=4; se escala ×ws/4")
    p.add_argument("--lr_min", type=float, default=3e-6)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", help="mixed precision fp16")
    p.add_argument("--no_lr_scale", action="store_true", help="no escalar LR por world_size/4")
    # Checkpointing / logging
    p.add_argument("--out_dir", type=str, default="runs/rife")
    p.add_argument("--resume", type=str, default=None, help="ruta a last.pth (o carpeta con él)")
    p.add_argument("--save_every", type=int, default=1000, help="pasos entre guardados de last.pth")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_every_epochs", type=int, default=1)
    p.add_argument("--time_limit", type=float, default=None, help="horas; para limpio antes de agotar la sesión")
    p.add_argument("--seed", type=int, default=42)
    # Smoke test
    p.add_argument("--smoke", action="store_true", help="config reducida para validar el pipeline")
    args = p.parse_args()

    if args.smoke:
        # Overrides razonables para una 3060 (~5-10 min).
        args.epochs = min(args.epochs, 3)
        args.max_samples = args.max_samples or 512
        args.max_val_samples = args.max_val_samples or 64
        args.warmup_steps = min(args.warmup_steps, 50)
        args.save_every = min(args.save_every, 50)
        args.log_every = min(args.log_every, 10)
        args.batch_size = min(args.batch_size, 8)
    if not args.synthetic and args.data_root is None:
        p.error("indica --data_root o usa --synthetic")
    return args


# ---------------------------------------------------------------------------
# Distribuido
# ---------------------------------------------------------------------------
def setup_distributed() -> tuple[int, int, int]:
    """Devuelve (rank, local_rank, world_size).  Funciona también sin torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def is_main(rank: int) -> bool:
    return rank == 0


def reduce_mean(t: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1:
        t = t.clone()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= world_size
    return t


# ---------------------------------------------------------------------------
# Sampler reanudable
# ---------------------------------------------------------------------------
class ResumableSampler(Sampler):
    """Envuelve un DistributedSampler y permite saltar los primeros `skip`
    índices de la época actual.

    DistributedSampler es determinista dado (seed, epoch), así que al
    reanudar en mitad de una época podemos regenerar exactamente la misma
    permutación y descartar los índices ya consumidos.  Así no perdemos ni
    repetimos datos.  Lo usamos también con world_size=1 (rank 0 de 1) para
    que el comportamiento sea idéntico en local y en Kaggle.
    """

    def __init__(self, base: DistributedSampler):
        self.base = base
        self.skip = 0

    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)

    def __iter__(self):
        indices = list(self.base)
        skip, self.skip = self.skip, 0  # sólo se salta una vez
        return iter(indices[skip:])

    def __len__(self) -> int:
        return len(self.base) - self.skip


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------
def lr_at(step: int, total_steps: int, base_lr: float, min_lr: float, warmup: int) -> float:
    """Warmup lineal → coseno.  Puro en función de `step` → reanudable."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(1.0, progress)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(path: Path, model, optimizer, scaler, step, epoch, best_psnr, args):
    raw = model.module if isinstance(model, DDP) else model
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": raw.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": step,
            "epoch": epoch,
            "best_psnr": best_psnr,
            "args": vars(args),
        },
        tmp,
    )
    # Escritura atómica: si Kaggle mata el proceso a mitad del save, no
    # corrompemos el checkpoint anterior.
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, model, optimizer=None, scaler=None, device="cpu") -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / "last.pth"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw = model.module if isinstance(model, DDP) else model
    raw.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


# ---------------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, device, world_size: int, amp: bool) -> dict:
    raw = model.module if isinstance(model, DDP) else model
    raw.eval()
    tot_psnr = torch.zeros((), device=device)
    tot_ssim = torch.zeros((), device=device)
    tot_psnr_avg = torch.zeros((), device=device)  # baseline: promedio de frames
    n = torch.zeros((), device=device)
    for batch in loader:
        img0 = batch["img0"].to(device, non_blocking=True)
        img1 = batch["img1"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            pred = raw.inference(img0, img1)
        pred = pred.float()
        tot_psnr += psnr(pred, gt).sum()
        tot_ssim += ssim(pred, gt).sum()
        tot_psnr_avg += psnr((img0 + img1) / 2, gt).sum()
        n += img0.shape[0]
    if world_size > 1:
        for t in (tot_psnr, tot_ssim, tot_psnr_avg, n):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    raw.train()
    return {
        "psnr": (tot_psnr / n).item(),
        "ssim": (tot_ssim / n).item(),
        "psnr_avg_baseline": (tot_psnr_avg / n).item(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    amp = args.amp and device.type == "cuda"

    # Semillas distintas por rank para que la augmentación no sea idéntica.
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # tamaños fijos → cudnn elige el mejor kernel

    out_dir = Path(args.out_dir)
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[train] world_size={world_size} device={device} amp={amp}")
        print(f"[train] args: {vars(args)}")

    # ---------------- Datos ----------------
    train_ds = build_dataset(args, "train")
    val_ds = build_dataset(args, "test")
    train_sampler = ResumableSampler(
        DistributedSampler(train_ds, world_size, rank, shuffle=True, drop_last=True, seed=args.seed)
    )
    val_sampler = DistributedSampler(val_ds, world_size, rank, shuffle=False) if world_size > 1 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    steps_per_epoch = len(train_sampler) // args.batch_size  # drop_last
    total_steps = steps_per_epoch * args.epochs
    if is_main(rank):
        print(f"[train] train={len(train_ds)} val={len(val_ds)} steps/epoch={steps_per_epoch} total={total_steps}")

    # ---------------- Modelo ----------------
    model = RIFE(
        ifnet_widths=tuple(args.ifnet_widths), refine_c=args.refine_c, use_checkpoint=args.grad_checkpoint
    ).to(device)
    if is_main(rank):
        n_student = sum(p.numel() for p in model.student_parameters()) / 1e6
        print(f"[train] parámetros estudiante: {n_student:.2f}M")
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    # LR escalado linealmente con el nº de GPUs (referencia: 4 GPUs en el paper).
    base_lr = args.lr if args.no_lr_scale else args.lr * world_size / 4
    min_lr = args.lr_min if args.no_lr_scale else args.lr_min * world_size / 4
    # AdamW: weight decay DESACOPLADO del gradiente (a diferencia de Adam +
    # L2).  Con Adam clásico, la L2 se divide por sqrt(v) y pierde efecto en
    # pesos con gradientes grandes; AdamW aplica el decay directamente.
    # Ayuda a evitar que los pesos del flujo exploten (→ NaN).
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    # ---------------- Resume ----------------
    step, start_epoch, best_psnr = 0, 0, -1.0
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, scaler, device)
        step = ckpt["step"]
        start_epoch = ckpt["epoch"]
        best_psnr = ckpt.get("best_psnr", -1.0)
        if is_main(rank):
            print(f"[train] reanudado desde {args.resume}: step={step} epoch={start_epoch} best={best_psnr:.2f}")
        # Reanudación exacta: si el checkpoint es de mitad de época, saltamos
        # las muestras ya vistas de esa época (ver ResumableSampler).
        start_epoch = step // steps_per_epoch
        batches_done = step % steps_per_epoch
        if batches_done:
            train_sampler.skip = batches_done * args.batch_size
            if is_main(rank):
                print(f"[train] reanudando a mitad de época {start_epoch}: salto {batches_done} batches")

    writer = None
    if is_main(rank):
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(out_dir / "tb")
        except ImportError:
            print("[train] tensorboard no disponible; sin logging TB")

    t_start = time.time()
    time_limit_s = args.time_limit * 3600 if args.time_limit else None
    stop_early = False

    # ---------------- Bucle ----------------
    model.train()
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)  # shuffle distinto (y reproducible) por época
        t_epoch = time.time()
        for batch in train_loader:
            lr = lr_at(step, total_steps, base_lr, min_lr, args.warmup_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr

            img0 = batch["img0"].to(device, non_blocking=True)
            img1 = batch["img1"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                out = model(img0, img1, gt)
                loss = out["loss"]

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            step += 1

            if not torch.isfinite(loss):
                print(f"[rank {rank}] loss no finita en step {step}; abortando", file=sys.stderr)
                if world_size > 1:
                    dist.barrier()
                sys.exit(1)

            # ---- Logging ----
            if step % args.log_every == 0:
                stats = torch.stack([loss.detach(), out["loss_rec"].detach(), out["loss_rec_teacher"].detach(), out["loss_distill"].detach()])
                stats = reduce_mean(stats, world_size)
                if is_main(rank):
                    with torch.no_grad():
                        p = psnr(out["pred"].float(), gt).mean().item()
                    elapsed = time.time() - t_start
                    print(
                        f"ep {epoch} step {step}/{total_steps} loss {stats[0]:.4f} "
                        f"rec {stats[1]:.4f} tea {stats[2]:.4f} dist {stats[3]:.4f} "
                        f"psnr {p:.2f} lr {lr:.2e} gnorm {grad_norm:.2f} t {elapsed/60:.1f}m"
                    )
                    if writer:
                        writer.add_scalar("train/loss", stats[0], step)
                        writer.add_scalar("train/loss_rec", stats[1], step)
                        writer.add_scalar("train/loss_rec_teacher", stats[2], step)
                        writer.add_scalar("train/loss_distill", stats[3], step)
                        writer.add_scalar("train/psnr", p, step)
                        writer.add_scalar("train/lr", lr, step)
                        writer.add_scalar("train/grad_norm", grad_norm, step)

            # ---- Guardado periódico ----
            if step % args.save_every == 0 and is_main(rank):
                save_checkpoint(out_dir / "last.pth", model, optimizer, scaler, step, epoch, best_psnr, args)

            # ---- Límite de tiempo ----
            if time_limit_s and (time.time() - t_start) > time_limit_s:
                stop_early = True
                break

        # Sincronizamos la decisión de parar para que todos los ranks salgan juntos.
        if world_size > 1:
            flag = torch.tensor([int(stop_early)], device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            stop_early = bool(flag.item())

        if stop_early:
            if is_main(rank):
                print(f"[train] límite de tiempo alcanzado en step {step}; guardando y saliendo")
                save_checkpoint(out_dir / "last.pth", model, optimizer, scaler, step, epoch, best_psnr, args)
            break

        # ---- Fin de época: validación + checkpoint ----
        if (epoch + 1) % args.val_every_epochs == 0 or epoch + 1 == args.epochs:
            metrics = validate(model, val_loader, device, world_size, amp)
            if is_main(rank):
                print(
                    f"[val] epoch {epoch} psnr {metrics['psnr']:.2f} ssim {metrics['ssim']:.4f} "
                    f"(baseline promedio: {metrics['psnr_avg_baseline']:.2f} dB) "
                    f"epoch_time {(time.time()-t_epoch)/60:.1f}m"
                )
                if writer:
                    writer.add_scalar("val/psnr", metrics["psnr"], step)
                    writer.add_scalar("val/ssim", metrics["ssim"], step)
                    writer.add_scalar("val/psnr_avg_baseline", metrics["psnr_avg_baseline"], step)
                if metrics["psnr"] > best_psnr:
                    best_psnr = metrics["psnr"]
                    save_checkpoint(out_dir / "best.pth", model, optimizer, scaler, step, epoch + 1, best_psnr, args)
                    print(f"[val] nuevo mejor PSNR {best_psnr:.2f} → best.pth")
                # Sanity check del smoke test (ver README): si tras entrenar no
                # superamos el baseline de promediar frames, algo va mal.
                if args.smoke and epoch + 1 == args.epochs:
                    if metrics["psnr"] < metrics["psnr_avg_baseline"]:
                        print(
                            "[SMOKE ✗] El modelo está POR DEBAJO del baseline de promediar frames. "
                            "Revisa el pipeline (warp, signos del flujo, normalización de datos)."
                        )
                    else:
                        print(f"[SMOKE ✓] modelo {metrics['psnr']:.2f} dB > baseline {metrics['psnr_avg_baseline']:.2f} dB")
        if is_main(rank):
            save_checkpoint(out_dir / "last.pth", model, optimizer, scaler, step, epoch + 1, best_psnr, args)

    if is_main(rank):
        # Exportamos también pesos "de inferencia" (sin teacher, sin optimizador): más ligero.
        raw = model.module if isinstance(model, DDP) else model
        torch.save({"model": raw.export_inference_state_dict(), "args": vars(args)}, out_dir / "inference.pth")
        print(f"[train] fin.  best_psnr={best_psnr:.2f}  checkpoints en {out_dir}")
        if writer:
            writer.close()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
