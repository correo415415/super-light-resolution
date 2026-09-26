"""
Verifica la integridad de un Vimeo90K (triplet o septuplet) antes de entrenar.
=============================================================================

Uso:
    python scripts/check_dataset.py --data_root "/kaggle/input/.../vimeo_septuplet" --workers 8
    python scripts/check_dataset.py --data_root ... --full          # además decodifica cada PNG (lento)
    python scripts/check_dataset.py --data_root ... --write_clean_lists out_dir   # listas sin las secuencias malas

Por defecto sólo comprueba existencia y tamaño > 0 de cada imK.png (rápido incluso
sobre el disco de red de Kaggle: ~640k stats).  Con --full decodifica con OpenCV
para cazar PNGs truncados ("libpng error: IDAT: CRC error"); tarda mucho más.

Los mirrors públicos de Kaggle a veces tienen ficheros corruptos o faltantes.  El
dataset de entrenamiento ya los tolera (sustituye la muestra), pero saber cuántos
hay te dice si el mirror es fiable o hay que buscar otro.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def check_seq(args):
    root, seq, n_frames, full = args
    d = root / "sequences" / seq
    bad = []
    for k in range(1, n_frames + 1):
        p = d / f"im{k}.png"
        try:
            if p.stat().st_size == 0:
                bad.append((p.name, "vacío"))
                continue
        except FileNotFoundError:
            bad.append((p.name, "falta"))
            continue
        if full:
            import cv2

            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                bad.append((p.name, "no decodifica"))
    return seq, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--full", action="store_true", help="decodificar cada PNG (lento)")
    ap.add_argument("--max_seqs", type=int, default=None, help="muestrear sólo N secuencias")
    ap.add_argument("--write_clean_lists", type=str, default=None, help="carpeta donde escribir *_trainlist/testlist limpios")
    a = ap.parse_args()

    root = Path(a.data_root)
    if (root / "sep_trainlist.txt").exists():
        prefix, n_frames = "sep", 7
    elif (root / "tri_trainlist.txt").exists():
        prefix, n_frames = "tri", 3
    else:
        sys.exit(f"no encuentro sep_/tri_trainlist.txt en {root}")
    print(f"[check] {prefix}: {n_frames} frames/secuencia  root={root}")

    splits = ["train", "test"] if a.split == "both" else [a.split]
    summary = {}
    for split in splits:
        lst = root / f"{prefix}_{split}list.txt"
        seqs = [line.strip() for line in open(lst) if line.strip()]
        if a.max_seqs:
            rng = np.random.default_rng(0)
            seqs = list(rng.choice(seqs, size=min(a.max_seqs, len(seqs)), replace=False))
        t0 = time.time()
        bad_seqs = {}
        with ThreadPoolExecutor(a.workers) as ex:
            for i, (seq, bad) in enumerate(ex.map(check_seq, ((root, s, n_frames, a.full) for s in seqs), chunksize=64)):
                if bad:
                    bad_seqs[seq] = bad
                if (i + 1) % 5000 == 0:
                    print(f"  {split}: {i+1}/{len(seqs)}  malas={len(bad_seqs)}  {time.time()-t0:.0f}s", flush=True)
        print(f"[check] {split}: {len(seqs)} secuencias, {len(bad_seqs)} con problemas ({time.time()-t0:.0f}s)")
        for seq, bad in list(bad_seqs.items())[:20]:
            print(f"    {seq}: {bad}")
        if len(bad_seqs) > 20:
            print(f"    ... y {len(bad_seqs)-20} más")
        summary[split] = (seqs, bad_seqs)

        if a.write_clean_lists:
            out = Path(a.write_clean_lists)
            out.mkdir(parents=True, exist_ok=True)
            clean = [s for s in seqs if s not in bad_seqs]
            (out / f"{prefix}_{split}list.txt").write_text("\n".join(clean) + "\n")
            print(f"[check] escrita lista limpia {out / f'{prefix}_{split}list.txt'} ({len(clean)} secuencias)")

    total_bad = sum(len(b) for _, b in summary.values())
    total = sum(len(s) for s, _ in summary.values())
    pct = 100 * total_bad / max(1, total)
    print(f"\n[check] TOTAL: {total_bad}/{total} secuencias con problemas ({pct:.3f} %)")
    if pct > 1:
        print("[check] ⚠ más del 1 % dañado: considera otro mirror o usa --write_clean_lists")
    elif total_bad:
        print("[check] OK: pocas secuencias dañadas; el dataset de entrenamiento las sustituye automáticamente")
    else:
        print("[check] OK: dataset íntegro")


if __name__ == "__main__":
    main()
