"""Genera un vimeo_septuplet sintético minúsculo (círculos en movimiento) para
smoke tests en CPU.  Uso: python scripts/make_fake_septuplet.py runs/fake_sep [n_train] [n_test]"""
import sys
from pathlib import Path

import cv2
import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/fake_sep")
n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 24
n_test = int(sys.argv[3]) if len(sys.argv) > 3 else 8
rng = np.random.default_rng(0)
H, W = 96, 128
lists = {"sep_trainlist.txt": [], "sep_testlist.txt": []}
for name, n in (("sep_trainlist.txt", n_train), ("sep_testlist.txt", n_test)):
    for i in range(n):
        seq = f"{i//10:05d}/{i%10:04d}"
        d = root / "sequences" / seq
        d.mkdir(parents=True, exist_ok=True)
        x0, y0 = rng.integers(20, W - 20), rng.integers(20, H - 20)
        vx, vy = rng.integers(-6, 7), rng.integers(-4, 5)
        bg = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
        bg = cv2.GaussianBlur(bg, (0, 0), 3)
        for k in range(7):
            img = bg.copy()
            cv2.circle(img, (int(x0 + vx * k), int(y0 + vy * k)), 12, (255, 0, 0), -1)
            cv2.imwrite(str(d / f"im{k+1}.png"), img)
        lists[name].append(seq)
for name, seqs in lists.items():
    (root / name).write_text("\n".join(seqs) + "\n")
print(f"fake septuplet en {root}: {n_train} train, {n_test} test")
