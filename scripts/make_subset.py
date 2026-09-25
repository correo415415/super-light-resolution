"""
Crea un subset de Vimeo90K triplet (para smoke tests locales o para subir a Kaggle).

    python scripts/make_subset.py --src /data/vimeo_triplet --dst /data/vimeo_small --n_train 2000 --n_test 200

Copia sólo las carpetas de `sequences/` referenciadas y escribe listas recortadas,
manteniendo la misma estructura que el dataset completo (así train.py funciona igual).
"""
import argparse
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--n_train", type=int, default=2000)
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--symlink", action="store_true", help="enlaces simbólicos en vez de copiar")
    a = p.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for name, n in (("tri_trainlist.txt", a.n_train), ("tri_testlist.txt", a.n_test)):
        lines = [l.strip() for l in open(src / name) if l.strip()][:n]
        (dst / name).write_text("\n".join(lines) + "\n")
        for l in lines:
            s, d = src / "sequences" / l, dst / "sequences" / l
            if d.exists():
                continue
            d.parent.mkdir(parents=True, exist_ok=True)
            if a.symlink:
                d.symlink_to(s.resolve())
            else:
                shutil.copytree(s, d)
        print(f"{name}: {len(lines)} tríos")
    print(f"subset en {dst}")


if __name__ == "__main__":
    main()
