"""
Tests del núcleo del modelo.  Ejecutar con:  python -m pytest tests/ -v
(o simplemente `python tests/test_core.py`).

Cada test verifica una propiedad que, si falla, indica un bug concreto:
  - warp con flujo cero = identidad          → convención de coordenadas
  - warp con flujo constante = traslación    → orden (x, y) y signo
  - IFBlock devuelve resolución completa     → factores de interpolación
  - forward completo produce shapes y loss finita → cableado de canales
  - gradientes llegan a todos los parámetros → nada desconectado del grafo
  - el teacher no recibe gradiente de L_distill → detach correcto
  - inference funciona con tamaños no múltiplos de 32 → padding
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import RIFE, IFBlock, IFNet, LapLoss, distillation_loss, psnr, ssim, warp  # noqa: E402

torch.manual_seed(0)


def test_warp_identity():
    img = torch.rand(2, 3, 16, 24)
    flow = torch.zeros(2, 2, 16, 24)
    out = warp(img, flow)
    # linspace introduce un error de ~1e-6 en las coordenadas normalizadas.
    assert torch.allclose(out, img, atol=1e-5), "warp con flujo 0 debe ser la identidad"


def test_warp_translation():
    """Flujo constante dx=+2: out[y, x] = img[y, x+2]  (backward warping)."""
    img = torch.rand(1, 1, 8, 12)
    flow = torch.zeros(1, 2, 8, 12)
    flow[:, 0] = 2.0  # dx
    out = warp(img, flow, padding_mode="zeros")
    # Comparamos en la zona interior (evitamos el borde donde x+2 se sale).
    assert torch.allclose(out[..., :, :-2], img[..., :, 2:], atol=1e-5)

    flow = torch.zeros(1, 2, 8, 12)
    flow[:, 1] = 1.0  # dy
    out = warp(img, flow, padding_mode="zeros")
    assert torch.allclose(out[..., :-1, :], img[..., 1:, :], atol=1e-5)


def test_warp_gradient_flows_to_flow():
    img = torch.rand(1, 3, 16, 16)
    flow = torch.zeros(1, 2, 16, 16, requires_grad=True)
    warp(img, flow).sum().backward()
    assert flow.grad is not None and torch.isfinite(flow.grad).all()


def test_ifblock_output_resolution():
    for scale, in_ch in ((4.0, 6), (2.0, 17), (1.0, 17)):
        block = IFBlock(in_ch, 32)
        # El bloque concatena el flujo (4 ch) internamente, así que la x
        # que le pasamos tiene in_ch - 4 canales.
        flow = None if in_ch == 6 else torch.zeros(1, 4, 64, 96)
        x = torch.rand(1, in_ch if flow is None else in_ch - 4, 64, 96)
        fd, md = block(x, flow, scale)
        assert fd.shape == (1, 4, 64, 96), (scale, fd.shape)
        assert md.shape == (1, 1, 64, 96), (scale, md.shape)


def test_ifnet_forward_shapes():
    net = IFNet(widths=(32, 24, 16))
    img0, img1, gt = (torch.rand(2, 3, 64, 64) for _ in range(3))
    out = net(img0, img1, gt=gt)
    assert len(out["flows"]) == 3 and all(f.shape == (2, 4, 64, 64) for f in out["flows"])
    assert all(m.shape == (2, 1, 64, 64) for m in out["masks"])
    assert all((m >= 0).all() and (m <= 1).all() for m in out["masks"]), "máscara debe estar en [0,1]"
    assert out["flow_teacher"].shape == (2, 4, 64, 64)
    assert out["merged_teacher"].shape == (2, 3, 64, 64)
    # Sin gt no hay teacher.
    out_inf = net(img0, img1, gt=None)
    assert "flow_teacher" not in out_inf


def test_rife_forward_loss_and_grads():
    model = RIFE(ifnet_widths=(32, 24, 16), refine_c=8)
    img0, img1, gt = (torch.rand(2, 3, 64, 64) for _ in range(3))
    out = model(img0, img1, gt)
    assert out["pred"].shape == (2, 3, 64, 64)
    assert (out["pred"] >= 0).all() and (out["pred"] <= 1).all()
    for k in ("loss", "loss_rec", "loss_rec_teacher", "loss_distill"):
        assert torch.isfinite(out[k]), k
    out["loss"].backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"parámetros sin gradiente: {missing}"


def test_teacher_receives_no_gradient_from_distillation():
    """L_distill sólo debe entrenar al estudiante."""
    model = RIFE(ifnet_widths=(32, 24, 16), refine_c=8)
    img0, img1, gt = (torch.rand(1, 3, 64, 64) for _ in range(3))
    out = model.ifnet(img0, img1, gt=gt)
    loss_d = distillation_loss(out["flows"], out["merged"], out["flow_teacher"], out["merged_teacher"], gt)
    loss_d.backward()
    teacher_grads = [p.grad for p in model.ifnet.teacher.parameters() if p.grad is not None]
    assert all(torch.count_nonzero(g) == 0 for g in teacher_grads), "el teacher recibió gradiente de L_distill"


def test_gradient_checkpointing_matches():
    torch.manual_seed(1)
    a = RIFE(ifnet_widths=(32, 24, 16), refine_c=8, use_checkpoint=False)
    b = RIFE(ifnet_widths=(32, 24, 16), refine_c=8, use_checkpoint=True)
    b.load_state_dict(a.state_dict())
    img0, img1, gt = (torch.rand(1, 3, 64, 64) for _ in range(3))
    la = a(img0, img1, gt)["loss"]
    lb = b(img0, img1, gt)["loss"]
    assert torch.allclose(la, lb, atol=1e-5)


def test_inference_arbitrary_size():
    model = RIFE(ifnet_widths=(32, 24, 16), refine_c=8).eval()
    img0, img1 = torch.rand(1, 3, 101, 187), torch.rand(1, 3, 101, 187)
    pred, aux = model.inference(img0, img1, return_aux=True)
    assert pred.shape == (1, 3, 101, 187)
    assert aux["flow"].shape == (1, 4, 101, 187)
    pred2 = model.inference(img0, img1, scales=(8.0, 4.0, 2.0))
    assert pred2.shape == (1, 3, 101, 187)


def test_laploss_zero_for_identical():
    lap = LapLoss(5)
    x = torch.rand(2, 3, 64, 64)
    assert lap(x, x).item() < 1e-6
    assert lap(x, torch.rand_like(x)).item() > 0


def test_metrics_sanity():
    x = torch.rand(2, 3, 64, 64)
    assert (psnr(x, x) > 90).all()
    assert torch.allclose(ssim(x, x), torch.ones(2), atol=1e-4)
    noisy = (x + 0.05 * torch.randn_like(x)).clamp(0, 1)
    assert (psnr(x, noisy) < psnr(x, x)).all()
    assert (ssim(x, noisy) < 1.0).all()


def test_parameter_count_full_model():
    """Sanity check del tamaño: RIFE completo ~10M parámetros en el estudiante."""
    model = RIFE()
    n_student = sum(p.numel() for p in model.student_parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"\nparámetros estudiante: {n_student/1e6:.2f}M  total (con teacher): {n_total/1e6:.2f}M")
    assert 5e6 < n_student < 20e6


if __name__ == "__main__":
    import inspect

    tests = [f for n, f in sorted(globals().items()) if n.startswith("test_") and inspect.isfunction(f)]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} tests pasaron")
