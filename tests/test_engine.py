import torch

from afrans.engine import clip_grad_norm_


def _make_grad(shape, dtype, device="cpu"):
    p = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    if torch.is_complex(p):
        (p.abs() ** 2).sum().backward()
    else:
        (p ** 2).sum().backward()
    return p


def test_clip_grad_norm_matches_torch_builtin_on_real_params():
    """No regression on the plain-real case: must agree with PyTorch's own
    implementation, not just avoid crashing."""
    torch.manual_seed(0)
    p1 = _make_grad((8, 8), torch.float32)
    p2 = _make_grad((4,), torch.float32)

    p1_ref = p1.detach().clone().requires_grad_(); p1_ref.grad = p1.grad.clone()
    p2_ref = p2.detach().clone().requires_grad_(); p2_ref.grad = p2.grad.clone()

    expected = torch.nn.utils.clip_grad_norm_([p1_ref, p2_ref], max_norm=1.0)
    actual = clip_grad_norm_([p1, p2], max_norm=1.0)

    assert torch.allclose(actual, expected, atol=1e-5)
    assert torch.allclose(p1.grad, p1_ref.grad, atol=1e-5)
    assert torch.allclose(p2.grad, p2_ref.grad, atol=1e-5)


def test_clip_grad_norm_handles_complex_and_matches_torch_builtin():
    """torch.nn.utils.clip_grad_norm_ calls torch.linalg.vector_norm
    directly on complex gradients, which MPS doesn't support (the FNO's
    spectral conv weights are complex64). This is the exact mixed
    real+complex case verified against PyTorch's own CPU implementation
    when the workaround was written -- pinned here so it can't regress."""
    torch.manual_seed(0)
    real_p = _make_grad((8, 8), torch.float32)
    cplx_p = _make_grad((4, 4), torch.complex64)

    real_ref = real_p.detach().clone().requires_grad_(); real_ref.grad = real_p.grad.clone()
    cplx_ref = cplx_p.detach().clone().requires_grad_(); cplx_ref.grad = cplx_p.grad.clone()

    expected = torch.nn.utils.clip_grad_norm_([real_ref, cplx_ref], max_norm=1.0)
    actual = clip_grad_norm_([real_p, cplx_p], max_norm=1.0)

    assert torch.allclose(actual, expected, atol=1e-5)
    assert torch.allclose(real_p.grad, real_ref.grad, atol=1e-5)
    assert torch.allclose(cplx_p.grad, cplx_ref.grad, atol=1e-5)


def test_clip_grad_norm_no_grads_returns_zero():
    p = torch.randn(4, requires_grad=True)  # never backpropped -- p.grad is None
    result = clip_grad_norm_([p], max_norm=1.0)
    assert result.item() == 0.0
