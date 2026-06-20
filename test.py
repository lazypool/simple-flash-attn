import torch
from flash_attn import flash_attn_func, flash_attn_backward

def manual_attention(q, k, v):
    import math
    scale = 1.0 / math.sqrt(q.size(-1))
    attn = (q @ k.transpose(-2, -1)) * scale
    mask = torch.tril(torch.ones(q.size(-2), q.size(-2), device=q.device))
    attn = attn.masked_fill(mask == 0, float('-inf'))
    attn = torch.softmax(attn, dim=-1)
    return attn @ v

def manual_lse(q, k):
    import math
    scale = 1.0 / math.sqrt(q.size(-1))
    s = (q @ k.transpose(-2, -1)) * scale
    mask = torch.tril(torch.ones(q.size(-2), q.size(-2), device=q.device))
    s = s.masked_fill(mask == 0, float('-inf'))
    m = s.max(dim=-1, keepdim=True)[0]
    return (m + torch.log(torch.exp(s - m).sum(dim=-1, keepdim=True))).squeeze(-1)

def test_forward():
    torch.manual_seed(42)
    B, H, N, D = 2, 4, 64, 64
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

    ref_o = manual_attention(q, k, v)               # fp16
    ref_l = manual_lse(q, k).float()                # kernel returns fp32, must cast

    o, l = flash_attn_func(q, k, v)
    assert torch.allclose(o, ref_o, atol=1e-2, rtol=1e-3), "Forward O mismatch"
    assert torch.allclose(l, ref_l, atol=1e-2, rtol=1e-3), "Forward LSE mismatch"
    print("Forward test passed.")

def test_backward():
    torch.manual_seed(123)
    B, H, N, D = 2, 4, 64, 64
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, requires_grad=True)

    ref_out = manual_attention(q, k, v)
    dout = torch.randn_like(ref_out)
    ref_out.backward(dout)
    dq_ref, dk_ref, dv_ref = q.grad.clone(), k.grad.clone(), v.grad.clone()

    q_cu = q.detach().clone().requires_grad_(False)
    k_cu = k.detach().clone().requires_grad_(False)
    v_cu = v.detach().clone().requires_grad_(False)
    o_cu, lse_cu = flash_attn_func(q_cu, k_cu, v_cu)
    dq_cu, dk_cu, dv_cu = flash_attn_backward(q_cu, k_cu, v_cu, o_cu, dout, lse_cu)

    assert torch.allclose(dq_cu, dq_ref, atol=1e-2, rtol=1e-3), "dQ mismatch"
    assert torch.allclose(dk_cu, dk_ref, atol=1e-2, rtol=1e-3), "dK mismatch"
    assert torch.allclose(dv_cu, dv_ref, atol=1e-2, rtol=1e-3), "dV mismatch"
    print("Backward test passed.")

if __name__ == "__main__":
    test_forward()
    test_backward()
