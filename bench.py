import math, torch
from torch.nn.functional import scaled_dot_product_attention as torch_sdpa
from flash_attn import flash_attn_func, flash_attn_backward
from visual import plot_results

def naive_attention(q, k, v):
    scale = 1.0 / math.sqrt(q.size(-1))
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    return attn @ v

def benchmark(fn, warmup=10, rep=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep): fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep

def run_one_config(B, H, N, D, dtype=torch.float16):
    device = "cuda"
    q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype)

    # Naive
    q_na = q.detach().clone().requires_grad_(True)
    k_na = k.detach().clone().requires_grad_(True)
    v_na = v.detach().clone().requires_grad_(True)
    out_na = naive_attention(q_na, k_na, v_na)
    dout_na = torch.randn_like(out_na)
    t_na_fwd = benchmark(lambda: naive_attention(q_na, k_na, v_na))
    t_na_bwd = benchmark(lambda: out_na.backward(dout_na, retain_graph=True))

    # PyTorch SDPA
    q_sd = q.detach().clone().requires_grad_(True)
    k_sd = k.detach().clone().requires_grad_(True)
    v_sd = v.detach().clone().requires_grad_(True)
    out_sd = torch_sdpa(q_sd, k_sd, v_sd)
    dout_sd = torch.randn_like(out_sd)
    t_sd_fwd = benchmark(lambda: torch_sdpa(q_sd, k_sd, v_sd))
    t_sd_bwd = benchmark(lambda: out_sd.backward(dout_sd, retain_graph=True))

    # Our Flash
    q_cu = q.detach().clone()
    k_cu = k.detach().clone()
    v_cu = v.detach().clone()
    o_cu, lse_cu = flash_attn_func(q_cu, k_cu, v_cu)
    dout_cu = torch.randn_like(o_cu)
    t_cu_fwd = benchmark(lambda: flash_attn_func(q_cu, k_cu, v_cu))
    t_cu_bwd = benchmark(lambda: flash_attn_backward(q_cu, k_cu, v_cu, o_cu, dout_cu, lse_cu))

    print(f"B={B:2d} H={H:2d} N={N:5d} D={D:3d}  "
          f"Naive fwd:{t_na_fwd:7.2f}ms bwd:{t_na_bwd:7.2f}ms  "
          f"SDPA fwd:{t_sd_fwd:7.2f}ms bwd:{t_sd_bwd:7.2f}ms  "
          f"MyFlash fwd:{t_cu_fwd:7.2f}ms bwd:{t_cu_bwd:7.2f}ms")

    return {
        "B": B, "H": H, "N": N, "D": D,
        "naive_fwd": t_na_fwd, "naive_bwd": t_na_bwd,
        "sdpa_fwd": t_sd_fwd, "sdpa_bwd": t_sd_bwd,
        "custom_fwd": t_cu_fwd, "custom_bwd": t_cu_bwd
    }

if __name__ == "__main__":
    print("Running full benchmark suite...\n")
    results = [
        run_one_config(1, 1, 64, 32),
        run_one_config(2, 4, 128, 64),
        run_one_config(4, 8, 256, 64),
        run_one_config(4, 8, 512, 64),
        run_one_config(4, 8, 1024, 64),
        run_one_config(4, 8, 2048, 64),
        run_one_config(2, 4, 4096, 64),
        run_one_config(1, 1, 8192, 64),
    ]
    plot_results(results, out_dir="fig")
    print(f"\nPlots saved to fig/")
