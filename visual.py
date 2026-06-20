import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

def plot_results(results, out_dir="fig"):
    os.makedirs(out_dir, exist_ok=True)
    labels = [f"({r['B']},{r['H']},{r['N']},{r['D']})" for r in results]
    x = range(len(results))

    # Forward bars
    plt.figure(figsize=(12, 6))
    plt.bar([i - 0.25 for i in x], [r["naive_fwd"] for r in results], width=0.25, label='Naive')
    plt.bar(x, [r["sdpa_fwd"] for r in results], width=0.25, label='PyTorch SDPA')
    plt.bar([i + 0.25 for i in x], [r["custom_fwd"] for r in results], width=0.25, label='My Flash (V5)')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('Time (ms)')
    plt.title('Forward Pass Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "forward_comparison.png"), dpi=150)
    plt.close()

    # Backward bars
    plt.figure(figsize=(12, 6))
    plt.bar([i - 0.25 for i in x], [r["naive_bwd"] for r in results], width=0.25, label='Naive')
    plt.bar(x, [r["sdpa_bwd"] for r in results], width=0.25, label='PyTorch SDPA')
    plt.bar([i + 0.25 for i in x], [r["custom_bwd"] for r in results], width=0.25, label='My Flash (V5)')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('Time (ms)')
    plt.title('Backward Pass Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "backward_comparison.png"), dpi=150)
    plt.close()

    groups = {}
    for r in results:
        key = (r["B"], r["H"], r["D"])
        groups.setdefault(key, []).append((r["N"], r))
    for (B, H, D), points in groups.items():
        points.sort(key=lambda x: x[0])
        Ns = [p[0] for p in points]
        naive_fwd = [p[1]["naive_fwd"] for p in points]
        sdpa_fwd  = [p[1]["sdpa_fwd"] for p in points]
        cust_fwd  = [p[1]["custom_fwd"] for p in points]
        naive_bwd = [p[1]["naive_bwd"] for p in points]
        sdpa_bwd  = [p[1]["sdpa_bwd"] for p in points]
        cust_bwd  = [p[1]["custom_bwd"] for p in points]

        # Forward line
        plt.figure(figsize=(10, 5))
        plt.plot(Ns, naive_fwd, 'o-', label='Naive')
        plt.plot(Ns, sdpa_fwd,  's-', label='PyTorch SDPA')
        plt.plot(Ns, cust_fwd,  '^-', label='My Flash (V5)')
        plt.xlabel('Sequence Length N')
        plt.ylabel('Forward Time (ms)')
        plt.title(f'Forward vs Sequence Length (B={B}, H={H}, D={D})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"forward_B{B}_H{H}_D{D}.png"), dpi=150)
        plt.close()

        # Backward line
        plt.figure(figsize=(10, 5))
        plt.plot(Ns, naive_bwd, 'o-', label='Naive')
        plt.plot(Ns, sdpa_bwd,  's-', label='PyTorch SDPA')
        plt.plot(Ns, cust_bwd,  '^-', label='My Flash (V5)')
        plt.xlabel('Sequence Length N')
        plt.ylabel('Backward Time (ms)')
        plt.title(f'Backward vs Sequence Length (B={B}, H={H}, D={D})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"backward_B{B}_H{H}_D{D}.png"), dpi=150)
        plt.close()
