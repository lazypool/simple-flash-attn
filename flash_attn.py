import os
import torch
from torch.utils.cpp_extension import load

_src_dir = os.path.dirname(os.path.abspath(__file__))

_flash_attn_cuda = load(
    name="flash_attn_cuda",
    sources=[
        os.path.join(_src_dir, "bindings.cpp"),
        os.path.join(_src_dir, "flash_attn.cu"),
    ],
    extra_cuda_cflags=["-O2"],
    verbose=False,
)


def flash_attn_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """
    Return the forward output (O, LSE) where LSE = log-sum-exp per row.
    """
    return _flash_attn_cuda.forward(q.contiguous(), k.contiguous(), v.contiguous())


def flash_attn_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
):
    """
    Return gradients (dQ, dK, dV).
    """
    return _flash_attn_cuda.backward(
        q.contiguous(), k.contiguous(), v.contiguous(),
        o.contiguous(), dout.contiguous(), lse.contiguous(),
    )
