#include <torch/extension.h>

// ---- V5 kernel declarations ----
std::vector<torch::Tensor> flash_forward_fp16_tc(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V);

std::vector<torch::Tensor> flash_backward_fp16_tc(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor O, torch::Tensor dO, torch::Tensor L);

// ---- wrappers ----
std::vector<torch::Tensor> forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V)
{
    auto orig_dtype = Q.scalar_type();          // remember input dtype
    Q = Q.to(torch::kFloat16);
    K = K.to(torch::kFloat16);
    V = V.to(torch::kFloat16);
    auto res = flash_forward_fp16_tc(Q, K, V);  // O (half), L (float)
    res[0] = res[0].to(orig_dtype);             // convert O back to original dtype
    return res;
}

std::vector<torch::Tensor> backward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor O, torch::Tensor dO, torch::Tensor L)
{
    auto orig_dtype = Q.scalar_type();
    Q  = Q.to(torch::kFloat16);
    K  = K.to(torch::kFloat16);
    V  = V.to(torch::kFloat16);
    O  = O.to(torch::kFloat16);
    dO = dO.to(torch::kFloat16);
    auto grads = flash_backward_fp16_tc(Q, K, V, O, dO, L);
    grads[0] = grads[0].to(orig_dtype);         // dQ, dK, dV back to original
    grads[1] = grads[1].to(orig_dtype);
    grads[2] = grads[2].to(orig_dtype);
    return grads;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &forward,  "FlashAttention forward (fp16 tensor core)");
    m.def("backward", &backward, "FlashAttention backward (fp16 tensor core)");
}
