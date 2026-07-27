import torch

print("CUDA:", torch.cuda.is_available())
print("BF16 supported:", torch.cuda.is_bf16_supported())