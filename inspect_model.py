import torch

model = torch.jit.load('apps/android/model_standard.pt')
print(model.code)
