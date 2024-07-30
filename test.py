import torch

device = torch.device("cuda:6") if torch.cuda.is_available() else torch.device("cpu")
print(device)