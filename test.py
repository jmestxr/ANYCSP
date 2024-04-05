import torch
from glob import glob


print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(glob("data/test/GSET\\G1.mtx"))