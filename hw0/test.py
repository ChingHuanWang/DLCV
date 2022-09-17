import torch 
import numpy as np
import os

# data = [[1, 2], [3, 4]]
# x_data = torch.tensor(data)

# # print(x_data)

# # # x_ones = torch.ones_like()

# # shape = (2, 3)
# # x_ones = torch.ones(shape)

# # print(x_ones)

# # print(f"shape = {x_data.shape}")
# # print(f"datatype = {x_data.dtype}")
# # print(f"device tensor = {x_data.device}")

# print(torch.cuda.is_available())

train_data_path = "/hw1_data/p1_data/train_50/*.png"

print(os.path.join(train_data_path))