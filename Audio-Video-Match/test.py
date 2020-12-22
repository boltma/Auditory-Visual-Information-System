import os
import torch
import pickle
from PIL import Image
import numpy as np
# files = os.listdir("dataset/train/061_foam_brick/1/mask/")
# c = np.array([])
# files.sort()
# for a in files:
#     print(a)
#     b = Image.open(os.path.join("dataset/train/061_foam_brick/1/mask/", a))
#     b = np.array(b)
#     print(b.shape)
#     if c.shape == (0,):
#         c = b
#     else:
#         c = np.dstack((c, b))

# print(c.shape)
a = np.load("dataset/train/green_basketball/2/mask/centroid.npy")
print(a)