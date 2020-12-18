import os
import torch
import pickle
from PIL import Image
import numpy as np
files = os.listdir("dataset/train/061_foam_brick/1/mask/")
c = np.array([])
for a in files:
    b = Image.open(os.path.join("dataset/train/061_foam_brick/1/mask/", a))
    b = np.array(b)
    print(b.shape)
    if c.shape == (0,):
        c = b
    else:
        c = np.dstack((c, b))

print(c.shape)