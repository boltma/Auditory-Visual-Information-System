import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
from PIL import Image

def centroid(img):
	count = (img == 255).sum()
	x_center, y_center = np.argwhere(img == 255).sum(0) / count
	return x_center, y_center
