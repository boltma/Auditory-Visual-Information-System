import numpy as np
import matplotlib.pyplot as plt
x = np.load('loss_50.npy', allow_pickle=True)
y = np.zeros(80)
n = np.arange(80)
y = y/100
plt.plot(x[0:79])
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.savefig('loss_50.pdf')