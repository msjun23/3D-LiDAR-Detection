import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

pc_data = np.load('my_data.npy')
print(pc_data)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d') # Axe3D object

x_data = pc_data[:, 0]
y_data = pc_data[:, 1]
z_data = pc_data[:, 2]

ax.scatter(x_data, y_data, z_data, 'o', s=0.1, label='test')

plt.show()