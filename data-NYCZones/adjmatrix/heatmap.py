import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

with open("W_od_taxi.csv") as f:
    arr = np.loadtxt(f,delimiter=',')
arr = np.array(arr)
arr = np.delete(arr,0,0)
a = sns.heatmap(arr,center=0.75)
plt.title('The visualization of OD Matrix')
plt.show()

with open("W_adj_matrix.csv") as f:
    arr = np.loadtxt(f,delimiter=',')
arr = np.array(arr)
arr = np.delete(arr,0,0)
a = sns.heatmap(arr,center=3)
plt.title('The visualization of 0-1 Matrix')
plt.show()